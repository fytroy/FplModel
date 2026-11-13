"""Replay 2025/26 with the whole system and compare it with real managers.

Usage:
    python simulate.py                  # full run: predictions (~20 min, cached) + 8 planner runs
    python simulate.py --quick          # GW1-6, default settings only (smoke test)
    python simulate.py --workers 4      # planner runs in parallel
Run pull_history.py and pull_benchmarks.py first.
"""
import argparse
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from fpl import benchmarks, matches, rules, simulate, storage
from fpl.settings import PlanSettings

SEASON = "2025-26"
DEFAULT = PlanSettings()
VARIANTS = [("default", DEFAULT),
            ("discount 0.8", PlanSettings(discount=0.8)), ("discount 0.9", PlanSettings(discount=0.9)),
            ("hit margin 0", PlanSettings(hit_margin=0.0)), ("hit margin 2", PlanSettings(hit_margin=2.0)),
            ("FT value 1", PlanSettings(ft_end_value=1.0)), ("FT value 2", PlanSettings(ft_end_value=2.0))]
FINAL_LABEL = "default, final fixtures"
ORDER = [label for label, _ in VARIANTS] + [FINAL_LABEL]
OPTIMISTIC = [
    "Fixture copies may miss changes made between copies.",
    "Player positions and the player list are end-of-season versions (mid-season signings "
    "are only buyable from their first match).",
    "The xP model's settings were chosen in part 2 partly on 2025/26 data.",
    "The planner settings are tested on the same season they are reported for (7 runs only).",
]
PESSIMISTIC = [
    "No injury news: the simulated manager can't see flags real managers saw.",
    "No press-conference news.",
    "Fixture copies (gaps up to 2-3 months) may lag what managers knew.",
]


def _run(job):
    label, data, xp, settings, *limit = job
    kwargs = {"time_limit": limit[0]} if limit and limit[0] else {}
    return label, simulate.run_season(data, xp, settings, log=lambda m: print(f"[{label}]{m}", flush=True),
                                      **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="GW1-6, default settings only")
    parser.add_argument("--runs", choices=["default", "variants", "all"], default="all",
                        help="default: the two default-settings runs (dated + final fixtures); "
                             "variants: the six settings variants on dated fixtures; all: everything. "
                             "Each run is saved, so a later --runs variants adds to an earlier --runs default")
    parser.add_argument("--only", nargs="+", metavar="RUN",
                        help="rerun just these runs by label, e.g. --only 'discount 0.95'")
    parser.add_argument("--discounts", nargs="+", type=float, metavar="D",
                        help="like-for-like discount check: one dated-fixture run per value, "
                             "labelled e.g. 'discount 0.85 (180s)'; use with --time-limit")
    parser.add_argument("--time-limit", type=int, default=None,
                        help="seconds per main solve (default: settings.SIM_TIME_LIMIT)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rebuild", action="store_true", help="recompute the cached predictions")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    bootstrap = storage.load_benchmark_json(f"bootstrap_{SEASON}.json")
    deadlines = {e["id"]: pd.Timestamp(e["deadline_time"]).as_unit("ns") for e in bootstrap["events"]}
    gws = list(range(1, 7)) if args.quick else list(range(1, 39))
    history, fixtures_all = matches.load_all_matches(), matches.load_all_fixtures()
    cache = storage.DATA_DIR / "processed" / "simulation_xp.csv"
    if cache.exists() and not args.rebuild:
        xp = storage.load_table("simulation_xp")
    else:
        print("Predicting every gameweek from its own deadline (about 20 minutes)...", flush=True)
        xp = simulate.build_predictions(history, fixtures_all, storage.load_snapshots(SEASON),
                                        storage.load_vaastav(SEASON, "teams.csv"), SEASON, deadlines,
                                        gws=range(1, 39))
        storage.save_table("simulation_xp", xp)
    data = simulate.SimData(season=SEASON, gws=gws, deadlines=deadlines,
                            chips=rules.chip_windows(bootstrap["chips"]),
                            players=simulate.season_players(history, SEASON),
                            actual=simulate.weekly_actual(history, SEASON))
    dated, final = xp[xp["variant"] == "dated"], xp[xp["variant"] == "final"]
    if args.quick:
        jobs = [("default", data, dated, DEFAULT), (FINAL_LABEL, data, final, DEFAULT)]
    else:
        jobs = []
        if args.runs in ("default", "all"):
            jobs += [("default", data, dated, DEFAULT), (FINAL_LABEL, data, final, DEFAULT)]
        if args.runs in ("variants", "all"):
            jobs += [(label, data, dated, s) for label, s in VARIANTS[1:]]
    if args.only:
        every = {"default": (dated, DEFAULT), FINAL_LABEL: (final, DEFAULT),
                 **{label: (dated, s) for label, s in VARIANTS[1:]}}
        unknown = [label for label in args.only if label not in every]
        if unknown:
            raise SystemExit(f"Unknown run(s) {unknown}; choose from {list(every)}")
        jobs = [(label, data, *every[label]) for label in args.only]
    if args.discounts:
        tag = f" ({args.time_limit}s)" if args.time_limit else ""
        jobs = [(f"discount {d:g}{tag}", data, dated, PlanSettings(discount=d)) for d in args.discounts]
    jobs = [(*job, args.time_limit) for job in jobs]
    results, failed = {}, {}
    with ProcessPoolExecutor(args.workers) as pool:
        futures = {pool.submit(_run, job): job[0] for job in jobs}
        for future in as_completed(futures):             # save each run the moment it finishes
            label = futures[future]
            try:
                results[label] = future.result()[1]
            except Exception as error:                   # one failed run must not lose the others
                failed[label] = error
                print(f"RUN FAILED [{label}]: {error!r}", flush=True)
                continue
            if not args.quick:
                storage.save_run(label, results[label])
    if failed:
        print(f"\nFailed runs (the rest are saved): {', '.join(failed)}")
    if not args.quick:
        results = storage.load_runs()                     # this run plus any earlier ones
    if "default" not in results:
        print("The default run failed; see above." if "default" in failed
              else "Saved. Run --runs default first to get the headline report.")
        sys.exit(1 if failed else 0)

    if args.quick:
        weeks = results["default"]
        per_week = (weeks["scan_seconds"] + weeks["solve_seconds"]).mean()
        def hours(runs):                                   # runs go in batches of `workers`
            return per_week * 38 * math.ceil(runs / args.workers) / 3600
        print("\nTiming per week (seconds):")
        print(weeks[["gw", "scan_seconds", "solve_seconds", "status"]].to_string(index=False))
        print(f"\nTiming: chip scan {weeks['scan_seconds'].mean():.0f}s, main solve "
              f"{weeks['solve_seconds'].mean():.0f}s per week on average.")
        print(f"Estimated full runs ({args.workers} workers): --runs default about {hours(2):.1f} hours; "
              f"--runs variants about {hours(len(VARIANTS) - 1):.1f} hours; --runs all about "
              f"{hours(len(VARIANTS) + 1):.1f} hours. (GW1-6 may not represent later weeks.)")

    average = benchmarks.average_manager(bootstrap["events"])
    sample = pd.read_csv(storage.benchmark_path(f"rank_sample_{SEASON}.csv"))
    user = benchmarks.USER_TOTALS[SEASON]
    report, all_weeks = simulate.build_report(results, ORDER, average, sample, user, data.chips)
    if not args.quick:
        storage.save_table("simulation_report", report)
        storage.save_table("simulation_weeks", all_weeks)

    main_run = report.iloc[0]
    print(f"\n2025/26 simulation (default settings, fixtures as known at each deadline):")
    print(f"  Season total:   {main_run.total}")
    print(f"  Average manager {main_run.average_total}  ({main_run.vs_average:+d})")
    if main_run.full_season:
        print(f"  Your 2025/26    {int(main_run.user_total)}  ({int(main_run.vs_user):+d})")
        print(f"  Estimated overall rank: {int(main_run['rank']):,} ({main_run.rank_note})")
    else:
        print(f"  ({main_run.rank_note}; no comparison with your 2025/26 total)")
    print(f"  Chips added {main_run.chip_points:.0f} pts; without them: {main_run.total_without_chips:.0f}")
    print(f"  Hits: {main_run.hits}, transfers: {main_run.transfers}, "
          f"captain was the best starter {main_run.captain_best_rate:.0%} of weeks")
    print("\nWhen each chip was played (a chip never played is wasted):")
    print(simulate.describe_chip_weeks(simulate.chip_weeks(results["default"], data.chips), data.chips))
    if args.quick:
        print("  (quick run covers GW1-6 only, so later chips show as not played)")
    print("\nAll runs:")
    print(report[["run", "total", "vs_average", "vs_user", "rank", "chip_points", "hits", "proven_best_rate"]]
          .to_string(index=False))
    print("\nWeek by week (default):")
    print(results["default"].to_string(index=False))
    print("\nWhat may make this result too GOOD:")
    print("\n".join(f"  - {line}" for line in OPTIMISTIC))
    print("What may make it too BAD:")
    print("\n".join(f"  - {line}" for line in PESSIMISTIC))
    print("\nSettings rule: keep the defaults unless another value wins by more than ~30 points "
          "AND its neighbouring values also do well.")

    if failed:
        sys.exit(1)                                       # some runs failed: say so to scripts and monitors


if __name__ == "__main__":
    main()
