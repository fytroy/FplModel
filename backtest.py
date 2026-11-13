"""Walk-forward backtest of the xP model against model B and a last-5 baseline.

Usage:
    python backtest.py                                    # 2025/26 GW1-38 + 2026/27 check (20-45 min)
    python backtest.py --season 2025-26 --gws 1-5 --horizon 1   # quick run
"""
import argparse

import pandas as pd

from fpl import backtest, matches, storage
from fpl.seasons import CURRENT_SEASON


def parse_gws(text: str) -> list[int]:
    """'1-38' -> [1, ..., 38]; '7' -> [7]."""
    first, _, last = text.partition("-")
    return list(range(int(first), int(last or first) + 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--season", help="one season to test, e.g. 2025-26")
    parser.add_argument("--gws", default="1-38", help="gameweeks with --season, e.g. 1-10")
    parser.add_argument("--horizon", type=int, default=6, help="weeks ahead to score (1-6)")
    args = parser.parse_args()

    history = matches.load_all_matches()
    fixtures_all = matches.load_all_fixtures()
    if args.season:
        runs = [(args.season, parse_gws(args.gws))]
    else:
        current = [g for g in matches.finished_gws(fixtures_all, CURRENT_SEASON) if g >= 2]
        runs = [("2025-26", list(range(1, 39)))]
        if current:
            runs.append((CURRENT_SEASON, current))
        else:
            print(f"No {CURRENT_SEASON} gameweek from GW2 on has finished yet: "
                  "skipping the current-season sanity run.")

    preds, summaries, calibrations = [], [], []
    for season, gws in runs:
        print(f"Backtesting {season}, GW{gws[0]}-{gws[-1]}...", flush=True)
        p = backtest.walk_forward(history, fixtures_all, season, gws, args.horizon)
        preds.append(p)
        summaries.append(backtest.summarize(p).assign(season=season))
        calibrations.append(backtest.calibration(p).assign(season=season))

    summary = pd.concat(summaries, ignore_index=True)
    storage.save_table("backtest_predictions", pd.concat(preds, ignore_index=True))
    storage.save_table("backtest_summary", summary)
    storage.save_table("backtest_calibration", pd.concat(calibrations, ignore_index=True))

    print()
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    first = summary[summary["horizon"] == 1].set_index(["season", "method"])
    for season in first.index.get_level_values("season").unique():
        a, base = first.loc[(season, "A (parts)")], first.loc[(season, "Baseline (last 5)")]
        print(f"\n{season}, next gameweek: A beats baseline on MAE: "
              f"{'yes' if a['mae'] < base['mae'] else 'NO'}; "
              f"on top-{backtest.TOP_N} points: "
              f"{'yes' if a[f'top{backtest.TOP_N}_points'] > base[f'top{backtest.TOP_N}_points'] else 'NO'}")

    for p in preds:
        season = p["season"].iloc[0]
        cs = backtest.clean_sheet_report(p)
        print(f"\n{season} clean sheets: P(CS) 5th-95th percentile {cs['p5']:.2f}-{cs['p95']:.2f}, "
              f"log loss {cs['log_loss']:.4f}, actual rate by predicted fifth {cs['actual_by_fifth']}")


if __name__ == "__main__":
    main()
