"""Replay a past season with the whole system: predict -> plan -> play -> score, week by week.

At each deadline the simulated manager knows only: matches already played,
prices that week, the newest fixture-list copy saved before the deadline, and
no injury news (none was stored). See the part 3 spec §7.
"""
import time
from dataclasses import dataclass

import pandas as pd

from fpl import benchmarks, chips, matches, models, optimizer, rules
from fpl.settings import CHIP_SCAN_WEEKS, SIM_TIME_LIMIT, PlanSettings


@dataclass
class SimData:
    season: str
    gws: list[int]
    deadlines: dict                 # gw -> UTC deadline
    chips: list                     # rules.Chip windows for the season
    players: pd.DataFrame           # season_players()
    actual: pd.DataFrame            # weekly_actual()


def season_players(history: pd.DataFrame, season: str) -> pd.DataFrame:
    """Who could be bought each gameweek, and at what price (tenths of £1m).

    The price is his price at the gameweek's first match (in a double gameweek it
    can rise before the second): the closest to the deadline price. A player counts
    from his first match row; in a gameweek his team blanks, his previous
    gameweek's row is used.
    """
    h = history[history["season"] == season].sort_values("kickoff_time")
    rows = h.groupby(["gw", "player_code"], as_index=False).first()
    frames = []
    for gw in range(1, int(h["gw"].max()) + 1):
        now = rows[rows["gw"] == gw]
        prev = rows[(rows["gw"] == gw - 1) & ~rows["player_code"].isin(now["player_code"])]
        frames.append(pd.concat([now, prev]).assign(gw=gw))
    out = pd.concat(frames, ignore_index=True)
    out["price"] = (out["price"] * 10).round().astype(int)
    return out[["gw", "player_code", "player_name", "position", "team", "price"]]


def weekly_actual(history: pd.DataFrame, season: str) -> pd.DataFrame:
    """Real points and minutes per player per gameweek (double gameweeks summed)."""
    h = history[history["season"] == season]
    return h.groupby(["gw", "player_code"], as_index=False).agg(
        points=("total_points", "sum"), minutes=("minutes", "sum"))


def snapshot_for(deadline: pd.Timestamp, snapshots: list):
    """The newest saved fixture list from before `deadline`."""
    before = [table for stamp, table in snapshots if stamp < deadline]
    if not before:
        raise ValueError(f"no fixture snapshot from before {deadline}")
    return before[-1]


def build_predictions(history, fixtures_all, snapshots, teams, season, deadlines,
                      gws=range(1, 39), weeks=CHIP_SCAN_WEEKS, train=models.train_and_predict,
                      log=print) -> pd.DataFrame:
    """xP per player per gameweek as seen at each deadline, for both fixture versions.

    'dated': gameweeks from the newest fixture copy before the deadline;
    'final': gameweeks from the final fixture list. Each fixture is predicted
    once (a fixture's prediction doesn't depend on which gameweek it's in).
    """
    players = season_players(history, season)
    cutoffs = matches.gw_cutoffs(fixtures_all)
    final = fixtures_all[fixtures_all["season"] == season]
    out = []
    for g in gws:
        deadline = pd.Timestamp(deadlines[g]).as_unit("ns")      # same clock unit as the match data
        past = history[history["kickoff_time"] < deadline]
        targets = [t for t in range(g, g + weeks) if t <= 38]
        dated = matches.fixture_list(season, snapshot_for(deadline, snapshots), teams)
        dated = dated[dated["gw"].isin(targets)]
        fin = final[final["gw"].isin(targets)]
        fx = pd.concat([dated, fin], ignore_index=True).drop_duplicates(["fixture_id", "team"])
        team_rows = fx[matches.TEAM_ROW_COLUMNS].assign(asof=deadline)
        reg = players[players["gw"] == g].drop(columns="gw").assign(price=lambda d: d["price"] / 10)
        player_rows = (fx[matches.TEAM_ROW_COLUMNS].merge(reg, on="team")
                       [matches.PLAYER_ROW_COLUMNS].assign(asof=deadline))
        pred = train(past, player_rows, team_rows, cutoffs, yardstick=False)
        for variant, table in (("dated", dated), ("final", fin)):
            gw_of = table.drop_duplicates("fixture_id").set_index("fixture_id")["gw"]
            p = pred.assign(gw=pred["fixture_id"].map(gw_of)).dropna(subset=["gw"])
            out.append(p.groupby(["gw", "player_code"], as_index=False)["xp"].sum()
                       .assign(variant=variant, origin_gw=g))
        log(f"  predicted from GW{g} ({len(past):,} past rows)")
    result = pd.concat(out, ignore_index=True)
    result["gw"] = result["gw"].astype(int)
    return result


def _picks(week, positions, actual_g) -> pd.DataFrame:
    order = list(week.starters) + list(week.bench)
    return pd.DataFrame({
        "position": positions.reindex(order).to_numpy(),
        "points": actual_g["points"].reindex(order).fillna(0).to_numpy(),
        "minutes": actual_g["minutes"].reindex(order).fillna(0).to_numpy(),
        "is_captain": [p == week.captain for p in order],
        "is_vice": [p == week.vice for p in order]})


def _normal_week(squad, xp_g, positions):
    """What the owned squad would have fielded (best XI by xP) – the Free Hit comparison."""
    xi = chips.best_xi(xp_g.reindex(squad).fillna(0), positions)
    bench = [p for p in squad if p not in xi]
    bench = sorted(bench, key=lambda p: (positions[p] != "GK", -xp_g.get(p, 0)))
    ranked = sorted(xi, key=lambda p: -xp_g.get(p, 0))
    return optimizer.WeekPlan(gw=0, chip=None, transfers_in=[], transfers_out=[], free_transfers=0,
                              paid=0, owned=squad, squad=squad, starters=xi, bench=bench,
                              captain=ranked[0], vice=ranked[1], xp=0.0, bank=0)


def _solve_with_retry(plan_input, settings, time_limit, gw, log):
    """Solve; if no plan is found in time, try once more with double the time."""
    try:
        return optimizer.solve(plan_input, settings, time_limit=time_limit)
    except ValueError as first:
        log(f"  GW{gw}: no plan within {time_limit}s ({first}); retrying with {2 * time_limit}s")
        try:
            return optimizer.solve(plan_input, settings, time_limit=2 * time_limit)
        except ValueError as second:
            raise ValueError(f"GW{gw}: the planner found no plan even with {2 * time_limit}s: {second}") from second


def run_season(data: SimData, xp: pd.DataFrame, settings: PlanSettings = PlanSettings(),
               time_limit: int = SIM_TIME_LIMIT, log=print) -> pd.DataFrame:
    """Manage a team from the first to the last gameweek; one result row per week."""
    squad, purchase, bank, free = [], {}, rules.BUDGET, 0
    chips_left = list(data.chips)
    rows = []
    for g in data.gws:
        known = data.players[data.players["gw"] <= g].groupby("player_code").last()
        registered = set(data.players.loc[data.players["gw"] == g, "player_code"])
        info = known.loc[sorted(registered | set(squad)), ["player_name", "position", "team", "price"]]
        info = info.rename(columns={"player_name": "name"})
        info["sell"] = [rules.selling_price(purchase[p], info.at[p, "price"]) if p in purchase
                        else info.at[p, "price"] for p in info.index]
        horizon_end = min(int(xp["gw"].max()), g + CHIP_SCAN_WEEKS - 1)
        week_xp = (xp[xp["origin_gw"] == g]
                   .pivot_table(index="player_code", columns="gw", values="xp", aggfunc="sum")
                   .reindex(index=info.index, columns=range(g, horizon_end + 1)).fillna(0.0))
        plan_gws = list(week_xp.columns[:settings.horizon])
        keep = optimizer.candidate_pool(week_xp[plan_gws], info, squad, settings.discount)
        scan_squad = squad
        if not squad:
            first = _solve_with_retry(optimizer.PlanInput(info.loc[keep], week_xp.loc[keep, plan_gws], [],
                                                          bank, 0, [], data.season), settings,
                                      time_limit, g, log)
            scan_squad = first.weeks[-1].owned
        started = time.perf_counter()
        holds = chips.hold_values(
            chips.scan(week_xp, info, scan_squad, bank, chips_left, plan_gws[-1], data.season),
            chips_left)
        scanned = time.perf_counter()
        plan_input = optimizer.PlanInput(info.loc[keep], week_xp.loc[keep, plan_gws], squad,
                                         bank, free, chips_left, data.season, holds)
        plan = _solve_with_retry(plan_input, settings, time_limit, g, log)
        solved = time.perf_counter()
        week = plan.weeks[0]
        for p in week.transfers_out:
            bank += int(info.at[p, "sell"])
            purchase.pop(p, None)
        for p in week.transfers_in:
            bank -= int(info.at[p, "price"])
            purchase[p] = int(info.at[p, "price"])
        before = squad
        squad = week.owned
        if week.chip:
            chips_left = rules.chips_left(chips_left, [(week.chip, g)])
        actual_g = data.actual[data.actual["gw"] == g].set_index("player_code")
        positions = info["position"]
        points = rules.score_week(_picks(week, positions, actual_g), week.chip, week.paid)
        chip_gain = None
        if week.chip in ("bboost", "3xc"):
            chip_gain = points - rules.score_week(_picks(week, positions, actual_g), None, week.paid)
        elif week.chip == "freehit":
            normal = _normal_week(before, week_xp[g], positions)
            chip_gain = points - rules.score_week(_picks(normal, positions, actual_g), None, 0)
        starters_pts = actual_g["points"].reindex(week.starters).fillna(0)
        rows.append({
            "gw": g, "points": points, "xp": round(week.xp, 2), "transfers": len(week.transfers_in),
            "hits": week.paid, "chip": week.chip or "", "chip_gain": chip_gain,
            "captain": info.at[week.captain, "name"],
            "captain_points": int(actual_g["points"].get(week.captain, 0)),
            "captain_best": bool(actual_g["points"].get(week.captain, 0) >= starters_pts.max()),
            "bank": bank, "status": plan.status,
            "scan_seconds": round(scanned - started, 1), "solve_seconds": round(solved - scanned, 1)})
        free = 1 if week.free_transfers is None else rules.next_free_transfers(
            week.free_transfers, len(week.transfers_in), week.chip)
        log(f"  GW{g}: {points} pts ({week.chip or 'no chip'}, {len(week.transfers_in)} transfers, "
            f"{week.paid} hits); chip scan {scanned - started:.0f}s, main solve {solved - scanned:.0f}s")
    return pd.DataFrame(rows)


def chip_weeks(weeks: pd.DataFrame, windows: list) -> dict:
    """Gameweek each chip copy was played (key like 'wildcard_2'), or None if never played."""
    left, out = list(windows), {c.key: None for c in windows}
    for gw, name in zip(weeks["gw"], weeks["chip"]):
        if not isinstance(name, str) or not name:      # '' or NaN (a saved run read back from CSV)
            continue
        used = next((c for c in left if c.name == name and c.allows(int(gw))), None)
        if used is not None:
            left.remove(used)
            out[used.key] = int(gw)
    return out


CHIP_LABELS = {"wildcard": "Wildcard", "freehit": "Free Hit", "bboost": "Bench Boost", "3xc": "Triple Captain"}


def describe_chip_weeks(played: dict, windows: list) -> str:
    """One line per window: each chip's gameweek, flagging chips never played."""
    lines = []
    for stop in sorted({c.stop for c in windows}):
        parts = []
        for c in (c for c in windows if c.stop == stop):
            label = CHIP_LABELS.get(c.name, c.name)
            gw = played.get(c.key)
            parts.append(f"{label} GW{gw}" if gw is not None else f"{label} NEVER PLAYED (wasted)")
        lines.append(f"  window ends GW{stop}: " + ", ".join(parts))
    return "\n".join(lines)


SEASON_GWS = 38


def summarize(weeks: pd.DataFrame, average: pd.DataFrame, rank_sample: pd.DataFrame,
              user_total: int, chips: list = ()) -> dict:
    total = int(weeks["points"].sum())
    average_total = int(average[average["gw"].isin(weeks["gw"])]["average"].sum())
    chip_points = float(pd.to_numeric(weeks["chip_gain"], errors="coerce").fillna(0).sum())
    full = set(weeks["gw"]) == set(range(1, SEASON_GWS + 1))
    if full:
        rank, note = benchmarks.estimate_rank(total, rank_sample)
    else:                       # a rank or a season-total comparison only means something for GW1-38
        rank, user_total = None, None
        note = f"partial season (GW{int(weeks['gw'].min())}-{int(weeks['gw'].max())}): no rank estimate"
    return {"total": total, "average_total": average_total, "vs_average": total - average_total,
            "full_season": full, "user_total": user_total,
            "vs_user": None if user_total is None else total - user_total, "rank": rank, "rank_note": note,
            "chip_points": chip_points, "total_without_chips": total - chip_points,
            "hits": int(weeks["hits"].sum()), "transfers": int(weeks["transfers"].sum()),
            "captain_best_rate": float(weeks["captain_best"].mean()),
            # share of weeks where the solver proved its plan best (not stopped by the time limit)
            "proven_best_rate": float((weeks["status"] == "Optimal").mean()) if "status" in weeks else None,
            **chip_weeks(weeks, list(chips))}


def build_report(runs: dict, order: list, average: pd.DataFrame, rank_sample: pd.DataFrame,
                 user_total: int, chips: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(one summary row per run, every run's weeks stacked), for the saved runs in `order`
    (runs not in `order` follow, alphabetically)."""
    labels = [r for r in order if r in runs] + sorted(r for r in runs if r not in order)
    report = pd.DataFrame([{"run": label, **summarize(runs[label], average, rank_sample, user_total, chips)}
                           for label in labels])
    weeks = pd.concat([runs[label].assign(run=label) for label in labels], ignore_index=True)
    return report, weeks
