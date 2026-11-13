"""Your best FPL moves this week, in plain English.

Usage:
    python recommend.py                  # your team (FPL_TEAM_ID), 6-week plan
    python recommend.py --horizon 4      # shorter plan
    python recommend.py --team 123456    # another team
Advice only: making the moves is alert.py's job (fpl/autopilot.py). Refreshes FPL data if it is
more than 6 hours old, then trains the model (a few minutes).
"""
import argparse
import json
import sys
from dataclasses import replace
from functools import partial
from pathlib import Path

import pandas as pd

import pull_data
from fpl import advice, api, chips, forecast, matches, models, myteam, optimizer, storage
from fpl.seasons import CURRENT_SEASON
from fpl.settings import (AUTOPILOT_MAX_TRANSFERS, CHIP_SCAN_WEEKS,DATA_MAX_AGE_HOURS, SOLVER_TIME_LIMIT, TEAM_ID,
                          PlanSettings, require_team)


def build_advice(state, players, teams, fixtures_all, xp, avail, deadline,
                 settings: PlanSettings, time_limit: int = SOLVER_TIME_LIMIT,
                 safe: bool = False) -> advice.Advice:
    """Plan and chip scan from predictions already made (no network, no training).

    safe: also work out the plan autopilot may carry out (no week-1 hit, no chips).
    """
    info = advice.pool_info(players, teams, state)
    xp = xp.reindex(info.index).fillna(0.0)                 # no prediction (left the club): 0
    plan_gws = list(xp.columns[:settings.horizon])
    squad = state.codes
    holds = chips.hold_values(
        chips.scan(xp, info, squad, state.bank, state.chips, plan_gws[-1], CURRENT_SEASON),
        state.chips)
    keep = optimizer.candidate_pool(xp[plan_gws], info, squad, settings.discount)
    inp = optimizer.PlanInput(pool=info.loc[keep], xp=xp.loc[keep, plan_gws], squad=squad,
                              bank=state.bank, free_transfers=state.free_transfers,
                              chips=state.chips, season=CURRENT_SEASON, hold_values=holds)
    plan = optimizer.solve(inp, settings, time_limit=time_limit)
    # Solve again with week 1's hit decision flipped, to explain it. Both solves stop within
    # SOLVER_GAP (or at the time limit), so the second can come out ahead: then it is the advice.
    try:
        other = optimizer.solve(inp, settings, week1_hits="none" if plan.weeks[0].paid else "some",
                                time_limit=time_limit)
    except ValueError:
        other = None
    plan, other, hit_gain, comparison_better = pick_plan(plan, other, settings)
    safe_plan = None
    if safe:
        first = plan.weeks[0]
        allowed = len(first.transfers_in) <= AUTOPILOT_MAX_TRANSFERS and not (first.chip or first.paid)
        safe_plan = plan if allowed else optimizer.solve(  # the best moves within the cap, not all or nothing
            replace(inp, chips=[], hold_values={}, free_transfers=min(inp.free_transfers, AUTOPILOT_MAX_TRANSFERS)),
            settings, week1_hits="none", time_limit=time_limit)
    end = plan.weeks[-1]
    final_scan = chips.scan(xp, info, end.owned, end.bank, state.chips, plan_gws[-1], CURRENT_SEASON)
    return advice.Advice(state=state, plan=plan, hit_gain=hit_gain, info=info, xp=xp, avail=avail,
                         scan=final_scan, counts=advice.fixture_counts(fixtures_all, teams, CURRENT_SEASON),
                         deadline=deadline, horizon=settings.horizon, comparison_better=comparison_better,
                         alternative=other, safe_plan=safe_plan)


def pick_plan(plan, other, settings: PlanSettings):
    """(the plan to advise, the other plan, hit gain, whether it came from the comparison solve).

    other: the plan with week 1's hit decision flipped (None if there is none). The
    higher objective wins, so the verdict always agrees with the advice. Hit gain is
    in expected points: the plan with a week-1 hit minus the one without, after paying
    the 4-point hits but not the planner's extra safety margin per hit (hit_margin).
    """
    if other is None:
        return plan, None, None, False
    better = other.objective > plan.objective
    if better:
        plan, other = other, plan
    with_hit, without = (plan, other) if plan.weeks[0].paid else (other, plan)
    margin = sum(settings.discount ** i * settings.hit_margin * (h.paid - n.paid)
                 for i, (h, n) in enumerate(zip(with_hit.weeks, without.weeks)))
    return plan, other, with_hit.objective - without.objective + margin, better


def plan_error_message(error: ValueError) -> str:
    """What to print when no plan could be made."""
    if isinstance(error, optimizer.SolverTimeout):
        return f"Couldn't make a plan: {error} (SOLVER_TIME_LIMIT in fpl/settings.py)."
    return f"Couldn't make a plan: {error}"


def load_team(team_id: int, players: pd.DataFrame) -> myteam.TeamState:
    bootstrap = json.loads(sorted(Path(storage.DATA_DIR / "raw").glob("*/bootstrap_static.json"))[-1]
                           .read_text(encoding="utf-8"))
    return myteam.load_team_state(
        team_id, api.fetch_entry(team_id), api.fetch_entry_history(team_id),
        api.fetch_entry_transfers(team_id), partial(api.fetch_entry_picks, team_id), players,
        storage.load_table("player_history"), bootstrap["chips"])


def run(team_id: int = TEAM_ID, horizon: int = 6, log=partial(print, flush=True), live=None) -> advice.Advice:
    """Refresh data if stale, load the team, train, predict and plan: this week's advice.

    log: called with each progress line (alert.py adds timings).
    live: live(state, players) -> state swaps in the logged-in view of the team, and
    asks for a plan autopilot may carry out.
    """
    if storage.table_age_hours("player_history") > DATA_MAX_AGE_HOURS:
        log("FPL data is more than 6 hours old: refreshing (a few minutes)...")
        pull_data.pull()
    players = storage.load_table("players")
    teams = storage.load_table("teams")
    gameweeks = storage.load_table("gameweeks")
    upcoming = gameweeks.loc[gameweeks["is_next"].astype(bool), "id"]
    if upcoming.empty:
        raise SystemExit("FPL lists no next gameweek (season over?).")
    next_gw = int(upcoming.iloc[0])

    log(f"Loading team {team_id}...")
    state = load_team(team_id, players)
    if live is not None:
        state = live(state, players)
    gws = [g for g in range(next_gw, next_gw + CHIP_SCAN_WEEKS) if g <= 38]
    history, fixtures_all = matches.load_all_matches(), matches.load_all_fixtures()
    log("Training the model and predicting the next weeks (a few minutes)...")
    pred, avail = forecast.live_forecast(history, fixtures_all, players, teams, gameweeks, gws)
    xp = models.weekly_xp(pred, gws)
    log("Planning...")
    deadline = forecast.deadlines_from(gameweeks)[next_gw]
    try:
        return build_advice(state, players, teams, fixtures_all, xp, avail, deadline,
                            PlanSettings(horizon=horizon), safe=live is not None)
    except ValueError as error:
        raise SystemExit(plan_error_message(error))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--team", type=int, default=TEAM_ID, help="FPL team ID")
    parser.add_argument("--horizon", type=int, default=6, choices=range(4, 7), help="weeks to plan (4-6)")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    result = run(require_team(args.team), args.horizon)
    print()
    print(advice.render(result))
    path = storage.save_table("plan", advice.plan_table(result))
    print(f"\nFull week-by-week plan saved to {path}")


if __name__ == "__main__":
    main()
