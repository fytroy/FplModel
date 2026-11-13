"""Remind you to switch your team before the FPL deadline.

Compares your current lineup (starters, bench, captain) against the optimal recommendations
and lists all switches to make, with deadline countdown in Nairobi time.

Usage:
    python remind.py                     # check switches for your team (FPL_TEAM_ID)
    python remind.py --team 123456       # check another team
    python remind.py --notify            # send reminder to your phone via ntfy
"""
import argparse
import os
import sys

import pandas as pd

from fpl import advice, optimal, reminder, storage
from fpl.settings import NTFY_TOPIC, TEAM_ID, require_team


def run_remind(team_id: int, notify: bool = False, horizon: int = 4, fast: bool = False, topic: str | None = None) -> None:
    players = storage.load_table("players")
    teams = storage.load_table("teams")
    gameweeks = storage.load_table("gameweeks")

    upcoming = gameweeks.loc[gameweeks["is_next"].astype(bool), "id"]
    if upcoming.empty:
        upcoming = gameweeks.loc[gameweeks["is_current"].astype(bool), "id"]
    next_gw = int(upcoming.iloc[0]) if not upcoming.empty else 1

    deadline_str = gameweeks.loc[gameweeks["id"] == next_gw, "deadline_time"].iloc[0]
    deadline_utc = pd.Timestamp(deadline_str, tz="UTC")

    state = None
    plan = None
    info = None
    xp = None

    if not fast:
        print(f"Analyzing team {team_id} and checking upcoming gameweek {next_gw} switches (full model)...")
        try:
            import recommend
            adv = recommend.run(team_id, horizon=horizon, log=lambda s: None)
            state = adv.state
            plan = adv.plan
            info = adv.info
            xp = adv.xp
        except Exception as error:
            print(f"Full model failed ({error}); using the fast estimate instead.")
            fast = True

    if fast or plan is None:
        print(f"Analyzing team {team_id} and checking upcoming gameweek {next_gw} switches (fast mode)...")
        # Build fast state
        import recommend
        state = recommend.load_team(team_id, players)   # no team, no advice: never guess a squad

        form = pd.to_numeric(players["form"], errors="coerce").fillna(0.0)
        tot = pd.to_numeric(players["total_points"], errors="coerce").fillna(0.0)
        est = (form * 0.7 + (tot / max(1, next_gw - 1)) * 0.3).clip(lower=0.5)
        xp = pd.DataFrame({next_gw: est.to_numpy()}, index=pd.Index(players["code"].astype(int)))
        info = advice.pool_info(players, teams, state)

        from fpl import optimizer
        keep = optimizer.candidate_pool(xp, info, state.codes, discount=1.0)
        inp = optimizer.PlanInput(
            pool=info.loc[keep], xp=xp.loc[keep, [next_gw]], squad=state.codes,
            bank=state.bank, free_transfers=state.free_transfers, chips=[], season="2026-27"
        )
        plan = optimizer.solve(inp, settings=optimizer.PlanSettings(horizon=1), time_limit=5)

    # Solve optimal dream team for comparison
    pool = optimal.build_player_pool(players, teams)
    gw_col = next_gw if next_gw in xp.columns else xp.columns[0]
    dt_squad = optimal.solve_optimal_squad(xp[gw_col], pool, gw=next_gw)
    dream_xp = dt_squad.starting_points

    rem = reminder.generate_switch_reminder(
        current_state=state,
        plan=plan,
        info=info,
        xp=xp,
        deadline_utc=deadline_utc,
        dream_team_xp=dream_xp,
    )

    print()
    print(reminder.render_switch_reminder(rem))

    if notify:
        ntfy_topic = (topic or os.environ.get("NTFY_TOPIC") or NTFY_TOPIC).strip()
        if not ntfy_topic:
            print("\nNotification skipped: NTFY_TOPIC environment variable is not set.")
        else:
            success = reminder.send_switch_notification(rem, ntfy_topic)
            if success:
                print(f"\nReminder notification successfully sent to ntfy topic: {ntfy_topic}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--team", type=int, default=TEAM_ID, help="FPL team ID")
    parser.add_argument("--notify", action="store_true", help="Send switch reminder notification via ntfy")
    parser.add_argument("--topic", type=str, default=None, help="ntfy topic (defaults to NTFY_TOPIC)")
    parser.add_argument("--horizon", type=int, default=4, help="Weeks to plan (default 4)")
    parser.add_argument("--fast", action="store_true", help="Fast calculation without retraining LightGBM")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    run_remind(require_team(args.team), args.notify, args.horizon, args.fast, topic=args.topic)


if __name__ == "__main__":
    main()
