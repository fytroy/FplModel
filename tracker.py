"""Track points and performance of your team at the end of each gameweek.

Usage:
    python tracker.py                    # season performance table for default team
    python tracker.py --gw 6             # detailed breakdown of GW6 squad, captain and bench
    python tracker.py --team 123456      # track another team ID
"""
import argparse
import sys

import pandas as pd

from fpl import api, optimal, storage, tracker
from fpl.settings import TEAM_ID, require_team


def run_tracker(team_id: int, gw: int | None = None, offline: bool = False) -> None:
    players = storage.load_table("players")
    teams = storage.load_table("teams")
    player_history = storage.load_table("player_history")

    entry_data = None
    history_data = None

    if offline:
        cached = tracker.load_cached_team_history(team_id)
        if cached:
            entry_data, history_data = cached
        else:
            entry_data = {"name": f"Manager {team_id}", "summary_overall_points": 0, "summary_overall_rank": 0}
            history_data = {"current": [], "chips": []}
    else:
        try:
            entry_data = api.fetch_entry(team_id)
            history_data = api.fetch_entry_history(team_id)
            tracker.save_team_history_cache(team_id, entry_data, history_data)
        except Exception:
            cached = tracker.load_cached_team_history(team_id)
            if cached:
                entry_data, history_data = cached
                print(f"[Offline mode] Loaded cached team history for team {team_id}")
            else:
                print(f"Notice: Network unavailable and no cached history for {team_id}. Using current standings.")
                entry_data = {"name": f"Manager {team_id}", "summary_overall_points": 0, "summary_overall_rank": 0}
                history_data = {"current": [], "chips": []}

    perf = tracker.parse_team_performance(team_id, entry_data, history_data)

    if gw is None:
        print()
        print(tracker.render_history_table(perf))
        return

    # Specific gameweek requested
    summary = next((s for s in perf.gameweeks if s.gw == gw), None)
    picks_data = None
    live_data = None

    try:
        picks_data = api.fetch_entry_picks(team_id, gw)
    except Exception:
        # If offline or picks not available, reconstruct from player_history or mock
        picks_data = None

    if picks_data is None:
        print(f"\nGameweek {gw} picks data not available online or cached.")
        if summary:
            print(f"GW{gw} Gross Score: {summary.points} pts | Hits: -{summary.hits} pts | Net: {summary.net_points} pts")
        return

    try:
        live_data = api.fetch_event_live(gw)
    except Exception:
        live_data = None

    # Solve dream team for that past gameweek to compare
    dream_pts = None
    try:
        pool = optimal.build_player_pool(players, teams)
        gw_pts_map = tracker.resolve_player_points_for_gw(gw, player_history=player_history, live_data=live_data)
        if gw_pts_map:
            # map element to player_code
            el_to_code = dict(zip(players["id"], players["code"]))
            pts_series = pd.Series({el_to_code[el]: pts for el, pts in gw_pts_map.items() if el in el_to_code})
            dt_squad = optimal.solve_optimal_squad(pts_series, pool, gw=gw)
            dream_pts = dt_squad.starting_points
    except Exception:
        dream_pts = None

    breakdown = tracker.parse_gameweek_breakdown(
        gw=gw,
        picks_data=picks_data,
        players=players,
        teams=teams,
        player_history=player_history,
        live_data=live_data,
        summary=summary,
        dream_team_points=dream_pts,
    )

    print()
    print(tracker.render_gameweek_breakdown(breakdown))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--team", type=int, default=TEAM_ID, help="FPL team ID")
    parser.add_argument("--gw", type=int, default=None, help="Gameweek number to inspect in detail")
    parser.add_argument("--offline", action="store_true", help="Use local cache without attempting network requests")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    run_tracker(require_team(args.team), args.gw, args.offline)


if __name__ == "__main__":
    main()
