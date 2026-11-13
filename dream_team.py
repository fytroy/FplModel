"""Find the optimal 15-player squad and Starting XI giving the highest gameweek points.

Solves the integer program (PuLP) within £100m budget and 3-per-club constraints to determine
the theoretical highest-scoring lineup, formation, captain, vice-captain and bench for any gameweek.
Also compares with your current squad if --compare is set.

Usage:
    python dream_team.py                     # optimal squad for upcoming gameweek
    python dream_team.py --gw 8              # optimal squad for GW8
    python dream_team.py --team 123456       # compare against current team
    python dream_team.py --budget 1000       # budget in tenths of £1m (1000 = £100.0m)
"""
import argparse
import sys

import pandas as pd

from fpl import forecast, matches, models, optimal, storage
from fpl.settings import TEAM_ID


def run_dream_team(gw: int | None = None, team_id: int | None = None, budget: int = 1000, fast: bool = False) -> None:
    players = storage.load_table("players")
    teams = storage.load_table("teams")
    gameweeks = storage.load_table("gameweeks")

    if gw is None:
        upcoming = gameweeks.loc[gameweeks["is_next"].astype(bool), "id"]
        if upcoming.empty:
            upcoming = gameweeks.loc[gameweeks["is_current"].astype(bool), "id"]
        gw = int(upcoming.iloc[0]) if not upcoming.empty else 1

    pool = optimal.build_player_pool(players, teams)

    # Get expected points for gw
    if not fast:
        try:
            history = matches.load_all_matches()
            fixtures_all = matches.load_all_fixtures()
            pred, _ = forecast.live_forecast(history, fixtures_all, players, teams, gameweeks, [gw])
            xp = models.weekly_xp(pred, [gw])
            pts_series = xp[gw]
        except Exception as e:
            fast = True

    if fast:
        # Fast form and point averages
        form = pd.to_numeric(players["form"], errors="coerce").fillna(0.0)
        tot = pd.to_numeric(players["total_points"], errors="coerce").fillna(0.0)
        est = (form * 0.7 + (tot / max(1, gw - 1)) * 0.3).clip(lower=0.5)
        pts_series = pd.Series(est.to_numpy(), index=pd.Index(players["code"].astype(int)))

    squad = optimal.solve_optimal_squad(pts_series, pool, gw=gw, budget=budget)

    print()
    print(optimal.render_optimal_squad(squad, label="OPTIMAL DREAM TEAM (HIGHEST GAMEWEEK POINTS)"))

    if team_id is not None:
        # Compare with user team if available
        try:
            from fpl import myteam, api
            # Try load team state
            state = myteam.load_team_state(
                team_id, api.fetch_entry(team_id), api.fetch_entry_history(team_id),
                api.fetch_entry_transfers(team_id), lambda g: api.fetch_entry_picks(team_id, g),
                players, storage.load_table("player_history"), []
            )
            owned_codes = set(state.codes)
            overlap = [p.name for p in squad.starters if p.code in owned_codes]
            print(f"\nYour current team owns {len(overlap)}/11 optimal starters: {', '.join(overlap) if overlap else 'None'}")
            user_xp = sum(pts_series.get(c, 0.0) for c in state.codes[:11])
            diff = squad.starting_points - user_xp
            print(f"Optimal Starting XI xP: {squad.starting_points:.1f} pts vs Your Starting XI: {user_xp:.1f} pts (Gap: +{diff:.1f} pts)")
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gw", type=int, default=None, help="Gameweek number")
    parser.add_argument("--team", type=int, default=None, help="FPL team ID to compare with")
    parser.add_argument("--budget", type=int, default=1000, help="Budget in tenths of £1m (default 1000)")
    parser.add_argument("--fast", action="store_true", help="Fast calculation using rolling form")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    run_dream_team(args.gw, args.team or None, args.budget, args.fast)


if __name__ == "__main__":
    main()
