"""Scout players for the upcoming gameweek: top picks by position, captain candidates, differentials and value.

Usage:
    python scout.py                      # scout upcoming gameweek
    python scout.py --gw 8               # scout specific gameweek
    python scout.py --top 8              # show top 8 per category
"""
import argparse
import sys

import pandas as pd

from fpl import scout, storage
from fpl.settings import CHIP_SCAN_WEEKS


def run_scout(gw: int | None = None, top_n: int = 5, fast: bool = False) -> None:
    players = storage.load_table("players")
    teams = storage.load_table("teams")
    gameweeks = storage.load_table("gameweeks")

    if gw is None:
        upcoming = gameweeks.loc[gameweeks["is_next"].astype(bool), "id"]
        if upcoming.empty:
            upcoming = gameweeks.loc[gameweeks["is_current"].astype(bool), "id"]
        gw = int(upcoming.iloc[0]) if not upcoming.empty else 1

    gws = [g for g in range(gw, gw + 3) if g <= 38]
    if not gws:
        gws = [gw]

    fixtures_all = None
    avail = None
    xp = None

    if not fast:
        try:
            from fpl import forecast, matches, models
            history = matches.load_all_matches()
            fixtures_all = matches.load_all_fixtures()
            pred, avail = forecast.live_forecast(history, fixtures_all, players, teams, gameweeks, gws)
            xp = models.weekly_xp(pred, gws)
        except Exception:
            fast = True

    if fast or xp is None:
        try:
            fixtures_all = storage.load_table("fixtures")
        except Exception:
            fixtures_all = None
        avail = None
        form = pd.to_numeric(players["form"], errors="coerce").fillna(0.0)
        pts = pd.to_numeric(players["total_points"], errors="coerce").fillna(0.0)
        est = (form * 0.7 + (pts / max(1, gw - 1)) * 0.3).clip(lower=0.5)
        xp = pd.DataFrame({gw: est.to_numpy()}, index=pd.Index(players["code"].astype(int)))

    report = scout.build_scout_report(
        players=players,
        teams=teams,
        xp=xp,
        gw=gw,
        fixtures_all=fixtures_all,
        avail=avail,
        top_n=top_n,
    )

    print()
    print(scout.render_scout_report(report))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gw", type=int, default=None, help="Upcoming gameweek number")
    parser.add_argument("--top", type=int, default=5, help="Number of players to show per position")
    parser.add_argument("--fast", action="store_true", help="Fast evaluation using rolling form")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    run_scout(args.gw, args.top, args.fast)


if __name__ == "__main__":
    main()
