"""Download the latest FPL data and save it under data/.

Usage:
    python pull_data.py                 # everything (~700 player requests, a few minutes)
    python pull_data.py --skip-history  # just players, teams, gameweeks, fixtures (seconds)
"""
import argparse
import time
from datetime import date

from fpl import api, storage, transform

DELAY_BETWEEN_PLAYERS = 0.3  # seconds; be polite to the FPL servers


def pull(skip_history: bool = False) -> None:
    run_date = date.today().isoformat()

    print("Fetching players, teams and gameweeks...")
    bootstrap = api.fetch_bootstrap()
    storage.save_raw("bootstrap_static", bootstrap, run_date)

    print("Fetching fixtures...")
    fixtures = api.fetch_fixtures()
    storage.save_raw("fixtures", fixtures, run_date)

    players = transform.players_table(bootstrap)
    storage.save_table("players", players)
    storage.save_table("teams", transform.teams_table(bootstrap))
    storage.save_table("gameweeks", transform.gameweeks_table(bootstrap))
    storage.save_table("fixtures", transform.fixtures_table(fixtures, bootstrap))

    history_rows = 0
    if not skip_history:
        ids = list(players["id"])
        summaries = {}
        print(f"Fetching match history for {len(ids)} players...", flush=True)
        for i, player_id in enumerate(ids, start=1):
            summaries[int(player_id)] = api.fetch_player_history(int(player_id))
            if i % 50 == 0 or i == len(ids):
                # flush=True shows progress immediately, even when output goes to a log file.
                print(f"  {i}/{len(ids)}", flush=True)
            time.sleep(DELAY_BETWEEN_PLAYERS)
        storage.save_raw("element_summaries", summaries, run_date)
        history = transform.history_table(summaries)
        storage.save_table("player_history", history)
        history_rows = len(history)

    print()
    print(f"Saved {len(players)} players, {len(fixtures)} fixtures, "
          f"current GW {transform.current_gameweek(bootstrap)}.")
    if not skip_history:
        print(f"Saved {history_rows} player-match history rows.")
    print(f"Files are in {storage.DATA_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-history", action="store_true",
                        help="skip the slow per-player match history download")
    pull(parser.parse_args().skip_history)


if __name__ == "__main__":
    main()
