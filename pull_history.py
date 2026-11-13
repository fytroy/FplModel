"""Download past seasons from the vaastav/Fantasy-Premier-League GitHub repo.

Usage:
    python pull_history.py           # download any season files not already saved
    python pull_history.py --force   # re-download everything
"""
import argparse

from fpl import api, storage
from fpl.seasons import PAST_SEASONS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="re-download files already saved")
    args = parser.parse_args()

    for season in PAST_SEASONS:
        for path in storage.VAASTAV_FILES:
            if storage.vaastav_path(season, path).exists() and not args.force:
                print(f"  {season}/{path}: already saved")
                continue
            storage.save_vaastav(season, path, api.fetch_vaastav_csv(season, path))
            print(f"  {season}/{path}: downloaded", flush=True)
    print(f"Files are in {storage.DATA_DIR / 'raw' / 'vaastav'}")


if __name__ == "__main__":
    main()
