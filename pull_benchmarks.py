"""One-off download of what the 2025/26 season simulation needs.

Usage:
    python pull_benchmarks.py           # about 10-20 minutes (polite pace)
Saves under data/raw/benchmarks/ and data/raw/fixture_snapshots/2025-26/.
"""
import io
import json
import random
import time

import pandas as pd

from fpl import api, storage

SEASON = "2025-26"
ARCHIVED_BOOTSTRAP = "20260526000012"     # Internet Archive copy taken after GW38
RANK_SAMPLE_SIZE = 3000
FIXTURE_COLUMNS = ["id", "code", "event", "kickoff_time", "finished", "team_h", "team_a",
                   "team_h_score", "team_a_score"]


def pull_bootstrap() -> None:
    raw = api.fetch_wayback("https://fantasy.premierleague.com/api/bootstrap-static/", ARCHIVED_BOOTSTRAP)
    data = json.loads(raw.decode("utf-8"))
    assert data["events"][-1]["finished"], "archived copy is from before the season ended"
    storage.save_benchmark(f"bootstrap_{SEASON}.json", raw)
    print(f"Average manager total: {sum(e['average_entry_score'] for e in data['events'])}")


def pull_rank_sample() -> None:
    total = api.fetch_bootstrap()["total_players"]
    rows, tried = [], 0
    random.seed(2526)
    while len(rows) < RANK_SAMPLE_SIZE:
        tried += 1
        try:
            past = api.fetch_entry_history(random.randint(1, total)).get("past", [])
        except Exception:
            continue
        season = next((p for p in past if p["season_name"] == "2025/26"), None)
        if season:
            rows.append({"points": season["total_points"], "rank": season["rank"]})
        if tried % 250 == 0:
            print(f"  tried {tried}, kept {len(rows)}", flush=True)
        time.sleep(0.1)
    buffer = io.StringIO()
    pd.DataFrame(rows).to_csv(buffer, index=False)
    storage.save_benchmark(f"rank_sample_{SEASON}.csv", buffer.getvalue().encode())


def pull_fixture_snapshots() -> None:
    path = f"data/{SEASON}/fixtures.csv"
    for sha, when in api.fetch_vaastav_commits(path):
        table = pd.read_csv(io.BytesIO(api.fetch_vaastav_file_at(sha, path)))
        stamp = pd.Timestamp(when).strftime("%Y%m%dT%H%M%S")
        storage.save_snapshot(SEASON, stamp, "vaastav", table[[c for c in FIXTURE_COLUMNS if c in table]])
    url = "https://fantasy.premierleague.com/api/fixtures/"
    for ts in api.fetch_wayback_index(url, "20250701", "20260601"):
        table = pd.DataFrame(json.loads(api.fetch_wayback(url, ts).decode("utf-8")))
        stamp = pd.Timestamp(ts, tz="UTC").strftime("%Y%m%dT%H%M%S")
        storage.save_snapshot(SEASON, stamp, "wayback", table[[c for c in FIXTURE_COLUMNS if c in table]])
    print(f"Saved {len(storage.load_snapshots(SEASON))} fixture snapshots")


if __name__ == "__main__":
    pull_bootstrap()
    pull_fixture_snapshots()
    pull_rank_sample()
