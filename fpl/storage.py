"""Save and load files in the data/ folder.

This is the only file that reads or writes the disk.
"""
import json
import time
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# The vaastav files we use for each past season, as paths inside the repo.
VAASTAV_FILES = ("gws/merged_gw.csv", "players_raw.csv", "teams.csv", "fixtures.csv")


def save_raw(name: str, data, run_date: str) -> Path:
    """Save an untouched API response to data/raw/<run_date>/<name>.json."""
    folder = DATA_DIR / "raw" / run_date
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def save_table(name: str, table: pd.DataFrame) -> Path:
    """Save a cleaned table to data/processed/<name>.csv (overwrites the last run)."""
    folder = DATA_DIR / "processed"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.csv"
    table.to_csv(path, index=False, encoding="utf-8")
    return path


def vaastav_path(season: str, path: str) -> Path:
    """Where a vaastav file is cached: data/raw/vaastav/<season>/<path>."""
    return DATA_DIR / "raw" / "vaastav" / season / path


def save_vaastav(season: str, path: str, content: bytes) -> Path:
    target = vaastav_path(season, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return target


def read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV; older vaastav files use Latin-1 for accented names."""
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")


def load_vaastav(season: str, path: str) -> pd.DataFrame:
    return read_csv(vaastav_path(season, path))


def save_run(label: str, weeks: pd.DataFrame) -> Path:
    """Save one simulation run's weekly table to data/processed/simulation_runs/<label>.csv."""
    folder = DATA_DIR / "processed" / "simulation_runs"
    folder.mkdir(parents=True, exist_ok=True)
    slug = "".join(c if c.isalnum() else "_" for c in label)
    path = folder / f"{slug}.csv"
    weeks.assign(run=label).to_csv(path, index=False, encoding="utf-8")
    return path


def load_runs() -> dict[str, pd.DataFrame]:
    """Every saved simulation run, by label (the file's 'run' column)."""
    folder = DATA_DIR / "processed" / "simulation_runs"
    runs = {}
    for path in sorted(folder.glob("*.csv")):
        table = read_csv(path)
        runs[str(table["run"].iloc[0])] = table.drop(columns="run")
    return runs


def load_table(name: str) -> pd.DataFrame:
    """Load data/processed/<name>.csv."""
    return read_csv(DATA_DIR / "processed" / f"{name}.csv")


def benchmark_path(name: str) -> Path:
    """data/raw/benchmarks/<name>: downloaded once by pull_benchmarks.py."""
    return DATA_DIR / "raw" / "benchmarks" / name


def save_benchmark(name: str, content: bytes) -> Path:
    path = benchmark_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def load_benchmark_json(name: str) -> dict:
    return json.loads(benchmark_path(name).read_text(encoding="utf-8"))


def save_snapshot(season: str, stamp: str, source: str, table: pd.DataFrame) -> Path:
    """A dated copy of a season's fixture list: data/raw/fixture_snapshots/<season>/<stamp>_<source>.csv.

    stamp is UTC, 'YYYYmmddTHHMMSS'.
    """
    folder = DATA_DIR / "raw" / "fixture_snapshots" / season
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{stamp}_{source}.csv"
    table.to_csv(path, index=False, encoding="utf-8")
    return path


def load_snapshots(season: str) -> list[tuple[pd.Timestamp, pd.DataFrame]]:
    """Every saved fixture-list copy for a season, oldest first."""
    folder = DATA_DIR / "raw" / "fixture_snapshots" / season
    snaps = [(pd.Timestamp(p.stem.split("_")[0], tz="UTC"), read_csv(p))
             for p in folder.glob("*.csv")]
    return sorted(snaps, key=lambda s: s[0])


def save_alert(gw: int, record: dict) -> Path:
    """What the deadline alert sent for a gameweek: data/alerts/gw<N>.json."""
    folder = DATA_DIR / "alerts"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"gw{gw}.json"
    path.write_text(json.dumps(record, indent=1), encoding="utf-8")
    return path


def load_alert(gw: int) -> dict | None:
    """The saved alert record for a gameweek (None if nothing was sent yet)."""
    path = DATA_DIR / "alerts" / f"gw{gw}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def table_age_hours(name: str) -> float:
    """Hours since data/processed/<name>.csv was written (infinite if it doesn't exist)."""
    path = DATA_DIR / "processed" / f"{name}.csv"
    return (time.time() - path.stat().st_mtime) / 3600 if path.exists() else float("inf")
