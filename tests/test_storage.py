"""Offline tests for fpl/storage.py and fpl/seasons.py."""
import pandas as pd

from fpl import seasons, storage


def test_read_csv_falls_back_to_latin1(tmp_path):
    path = tmp_path / "names.csv"
    path.write_bytes("name\nMartín Ødegaard\n".encode("latin-1"))
    assert storage.read_csv(path)["name"].iloc[0] == "Martín Ødegaard"


def test_read_csv_reads_utf8(tmp_path):
    path = tmp_path / "names.csv"
    path.write_bytes("name\nMartín Ødegaard\n".encode("utf-8"))
    assert storage.read_csv(path)["name"].iloc[0] == "Martín Ødegaard"


def test_save_vaastav_keeps_the_repo_layout(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DATA_DIR", tmp_path)
    saved = storage.save_vaastav("2025-26", "gws/merged_gw.csv", b"a,b\n1,2\n")
    assert saved == tmp_path / "raw" / "vaastav" / "2025-26" / "gws" / "merged_gw.csv"
    assert list(storage.load_vaastav("2025-26", "gws/merged_gw.csv")["b"]) == [2]


def test_previous_season():
    assert seasons.previous_season("2026-27") == "2025-26"
    assert seasons.previous_season("2000-01") == "1999-00"


def test_snapshots_load_oldest_first(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DATA_DIR", tmp_path)
    storage.save_snapshot("2025-26", "20250901T120000", "vaastav", pd.DataFrame({"id": [2]}))
    storage.save_snapshot("2025-26", "20250801T120000", "wayback", pd.DataFrame({"id": [1]}))
    snaps = storage.load_snapshots("2025-26")
    assert [s[0] for s in snaps] == [pd.Timestamp("2025-08-01 12:00", tz="UTC"),
                                     pd.Timestamp("2025-09-01 12:00", tz="UTC")]
    assert list(snaps[0][1]["id"]) == [1]


def test_benchmark_round_trip_and_table_age(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DATA_DIR", tmp_path)
    storage.save_benchmark("x.json", b'{"a": 1}')
    assert storage.load_benchmark_json("x.json") == {"a": 1}
    assert storage.table_age_hours("players") == float("inf")
    storage.save_table("players", pd.DataFrame({"a": [1]}))
    assert 0 <= storage.table_age_hours("players") < 0.1


def test_alert_record_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DATA_DIR", tmp_path)
    assert storage.load_alert(8) is None
    path = storage.save_alert(8, {"stage": "early", "decisions": {"hit": 0}})
    assert path == tmp_path / "alerts" / "gw8.json"
    assert storage.load_alert(8) == {"stage": "early", "decisions": {"hit": 0}}
