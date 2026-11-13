"""Offline tests for fpl/matches.py, using tiny hand-made season files."""
import numpy as np
import pandas as pd
import pytest

from fpl import matches

ZERO_STATS = {c: 0 for c in matches.STAT_COLUMNS}


def season_frames():
    teams = pd.DataFrame({"id": [1, 2], "name": ["Arsenal", "Ipswich Town"],
                          "short_name": ["ARS", "IPS"]})
    fixtures = pd.DataFrame({
        "id": [10, 11, 12], "event": [1, 2, np.nan],
        "kickoff_time": ["2026-08-15T14:00:00Z", "2026-08-22T14:00:00Z", None],
        "team_h": [1, 2, 1], "team_a": [2, 1, 2],
        "team_h_score": [2, np.nan, np.nan], "team_a_score": [1, np.nan, np.nan],
        "finished": [True, False, False]})
    players = pd.DataFrame({"id": [1, 2, 3], "code": [1001, 1002, 1003],
                            "element_type": [3, 1, 5], "web_name": ["Saka", "Walton", "Boss"]})
    saka = {**ZERO_STATS, "element": 1, "fixture": 10, "was_home": True, "value": 100,
            "minutes": 90, "goals_scored": 1, "total_points": 7}
    gws = pd.DataFrame([
        saka,
        saka,  # duplicated row, as sometimes happens in vaastav files
        {**ZERO_STATS, "element": 2, "fixture": 10, "was_home": False, "value": 45,
         "minutes": 90, "saves": 4, "goals_conceded": 2, "total_points": 2},
        {**ZERO_STATS, "element": 3, "fixture": 10, "was_home": True, "value": 5},  # manager
        {**ZERO_STATS, "element": 1, "fixture": 11, "was_home": False, "value": 100},  # unfinished
    ])
    return gws, players, teams, fixtures


def test_build_season():
    table = matches.build_season("2026-27", *season_frames())
    assert list(table.columns) == matches.MATCH_COLUMNS
    assert len(table) == 2  # duplicate, manager and unfinished-fixture rows dropped
    saka = table.set_index("player_name").loc["Saka"]
    assert (saka["player_code"], saka["position"], saka["price"]) == (1001, "MID", 10.0)
    assert (saka["team"], saka["opponent"], saka["team_goals"], saka["opp_goals"]) == \
        ("Arsenal", "Ipswich", 2, 1)
    walton = table.set_index("player_name").loc["Walton"]
    assert (walton["team"], walton["position"], walton["was_home"]) == ("Ipswich", "GK", False)
    assert table["clearances_blocks_interceptions"].isna().all()
    assert (table["rules_era"] == "2026_27").all()
    assert str(table["kickoff_time"].dt.tz) == "UTC"


def test_fixture_list_has_two_rows_per_scheduled_fixture():
    _, _, teams, fixtures = season_frames()
    fl = matches.fixture_list("2026-27", fixtures, teams)
    assert len(fl) == 4  # the unscheduled fixture (no gameweek) is dropped
    assert set(fl["team"]) == {"Arsenal", "Ipswich"}
    assert fl["gw"].dtype.kind == "i"


def test_gw_cutoffs_and_finished_gws():
    _, _, teams, fixtures = season_frames()
    fl = matches.fixture_list("2026-27", fixtures, teams)
    cutoffs = matches.gw_cutoffs(fl).set_index("gw")["cutoff"]
    assert cutoffs.loc[1] == pd.Timestamp("2026-08-15 14:00", tz="UTC")
    assert matches.finished_gws(fl, "2026-27") == [1]


def test_with_asof_uses_each_rows_own_gameweek():
    _, _, teams, fixtures = season_frames()
    fl = matches.fixture_list("2026-27", fixtures, teams)
    rows = pd.DataFrame({"season": ["2026-27", "2026-27"], "gw": [2, 1]})
    out = matches.with_asof(rows, matches.gw_cutoffs(fl))
    assert list(out["asof"]) == [pd.Timestamp("2026-08-22 14:00", tz="UTC"),
                                 pd.Timestamp("2026-08-15 14:00", tz="UTC")]


def test_upcoming_player_rows_skip_players_who_left():
    _, _, teams, fixtures = season_frames()
    fl = matches.fixture_list("2026-27", fixtures, teams)
    players = pd.DataFrame({"code": [1001, 1004], "web_name": ["Saka", "Gone"],
                            "element_type": [3, 4], "now_cost": [100, 60], "team": [1, 2],
                            "status": ["a", "u"]})
    rows = matches.upcoming_player_rows(fl, players, teams, "2026-27", [2])
    assert list(rows.columns) == matches.PLAYER_ROW_COLUMNS
    assert list(rows["player_name"]) == ["Saka"]
    assert (rows["team"].iloc[0], rows["opponent"].iloc[0], rows["was_home"].iloc[0]) == \
        ("Arsenal", "Ipswich", False)
    teams_rows = matches.upcoming_team_rows(fl, "2026-27", [2])
    assert list(teams_rows.columns) == matches.TEAM_ROW_COLUMNS and len(teams_rows) == 2


def test_expected_stats_before_fpl_published_them_become_missing():
    """FPL published xG/xA/xGC only from Nov 2022: earlier rows hold 0, which means 'unknown'."""
    kick = pd.to_datetime(["2022-08-05", "2022-08-05", "2022-11-12", "2022-11-12",
                           "2023-08-11", "2023-08-11"], utc=True)
    table = pd.DataFrame({"season": ["2022-23"] * 4 + ["2023-24"] * 2, "kickoff_time": kick,
                          "expected_goals": [0.0, 0.0, 0.0, 0.4, 0.0, 0.2],
                          "expected_assists": [0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
                          "expected_goals_conceded": [0.0, 0.0, 1.2, 0.3, 0.9, 0.0]})
    out = matches.blank_unpublished_expected(table)
    for col in matches.EXPECTED_COLUMNS:
        assert out[col].iloc[:2].isna().all()             # before the first published value
    assert list(out["expected_goals"].iloc[2:]) == [0.0, 0.4, 0.0, 0.2]  # real zeros stay 0
    assert out["expected_assists"].iloc[2:].notna().all()


def test_a_season_with_no_expected_stats_at_all_is_left_alone():
    table = matches.build_season("2026-27", *season_frames())   # every xG is 0
    assert (table["expected_goals"] == 0).all()
