"""Offline tests for fpl/transform.py, using a tiny hand-made API response."""
import json
from pathlib import Path

import pytest

from fpl import transform

SAMPLE = Path(__file__).parent / "sample_bootstrap.json"


@pytest.fixture
def bootstrap():
    return json.loads(SAMPLE.read_text(encoding="utf-8"))


def test_players_table_adds_readable_columns(bootstrap):
    players = transform.players_table(bootstrap)
    assert len(players) == 3
    for col in ["id", "web_name", "team_name", "position", "price", "news",
                "chance_of_playing_next_round"]:
        assert col in players.columns
    saka = players.set_index("web_name").loc["Saka"]
    assert saka["price"] == 10.5
    assert saka["position"] == "MID"
    assert saka["team_name"] == "Arsenal"
    assert saka["chance_of_playing_next_round"] == 75


def test_teams_table(bootstrap):
    teams = transform.teams_table(bootstrap)
    assert list(teams["short_name"]) == ["ARS", "MCI"]


def test_gameweeks_table(bootstrap):
    gws = transform.gameweeks_table(bootstrap)
    assert len(gws) == 2
    assert transform.current_gameweek(bootstrap) == 2


def test_fixtures_table_adds_team_names(bootstrap):
    fixtures = [{"id": 10, "event": 2, "team_h": 1, "team_a": 2, "team_h_difficulty": 4,
                 "team_a_difficulty": 4, "kickoff_time": "2026-08-23T15:00:00Z",
                 "finished": False, "team_h_score": None, "team_a_score": None}]
    table = transform.fixtures_table(fixtures, bootstrap)
    row = table.iloc[0]
    assert row["home_team"] == "ARS"
    assert row["away_team"] == "MCI"


def test_history_table_combines_players():
    summaries = {
        2: {"history": [{"element": 2, "round": 1, "minutes": 90, "total_points": 8},
                        {"element": 2, "round": 2, "minutes": 70, "total_points": 2}]},
        3: {"history": [{"element": 3, "round": 1, "minutes": 90, "total_points": 13}]},
        4: {"history": []},
    }
    table = transform.history_table(summaries)
    assert len(table) == 3
    assert set(table["element"]) == {2, 3}
