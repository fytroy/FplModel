"""Offline test for the ranked table in predict.py."""
import pandas as pd
import pytest

from predict import ranked_table


def test_ranked_table_sums_double_gameweeks_and_handles_blanks():
    common = {"position": "MID", "price": 7.0, "p_plays": 1.0, "p_60": 1.0, "xg": 0.2,
              "xa": 0.1, "p_cs": 0.3, "p_defcon": 0.1, "e_bonus": 0.5, "baseline": 3.0}
    pred = pd.DataFrame([
        {**common, "player_code": 1, "player_name": "One", "team": "Alpha", "gw": 7,
         "opponent": "Bravo", "was_home": 1, "xp": 3.0},
        {**common, "player_code": 1, "player_name": "One", "team": "Alpha", "gw": 7,
         "opponent": "Charlie", "was_home": 0, "xp": 4.0},
        {**common, "player_code": 1, "player_name": "One", "team": "Alpha", "gw": 8,
         "opponent": "Bravo", "was_home": 0, "xp": 2.0},
        {**common, "player_code": 2, "player_name": "Two", "team": "Bravo", "gw": 8,
         "opponent": "Alpha", "was_home": 1, "xp": 5.0},
    ])
    teams = pd.DataFrame({"name": ["Alpha", "Bravo", "Charlie"],
                          "short_name": ["ALP", "BRA", "CHA"]})
    t = ranked_table(pred, teams, [7, 8]).set_index("player")
    assert t.loc["One", "xP"] == pytest.approx(7.0)
    assert t.loc["One", "next"] == "BRA (H), CHA (A)"
    assert t.loc["One", "xG"] == pytest.approx(0.4)
    assert (t.loc["Two", "xP"], t.loc["Two", "next"], t.loc["Two", "xP_gw7"]) == (0, "-", 0)
    assert (t.loc["One", "xP_gw8"], t.loc["Two", "xP_total"]) == (2.0, 5.0)
    assert list(t.index) == ["One", "Two"]  # 9.0 total beats 5.0


def test_ranked_table_lists_every_current_player_even_with_a_blank_gameweek():
    common = {"position": "MID", "price": 7.0, "p_plays": 1.0, "p_60": 1.0, "xg": 0.2,
              "xa": 0.1, "p_cs": 0.3, "p_defcon": 0.1, "e_bonus": 0.5, "baseline": 3.0}
    pred = pd.DataFrame([{**common, "player_code": 1, "player_name": "One", "team": "Alpha",
                          "gw": 7, "opponent": "Bravo", "was_home": 1, "xp": 3.0}])
    teams = pd.DataFrame({"id": [1, 2], "name": ["Alpha", "Bravo"], "short_name": ["ALP", "BRA"]})
    players = pd.DataFrame({"code": [1, 2, 3], "web_name": ["One", "Two", "Gone"],
                            "team": [1, 2, 1], "element_type": [3, 4, 3], "now_cost": [70, 85, 50],
                            "status": ["a", "a", "u"]})
    t = ranked_table(pred, teams, [7], players).set_index("player")
    assert list(t.index) == ["One", "Two"]          # the departed player is left out
    assert (t.loc["Two", "next"], t.loc["Two", "xP"]) == ("-", 0)
    assert (t.loc["Two", "team"], t.loc["Two", "pos"], t.loc["Two", "price"]) == ("BRA", "FWD", 8.5)
    assert t.loc["One", "next"] == "BRA (H)"
