import pandas as pd
import pytest

from fpl import scout


def test_build_scout_report():
    players = pd.DataFrame([
        {"id": 1, "code": 101, "web_name": "Raya", "element_type": 1, "team": 1, "now_cost": 55, "status": "a", "selected_by_percent": "15.0"},
        {"id": 2, "code": 102, "web_name": "Gabriel", "element_type": 2, "team": 1, "now_cost": 60, "status": "a", "selected_by_percent": "25.0"},
        {"id": 3, "code": 103, "web_name": "Salah", "element_type": 3, "team": 2, "now_cost": 125, "status": "a", "selected_by_percent": "35.0"},
        {"id": 4, "code": 104, "web_name": "Haaland", "element_type": 4, "team": 3, "now_cost": 150, "status": "a", "selected_by_percent": "50.0"},
        {"id": 5, "code": 105, "web_name": "Rogers", "element_type": 3, "team": 4, "now_cost": 50, "status": "a", "selected_by_percent": "8.0"},
    ])
    teams = pd.DataFrame([
        {"id": 1, "short_name": "ARS"},
        {"id": 2, "short_name": "LIV"},
        {"id": 3, "short_name": "MCI"},
        {"id": 4, "short_name": "AVL"},
    ])
    xp = pd.DataFrame({
        8: [5.0, 5.5, 9.0, 10.0, 6.0],
    }, index=pd.Index([101, 102, 103, 104, 105]))

    fixtures = pd.DataFrame([
        {"gw": 8, "team": "ARS", "opponent": "EVE", "was_home": True, "difficulty": 2},
        {"gw": 8, "team": "LIV", "opponent": "BHA", "was_home": True, "difficulty": 2},
        {"gw": 8, "team": "MCI", "opponent": "AVL", "was_home": False, "difficulty": 3},
        {"gw": 8, "team": "AVL", "opponent": "MCI", "was_home": True, "difficulty": 4},
    ])

    report = scout.build_scout_report(players, teams, xp, gw=8, fixtures_all=fixtures, top_n=3)

    assert report.gw == 8
    assert len(report.captains) > 0
    assert report.captains[0].player.name == "Haaland"
    assert report.top_by_position["FWD"][0].name == "Haaland"
    assert report.top_by_position["MID"][0].name == "Salah"

    # Rogers: price <= 60, xP = 6.0 -> value enabler and differential (ownership 8%)
    assert any(p.name == "Rogers" for p in report.value_picks)
    assert any(p.name == "Rogers" for p in report.differentials)

    rendered = scout.render_scout_report(report)
    assert "GAMEWEEK 8 PLAYER SCOUT" in rendered
    assert "Haaland" in rendered
    assert "Salah" in rendered
