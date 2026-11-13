import pandas as pd
import pytest

from fpl import optimal


def make_dummy_pool():
    # 2 GK, 5 DEF, 5 MID, 3 FWD plus extra
    players = [
        {"code": 1, "name": "Raya", "position": "GK", "team": "ARS", "price": 55},
        {"code": 2, "name": "Fabianski", "position": "GK", "team": "WHU", "price": 40},
        {"code": 3, "name": "Gabriel", "position": "DEF", "team": "ARS", "price": 60},
        {"code": 4, "name": "Saliba", "position": "DEF", "team": "ARS", "price": 60},
        {"code": 5, "name": "Gvardiol", "position": "DEF", "team": "MCI", "price": 60},
        {"code": 6, "name": "Alexander-Arnold", "position": "DEF", "team": "LIV", "price": 70},
        {"code": 7, "name": "Konsa", "position": "DEF", "team": "AVL", "price": 45},
        {"code": 8, "name": "Van de Ven", "position": "DEF", "team": "TOT", "price": 45},
        {"code": 9, "name": "Salah", "position": "MID", "team": "LIV", "price": 125},
        {"code": 10, "name": "Saka", "position": "MID", "team": "ARS", "price": 100},
        {"code": 11, "name": "Palmer", "position": "MID", "team": "CHE", "price": 105},
        {"code": 12, "name": "Rogers", "position": "MID", "team": "AVL", "price": 50},
        {"code": 13, "name": "Mbeumo", "position": "MID", "team": "BRE", "price": 70},
        {"code": 14, "name": "Winks", "position": "MID", "team": "LEI", "price": 45},
        {"code": 15, "name": "Haaland", "position": "FWD", "team": "MCI", "price": 150},
        {"code": 16, "name": "Watkins", "position": "FWD", "team": "AVL", "price": 90},
        {"code": 17, "name": "Wood", "position": "FWD", "team": "NFO", "price": 60},
        {"code": 18, "name": "Stewart", "position": "FWD", "team": "SOU", "price": 45},
    ]
    df = pd.DataFrame(players).set_index("code")
    return df


def test_solve_optimal_squad():
    pool = make_dummy_pool()
    points = pd.Series({
        1: 6.0, 2: 2.0,
        3: 5.5, 4: 5.0, 5: 5.2, 6: 6.5, 7: 4.0, 8: 3.5,
        9: 9.0, 10: 7.5, 11: 8.0, 12: 5.0, 13: 6.0, 14: 2.5,
        15: 10.0, 16: 6.5, 17: 5.0, 18: 2.0,
    })

    squad = optimal.solve_optimal_squad(points, pool, gw=1, budget=1000)
    assert squad.status == "Optimal"
    assert len(squad.starters) == 11
    assert len(squad.bench) == 4
    assert squad.captain.code == 15      # Haaland has 10.0 pts
    assert squad.vice.points >= 8.0      # Second highest in starting XI
    assert squad.total_cost <= 1000

    # Max 3 per club: ARS has 1, 3, 4, 10
    ars_count = sum(1 for p in squad.starters + squad.bench if p.team == "ARS")
    assert ars_count <= 3

    rendered = optimal.render_optimal_squad(squad)
    assert "OPTIMAL DREAM TEAM" in rendered
    assert "Haaland" in rendered
    assert "(C)" in rendered
