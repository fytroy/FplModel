"""Offline tests for fpl/scoring.py."""
import math

import numpy as np
import pandas as pd
import pytest

from fpl import scoring

STATS = ["minutes", "goals_scored", "assists", "clean_sheets", "goals_conceded", "saves",
         "yellow_cards", "red_cards", "penalties_saved", "penalties_missed", "own_goals", "bonus"]


def match(**values):
    """One match-table row: zeros unless given, DefCon columns unknown."""
    row = {"season": "2026-27", "position": "MID", **{s: 0 for s in STATS},
           "clearances_blocks_interceptions": np.nan, "tackles": np.nan, "recoveries": np.nan}
    row.update(values)
    return row


def points(*rows):
    return list(scoring.actual_points(pd.DataFrame(rows)))


def test_goalkeeper_goal_is_worth_more_from_2026_27():
    assert points(match(season="2025-26", position="GK", minutes=90, goals_scored=1),
                  match(season="2026-27", position="GK", minutes=90, goals_scored=1)) == [8, 12]


def test_defcon_thresholds_by_position():
    dc = dict(season="2025-26", minutes=90)
    rows = [
        match(position="DEF", clearances_blocks_interceptions=6, tackles=4, recoveries=0, **dc),  # 10 CBIT: yes
        match(position="DEF", clearances_blocks_interceptions=5, tackles=4, recoveries=9, **dc),  # 9 CBIT: no
        match(position="MID", clearances_blocks_interceptions=6, tackles=5, recoveries=1, **dc),  # 12 CBIRT: yes
        match(position="FWD", clearances_blocks_interceptions=6, tackles=5, recoveries=0, **dc),  # 11 CBIRT: no
        match(position="GK", clearances_blocks_interceptions=20, tackles=5, recoveries=5, **dc),  # GK: never
    ]
    hits = scoring.defcon_hit(pd.DataFrame(rows))
    assert list(hits.iloc[:4]) == [1.0, 0.0, 1.0, 0.0] and np.isnan(hits.iloc[4])
    assert points(*rows) == [4, 2, 4, 2, 2]


def test_goalkeeper_defcon_is_unknown_not_zero():
    gk = pd.DataFrame([match(season="2025-26", position="GK", minutes=90,
                             clearances_blocks_interceptions=20, tackles=5, recoveries=5)])
    assert scoring.defcon_hit(gk).isna().all()


def test_no_defcon_points_before_2025_26():
    row = match(season="2024-25", position="DEF", minutes=90,
                clearances_blocks_interceptions=20, tackles=5, recoveries=0)
    assert points(row) == [2]


def test_other_events():
    rows = [
        match(position="DEF", minutes=90, clean_sheets=1, assists=1, yellow_cards=1),  # 2+4+3-1
        match(position="GK", minutes=90, goals_conceded=3, saves=7, penalties_saved=1),  # 2-1+2+5
        match(position="FWD", minutes=30, red_cards=1, own_goals=1, penalties_missed=1, bonus=2),  # 1-3-2-2+2
        match(position="MID", minutes=0),
    ]
    assert points(*rows) == [8, 8, -4, 0]


def test_rules_era():
    assert [scoring.rules_era(s) for s in ["2022-23", "2025-26", "2026-27"]] == \
        ["pre_2025", "2025_26", "2026_27"]


@pytest.mark.parametrize("lam,k", [(0.3, 2), (1.0, 2), (2.7, 2), (3.5, 3), (6.0, 3)])
def test_expected_floor_div_matches_brute_force(lam, k):
    brute = sum((n // k) * math.exp(-lam) * lam ** n / math.factorial(n) for n in range(60))
    assert scoring.expected_floor_div([lam], k)[0] == pytest.approx(brute, rel=1e-9)


def test_expected_floor_div_of_zero_is_zero():
    assert scoring.expected_floor_div([0.0], 2)[0] == pytest.approx(0.0)


def prediction(**values):
    row = {"season": "2026-27", "position": "DEF", "p_plays": 1.0, "p_60": 1.0, "xg": 0.0,
           "xa": 0.0, "p_cs": 0.0, "gc_lambda": 0.0, "saves_lambda": 0.0, "p_yellow": 0.0,
           "p_red": 0.0, "p_defcon": 0.0, "e_bonus": 0.0}
    row.update(values)
    return row


def test_expected_points_hand_worked_defender():
    out = scoring.expected_points(pd.DataFrame([prediction(p_cs=0.5, p_defcon=0.5)]))
    # 2 for 60+ minutes, 0.5 x 4 clean sheet, 0.5 x 2 DefCon
    assert out["xp"].iloc[0] == pytest.approx(5.0)
    assert list(out.columns[-len(scoring.POINT_PARTS) - 1:]) == scoring.POINT_PARTS + ["xp"]


def test_clean_sheet_needs_sixty_minutes():
    out = scoring.expected_points(pd.DataFrame([prediction(p_plays=0.8, p_60=0.5, p_cs=0.4)]))
    assert out["pts_clean_sheet"].iloc[0] == pytest.approx(0.5 * 0.4 * 4)


def test_expected_points_use_that_seasons_goal_value():
    rows = [prediction(season="2025-26", position="GK", xg=0.1),
            prediction(season="2026-27", position="GK", xg=0.1)]
    assert list(scoring.expected_points(pd.DataFrame(rows))["pts_goals"]) == \
        pytest.approx([0.6, 1.0])


def test_goalkeepers_get_no_defcon_and_outfielders_no_saves():
    rows = [prediction(position="GK", p_defcon=0.9, saves_lambda=0.0),
            prediction(position="MID", saves_lambda=9.0)]
    out = scoring.expected_points(pd.DataFrame(rows))
    assert out["pts_defcon"].iloc[0] == 0 and out["pts_saves"].iloc[1] == 0


from pathlib import Path

REAL_SAMPLE = Path(__file__).parent / "sample_matches.csv"


def test_rules_rebuild_official_points_on_real_matches():
    sample = pd.read_csv(REAL_SAMPLE)
    assert {"2022-23", "2023-24", "2024-25", "2025-26", "2026-27"} <= set(sample["season"])
    assert (scoring.actual_points(sample) == sample["total_points"]).all()
