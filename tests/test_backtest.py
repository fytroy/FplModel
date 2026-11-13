"""Offline tests for fpl/backtest.py."""
import numpy as np
import pandas as pd
import pytest

from fpl import backtest

QUIET = dict(log=lambda message: None)


def test_walk_forward_only_trains_on_the_past(history, fixtures_all, monkeypatch):
    seen = []

    def fake_train_and_predict(past, player_rows, team_rows, cutoffs):
        seen.append((past.kickoff_time.max(), player_rows["asof"].iloc[0], set(player_rows.gw)))
        return player_rows.assign(xp=1.0, xp_b=1.0, baseline=1.0, p_plays=0.5, p_60=0.4,
                                  p_cs=0.3, p_defcon=0.2)

    monkeypatch.setattr(backtest, "train_and_predict", fake_train_and_predict)
    preds = backtest.walk_forward(history, fixtures_all, "2025-26", [2, 3], horizon=2, **QUIET)
    assert all(past_max < asof for past_max, asof, _ in seen)
    assert [gws for *_, gws in seen] == [{2, 3}, {3, 4}]
    assert set(preds.horizon) == {1, 2}
    assert preds.total_points.notna().all()


def test_summarize_scores_each_method():
    i = np.arange(25)
    preds = pd.DataFrame({"season": "2025-26", "origin_gw": 1, "horizon": 1, "gw": 1,
                          "fixture_id": i, "player_code": i, "position": "MID",
                          "xp": i.astype(float), "xp_b": (24 - i).astype(float),
                          "baseline": 1.0, "total_points": i})
    s = backtest.summarize(preds).set_index("method")
    assert s.loc["A (parts)", "mae"] == 0
    assert s.loc["A (parts)", "spearman"] == pytest.approx(1)
    assert s.loc["B (total points)", "spearman"] == pytest.approx(-1)
    assert s.loc["A (parts)", "top20_points"] == sum(range(5, 25))
    assert s.loc["B (total points)", "top20_points"] == sum(range(20))


def test_double_gameweek_fixtures_are_added_together():
    preds = pd.DataFrame({"season": "2025-26", "origin_gw": 1, "horizon": 1, "gw": 1,
                          "fixture_id": [1, 2], "player_code": 7, "position": "MID",
                          "xp": [2.0, 3.0], "xp_b": [1.0, 1.0], "baseline": [4.0, 4.0],
                          "total_points": [4, 5]})
    row = backtest.per_player_gw(preds).iloc[0]
    assert (row.xp, row.xp_b, row.baseline, row.total_points) == (5.0, 2.0, 8.0, 9)


def test_walk_forward_runs_end_to_end(history, fixtures_all):
    preds = backtest.walk_forward(history, fixtures_all, "2025-26", [3], horizon=2, **QUIET)
    assert set(preds.gw) == {3, 4}
    assert preds.xp.notna().all()
    assert len(backtest.summarize(preds)) == 2 * len(backtest.METHODS)
    calib = backtest.calibration(preds)
    assert set(calib.part) <= {"p_plays", "p_60", "p_cs", "p_defcon"}
    assert calib.predicted.between(0, 1).all()


def test_clean_sheet_report_on_hand_made_numbers():
    preds = pd.DataFrame({"horizon": 1, "season": "s", "fixture_id": range(10), "team": "T",
                          "p_cs": [0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5],
                          "opp_goals": [1, 1, 0, 0, 0, 1, 1, 1, 0, 0]})
    cs = backtest.clean_sheet_report(preds)
    assert cs["p5"] == pytest.approx(0.1) and cs["p95"] == pytest.approx(0.5)
    assert cs["actual_by_fifth"] == [0.0, 0.0, 0.5, 1.0, 1.0]
