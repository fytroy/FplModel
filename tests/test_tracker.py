import pandas as pd
import pytest

from fpl import tracker


def test_parse_team_performance():
    entry = {"name": "Test FC", "summary_overall_points": 200, "summary_overall_rank": 50000}
    history = {
        "current": [
            {"event": 1, "points": 60, "total_points": 60, "rank": 100000, "overall_rank": 100000, "event_transfers_cost": 0, "points_on_bench": 4, "bank": 10, "event_transfers": 0},
            {"event": 2, "points": 70, "total_points": 126, "rank": 80000, "overall_rank": 60000, "event_transfers_cost": 4, "points_on_bench": 8, "bank": 10, "event_transfers": 2},
        ],
        "chips": [{"name": "wildcard", "event": 2, "time": "2026-08-27T18:00:00Z"}]
    }

    perf = tracker.parse_team_performance(12345, entry, history)
    assert perf.team_id == 12345
    assert perf.name == "Test FC"
    assert len(perf.gameweeks) == 2
    assert perf.gameweeks[0].net_points == 60
    assert perf.gameweeks[1].net_points == 66  # 70 - 4
    assert perf.gameweeks[1].chip == "wildcard"
    assert perf.total_hits_cost == 4
    assert perf.total_bench_points == 12
    assert perf.avg_points == 63.0
    assert perf.best_gw == (2, 66)
    assert perf.worst_gw == (1, 60)

    table = tracker.render_history_table(perf)
    assert "TEST FC" in table
    assert "GW1" in table
    assert "GW2" in table
    assert "Wildcard" in table
