"""Offline tests for fpl/benchmarks.py."""
import pandas as pd

from fpl import benchmarks


def test_average_manager_from_events():
    events = [{"id": 1, "average_entry_score": 54}, {"id": 2, "average_entry_score": 51}]
    avg = benchmarks.average_manager(events)
    assert avg["average"].sum() == 105 and list(avg["gw"]) == [1, 2]


def test_estimate_rank_interpolates_and_is_monotonic():
    sample = pd.DataFrame({"points": [1800, 1900, 2000, 2100, 2061],
                           "rank": [8_000_000, 5_000_000, 3_000_000, 1_500_000, 2_927_134]})
    r2061, _ = benchmarks.estimate_rank(2061, sample)
    assert 1_500_000 < r2061 < 3_000_000
    assert benchmarks.estimate_rank(1950, sample)[0] > benchmarks.estimate_rank(2050, sample)[0]
    top, note = benchmarks.estimate_rank(2500, sample)
    assert top == 1_500_000 and "better than" in note
