"""Offline tests for weekly xP and live-forecast helpers."""
import pandas as pd

from fpl import forecast, models


def test_weekly_xp_sums_doubles_and_fills_blanks():
    pred = pd.DataFrame({"player_code": [1, 1, 1, 2], "gw": [7, 8, 8, 7], "xp": [2.0, 3.0, 1.5, 4.0]})
    xp = models.weekly_xp(pred, [7, 8, 9])
    assert xp.loc[1].tolist() == [2.0, 4.5, 0.0]
    assert xp.loc[2].tolist() == [4.0, 0.0, 0.0]


def test_deadlines_from_gameweeks_are_utc():
    gws = pd.DataFrame({"id": [7], "deadline_time": ["2026-10-17T10:00:00Z"]})
    assert forecast.deadlines_from(gws)[7] == pd.Timestamp("2026-10-17 10:00", tz="UTC")


def test_week_ends_from_fixtures_is_last_kickoff_of_that_season():
    fx = pd.DataFrame({"season": ["2026-27"] * 3 + ["2025-26"], "gw": [7, 7, 8, 7],
                       "kickoff_time": pd.to_datetime(["2026-10-17 12:30", "2026-10-19 20:00",
                                                       "2026-10-24 15:00", "2026-01-01 12:00"], utc=True)})
    ends = forecast.week_ends_from(fx, "2026-27")
    assert ends == {7: pd.Timestamp("2026-10-19 20:00", tz="UTC"),
                    8: pd.Timestamp("2026-10-24 15:00", tz="UTC")}


def test_a_gameweek_without_fixtures_ends_at_its_deadline():
    fx = pd.DataFrame({"season": ["2026-27"], "gw": [7],
                       "kickoff_time": pd.to_datetime(["2026-10-19 20:00"], utc=True)})
    deadlines = {7: pd.Timestamp("2026-10-17 10:00", tz="UTC"), 8: pd.Timestamp("2026-10-24 10:00", tz="UTC")}
    ends = forecast.week_ends_from(fx, "2026-27", deadlines)
    assert ends == {7: pd.Timestamp("2026-10-19 20:00", tz="UTC"), 8: deadlines[8]}
