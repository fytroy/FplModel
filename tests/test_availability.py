"""Offline tests for fpl/availability.py, using the user's four flagged players' real news."""
from datetime import date

import pandas as pd
import pytest

from fpl import availability

WEEK_ENDS = {gw: pd.Timestamp(t, tz="UTC") for gw, t in {
    7: "2026-10-17 10:00", 8: "2026-10-24 10:00", 9: "2026-10-31 11:00",
    10: "2026-11-07 11:00", 11: "2026-11-21 11:00", 12: "2026-11-28 11:00"}.items()}
GWS = list(WEEK_ENDS)
CURVE = pd.Series([0.5, 0.4, 0.3, 0.3, 0.2, 0.2, 0.2, 0.1], index=range(1, 9))


def player(code, status, news, chance, added="2026-10-02T15:00:11Z"):
    return {"code": code, "status": status, "news": news, "news_added": added,
            "chance_of_playing_next_round": chance}


PLAYERS = pd.DataFrame([
    player(1, "u", "Has joined York City on loan for the rest of the season", 0.0),      # McNally
    player(2, "i", "Hamstring injury - Expected back 24 Oct", 0.0),                       # Tzolis
    player(3, "i", "Thigh injury - Unknown return date", 0.0, "2026-09-27T10:30:09Z"),    # Isak
    player(4, "d", "Ankle injury - 75% chance of playing", 75.0),                         # Robinson
    player(5, "s", "Suspended until 31 Oct", 0.0),
    player(6, "a", "", None),
])


def table():
    avail = availability.weekly_availability(PLAYERS, GWS, WEEK_ENDS, CURVE, pd.Series({3: 2}))
    return avail.pivot(index="player_code", columns="gw", values="availability")


def test_each_news_form_gives_the_right_weekly_chance():
    t = table()
    assert list(t.loc[1]) == [0, 0, 0, 0, 0, 0]
    assert list(t.loc[2]) == [0, 0.8, 1, 1, 1, 1]
    assert list(t.loc[4]) == [0.75, 0.875, 1, 1, 1, 1]
    assert list(t.loc[5]) == [0, 0, 0.8, 1, 1, 1]
    assert 6 not in t.index                                     # fit players aren't listed


def test_unknown_return_uses_fpls_chance_then_the_recovery_curve():
    isak = table().loc[3]
    assert isak[7] == 0                                         # FPL says 0% for next week
    assert isak[8] == pytest.approx(0.3)                        # missed 2 + GW7 = 3: curve[3]
    assert isak[9] == pytest.approx(1 - 0.7 * 0.7)              # then curve[4]
    assert isak.is_monotonic_increasing and isak[12] < 1


def test_unknown_return_without_a_chance_uses_the_curve_from_next_week():
    nameless = PLAYERS.iloc[[2]].assign(chance_of_playing_next_round=None)
    avail = availability.weekly_availability(nameless, GWS, WEEK_ENDS, CURVE, pd.Series({3: 2}))
    assert avail["availability"].iloc[0] == pytest.approx(0.4)  # curve[2]


def test_return_date_handles_odd_text():
    added = pd.Timestamp("2026-12-20", tz="UTC")
    assert availability.return_date("Knee injury - Expected back 05 Jan", added) == date(2027, 1, 5)
    assert availability.return_date("Knee injury - Expected back 31 Feb", added) is None
    assert availability.return_date("Knee injury - Unknown return date", added) is None
    assert availability.return_date(None, added) is None


def history(minutes_by_player):
    rows = []
    for code, minutes in minutes_by_player.items():
        for i, m in enumerate(minutes):
            rows.append({"player_code": code, "kickoff_time": pd.Timestamp("2025-08-01", tz="UTC")
                         + pd.Timedelta(days=7 * i), "minutes": m, "fixture_id": i, "season": "2025-26"})
    return pd.DataFrame(rows)


def test_recovery_curve_from_regular_starters():
    h = history({1: [90] * 12 + [0, 0, 90], 2: [90] * 12 + [0, 90], 3: [10] * 12 + [0, 0, 0, 90]})
    curve = availability.recovery_curve(h)
    assert curve[1] == pytest.approx(0.5)                       # player 3 isn't a regular starter
    assert curve.is_monotonic_decreasing and curve.between(0, 1).all()


def test_matches_missed_and_trim_absences():
    h = history({1: [90, 90, 0, 0], 2: [90, 0, 90, 90]})
    assert availability.matches_missed(h, [1, 2]).to_dict() == {1: 2, 2: 0}
    trimmed = availability.trim_absences(h, [1])
    assert len(trimmed[trimmed.player_code == 1]) == 2 and len(trimmed[trimmed.player_code == 2]) == 4


def test_apply_scales_plays_and_sixty_per_week():
    base = {"season": "2026-27", "position": "MID", "p_plays": 0.9, "p_60": 0.6, "xg": 0.5,
            "xa": 0.2, "p_cs": 0.3, "gc_lambda": 1.0, "saves_lambda": 0.0, "p_yellow": 0.1,
            "p_red": 0.0, "p_defcon": 0.2, "e_bonus": 0.4}
    pred = pd.DataFrame([{**base, "player_code": 4, "gw": 7}, {**base, "player_code": 4, "gw": 9},
                         {**base, "player_code": 6, "gw": 7}])
    out = availability.apply(pred, availability.weekly_availability(
        PLAYERS, GWS, WEEK_ENDS, CURVE, pd.Series(dtype=float)))
    assert list(out.p_plays) == pytest.approx([0.675, 0.9, 0.9])
    assert out.p_60.iloc[0] == pytest.approx(0.45) and out.xp.iloc[0] < out.xp.iloc[2]


def test_a_planned_week_missing_from_week_ends_does_not_crash():
    """A gameweek with no fixtures has no 'last kickoff': treat that week as available."""
    ends = {gw: t for gw, t in WEEK_ENDS.items() if gw != 9}
    avail = availability.weekly_availability(PLAYERS, GWS, ends, CURVE, pd.Series({3: 2}))
    t = avail.pivot(index="player_code", columns="gw", values="availability")
    assert t.loc[2].tolist() == [0.0, availability.FIRST_WEEK_BACK, 1.0, 1.0, 1.0, 1.0]
    assert t.loc[5, 9] == 1.0 and t.loc[5, 10] == availability.FIRST_WEEK_BACK
