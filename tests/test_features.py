"""Offline tests for fpl/features.py on the synthetic league in conftest.py."""
import numpy as np
import pandas as pd
import pytest
from conftest import MOVER

from fpl.features import (PLAYER_FEATURES, PLAYER_HISTORY_FEATURES, RATE_FEATURES, RATE_STATS,
                          TEAM_FEATURES, WINDOWS, FeatureBuilder, price_band, team_matches)
from fpl.matches import gw_cutoffs, with_asof


def test_team_matches_pair_teams_within_the_same_season(history):
    t = team_matches(history)
    assert len(t) == 2 * len(history.drop_duplicates(["season", "fixture_id"]))
    for season in ["2024-25", "2025-26"]:  # fixture id 1 exists in both seasons
        alpha = t[(t.season == season) & (t.fixture_id == 1) & (t.team == "Alpha")].iloc[0]
        opp = t[(t.season == season) & (t.fixture_id == 1) & (t.team == alpha.opponent)].iloc[0]
        assert alpha.xg_against == pytest.approx(opp.xg_for)
        assert alpha.goals_against == opp.goals_for


def gw1_2025_team_features(history, fixtures_all):
    builder = FeatureBuilder(history)
    tm = builder.team_matches
    rows = with_asof(tm[(tm.season == "2025-26") & (tm.gw == 1)], gw_cutoffs(fixtures_all))
    return builder, builder.team_features(rows).set_index("team")


def test_promoted_team_starts_from_the_promoted_prior(history, fixtures_all):
    # The builder is deliberately given full history here to check how the promoted
    # prior is computed. In real use FeatureBuilder only receives history from before
    # the prediction cutoff, so priors never include future matches.
    builder, f = gw1_2025_team_features(history, fixtures_all)
    tm = builder.team_matches
    echo_first6 = tm[tm.team == "Echo"].head(6)
    assert f.loc["Echo", "team_xg_for"] == pytest.approx(echo_first6.xg_for.mean())
    assert f.loc["Echo", "team_goals_against"] == pytest.approx(echo_first6.goals_against.mean())
    assert builder.team_priors["promoted"]["xg_for"] == pytest.approx(echo_first6.xg_for.mean())
    assert f.loc["Echo", "team_xg_for"] != pytest.approx(builder.team_priors["league"]["xg_for"])


def test_established_team_blends_last_six_with_league_prior(history, fixtures_all):
    builder, f = gw1_2025_team_features(history, fixtures_all)
    tm = builder.team_matches
    last6 = tm[(tm.team == "Alpha") & (tm.season == "2024-25")].tail(6)
    league = tm.xg_for.mean()
    # n = 6 past matches in Alpha's current PL spell, K = 6
    assert f.loc["Alpha", "team_xg_for"] == pytest.approx((6 * last6.xg_for.mean() + 6 * league) / 12)


def test_team_features_have_all_columns_and_keep_order(history, fixtures_all):
    builder = FeatureBuilder(history)
    rows = with_asof(builder.team_matches, gw_cutoffs(fixtures_all)).sample(frac=1, random_state=3)
    f = builder.team_features(rows)
    assert (f.index == rows.index).all()
    assert list(f["team"]) == list(rows["team"])
    assert set(TEAM_FEATURES) <= set(f.columns)


def test_rows_without_asof_are_rejected(history):
    builder = FeatureBuilder(history)
    with pytest.raises(ValueError, match="asof"):
        builder.team_features(builder.team_matches)


def _features_with_future_changed(history, fixtures_all, change_gw, gw):
    """Features for GW `gw` rows, with every match from GW `change_gw`'s cutoff on altered."""
    cutoffs = gw_cutoffs(fixtures_all)
    asof = cutoffs[(cutoffs.season == "2025-26") & (cutoffs.gw == change_gw)]["cutoff"].iloc[0]
    changed = history.copy()
    future = changed["kickoff_time"] >= asof
    changed.loc[future, ["team_goals", "opp_goals"]] += 3
    changed.loc[future, "expected_goals"] += 1.0
    a, b = FeatureBuilder(history), FeatureBuilder(changed)
    b.team_priors = a.team_priors  # priors depend on the given history by design; pin them
    cols = [c for c in TEAM_FEATURES if c.startswith(("team_", "opp_"))]
    out = []
    for builder in (a, b):
        tm = builder.team_matches
        rows = with_asof(tm[(tm.season == "2025-26") & (tm.gw == gw)], cutoffs)
        out.append(builder.team_features(rows)[cols].reset_index(drop=True))
    return out


def test_team_features_ignore_matches_at_or_after_asof(history, fixtures_all):
    a, b = _features_with_future_changed(history, fixtures_all, 3, 3)
    pd.testing.assert_frame_equal(a, b)
    a5, b5 = _features_with_future_changed(history, fixtures_all, 3, 5)  # sensitivity: GW5 does see GW3 on
    assert not a5.equals(b5)

def player_rows(history, fixtures_all, season, gw, codes=None):
    rows = history[(history.season == season) & (history.gw == gw)]
    if codes is not None:
        rows = rows[rows.player_code.isin(codes)]
    return with_asof(rows, gw_cutoffs(fixtures_all))


def test_features_ignore_later_matches(history, fixtures_all):
    rows = player_rows(history, fixtures_all, "2025-26", 3)
    cutoff = rows["asof"].iloc[0]
    later = (history.kickoff_time >= cutoff) & ~history.player_code.between(500, 599)  # not newcomers
    changed = history.copy()
    changed.loc[later, "threat"] *= 10
    changed.loc[later, "bps"] += 25
    before = FeatureBuilder(history).player_features(rows)[PLAYER_FEATURES]
    after = FeatureBuilder(changed).player_features(rows)[PLAYER_FEATURES]
    pd.testing.assert_frame_equal(before, after)
    # ...and the change is visible to rows after it, so the test can fail:
    rows5 = player_rows(history, fixtures_all, "2025-26", 5)
    assert not FeatureBuilder(history).player_features(rows5)["threat90_3"].equals(
        FeatureBuilder(changed).player_features(rows5)["threat90_3"])


def test_newcomer_features_equal_the_prior(history, fixtures_all):
    builder = FeatureBuilder(history)
    f = builder.player_features(player_rows(history, fixtures_all, "2025-26", 1, [503])).iloc[0]
    prior = builder.player_priors.loc[("FWD", price_band(9.0))]
    assert f["matches_played"] == 0
    for name, *_ in RATE_STATS:
        for w in WINDOWS:
            assert f[f"{name}_{w}"] == pytest.approx(prior[name])
            assert not np.isnan(f[f"{name}_{w}"])


def test_prior_fades_with_total_matches(history, fixtures_all):
    builder = FeatureBuilder(history)
    rows = player_rows(history, fixtures_all, "2025-26", 4, [102])
    f = builder.player_features(rows).iloc[0]
    past = history[(history.player_code == 102) & (history.kickoff_time < rows["asof"].iloc[0])]
    n, own = len(past), past.sort_values("kickoff_time").minutes.tail(3).mean()
    prior = builder.player_priors.loc[("MID", price_band(7.5)), "minutes"]
    assert n == 9  # total matches, not capped at the window size of 3
    assert f["minutes_3"] == pytest.approx((n * own + 5 * prior) / (n + 5))


def test_season_and_team_counters(history, fixtures_all):
    f = FeatureBuilder(history).player_features(
        player_rows(history, fixtures_all, "2025-26", 3, [102, MOVER])).set_index("player_code")
    assert (f.loc[102, "matches_played"], f.loc[102, "matches_this_season"],
            f.loc[102, "matches_for_current_team"]) == (8, 2, 8)
    assert (f.loc[MOVER, "matches_played"], f.loc[MOVER, "matches_this_season"],
            f.loc[MOVER, "matches_for_current_team"]) == (8, 2, 2)


def test_transferred_player_uses_his_new_teams_strength(history, fixtures_all):
    f = FeatureBuilder(history).player_features(
        player_rows(history, fixtures_all, "2025-26", 1, [200, MOVER])).set_index("player_code")
    assert f.loc[MOVER, "team"] == "Bravo"
    assert f.loc[MOVER, "team_xg_for"] == pytest.approx(f.loc[200, "team_xg_for"])
    assert f.loc[MOVER, "matches_for_current_team"] == 0
    assert f.loc[MOVER, "matches_played"] == 6


def test_both_fixtures_of_a_double_gameweek_get_the_same_player_features(history, fixtures_all):
    rows = player_rows(history, fixtures_all, "2025-26", 3, [102])
    second = rows.assign(fixture_id=99, opponent="Echo")
    f = FeatureBuilder(history).player_features(pd.concat([rows, second], ignore_index=True))
    cols = RATE_FEATURES + PLAYER_HISTORY_FEATURES
    pd.testing.assert_series_equal(f.loc[0, cols], f.loc[1, cols], check_names=False)


def test_pts_last5_is_points_per_fixture_over_last_five(history, fixtures_all):
    rows = player_rows(history, fixtures_all, "2025-26", 3, [102])
    f = FeatureBuilder(history).player_features(rows).iloc[0]
    past = history[(history.player_code == 102) & (history.kickoff_time < rows["asof"].iloc[0])]
    assert f["pts_last5"] == pytest.approx(past.sort_values("kickoff_time").total_points.tail(5).mean())


def test_player_features_keep_the_input_row_order(history, fixtures_all):
    rows = with_asof(history[history.season == "2025-26"], gw_cutoffs(fixtures_all))
    rows = rows.sample(frac=1, random_state=1)
    f = FeatureBuilder(history).player_features(rows)
    assert (f.index == rows.index).all()
    assert (f["player_code"].to_numpy() == rows["player_code"].to_numpy()).all()
    assert set(PLAYER_FEATURES) <= set(f.columns)


def test_price_band():
    assert [price_band(p) for p in [4.0, 5.0, 5.5, 7.0, 7.5, 9.0, 9.5, 14.0]] == \
        [0, 0, 1, 1, 2, 2, 3, 3]
