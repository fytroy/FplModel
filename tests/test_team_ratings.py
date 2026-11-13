"""Offline tests for fpl/team_ratings.py and the rating features."""
import numpy as np
import pandas as pd
import pytest

from conftest import make_matches
from fpl import team_ratings
from fpl.features import FeatureBuilder
from fpl.matches import TEAM_ROW_COLUMNS


def league(seed=0, n_rounds=20):
    """6 teams; 'Wall' concedes little xG, 'Sieve' a lot."""
    rng = np.random.default_rng(seed)
    teams = ["Wall", "Sieve", "A", "B", "C", "D"]
    concede = {"Wall": 0.5, "Sieve": 2.5, "A": 1.3, "B": 1.3, "C": 1.3, "D": 1.3}
    rows, start = [], pd.Timestamp("2025-08-16", tz="UTC")
    fid = 0
    for r in range(n_rounds):
        order = rng.permutation(teams)
        for h, a in zip(order[::2], order[1::2]):
            fid += 1
            when = start + pd.Timedelta(days=7 * r)
            for team, opp, home in [(h, a, True), (a, h, False)]:
                rows.append({"season": "2025-26", "gw": r + 1, "fixture_id": fid, "kickoff_time": when,
                             "team": team, "opponent": opp, "was_home": home,
                             "xg_for": rng.gamma(4, concede[opp] / 4)})
    return pd.DataFrame(rows)


def ask(team, opponent, asof):
    return pd.DataFrame({"team": [team], "opponent": [opponent], "was_home": [True], "asof": [asof]})


def test_ratings_see_strong_and_weak_defences():
    tm = league()
    asof = pd.Timestamp("2026-06-01", tz="UTC")
    vs_wall = team_ratings.rating_values(ask("A", "Wall", asof), tm).iloc[0]
    vs_sieve = team_ratings.rating_values(ask("A", "Sieve", asof), tm).iloc[0]
    assert vs_wall["rating_xg_for"] < 0.8 < 1.8 < vs_sieve["rating_xg_for"]


def test_ratings_ignore_future_matches():
    tm = league()
    asof = pd.Timestamp("2025-12-01", tz="UTC")
    before = team_ratings.rating_values(ask("A", "Wall", asof), tm).iloc[0]
    changed = tm.copy()
    changed.loc[changed["kickoff_time"] >= asof, "xg_for"] = 9.0
    team_ratings._cache.clear()
    after = team_ratings.rating_values(ask("A", "Wall", asof), changed).iloc[0]
    assert after["rating_xg_for"] == pytest.approx(before["rating_xg_for"])


def test_too_little_history_gives_nan():
    tm = league(n_rounds=2)
    out = team_ratings.rating_values(ask("A", "Wall", pd.Timestamp("2026-01-01", tz="UTC")), tm)
    assert out.isna().all(axis=None)


def test_team_features_include_ratings_and_promoted_prior():
    """Echo is promoted for 2025-26: before its first match its rating is the promoted prior."""
    history = make_matches(gws=30)
    start = pd.Timestamp("2025-08-16", tz="UTC")           # 2025-26 opening kickoff in conftest
    builder = FeatureBuilder(history[history.kickoff_time < start])
    rows = pd.DataFrame([{"season": "2025-26", "gw": 1, "fixture_id": 1, "kickoff_time": start,
                          "team": team, "opponent": opp, "was_home": home, "asof": start}
                         for team, opp, home in [("Echo", "Alpha", True), ("Alpha", "Echo", False)]])
    out = builder.team_features(rows[TEAM_ROW_COLUMNS + ["asof"]])
    for col in team_ratings.RATING_FEATURES:
        assert col in out and out[col].notna().all()
    assert out["rating_cs"].between(0, 1).all()
    promoted = builder.team_priors["promoted"]
    assert out.loc[0, "rating_xg_against"] == pytest.approx(promoted["xg_against"])


def history_with_unpublished_xg():
    """make_matches, but 2024-25's first 10 gameweeks have zero xG (FPL hadn't published it)."""
    from fpl import matches
    h = make_matches(gws=30)
    early = (h["season"] == "2024-25") & (h["gw"] <= 10)
    h.loc[early, matches.EXPECTED_COLUMNS] = 0.0
    return matches.blank_unpublished_expected(h), early


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_unpublished_xg_is_ignored_by_ratings_and_features():
    h, early = history_with_unpublished_xg()
    builder = FeatureBuilder(h)
    tm = builder.team_matches
    gap = (tm["season"] == "2024-25") & (tm["gw"] <= 10)
    assert tm.loc[gap, ["xg_for", "xg_against"]].isna().all(axis=None)
    assert tm.loc[~gap, ["xg_for", "xg_against"]].notna().all(axis=None)
    asof = pd.Timestamp("2025-09-01", tz="UTC")
    team_ratings._cache.clear()
    fit = team_ratings.fit_ratings(tm, asof)
    assert np.isfinite(fit["base"])
    team_ratings._cache.clear()
    known_only = team_ratings.fit_ratings(tm[~gap], asof)
    assert fit["base"] == pytest.approx(known_only["base"])
    # Rolling team xG right after the gap uses only matches with known xG.
    state = builder.team_state[gap.to_numpy()]
    assert state["xg_for__sum"].isna().all()
    # A player's xG per 90 counts only minutes with known xG.
    rows = h[(h["season"] == "2024-25") & (h["gw"] == 14) & (h["position"] == "FWD")].head(1)
    code = rows["player_code"].iloc[0]
    rows = rows.assign(asof=rows["kickoff_time"])
    feats = builder.player_features(rows)
    past = h[(h["player_code"] == code) & (h["kickoff_time"] < rows["asof"].iloc[0])]
    known = past[past["expected_goals"].notna()].tail(38)
    assert known["minutes"].sum() > 0
    xg90_raw = known["expected_goals"].sum() / known["minutes"].sum() * 90
    n = (past["minutes"].where(past["expected_goals"].notna(), 0) > 0).sum()
    from fpl.features import PLAYER_K, price_band
    prior = builder.player_priors.loc[(rows["position"].iloc[0], price_band(rows["price"].iloc[0])),
                                      "xg90"]
    assert feats["xg90_38"].iloc[0] == pytest.approx((n * xg90_raw + PLAYER_K * prior) / (n + PLAYER_K))
