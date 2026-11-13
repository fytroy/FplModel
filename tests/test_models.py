"""Offline tests for fpl/models.py on a larger synthetic league."""
import numpy as np
import pandas as pd
import pytest

from conftest import make_fixtures, make_matches
from fpl import models
from fpl.features import team_matches
from fpl.matches import PLAYER_ROW_COLUMNS, TEAM_ROW_COLUMNS, gw_cutoffs


@pytest.fixture(scope="module")
def split():
    history = make_matches(gws=30)
    cutoffs = gw_cutoffs(make_fixtures(history))
    cutoff = cutoffs[(cutoffs.season == "2025-26") & (cutoffs.gw == 25)].cutoff.iloc[0]
    target = history[(history.season == "2025-26") & (history.gw == 25)]
    players = target[PLAYER_ROW_COLUMNS].assign(asof=cutoff)
    teams = team_matches(target)[TEAM_ROW_COLUMNS].assign(asof=cutoff)
    return history, history[history.kickoff_time < cutoff], players, teams, cutoffs


def test_train_and_predict_gives_sensible_parts(split):
    _, past, players, teams, cutoffs = split
    pred = models.train_and_predict(past, players, teams, cutoffs)
    assert len(pred) == len(players)
    assert pred.p_plays.between(0, 1).all() and (pred.p_60 <= pred.p_plays + 1e-12).all()
    assert pred.p_cs.between(0, 1).all()
    assert (pred.loc[pred.position == "GK", "p_defcon"] == 0).all()
    assert (pred.loc[pred.position != "GK", "saves_lambda"] == 0).all()
    assert np.isfinite(pred[["xp", "xp_b", "baseline"]].to_numpy()).all()


def test_a_brand_new_player_gets_a_finite_prediction(split):
    _, past, players, teams, cutoffs = split
    newcomer = players.iloc[[0]].assign(player_code=12345, player_name="New", price=6.0)
    pred = models.train_and_predict(past, pd.concat([players, newcomer], ignore_index=True),
                                    teams, cutoffs)
    new = pred[pred.player_code == 12345].iloc[0]
    assert np.isfinite(new.xp) and 0 < new.p_plays < 1


def test_train_and_predict_refuses_future_history(split):
    history, _, players, teams, cutoffs = split
    with pytest.raises(ValueError, match="at or after"):
        models.train_and_predict(history, players, teams, cutoffs)


def test_fit_part_falls_back_to_the_average_with_few_rows():
    part = models.fit_part("binary", pd.DataFrame({"a": [1, 2, 3]}), pd.Series([0, 1, 1]))
    assert part.predict(pd.DataFrame({"a": [9]}))[0] == pytest.approx(2 / 3)
    empty = models.fit_part("binary", pd.DataFrame({"a": []}), pd.Series([], dtype=float))
    assert empty.predict(pd.DataFrame({"a": [9]}))[0] == 0


def test_clean_sheet_model_does_not_memorise_noise():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(3000, 8)))
    y = pd.Series(rng.random(3000) < 0.25)   # no signal at all
    loose = models.fit_part("binary", X, y).predict(X)
    tight = models.fit_part("binary", X, y, params=models.CLEAN_SHEET_PARAMS).predict(X)
    assert loose.std() > 0.15                    # default params memorise
    assert tight.max() < 0.45 and tight.std() < 0.05


def test_xp_model_fits_clean_sheet_with_the_regularised_params(split):
    _, past, _, _, cutoffs = split
    b = models.FeatureBuilder(past)
    model = models.XPModel().fit(b.player_features(models.with_asof(b.history, cutoffs)),
                                 b.team_features(models.with_asof(b.team_matches, cutoffs)))
    assert model.clean_sheet.model.get_params()["num_leaves"] == models.CLEAN_SHEET_PARAMS["num_leaves"]
    assert model.conceded.model.get_params()["num_leaves"] == models.LGB_PARAMS["num_leaves"]


def test_defcon_model_does_not_memorise_noise():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(3000, 8)))
    y = pd.Series(rng.random(3000) < 0.15)   # no signal at all
    loose = models.fit_part("binary", X, y).predict(X)
    tight = models.fit_part("binary", X, y, params=models.DEFCON_PARAMS).predict(X)
    assert loose.std() > 0.15                    # default params memorise
    assert tight.std() < 0.08 and tight.max() < 0.6


def test_xp_model_fits_defcon_with_the_regularised_params(split):
    _, past, _, _, cutoffs = split
    b = models.FeatureBuilder(past)
    model = models.XPModel().fit(b.player_features(models.with_asof(b.history, cutoffs)),
                                 b.team_features(models.with_asof(b.team_matches, cutoffs)))
    assert model.defcon.model.get_params()["num_leaves"] == models.DEFCON_PARAMS["num_leaves"]
