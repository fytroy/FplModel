"""The xP models.

Approach A (XPModel): one small LightGBM model per scoring part, combined with
FPL's rules. Approach B (TotalPointsModel): one LightGBM predicting total
points directly, used only as a yardstick in the backtest.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

from fpl import scoring
from fpl.features import PLAYER_FEATURES, TEAM_FEATURES, FeatureBuilder
from fpl.matches import with_asof
from fpl.team_ratings import RATING_FEATURES

CS_FEATURES = TEAM_FEATURES + RATING_FEATURES  # clean-sheet and goals-conceded models only

LGB_PARAMS = {"n_estimators": 200, "learning_rate": 0.05, "num_leaves": 31,
              "min_child_samples": 50, "subsample": 0.8, "subsample_freq": 1,
              "colsample_bytree": 0.8, "random_state": 0, "verbose": -1}
# The clean-sheet model sees only ~3,000 team-fixture rows and a weak signal; with
# LGB_PARAMS it memorises them (in-sample calibration perfect, out-of-sample flat).
CLEAN_SHEET_PARAMS = {**LGB_PARAMS, "n_estimators": 60, "learning_rate": 0.03, "num_leaves": 3,
                      "min_child_samples": 150, "reg_lambda": 20}
# DefCon exists only in recent seasons (~280 rows early in 2025/26, ~10,000 late); with
# LGB_PARAMS it memorises them (in-sample 0.79 predicted -> 1.00 actual, out-of-sample
# 0.79 -> 0.50, logloss 0.366 vs 0.331 with these).
DEFCON_PARAMS = {**LGB_PARAMS, "n_estimators": 100, "learning_rate": 0.03, "num_leaves": 7,
                 "min_child_samples": 200, "reg_lambda": 20}
MIN_TRAIN_ROWS = 200            # fewer than this: predict the plain average
BONUS_WEIGHT_CURRENT_ERA = 3.0  # lean the bonus model towards the 2026/27 BPS rules


class _Constant:
    def __init__(self, value: float):
        self.value = float(value)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.value)


class _Fitted:
    def __init__(self, model, binary: bool):
        self.model, self.binary = model, binary

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1] if self.binary else self.model.predict(X)


def fit_part(objective: str, X: pd.DataFrame, y, weight=None, params=LGB_PARAMS):
    """Fit one part ('binary', 'poisson' or 'regression'). Too little data: use the average."""
    y = pd.Series(y).astype(float)
    if len(y) < MIN_TRAIN_ROWS or y.nunique() < 2:
        return _Constant(y.mean() if len(y) else 0.0)
    if objective == "binary":
        model = lgb.LGBMClassifier(**params)
        y = y.astype(int)
    else:
        model = lgb.LGBMRegressor(objective=objective, **params)
    model.fit(X, y, sample_weight=weight)
    return _Fitted(model, objective == "binary")


class XPModel:
    """Approach A: predict each scoring part, then apply FPL's rules."""

    def fit(self, players: pd.DataFrame, teams: pd.DataFrame) -> "XPModel":
        X = players[PLAYER_FEATURES]
        played = players["minutes"] > 0
        on = players[played]
        self.plays = fit_part("binary", X, played)
        self.sixty = fit_part("binary", X[played], on["minutes"] >= 60)   # P(60+ | plays)
        self.goals = fit_part("poisson", X[played], on["goals_scored"])
        self.assists = fit_part("poisson", X[played], on["assists"])
        keepers = played & (players["position"] == "GK")
        self.saves = fit_part("poisson", X[keepers], players.loc[keepers, "saves"])
        hit = scoring.defcon_hit(players)
        with_defcon = played & hit.notna()   # outfield players, DefCon seasons only
        self.defcon = fit_part("binary", X[with_defcon], hit[with_defcon], params=DEFCON_PARAMS)
        weight = np.where(on["rules_era"] == "2026_27", BONUS_WEIGHT_CURRENT_ERA, 1.0)
        self.bonus = fit_part("poisson", X[played], on["bonus"], weight)
        T = teams[CS_FEATURES]
        self.clean_sheet = fit_part("binary", T, teams["goals_against"] == 0,
                                     params=CLEAN_SHEET_PARAMS)
        self.conceded = fit_part("poisson", T, teams["goals_against"])
        return self

    def predict(self, players: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
        X = players[PLAYER_FEATURES]
        keep = ["season", "gw", "fixture_id", "player_code", "player_name", "position", "team",
                "opponent", "was_home", "price", "asof"]
        out = players[keep].reset_index(drop=True)
        is_gk = (out["position"] == "GK").to_numpy()
        out["p_plays"] = self.plays.predict(X)
        out["p_60"] = out["p_plays"] * self.sixty.predict(X)
        out["xg"] = self.goals.predict(X)
        out["xa"] = self.assists.predict(X)
        out["saves_lambda"] = np.where(is_gk, self.saves.predict(X), 0.0)
        out["p_defcon"] = np.where(is_gk, 0.0, self.defcon.predict(X))
        out["e_bonus"] = self.bonus.predict(X)
        # Cards: his own blended per-appearance rate, no model needed.
        out["p_yellow"] = players["yellow_rate_38"].fillna(0).clip(0, 1).to_numpy()
        out["p_red"] = players["red_rate_38"].fillna(0).clip(0, 1).to_numpy()
        T = teams[CS_FEATURES]
        team_pred = teams[["season", "fixture_id", "team"]].assign(
            p_cs=self.clean_sheet.predict(T), gc_lambda=self.conceded.predict(T))
        out = out.merge(team_pred, on=["season", "fixture_id", "team"], how="left",
                         validate="many_to_one")
        return scoring.expected_points(out)


class TotalPointsModel:
    """Approach B (yardstick): one LightGBM predicting total points directly."""

    def fit(self, players: pd.DataFrame) -> "TotalPointsModel":
        self.model = fit_part("regression", players[PLAYER_FEATURES], players["total_points"])
        return self

    def predict(self, players: pd.DataFrame) -> np.ndarray:
        return self.model.predict(players[PLAYER_FEATURES])


def train_and_predict(history: pd.DataFrame, player_rows: pd.DataFrame,
                      team_rows: pd.DataFrame, cutoffs: pd.DataFrame,
                      yardstick: bool = True) -> pd.DataFrame:
    """Train A and B on `history`, then predict the given fixtures.

    history:     the match table: everything known at prediction time, nothing later
    player_rows: PLAYER_ROW_COLUMNS + asof for the fixtures to predict (no outcomes)
    team_rows:   TEAM_ROW_COLUMNS + asof for the same fixtures
    cutoffs:     matches.gw_cutoffs(), gives each training row its own asof
    yardstick:   also fit model B (xp_b); the planner and simulation skip it to save time
    Returns one row per player per fixture: parts, pts_*, xp (A), xp_b (B), baseline.
    """
    if len(history) and history["kickoff_time"].max() >= player_rows["asof"].min():
        raise ValueError("history has matches at or after the prediction time; "
                         "that would leak the future into training")
    builder = FeatureBuilder(history)
    train_players = builder.player_features(with_asof(builder.history, cutoffs))
    train_teams = builder.team_features(with_asof(builder.team_matches, cutoffs))
    model_a = XPModel().fit(train_players, train_teams)

    players = builder.player_features(player_rows)
    teams = builder.team_features(team_rows)
    pred = model_a.predict(players, teams)  # same row order as `players`
    if yardstick:
        pred["xp_b"] = TotalPointsModel().fit(train_players).predict(players)
    pred["baseline"] = players["pts_last5"].to_numpy()
    return pred



def weekly_xp(pred: pd.DataFrame, gws) -> pd.DataFrame:
    """xP per player per gameweek: fixtures summed (double gameweeks), 0 for a blank."""
    return (pred.pivot_table(index="player_code", columns="gw", values="xp", aggfunc="sum")
            .reindex(columns=list(gws)).fillna(0.0))
