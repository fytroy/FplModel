"""Time-weighted attack and defence ratings for every team (the clean-sheet fix).

Part 2's team features average only each team's last 6 matches, which is too
noisy: on 2025/26 predicted clean-sheet chances spanned 17-33% while real rates
differed about twice as much. Here: log(xG) = base + attack[team] +
defence[opponent] + home, fitted on every team-match before the cutoff, with a
match's weight halving every RATING_HALF_LIFE_DAYS. xG is used rather than
goals because it is less noisy. Fits are cached by cutoff.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor

from fpl.settings import RATING_HALF_LIFE_DAYS

RATING_FEATURES = ["rating_xg_for", "rating_xg_against", "rating_cs"]
MIN_MATCHES = 40          # team-matches needed before a fit is trusted
_cache: dict = {}


def fit_ratings(team_matches: pd.DataFrame, asof: pd.Timestamp) -> dict | None:
    """Ratings from matches strictly before `asof`; None if there are too few."""
    past = team_matches[(team_matches["kickoff_time"] < asof) & team_matches["xg_for"].notna()]
    if len(past) < MIN_MATCHES:
        return None
    key = (asof, len(past), round(float(past["xg_for"].sum()), 6))
    if key in _cache:
        return _cache[key]
    teams = sorted(set(past["team"]) | set(past["opponent"]))
    index = {t: i for i, t in enumerate(teams)}
    n, rows = len(teams), np.arange(len(past))
    X = np.zeros((len(past), 2 * n + 1))
    X[rows, past["team"].map(index).to_numpy()] = 1
    X[rows, n + past["opponent"].map(index).to_numpy()] = 1
    X[:, 2 * n] = past["was_home"].astype(float).to_numpy()
    age_days = (asof - past["kickoff_time"]).dt.total_seconds().to_numpy() / 86400
    model = PoissonRegressor(alpha=1e-3, max_iter=300).fit(
        X, past["xg_for"].to_numpy(), sample_weight=0.5 ** (age_days / RATING_HALF_LIFE_DAYS))
    coef = model.coef_
    fit = {"base": float(model.intercept_), "home": float(coef[2 * n]),
           "attack": dict(zip(teams, coef[:n])), "defence": dict(zip(teams, coef[n:2 * n]))}
    _cache[key] = fit
    return fit


def rating_values(rows: pd.DataFrame, team_matches: pd.DataFrame) -> pd.DataFrame:
    """Ratings' expected goals for and against each row (team, opponent, was_home, asof).

    NaN where a team is unrated (no fit yet, or a team with no past matches).
    """
    out = np.full((len(rows), 2), np.nan)
    for asof, idx in rows.groupby("asof").indices.items():
        fit = fit_ratings(team_matches, asof)
        if fit is None:
            continue
        r = rows.iloc[idx]
        home = r["was_home"].astype(float).to_numpy()
        att_t = r["team"].map(fit["attack"]).to_numpy(dtype=float)
        att_o = r["opponent"].map(fit["attack"]).to_numpy(dtype=float)
        def_t = r["team"].map(fit["defence"]).to_numpy(dtype=float)
        def_o = r["opponent"].map(fit["defence"]).to_numpy(dtype=float)
        out[idx, 0] = np.exp(fit["base"] + att_t + def_o + fit["home"] * home)
        out[idx, 1] = np.exp(fit["base"] + att_o + def_t + fit["home"] * (1 - home))
    return pd.DataFrame(out, columns=["rating_xg_for", "rating_xg_against"], index=rows.index)
