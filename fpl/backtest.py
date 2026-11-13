"""Walk-forward backtest.

For each gameweek: train only on matches before it, predict it (and the next
few weeks), then compare with what actually happened.
"""
import numpy as np
import pandas as pd

from fpl import scoring
from fpl.features import team_matches
from fpl.matches import PLAYER_ROW_COLUMNS, TEAM_ROW_COLUMNS, gw_cutoffs
from fpl.models import train_and_predict

TOP_N = 20
METHODS = {"A (parts)": "xp", "B (total points)": "xp_b", "Baseline (last 5)": "baseline"}


def _print(message: str) -> None:
    print(message, flush=True)


def walk_forward(history: pd.DataFrame, fixtures_all: pd.DataFrame, season: str, gws,
                 horizon: int = 1, log=_print) -> pd.DataFrame:
    """Predict each gameweek in `gws` (and up to horizon-1 weeks after) from earlier data only."""
    cutoffs = gw_cutoffs(fixtures_all)
    cutoff_of = {(s, g): c for s, g, c in cutoffs.itertuples(index=False)}
    in_season = history[history["season"] == season]
    played_gws = set(in_season["gw"])
    all_team_rows = team_matches(in_season)
    results = []
    for gw in gws:
        cutoff = cutoff_of[(season, gw)]
        past = history[history["kickoff_time"] < cutoff]
        if past["kickoff_time"].max() >= cutoff:
            raise AssertionError(f"training data for GW{gw} includes matches after its cutoff")
        targets = [g for g in range(gw, gw + horizon) if g in played_gws]
        actual = in_season[in_season["gw"].isin(targets)]
        player_rows = actual[PLAYER_ROW_COLUMNS].assign(asof=cutoff)
        # Later weeks' prices aren't known yet: use each player's price at this gameweek.
        known_price = actual[actual["gw"] == gw].groupby("player_code")["price"].first()
        player_rows["price"] = player_rows["player_code"].map(known_price).fillna(player_rows["price"])
        team_rows = all_team_rows[all_team_rows["gw"].isin(targets)][TEAM_ROW_COLUMNS].assign(asof=cutoff)

        pred = train_and_predict(past, player_rows, team_rows, cutoffs)

        outcomes = actual[["season", "fixture_id", "player_code", "total_points", "minutes",
                           "opp_goals"]].assign(defcon_hit=scoring.defcon_hit(actual).to_numpy())
        pred = pred.merge(outcomes, on=["season", "fixture_id", "player_code"], how="left")
        pred["origin_gw"] = gw
        pred["horizon"] = pred["gw"] - gw + 1
        results.append(pred)
        log(f"  {season} GW{gw}: trained on {len(past):,} rows, predicted {len(pred):,}")
    return pd.concat(results, ignore_index=True)


def per_player_gw(preds: pd.DataFrame) -> pd.DataFrame:
    """Add up fixtures so each player has one row per gameweek (double gameweeks summed)."""
    keys = ["season", "origin_gw", "horizon", "gw", "player_code", "position"]
    return preds.groupby(keys, as_index=False)[list(METHODS.values()) + ["total_points"]].sum()


def _weekly_mean(frame: pd.DataFrame, score) -> float:
    return float(pd.Series([score(week) for _, week in frame.groupby("origin_gw")]).mean())


def _rank_corr(predicted: pd.Series, actual: pd.Series) -> float:
    """Spearman correlation; undefined (NaN) when either side is constant."""
    if predicted.nunique() < 2 or actual.nunique() < 2:
        return float("nan")
    return float(predicted.rank().corr(actual.rank()))


def summarize(preds: pd.DataFrame) -> pd.DataFrame:
    """MAE, rank correlation and top-N actual points, per horizon and method."""
    by_player = per_player_gw(preds)
    rows = []
    for horizon, frame in by_player.groupby("horizon"):
        for method, col in METHODS.items():
            rows.append({
                "horizon": horizon, "method": method,
                "mae": (frame[col] - frame["total_points"]).abs().mean(),
                "spearman": _weekly_mean(frame, lambda w: _rank_corr(w[col], w["total_points"])),
                f"top{TOP_N}_points": _weekly_mean(frame, lambda w: w.nlargest(TOP_N, col)["total_points"].sum()),
                "player_gws": len(frame),
            })
    return pd.DataFrame(rows)


def calibration(preds: pd.DataFrame, buckets: int = 10) -> pd.DataFrame:
    """Predicted vs actual rates for the probability parts, one-week-ahead only."""
    h1 = preds[preds["horizon"] == 1]
    checks = {
        "p_plays": (h1, h1["minutes"] > 0),
        "p_60": (h1, h1["minutes"] >= 60),
    }
    teams = h1.drop_duplicates(["season", "fixture_id", "team"])
    checks["p_cs"] = (teams, teams["opp_goals"] == 0)
    defcon_rows = h1[(h1["minutes"] > 0) & h1["defcon_hit"].notna()]  # p_defcon is "if he plays"
    checks["p_defcon"] = (defcon_rows, defcon_rows["defcon_hit"] == 1)

    tables = []
    for part, (frame, outcome) in checks.items():
        if frame.empty:
            continue
        bucket = np.minimum((frame[part] * buckets).astype(int), buckets - 1)
        table = (pd.DataFrame({"bucket": bucket, "predicted": frame[part],
                               "actual": outcome.astype(float)})
                 .groupby("bucket", as_index=False)
                 .agg(predicted=("predicted", "mean"), actual=("actual", "mean"),
                      rows=("actual", "size")))
        tables.append(table.assign(part=part)[["part", "bucket", "predicted", "actual", "rows"]])
    return pd.concat(tables, ignore_index=True)


def clean_sheet_report(preds: pd.DataFrame) -> dict:
    """Spread and accuracy of P(clean sheet) one week ahead, one row per team-fixture."""
    h1 = preds[preds["horizon"] == 1].drop_duplicates(["season", "fixture_id", "team"])
    kept = (h1["opp_goals"] == 0).astype(int)
    p = h1["p_cs"].clip(1e-6, 1 - 1e-6)
    fifths = pd.qcut(p, 5, labels=False, duplicates="drop")
    return {"p5": float(p.quantile(0.05)), "p95": float(p.quantile(0.95)),
            "log_loss": float(-(kept * np.log(p) + (1 - kept) * np.log(1 - p)).mean()),
            "actual_by_fifth": kept.groupby(fifths).mean().round(3).tolist()}
