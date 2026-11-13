"""FPL's scoring rules, season by season, in one place.

Used two ways: rebuilding official points from match stats (which checks the
rules are right), and turning the model's part predictions into expected
points. Nothing here touches the internet or the disk.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

POSITIONS = ("GK", "DEF", "MID", "FWD")
ASSIST_POINTS = 3
SAVES_PER_POINT = 3
GOALS_CONCEDED_PER_MINUS_POINT = 2
# Defensive actions needed for DefCon points. Goalkeepers can't earn them.
DEFCON_THRESHOLD = {"DEF": 10, "MID": 12, "FWD": 12}
RULES_ERA_CODE = {"pre_2025": 0, "2025_26": 1, "2026_27": 2}
POINT_PARTS = ["pts_minutes", "pts_goals", "pts_assists", "pts_clean_sheet",
               "pts_goals_conceded", "pts_saves", "pts_cards", "pts_defcon", "pts_bonus"]


@dataclass(frozen=True)
class Rules:
    goal: dict          # points per goal, by position
    clean_sheet: dict   # points per clean sheet, by position
    defcon_points: int  # points for reaching the DefCon threshold


def rules_for(season: str) -> Rules:
    year = int(season[:4])
    return Rules(
        goal={"GK": 10 if year >= 2026 else 6, "DEF": 6, "MID": 5, "FWD": 4},
        clean_sheet={"GK": 4, "DEF": 4, "MID": 1, "FWD": 0},
        defcon_points=2 if year >= 2025 else 0,
    )


def rules_era(season: str) -> str:
    """Which bonus/DefCon rule set a season uses. Later seasons reuse the newest."""
    year = int(season[:4])
    if year >= 2026:
        return "2026_27"
    return "2025_26" if year == 2025 else "pre_2025"


def defcon_actions(matches: pd.DataFrame) -> pd.Series:
    """Defensive actions that count for each row's position.

    DEF: clearances/blocks/interceptions + tackles (CBIT). MID/FWD: CBIT plus
    recoveries (CBIRT). NaN for goalkeepers and seasons without DefCon data.
    """
    cbit = matches["clearances_blocks_interceptions"] + matches["tackles"]
    cbirt = cbit + matches["recoveries"]
    position = matches["position"]
    return cbit.where(position == "DEF", cbirt.where(position.isin(["MID", "FWD"])))


def defcon_hit(matches: pd.DataFrame) -> pd.Series:
    """1.0 if the row reached its position's DefCon threshold, 0.0 if not, NaN if unknown."""
    actions = defcon_actions(matches)
    threshold = matches["position"].map(DEFCON_THRESHOLD)
    return (actions >= threshold).astype(float).where(actions.notna() & threshold.notna())


def _per_row(frame: pd.DataFrame, pick) -> pd.Series:
    """A by-position rule value for each row, using that row's season's rules."""
    return pd.Series([pick(rules_for(s))[p] for s, p in zip(frame["season"], frame["position"])],
                     index=frame.index, dtype=float)


def actual_points(matches: pd.DataFrame) -> pd.Series:
    """Rebuild official FPL points from match stats, using each row's season rules."""
    position = matches["position"]
    minutes = matches["minutes"]
    defcon_points = matches["season"].map(lambda s: rules_for(s).defcon_points)
    points = (
        np.where(minutes >= 60, 2, np.where(minutes > 0, 1, 0))
        + matches["goals_scored"] * _per_row(matches, lambda r: r.goal)
        + matches["assists"] * ASSIST_POINTS
        + matches["clean_sheets"] * _per_row(matches, lambda r: r.clean_sheet)
        - np.where(position.isin(["GK", "DEF"]),
                   matches["goals_conceded"] // GOALS_CONCEDED_PER_MINUS_POINT, 0)
        + np.where(position == "GK", matches["saves"] // SAVES_PER_POINT, 0)
        + 5 * matches["penalties_saved"] - 2 * matches["penalties_missed"]
        - matches["yellow_cards"] - 3 * matches["red_cards"] - 2 * matches["own_goals"]
        + matches["bonus"]
        + defcon_points * defcon_hit(matches).fillna(0)
    )
    return pd.Series(points, index=matches.index).round().astype(int)


def expected_floor_div(lam, k: int, max_count: int = 40) -> np.ndarray:
    """E[floor(X / k)] for X ~ Poisson(lam): the expected value of '1 point per k events'."""
    lam = np.clip(np.asarray(lam, dtype=float), 1e-12, None)[..., None]
    n = np.arange(max_count + 1)
    log_factorial = np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, max_count + 1)))])
    pmf = np.exp(n * np.log(lam) - lam - log_factorial)
    return (pmf * (n // k)).sum(axis=-1)


def expected_points(pred: pd.DataFrame) -> pd.DataFrame:
    """Turn per-fixture part predictions into expected FPL points.

    Needs one row per player per fixture with: season, position,
      p_plays, p_60     chance he plays at all / plays 60+ minutes
      xg, xa            expected goals / assists if he plays
      p_cs              chance his team keeps a clean sheet
      gc_lambda         his team's expected goals conceded
      saves_lambda      expected saves if he plays (goalkeepers)
      p_yellow, p_red   chance of a yellow / red card if he plays
      p_defcon          chance he reaches the DefCon threshold if he plays
      e_bonus           expected bonus points if he plays
    Returns a copy with one pts_* column per scoring part, then xp (their sum).
    """
    out = pred.drop(columns=POINT_PARTS + ["xp"], errors="ignore").copy()
    position = out["position"]
    plays, sixty = out["p_plays"], out["p_60"]
    defcon_points = out["season"].map(lambda s: rules_for(s).defcon_points)
    goals_conceded = expected_floor_div(out["gc_lambda"], GOALS_CONCEDED_PER_MINUS_POINT)
    saves = expected_floor_div(out["saves_lambda"], SAVES_PER_POINT)

    out["pts_minutes"] = (plays - sixty) * 1 + sixty * 2
    out["pts_goals"] = plays * out["xg"] * _per_row(out, lambda r: r.goal)
    out["pts_assists"] = plays * out["xa"] * ASSIST_POINTS
    # A clean sheet only counts if he plays 60+: P(clean sheet AND 60+ minutes).
    out["pts_clean_sheet"] = sixty * out["p_cs"] * _per_row(out, lambda r: r.clean_sheet)
    out["pts_goals_conceded"] = np.where(position.isin(["GK", "DEF"]), -sixty * goals_conceded, 0.0)
    out["pts_saves"] = np.where(position == "GK", plays * saves, 0.0)
    out["pts_cards"] = -plays * (out["p_yellow"] + 3 * out["p_red"])
    out["pts_defcon"] = np.where(position == "GK", 0.0, plays * out["p_defcon"] * defcon_points)
    out["pts_bonus"] = plays * out["e_bonus"]
    out["xp"] = out[POINT_PARTS].sum(axis=1)
    return out
