"""Turn FPL's injury and suspension news into a chance of being available each gameweek.

Part 2 only capped next week, so a 6-week plan treated an injured player as fit
from week 2. Each flagged player now gets availability a(w) for every planned
gameweek; P(plays) and P(60+) are multiplied by it. Live predictions only:
past news wasn't stored.
"""
import re
from datetime import date

import numpy as np
import pandas as pd

from fpl import scoring
from fpl.settings import FIRST_WEEK_BACK

MAX_MISSED = 8                       # recovery curve: 8+ missed matches pooled
REGULAR_WINDOW, REGULAR_MIN_60 = 10, 6
_DATE = re.compile(r"(?:Expected back|Suspended until)\s+(\d{1,2})\s+([A-Z][a-z]{2})")
_MONTHS = {m: i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], start=1)}
_DEFAULT_CURVE = pd.Series(np.linspace(0.5, 0.15, MAX_MISSED), index=range(1, MAX_MISSED + 1))


def return_date(news, added: pd.Timestamp) -> date | None:
    """'Hamstring injury - Expected back 24 Oct' -> 2026-10-24; None if there's no valid date.

    The year comes from when the news was added (a January date in December news is next year).
    """
    m = _DATE.search(news) if isinstance(news, str) else None
    if not m or m.group(2) not in _MONTHS or pd.isna(added):
        return None
    month = _MONTHS[m.group(2)]
    year = added.year + (1 if month < added.month else 0)
    try:
        return date(year, month, int(m.group(1)))
    except ValueError:
        return None


def recovery_curve(history: pd.DataFrame) -> pd.Series:
    """Chance a regular starter plays his next fixture after missing m in a row (m = 1..8).

    Regular = 60+ minutes in at least 6 of the 10 fixtures before the absence.
    Made non-increasing in m. It includes non-injury absences (dropped,
    rested), so it is a rough guide; recommend.py prints what it implies.
    """
    records = []
    h = history.sort_values(["player_code", "kickoff_time"])
    for _, minutes in h.groupby("player_code", sort=False)["minutes"]:
        mins = minutes.to_numpy()
        missed, regular = 0, False
        for i in range(len(mins) - 1):
            if mins[i] > 0:
                missed = 0
                continue
            if missed == 0:
                before = mins[max(0, i - REGULAR_WINDOW):i]
                regular = len(before) == REGULAR_WINDOW and (before >= 60).sum() >= REGULAR_MIN_60
            missed += 1
            if regular:
                records.append((min(missed, MAX_MISSED), mins[i + 1] > 0))
    if not records:
        return _DEFAULT_CURVE.copy()
    rates = pd.DataFrame(records, columns=["m", "back"]).groupby("m")["back"].mean()
    curve = rates.reindex(range(1, MAX_MISSED + 1)).ffill().fillna(rates.iloc[0])
    return pd.Series(np.minimum.accumulate(curve.to_numpy()), index=curve.index)


def matches_missed(history: pd.DataFrame, codes) -> pd.Series:
    """Fixtures each player has missed in a row since his last appearance (0 if he played last)."""
    h = history[history["player_code"].isin(list(codes))].sort_values("kickoff_time")

    def trailing_zeros(minutes: pd.Series) -> int:
        played = np.flatnonzero(minutes.to_numpy() > 0)
        return len(minutes) - (played[-1] + 1 if len(played) else 0)
    return h.groupby("player_code")["minutes"].apply(trailing_zeros).astype(int)


def trim_absences(history: pd.DataFrame, codes) -> pd.DataFrame:
    """Drop listed players' 0-minute matches after their last appearance.

    So a returning player is judged on his form before the injury, not on the
    empty weeks he missed.
    """
    h = history
    flagged = h["player_code"].isin(list(codes))
    last_on = h[flagged & (h["minutes"] > 0)].groupby("player_code")["kickoff_time"].max()
    after = h["kickoff_time"] > h["player_code"].map(last_on)
    return h[~(flagged & after.fillna(False))]


def _chance(row) -> float | None:
    c = row["chance_of_playing_next_round"]
    return None if pd.isna(c) else float(c) / 100


def _weeks(row, gws, week_ends, curve, missed) -> list[float]:
    n = len(gws)
    chance = _chance(row)
    if row["status"] == "u":
        return [0.0] * n
    back = return_date(row["news"], pd.Timestamp(row["news_added"]))
    if back is not None:
        out, returned = [], False
        for gw in gws:
            if gw not in week_ends:          # no fixtures that week: nothing to miss
                out.append(1.0)
            elif week_ends[gw].date() < back:
                out.append(0.0)
            else:
                out.append(1.0 if returned else FIRST_WEEK_BACK)
                returned = True
        return out
    if chance is not None and 0 < chance < 1:
        return [chance, (1 + chance) / 2] + [1.0] * (n - 2) if n >= 2 else [chance]
    if row["status"] in ("i", "s"):
        # Next week: FPL's own chance when given (e.g. Isak 0%); after that, the recovery curve.
        m = max(int(missed.get(row["code"], 0)), 1)
        still_out, out = 1.0, []
        for k in range(n):
            if k == 0 and chance is not None:
                still_out = 1 - chance
            else:
                still_out *= 1 - curve[min(m + k, MAX_MISSED)]
            out.append(1 - still_out)
        return out
    if chance is not None:
        return [chance] + [1.0] * (n - 1)
    return [1.0] * n


def weekly_availability(players, gws, week_ends, curve, missed) -> pd.DataFrame:
    """One row per flagged player per gameweek: player_code, gw, availability, news.

    players: players.csv rows (code, status, news, news_added, chance_of_playing_next_round).
    A player who isn't listed is fully available. See the part 3 spec §2 for the rules.
    """
    flagged = players[(players["status"] != "a") | players["chance_of_playing_next_round"].notna()]
    rows = []
    for _, row in flagged.iterrows():
        for gw, a in zip(gws, _weeks(row, gws, week_ends, curve, missed)):
            rows.append({"player_code": int(row["code"]), "gw": gw, "availability": a,
                         "news": row["news"]})
    return pd.DataFrame(rows, columns=["player_code", "gw", "availability", "news"])


def apply(pred: pd.DataFrame, avail: pd.DataFrame) -> pd.DataFrame:
    """Multiply per-fixture P(plays) and P(60+) by that week's availability; recompute xP."""
    out = pred.merge(avail[["player_code", "gw", "availability"]], on=["player_code", "gw"],
                     how="left")
    factor = out.pop("availability").fillna(1.0)
    out["p_plays"] = out["p_plays"] * factor
    out["p_60"] = out["p_60"] * factor
    return scoring.expected_points(out)
