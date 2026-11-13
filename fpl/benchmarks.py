"""2025/26 benchmarks for the season simulation: the average manager and a points-to-rank curve.

Rank curve: every sampled manager gives one exact (season points, final rank)
pair, so interpolating between them is unbiased; precision is limited by
sample size, worst at very high ranks. Nothing here touches the internet.
"""
import numpy as np
import pandas as pd

USER_TOTALS = {"2025-26": 2061}     # team 5234632, from /entry/5234632/history/ 'past'


def average_manager(events: list) -> pd.DataFrame:
    """FPL's average score per gameweek (bootstrap-static 'events')."""
    return pd.DataFrame({"gw": [e["id"] for e in events],
                         "average": [e["average_entry_score"] for e in events]})


def estimate_rank(points: float, sample: pd.DataFrame) -> tuple[int, str]:
    """Estimated overall rank for a season total, interpolated on log rank between sampled managers."""
    curve = (sample.assign(log_rank=np.log(sample["rank"])).groupby("points")["log_rank"].median()
             .sort_index())
    xs, ys = curve.index.to_numpy(dtype=float), curve.to_numpy()
    ys = np.minimum.accumulate(ys)                       # more points never means a worse rank
    note = "estimated, roughly ±10-20%"
    if points > xs[-1]:
        note = f"better than every sampled manager ({int(xs[-1])} pts); rank is at most this"
    elif points < xs[0]:
        note = f"below every sampled manager ({int(xs[0])} pts)"
    return int(round(float(np.exp(np.interp(points, xs, ys))))), note
