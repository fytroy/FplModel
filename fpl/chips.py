"""Chip scan: what each unused chip is worth in each gameweek after the plan.

Rough by design (spec §5). It holds today's squad fixed and uses the
predictions as they are; predictions far ahead are flatter than near ones, so
these values run a little low. hold value = the best later week's value,
undiscounted; it is 0 when no week is left in the chip's window, so a
chip about to expire gets played.
"""
import pandas as pd

from fpl import optimizer, rules
from fpl.settings import PlanSettings

SCAN_POOL = {"per_position": 15, "value_per_position": 5, "cheapest": 3}


def best_xi(xp_week: pd.Series, positions: pd.Series) -> list[int]:
    """The best legal starting 11 from these players for one week."""
    order = list(xp_week.sort_values(ascending=False).index)
    picked = [p for p in order if positions[p] == "GK"][:1]
    for position, (lo, _) in rules.XI_LIMITS.items():
        if position != "GK":
            picked += [p for p in order if positions[p] == position][:lo]
    rest = [p for p in order if p not in picked and positions[p] != "GK"]
    return picked + rest[:11 - len(picked)]


def _xi_points(xp_week: pd.Series, positions: pd.Series) -> float:
    xi = best_xi(xp_week, positions)
    return float(xp_week[xi].sum() + xp_week[xi].max())         # captain counted twice


def scan(xp, pool, squad, bank, chips, after_gw, season, wildcard_weeks=6) -> pd.DataFrame:
    """Value of each chip copy in each week after `after_gw` that its window and xp cover."""
    positions = pool["position"]
    budget = int(bank + pool.loc[squad, "sell"].sum())
    weeks = [int(w) for w in xp.columns if int(w) > after_gw]
    rows, best_cache = [], {}

    def best(window):
        if window not in best_cache:
            sub = xp[list(window)]
            keep = optimizer.candidate_pool(sub, pool, squad, 1.0, **SCAN_POOL)
            best_cache[window] = (optimizer.best_squad(sub.loc[keep], pool.loc[keep], budget, season), keep)
        return best_cache[window]

    def reachable(window):
        """Best squad you could reach from today's with half a Wildcard window of free transfers."""
        _, keep = best(window)
        inp = optimizer.PlanInput(pool=pool.loc[keep], xp=xp.loc[keep, list(window)], squad=list(squad),
                                  bank=bank, free_transfers=len(window) // 2, chips=[], season=season)
        flat = PlanSettings(discount=1.0, hit_margin=0.0, ft_end_value=0.0, bench_weight=0.0)
        return optimizer.solve(inp, flat, week1_hits="none", fixed_squad=True).value

    for ch in chips:
        for w in weeks:
            if not ch.allows(w):
                continue
            mine = xp.loc[squad, w]
            if ch.name == "bboost":
                xi = best_xi(mine, positions)
                value = float(mine.drop(xi).sum())
            elif ch.name == "3xc":
                value = float(mine[best_xi(mine, positions)].max())
            elif ch.name == "freehit":
                value = best((w,))[0].value - _xi_points(mine, positions)
            else:                                               # wildcard
                window = tuple(v for v in weeks if w <= v < w + wildcard_weeks)
                value = best(window)[0].value - reachable(window)
            rows.append({"key": ch.key, "name": ch.name, "gw": w, "value": max(0.0, float(value))})
    return pd.DataFrame(rows, columns=["key", "name", "gw", "value"])


def hold_values(scan_df: pd.DataFrame, chips) -> dict[str, float]:
    """Points for keeping each chip past the plan: its best scanned week, undiscounted (0 if none)."""
    out = {}
    for ch in chips:
        mine = scan_df[scan_df["key"] == ch.key]
        out[ch.key] = float(mine["value"].max()) if len(mine) else 0.0
    return out
