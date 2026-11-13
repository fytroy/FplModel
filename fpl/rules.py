"""FPL game rules shared by the planner and the season simulation.

Prices are FPL units (tenths of £1m: 55 = £5.5m) so rounding matches FPL.
Nothing here touches the internet or the disk.
"""
from dataclasses import dataclass

import pandas as pd

SQUAD_SHAPE = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
XI_LIMITS = {"GK": (1, 1), "DEF": (3, 5), "MID": (2, 5), "FWD": (1, 3)}
MAX_PER_CLUB = 3
BUDGET = 1000                       # £100.0m
HIT_COST = 4
MAX_FREE_TRANSFERS = 5
TRANSFER_CHIPS = ("wildcard", "freehit")
CHIP_LABELS = {"wildcard": "Wildcard", "freehit": "Free Hit", "bboost": "Bench Boost",
               "3xc": "Triple Captain"}
# Season-wide top-ups: {season: {gameweek: free transfers}}. AFCON 2025/26, per
# premierleague.com/en/news/4362102 ("topped up to the maximum possible number of five in GW16").
FREE_TRANSFER_TOP_UPS = {"2025-26": {16: 5}}


def selling_price(purchase: int, now: int) -> int:
    """A rise is shared 50/50 with FPL (your half rounded down); a fall is all yours."""
    return now if now <= purchase else purchase + (now - purchase) // 2


def paid_transfers(free: int, transfers: int, chip: str | None) -> int:
    """How many of this gameweek's transfers cost a hit."""
    return 0 if chip in TRANSFER_CHIPS else max(0, transfers - free)


def next_free_transfers(free: int, transfers: int, chip: str | None) -> int:
    """Free transfers for the next gameweek.

    Normal week: unused ones roll over (max 5) and one is added. Wildcard or
    Free Hit week: the count is kept but none is added. Checked against 478
    real manager-weeks in 2026/27 (see the part 3 spec).
    """
    if chip in TRANSFER_CHIPS:
        return free
    return min(MAX_FREE_TRANSFERS, max(free - transfers, 0) + 1)


def free_transfers_at(gw: int, rolled: int, season: str) -> int:
    """The rolled-over count, unless FPL topped everyone up that gameweek."""
    return FREE_TRANSFER_TOP_UPS.get(season, {}).get(gw, rolled)


@dataclass(frozen=True)
class Chip:
    """One copy of a chip, playable from `start` to `stop` inclusive (FPL's chips data)."""
    name: str
    start: int
    stop: int

    @property
    def key(self) -> str:
        return f"{self.name}_{self.start}"

    def allows(self, gw: int) -> bool:
        return self.start <= gw <= self.stop


def chip_windows(bootstrap_chips: list) -> list[Chip]:
    """bootstrap-static['chips'] -> one Chip per copy."""
    return [Chip(c["name"], int(c["start_event"]), int(c["stop_event"])) for c in bootstrap_chips]


def chips_left(windows: list[Chip], played: list[tuple[str, int]]) -> list[Chip]:
    """Drop the copy each played (name, gameweek) used up."""
    left = list(windows)
    for name, gw in played:
        used = next((c for c in left if c.name == name and c.allows(gw)), None)
        if used is not None:
            left.remove(used)
    return left


def valid_xi(positions) -> bool:
    counts = pd.Series(list(positions)).value_counts()
    return all(lo <= counts.get(p, 0) <= hi for p, (lo, hi) in XI_LIMITS.items())


def score_week(picks: pd.DataFrame, chip: str | None = None, hits: int = 0) -> int:
    """Score one gameweek as FPL does.

    picks: 15 rows in pick order (11 starters, the bench goalkeeper, then the
    outfield bench in order) with position, points, minutes (his total that
    gameweek; 0 if he had no match), is_captain, is_vice.
    Auto-subs: a starter with 0 minutes is replaced by the first bench player
    who played and keeps a legal formation (goalkeeper only by goalkeeper).
    The captain's points are doubled (tripled with Triple Captain), or the
    vice-captain's if the captain didn't play. Bench Boost counts all 15.
    """
    p = picks.reset_index(drop=True)
    played = p["minutes"] > 0
    if chip == "bboost":
        counted = list(range(15))
    else:
        counted = list(range(11))
        bench = [11, 12, 13, 14]
        for i in range(11):
            if played[counted[i]]:
                continue
            out_gk = p.at[counted[i], "position"] == "GK"
            for b in bench:
                if not played[b] or (p.at[b, "position"] == "GK") != out_gk:
                    continue
                trial = [p.at[j, "position"] for j in counted if j != counted[i]] + [p.at[b, "position"]]
                if valid_xi(trial):
                    bench.remove(b)
                    counted[i] = b
                    break
    total = int(p.loc[counted, "points"].sum())
    captain = int(p.index[p["is_captain"]][0])
    vice = int(p.index[p["is_vice"]][0])
    armband = captain if played[captain] else (vice if played[vice] else None)
    if armband is not None and armband in counted:
        total += (2 if chip == "3xc" else 1) * int(p.at[armband, "points"])
    return total - HIT_COST * hits
