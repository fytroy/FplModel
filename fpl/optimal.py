"""Optimal squad and Dream Team solver: finds the highest-scoring squad and Starting XI.

Solves an integer linear program (PuLP) to choose 15 players within budget (default £100m)
and club limits (max 3 per club) with the highest gameweek points (or expected points xP).
Selects the best 11 starters, valid formation, captain (2x), vice-captain, and ordered bench.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pulp

from fpl import rules
from fpl.matches import POSITION_BY_TYPE

try:
    import highspy  # noqa: F401
    HIGHS = True
except ImportError:
    HIGHS = False


@dataclass
class OptimalPlayer:
    code: int
    name: str
    position: str
    team: str
    price: int       # tenths of £1m
    points: float    # gameweek points or xP
    is_starter: bool
    is_captain: bool
    is_vice: bool
    bench_order: int | None = None   # 0 for backup GK, 1-3 for outfield bench


@dataclass
class OptimalSquad:
    gw: int
    formation: str
    starters: list[OptimalPlayer]
    bench: list[OptimalPlayer]
    captain: OptimalPlayer
    vice: OptimalPlayer
    total_cost: int              # tenths of £1m
    starting_points: float       # sum of starters + captain bonus
    bench_points: float          # sum of bench points
    total_points: float          # starting_points
    status: str                  # "Optimal" or solver status


def build_player_pool(players: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """Standardized pool indexed by player_code."""
    short = dict(zip(teams["id"], teams["short_name"]))
    p = players[players["status"] != "u"].copy()
    pool = pd.DataFrame({
        "element": p["id"].astype(int).to_numpy(),
        "name": p["web_name"].to_numpy(),
        "position": p["element_type"].map(POSITION_BY_TYPE).to_numpy(),
        "team": p["team"].map(short).to_numpy(),
        "price": p["now_cost"].astype(int).to_numpy(),
    }, index=pd.Index(p["code"].astype(int), name="player_code"))
    return pool[pool["position"].notna()]


def filter_candidates(pts: pd.Series, pool: pd.DataFrame,
                      top_per_pos: int = 25, value_per_pos: int = 15,
                      cheapest_per_pos: int = 5) -> list[int]:
    """Select the most promising players per position to keep ILP solve instant."""
    chosen = set()
    for _, group in pool.groupby("position"):
        s = pts.reindex(group.index).fillna(0.0)
        chosen |= set(s.nlargest(top_per_pos).index)
        val = s / (group["price"].clip(lower=35) / 10.0)
        chosen |= set(val.nlargest(value_per_pos).index)
        chosen |= set(group["price"].nsmallest(cheapest_per_pos).index)
    return sorted(int(c) for c in chosen)


def solve_optimal_squad(points: pd.Series, pool: pd.DataFrame, gw: int = 1,
                        budget: int = 1000, max_per_club: int = rules.MAX_PER_CLUB,
                        time_limit: float = 10.0, filter_pool: bool = True) -> OptimalSquad:
    """Find the best 15-player squad, Starting XI, captain, and bench maximizing points.

    points: Series indexed by player_code (floats).
    pool: DataFrame indexed by player_code with 'name', 'position', 'team', 'price'.
    budget: Tenths of £1m (default 1000 = £100m).
    """
    valid_codes = pool.index.intersection(points.index)
    pool = pool.loc[valid_codes].copy()
    pts = points.reindex(valid_codes).fillna(0.0).astype(float)

    if filter_pool and len(pool) > 100:
        candidates = filter_candidates(pts, pool)
        pool = pool.loc[candidates]
        pts = pts.loc[candidates]

    P = [int(c) for c in pool.index]
    pos = pool["position"]
    club = pool["team"]
    price = pool["price"].astype(int)

    prob = pulp.LpProblem("fpl_optimal_squad", pulp.LpMaximize)

    # Binaries: in squad (x), in starting XI (s), is captain (c), is vice (v)
    x = {p: pulp.LpVariable(f"x_{p}", cat="Binary") for p in P}
    s = {p: pulp.LpVariable(f"s_{p}", cat="Binary") for p in P}
    c = {p: pulp.LpVariable(f"c_{p}", cat="Binary") for p in P}
    v = {p: pulp.LpVariable(f"v_{p}", cat="Binary") for p in P}

    for p in P:
        prob += s[p] <= x[p]
        prob += c[p] <= s[p]
        prob += v[p] <= s[p]
        prob += c[p] + v[p] <= 1

    # Squad limits: 2 GK, 5 DEF, 5 MID, 3 FWD (total 15)
    for position, count in rules.SQUAD_SHAPE.items():
        prob += pulp.lpSum(x[p] for p in P if pos[p] == position) == count

    # Starting XI limits: 1 GK, 3-5 DEF, 2-5 MID, 1-3 FWD (total 11)
    prob += pulp.lpSum(s[p] for p in P) == 11
    prob += pulp.lpSum(s[p] for p in P if pos[p] == "GK") == 1
    for position, (lo, hi) in rules.XI_LIMITS.items():
        prob += pulp.lpSum(s[p] for p in P if pos[p] == position) >= lo
        prob += pulp.lpSum(s[p] for p in P if pos[p] == position) <= hi

    # Captain and Vice
    prob += pulp.lpSum(c[p] for p in P) == 1
    prob += pulp.lpSum(v[p] for p in P) == 1

    # Max per club
    for team in club.unique():
        members = [p for p in P if club[p] == team]
        prob += pulp.lpSum(x[p] for p in members) <= max_per_club

    # Budget
    prob += pulp.lpSum(price[p] * x[p] for p in P) <= budget

    # Objective:
    # Starting XI gets full points
    # Captain gets extra 1x points
    # Bench gets 0.05x tie-breaker so strongest bench is selected
    # Vice gets 0.01x tie-breaker so best alternative is chosen
    # Cheaper players get microscopic tie-breaker so cost is minimized on ties
    obj = pulp.lpSum(
        pts[p] * s[p]
        + pts[p] * c[p]
        + 0.05 * pts[p] * (x[p] - s[p])
        + 0.01 * pts[p] * v[p]
        - 0.0001 * price[p] * x[p]
        for p in P
    )
    prob += obj

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit)
    prob.solve(solver)

    status = pulp.LpStatus[prob.status]
    if status != "Optimal":
        raise ValueError(f"Solver failed to find an optimal squad: {status}")

    squad_codes = [p for p in P if pulp.value(x[p]) > 0.5]
    starter_codes = [p for p in P if pulp.value(s[p]) > 0.5]
    bench_codes = [p for p in squad_codes if p not in starter_codes]
    captain_code = next(p for p in P if pulp.value(c[p]) > 0.5)
    vice_code = next(p for p in P if pulp.value(v[p]) > 0.5)

    pos_order = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
    starters_sorted = sorted(starter_codes, key=lambda p: (pos_order[pos[p]], -pts[p]))

    def_count = sum(1 for p in starter_codes if pos[p] == "DEF")
    mid_count = sum(1 for p in starter_codes if pos[p] == "MID")
    fwd_count = sum(1 for p in starter_codes if pos[p] == "FWD")
    formation = f"{def_count}-{mid_count}-{fwd_count}"

    # Bench order: GK first (bench_order=0), then outfield by points descending (1, 2, 3)
    bench_gk = [p for p in bench_codes if pos[p] == "GK"]
    bench_outfield = sorted([p for p in bench_codes if pos[p] != "GK"], key=lambda p: -pts[p])

    bench_ordered = []
    if bench_gk:
        bench_ordered.append((bench_gk[0], 0))
    for idx, p in enumerate(bench_outfield, 1):
        bench_ordered.append((p, idx))

    starters_obj = [
        OptimalPlayer(
            code=p,
            name=pool.at[p, "name"],
            position=pool.at[p, "position"],
            team=pool.at[p, "team"],
            price=int(price[p]),
            points=float(pts[p]),
            is_starter=True,
            is_captain=(p == captain_code),
            is_vice=(p == vice_code),
        )
        for p in starters_sorted
    ]

    bench_obj = [
        OptimalPlayer(
            code=p,
            name=pool.at[p, "name"],
            position=pool.at[p, "position"],
            team=pool.at[p, "team"],
            price=int(price[p]),
            points=float(pts[p]),
            is_starter=False,
            is_captain=False,
            is_vice=False,
            bench_order=order,
        )
        for p, order in bench_ordered
    ]

    captain_obj = next(p for p in starters_obj if p.code == captain_code)
    vice_obj = next(p for p in starters_obj if p.code == vice_code)

    total_cost = sum(price[p] for p in squad_codes)
    starting_points = sum(p.points for p in starters_obj) + captain_obj.points
    bench_points = sum(p.points for p in bench_obj)

    return OptimalSquad(
        gw=gw,
        formation=formation,
        starters=starters_obj,
        bench=bench_obj,
        captain=captain_obj,
        vice=vice_obj,
        total_cost=total_cost,
        starting_points=round(starting_points, 1),
        bench_points=round(bench_points, 1),
        total_points=round(starting_points, 1),
        status=status,
    )


def money(tenths: int) -> str:
    return f"£{tenths / 10:.1f}m"


def render_optimal_squad(squad: OptimalSquad, label: str = "OPTIMAL DREAM TEAM") -> str:
    """Format the squad into a clean presentation."""
    header = f"=== GAMEWEEK {squad.gw} {label} ==="
    subhead = f"Formation: {squad.formation} | Cost: {money(squad.total_cost)} | Projected: {squad.starting_points:.1f} pts"
    sep = "=" * max(len(header), len(subhead), 65)

    lines = [sep, f"{header:^65}", f"{subhead:^65}", sep, "", "STARTING XI:"]
    for p in squad.starters:
        role = ""
        if p.is_captain:
            role = " (C) "
        elif p.is_vice:
            role = " (V) "
        else:
            role = "     "

        pts_str = f"{p.points * 2:.1f} pts (2x {p.points:.1f})" if p.is_captain else f"{p.points:.1f} pts"
        lines.append(f"  {p.position:<4} {p.name:<18} ({p.team:<3}){role} {money(p.price):>6}  {pts_str:>16}")

    lines += ["", "BENCH:"]
    for p in squad.bench:
        prefix = "GK  " if p.bench_order == 0 else f"Sub{p.bench_order}"
        lines.append(f"  {prefix:<4} {p.name:<18} ({p.team:<3})      {money(p.price):>6}  {p.points:>6.1f} pts")

    lines += [sep]
    return "\n".join(lines)
