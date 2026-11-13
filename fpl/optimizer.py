"""The multi-week planner: squad, transfers, captain and chips over the next few gameweeks.

One integer linear program (PuLP, solved by HiGHS or else CBC) chooses every
planned week at once, so it can weigh "transfer now" against "save it for a
double move later". A branch and bound over chip weeks (_chip_search) proves it.
Prices are FPL units (tenths of £1m) and assumed constant over the plan.
The rules and the objective are in the part 3 spec, section 4.
"""
import heapq
import itertools
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pulp

from fpl import rules
from fpl.settings import (CANDIDATES_PER_POSITION, CHEAPEST_PER_POSITION, SOLVER_GAP, SOLVER_TIME_LIMIT,
                          VALUE_CANDIDATES_PER_POSITION, PlanSettings)

try:                                      # HiGHS solves these models faster than CBC (spec §4)
    import highspy  # noqa: F401
    HIGHS = True
except ImportError:                       # CBC ships with pulp
    HIGHS = False

BIG = 15                                  # more than any week's transfer count
BRANCH_CHIPS = ("wildcard", "freehit", "bboost")    # chips the search fixes week by week; Triple Captain stays in the MIP
OUT_OF_TIME = (None, None, None)          # a solve that stopped before finding (or proving) anything
TIE_BREAK = 0.01                          # per transfer: never make a move that gains nothing


class SolverTimeout(ValueError):
    """The time limit ran out before the solver found any valid plan."""


@dataclass
class PlanInput:
    pool: pd.DataFrame                    # index player_code: name, position, team, price, sell
    xp: pd.DataFrame                      # index = pool index; one column per gameweek
    squad: list[int]                      # current 15 player codes ([] = pick a new team)
    bank: int
    free_transfers: int                   # for the first planned gameweek
    chips: list                           # rules.Chip copies still available
    season: str
    hold_values: dict = field(default_factory=dict)   # chip key -> points if kept past the plan


@dataclass
class WeekPlan:
    gw: int
    chip: str | None
    transfers_in: list[int]
    transfers_out: list[int]
    free_transfers: int | None            # None: a new team's first week (unlimited)
    paid: int
    owned: list[int]                      # your squad after this week's transfers
    squad: list[int]                      # the 15 who play (Free Hit squad in a Free Hit week)
    starters: list[int]
    bench: list[int]                      # [backup GK, first outfield sub, second, third]
    captain: int
    vice: int
    xp: float                             # expected points incl. captain and chip, minus hits
    bank: int


@dataclass
class Plan:
    weeks: list[WeekPlan]
    value: float                          # discounted xP minus hits; Triple Captain / Bench Boost extras
                                          # (the whole bench for BB) at weight 1; no margin, end value or holds
    status: str                           # "Optimal" (within SOLVER_GAP) or "time limit (gap 1.2%)"
    objective: float = 0.0                # the solver's objective (value + margins, end value, holds)
    gap: float = 0.0                      # how far the best possible plan could be above this one


def candidate_pool(xp: pd.DataFrame, pool: pd.DataFrame, keep, discount: float,
                   per_position: int = CANDIDATES_PER_POSITION,
                   value_per_position: int = VALUE_CANDIDATES_PER_POSITION,
                   cheapest: int = CHEAPEST_PER_POSITION) -> list[int]:
    """Players worth giving the solver: best by discounted xP, by xP per £m, the cheapest, and `keep`."""
    weights = discount ** np.arange(xp.shape[1])
    score = (xp * weights).sum(axis=1)
    chosen = set(int(k) for k in keep)
    for _, group in pool.groupby("position"):
        s = score.reindex(group.index).fillna(0)
        chosen |= set(s.nlargest(per_position).index)
        chosen |= set((s / group["price"]).nlargest(value_per_position).index)
        chosen |= set(group["price"].nsmallest(cheapest).index)
    return sorted(int(c) for c in chosen)


def solve(inp: PlanInput, settings: PlanSettings = PlanSettings(), week1_hits: str = "free",
          fixed_squad: bool = False, time_limit: float = SOLVER_TIME_LIMIT,
          gap: float = SOLVER_GAP) -> Plan:
    """Best plan for the gameweeks in inp.xp's columns.

    week1_hits: 'free' (the solver decides), 'none' (no hit in week 1) or 'some'
    (at least one transfer beyond the free ones, no Wildcard/Free Hit), used to
    explain the hit verdict. fixed_squad: no transfers after week 1 (chip scan's
    best squad). gap: stop once the plan is proven within this fraction of the
    best possible. Raises ValueError if no plan is valid, SolverTimeout (a ValueError)
    if time ran out before any plan was found.
    """
    model = _Model(inp, settings, week1_hits, fixed_squad)
    found, bound, proven = _chip_search(model, time.monotonic() + time_limit, gap)
    if found is None and bound == float("inf"):
        raise SolverTimeout(f"no plan found within {time_limit:g} s; try again or raise the time limit")
    if found is None:
        raise ValueError("no valid plan: check the budget and squad")
    objective, values = found
    for v in model.prob.variables():
        v.varValue = values.get(v.name, 0.0)
    plan_gap = max(0.0, bound - objective) / max(abs(objective), 1e-9)
    status = "Optimal" if proven else f"time limit (gap {100 * plan_gap:.2f}%)"
    plan = _read_plan(inp, model, settings, status)
    plan.objective, plan.gap = objective, plan_gap
    return plan


class _Model:
    """The integer program for one PlanInput (spec §4). Built once, solved many times.

    Chip choices (`use`) are free here; _chip_search fixes them by changing their
    bounds. Products of binaries that the objective pushes up (the tripled captain,
    boosted bench players, the kept squad in a Free Hit week, Wildcard transfers)
    are continuous: they come out 0/1 whenever the binaries do. The sum constraints
    on them are redundant for whole solutions but keep the LP relaxation tight,
    and the LP bound is what the chip search prunes with.
    """

    def __init__(self, inp: PlanInput, settings: PlanSettings, week1_hits: str, fixed_squad: bool):
        self.inp = inp
        gws = self.gws = [int(g) for g in inp.xp.columns]
        P = self.P = [int(p) for p in inp.pool.index]
        pos, club = inp.pool["position"], inp.pool["team"]
        price = inp.pool["price"].astype(int)
        sale = inp.pool["sell"].astype(int)
        xp = inp.xp
        new_team = not inp.squad
        owned0 = set(inp.squad)
        # Bank + squad value never grows (selling price <= price), so this caps any Free Hit budget.
        budget_cap = (inp.bank + sum(int(sale[p]) for p in owned0)
                      + sum(max(0, int(sale[p] - price[p])) for p in P))
        prob = self.prob = pulp.LpProblem("fpl_plan", pulp.LpMaximize)

        def binary(name):
            return pulp.LpVariable(name, cat="Binary")

        def unit(name):
            return pulp.LpVariable(name, 0, 1)

        use = self.use = {(ch.key, w): binary(f"use_{ch.key}_{w}")
                          for ch in inp.chips for w in gws if ch.allows(w)}

        def chip_flag(name, w):
            return pulp.lpSum(v for (k, ww), v in use.items() if ww == w and k.startswith(name + "_"))

        def chip_possible(name, w):
            return any(k.startswith(name + "_") and ww == w for k, ww in use)

        x, tin, tout, z, s, c, y, tcp, bbp = ({} for _ in range(9))
        paid, ft, n_in, counted = {}, {}, {}, {}
        objective = []
        prev_x = {p: int(p in owned0) for p in P}
        bank_prev = inp.bank
        for i, w in enumerate(gws):
            for p in P:
                for d, tag in ((x, "x"), (tin, "in"), (tout, "out"), (s, "s"), (c, "c")):
                    d[p, w] = binary(f"{tag}_{p}_{w}")
                prob += x[p, w] == prev_x[p] + tin[p, w] - tout[p, w]
                prob += tin[p, w] + tout[p, w] <= 1
            for position, k in rules.SQUAD_SHAPE.items():
                prob += pulp.lpSum(x[p, w] for p in P if pos[p] == position) == k
            for team in club.unique():
                members = [p for p in P if club[p] == team]
                count = pulp.lpSum(x[p, w] for p in members)
                held = sum(1 for p in owned0 if club[p] == team)
                if held <= rules.MAX_PER_CLUB:
                    prob += count <= rules.MAX_PER_CLUB
                else:
                    # A mid-season move left more than 3 from this club: FPL lets you keep them,
                    # but a purchase from the club must leave you with 3 or fewer. Without a
                    # purchase the count can only fall, so it never exceeds `held`.
                    for p in members:
                        prob += count <= rules.MAX_PER_CLUB + (held - rules.MAX_PER_CLUB) * (1 - tin[p, w])
            bank_w = bank_prev + pulp.lpSum(sale[p] * tout[p, w] - price[p] * tin[p, w] for p in P)
            prob += bank_w >= 0
            n_in[w] = pulp.lpSum(tin[p, w] for p in P)
            paid[w] = pulp.LpVariable(f"paid_{w}", lowBound=0, cat="Integer")
            prob += paid[w] <= n_in[w]
            wc, fh = chip_flag("wildcard", w), chip_flag("freehit", w)
            if fixed_squad and i > 0:
                prob += n_in[w] == 0

            # Transfers that count against free transfers: all of them, except under a Wildcard.
            if chip_possible("wildcard", w):
                wc_in = {p: unit(f"wcin_{p}_{w}") for p in P}           # bought with the Wildcard
                for p in P:
                    prob += wc_in[p] <= tin[p, w]
                    prob += wc_in[p] <= wc
                    prob += tin[p, w] - wc_in[p] <= 1 - wc
                prob += pulp.lpSum(wc_in.values()) <= BIG * wc
                counted[w] = n_in[w] - pulp.lpSum(wc_in.values())
            else:
                counted[w] = n_in[w]

            # Free transfers: see rules.next_free_transfers. ft[w] is the count for week w.
            if new_team and i == 0:
                prob += paid[w] == 0
                ft[w] = None
            else:
                ft[w] = pulp.LpVariable(f"ft_{w}", 0, rules.MAX_FREE_TRANSFERS, cat="Integer")
                top_up = rules.FREE_TRANSFER_TOP_UPS.get(inp.season, {}).get(w)
                if top_up is not None:
                    prob += ft[w] == top_up
                elif i == 0:
                    prob += ft[w] == inp.free_transfers
                elif ft[gws[i - 1]] is None:
                    prob += ft[w] == 1
                else:
                    _link_free_transfers(prob, ft[w], gws[i - 1], ft, counted, paid, chip_flag)
                # paid is exactly max(0, transfers - free) in a normal week and 0 in a Wildcard week,
                # so _link_free_transfers can't be gamed by "paying" for a transfer that was free.
                hit = binary(f"hit_{w}")
                prob += paid[w] >= counted[w] - ft[w]
                prob += paid[w] <= counted[w] - ft[w] + BIG * (1 - hit)
                prob += paid[w] <= BIG * hit
                prob += paid[w] <= BIG * (1 - wc)

            # The 15 who play: your squad, or the Free Hit squad (y) in a Free Hit week.
            if chip_possible("freehit", w):
                kept = {p: unit(f"kept_{p}_{w}") for p in P}            # = x, or 0 in a Free Hit week
                for p in P:
                    prob += tin[p, w] <= 1 - fh                          # no transfers with a Free Hit
                    y[p, w] = binary(f"y_{p}_{w}")
                    prob += y[p, w] <= fh
                    prob += kept[p] <= x[p, w]
                    prob += kept[p] <= 1 - fh
                    prob += kept[p] >= x[p, w] - fh
                    z[p, w] = kept[p] + y[p, w]
                for position, k in rules.SQUAD_SHAPE.items():
                    prob += pulp.lpSum(y[p, w] for p in P if pos[p] == position) == k * fh
                    prob += pulp.lpSum(kept[p] for p in P if pos[p] == position) == k * (1 - fh)
                for team in club.unique():
                    prob += pulp.lpSum(y[p, w] for p in P if club[p] == team) <= rules.MAX_PER_CLUB * fh
                # Budget = what the squad is worth plus the bank.
                prob += (pulp.lpSum(sale[p] * y[p, w] for p in P)
                         <= bank_prev + pulp.lpSum(sale[p] * prev_x[p] for p in P))
                prob += pulp.lpSum(sale[p] * y[p, w] for p in P) <= budget_cap * fh
            else:
                for p in P:
                    z[p, w] = x[p, w]

            prob += pulp.lpSum(s[p, w] for p in P) == 11
            prob += pulp.lpSum(c[p, w] for p in P) == 1
            for p in P:
                prob += s[p, w] <= z[p, w]
                prob += c[p, w] <= s[p, w]
            for position, (lo, hi) in rules.XI_LIMITS.items():
                starters = pulp.lpSum(s[p, w] for p in P if pos[p] == position)
                prob += starters >= lo
                prob += starters <= hi
            if chip_possible("3xc", w):                                  # tcp: the tripled captain
                for p in P:
                    tcp[p, w] = unit(f"tc_{p}_{w}")
                    prob += tcp[p, w] <= c[p, w]
                prob += pulp.lpSum(tcp[p, w] for p in P) <= chip_flag("3xc", w)
            if chip_possible("bboost", w):                               # bbp: a boosted bench player
                for p in P:
                    bbp[p, w] = unit(f"bb_{p}_{w}")
                    prob += bbp[p, w] <= z[p, w] - s[p, w]
                prob += pulp.lpSum(bbp[p, w] for p in P) <= 4 * chip_flag("bboost", w)
            prob += pulp.lpSum(v for (k, ww), v in use.items() if ww == w) <= 1

            weight = settings.discount ** i
            points = pulp.lpSum(
                float(xp.at[p, w]) * (s[p, w] + c[p, w] + settings.bench_weight * (z[p, w] - s[p, w]))
                for p in P)
            # Triple Captain / Bench Boost extras count in full in every week (weight 1), so
            # discounting never pushes a chip towards week 1 just because it is sooner.
            chip_extra = pulp.lpSum(
                float(xp.at[p, w]) * (tcp.get((p, w), 0) + (1 - settings.bench_weight) * bbp.get((p, w), 0))
                for p in P)
            objective.append(weight * (points - (rules.HIT_COST + settings.hit_margin) * paid[w]) + chip_extra
                             - TIE_BREAK * n_in[w])
            prev_x = {p: x[p, w] for p in P}
            bank_prev = bank_w

        first = gws[0]
        if week1_hits == "none":
            prob += paid[first] == 0
        elif week1_hits == "some":
            # A genuine extra transfer: more transfers than free ones, with no chip making them free.
            prob += n_in[first] >= ft[first] + 1
            prob += chip_flag("wildcard", first) + chip_flag("freehit", first) == 0
        for ch in inp.chips:
            uses = [v for (k, _), v in use.items() if k == ch.key]
            if uses:
                prob += pulp.lpSum(uses) <= 1
            objective.append(inp.hold_values.get(ch.key, 0.0) * (1 - pulp.lpSum(uses)))
        last = gws[-1]
        ft_end = pulp.LpVariable("ft_end", 0, rules.MAX_FREE_TRANSFERS, cat="Integer")
        if ft[last] is None:
            prob += ft_end == 1
        else:
            _link_free_transfers(prob, ft_end, last, ft, counted, paid, chip_flag)
        objective.append(settings.ft_end_value * ft_end)
        prob += pulp.lpSum(objective)
        self.x, self.tin, self.tout, self.z, self.s, self.c = x, tin, tout, z, s, c
        self.paid, self.ft = paid, ft
        self.integers = [v for v in prob.variables() if v.cat == pulp.LpInteger]
        self.highs = None                 # HiGHS copy of the model, built on first use and kept

    def chip_keys(self) -> list[str]:
        """Chip copies the plan could play; Wildcard and Free Hit first (they weaken the LP most)."""
        return sorted({k for k, _ in self.use if k.split("_")[0] in BRANCH_CHIPS},
                      key=lambda k: (BRANCH_CHIPS.index(k.split("_")[0]), k))

    def fix(self, fixed: dict) -> None:
        """Fix chips: key -> the week it is played, or None (not in this plan). The rest are free."""
        for (k, w), v in self.use.items():
            v.lowBound, v.upBound = (int(fixed[k] == w),) * 2 if k in fixed else (0, 1)
            if self.highs is not None:
                self.highs.changeColBounds(v.index, v.lowBound, v.upBound)

    def run(self, seconds: float, gap: float, relax: bool = False, cutoff: float | None = None,
            scale: float | None = None):
        """(objective, best bound, values); None if no plan here is valid (or beats `cutoff`);
        OUT_OF_TIME if time ran out first. With relax, the LP relaxation: its objective is a bound.
        scale (e.g. the LP bound): stop once within gap * scale points, so the gap is measured on the
        whole objective as _chip_search measures it."""
        seconds = max(1.0, seconds)
        if HIGHS:
            return self._run_highs(seconds, gap, relax, cutoff, scale)
        return _run_cbc(self.prob, seconds, gap, relax, self.integers)

    def _run_highs(self, seconds, gap, relax, cutoff, scale):
        """One persistent HiGHS model: a chip-fix is a bound change and each LP starts from the last basis."""
        if self.highs is None:
            solver = pulp.HiGHS(msg=False)
            solver.createAndConfigureSolver(self.prob)
            solver.buildSolverModel(self.prob)     # sets each variable's .index
            self.highs = self.prob.solverModel
        h, constant = self.highs, self.prob.objective.constant     # HiGHS minimises -objective
        h.setOptionValue("time_limit", float(seconds))
        h.setOptionValue("mip_rel_gap", float(gap if scale is None else 0.0))
        h.setOptionValue("mip_abs_gap", float(0.0 if scale is None else gap * abs(scale)))
        h.setOptionValue("solve_relaxation", bool(relax))
        h.setOptionValue("objective_bound", float("inf") if cutoff is None else constant - cutoff)
        h.run()
        info, status = h.getInfo(), h.getModelStatus()
        if status in (highspy.HighsModelStatus.kInfeasible, highspy.HighsModelStatus.kObjectiveBound):
            return None
        if relax and status != highspy.HighsModelStatus.kOptimal or info.primal_solution_status != 2:
            return OUT_OF_TIME                    # 2 = a feasible solution is available
        objective = constant - info.objective_function_value
        if relax:
            return objective, objective, None
        bound = max(objective, constant - info.mip_dual_bound)
        cols = h.getSolution().col_value
        return objective, bound, {v.name: cols[v.index] for v in self.prob.variables()}


def _chip_search(model: _Model, deadline: float, gap: float):
    """Best plan by branch and bound over chip weeks, best LP bound first.

    One MIP with every chip free proves slowly: a fractional Wildcard or Free Hit
    in the LP relaxation looks far better than any real one, so the bound barely
    moves. Instead each chip is fixed in turn (to a week, or to "not in this plan"),
    the LP with those chips fixed bounds the branch, branches that can't beat the
    best plan by more than `gap` are dropped, and a MIP is solved only once every
    chip is fixed. Returns ((objective, values) or None, best bound, proven): with no plan
    found, the bound is +inf if time ran out and -inf if no plan is valid. Proven means
    the search finished within the time limit, every plan left out being at most `gap`
    (a fraction of its LP bound) better than the one returned.
    """
    keys = model.chip_keys()
    if not keys:
        result = model.run(deadline - time.monotonic(), gap)
        if result is None or result is OUT_OF_TIME:
            return None, float("inf" if result is OUT_OF_TIME else "-inf"), False
        objective, bound, values = result
        return (objective, values), bound, bound <= objective + gap * abs(objective) + 1e-6
    weeks = {k: [w for kk, w in model.use if kk == k] for k in keys}
    heap, best, unproven, order = [], None, [], itertools.count()
    ceiling = -float("inf")                    # the best bound among branches dropped as beaten

    def beaten(bound):
        nonlocal ceiling
        if best is not None and bound <= best[0] + gap * abs(best[0]) + 1e-9:
            ceiling = max(ceiling, bound)
            return True
        return False

    def push(fixed, parent_bound):
        result = OUT_OF_TIME
        if deadline - time.monotonic() > 1:
            model.fix(fixed)
            result = model.run(deadline - time.monotonic(), gap, relax=True)
        if result is None:
            return                                                       # no valid plan here
        bound = parent_bound if result is OUT_OF_TIME else result[0]
        if not beaten(bound):
            heapq.heappush(heap, (-bound, next(order), fixed))

    def solve_leaf(fixed, lp_bound):
        """The MIP with every branched chip fixed; updates best, unproven and ceiling."""
        nonlocal best, ceiling
        model.fix(fixed)
        cutoff = None if best is None else best[0] + gap * abs(best[0])
        result = model.run(deadline - time.monotonic(), gap, cutoff=cutoff, scale=lp_bound)
        if result is None:                                               # can't beat the best plan
            ceiling = max(ceiling, -float("inf") if cutoff is None else cutoff)
            return
        if result is OUT_OF_TIME:
            unproven.append(lp_bound)
            return
        objective, bound, values = result
        if best is None or objective > best[0]:
            best = (objective, values)
        bound = min(bound, lp_bound)                   # the leaf's LP bound holds too
        if bound > objective + gap * abs(lp_bound) + 1e-6 and (cutoff is None or bound > cutoff + 1e-6):
            unproven.append(bound)                     # stopped at the time limit, still might win
        ceiling = max(ceiling, bound)

    push({}, float("inf"))
    while heap:
        neg_bound, _, fixed = heapq.heappop(heap)
        if beaten(-neg_bound):
            continue
        if deadline - time.monotonic() <= 1:
            unproven += [-neg_bound] + [-b for b, _, _ in heap]
            break
        if len(fixed) < len(keys):
            key = keys[len(fixed)]
            taken = set(fixed.values())
            for option in [None] + [w for w in weeks[key] if w not in taken]:
                push({**fixed, key: option}, -neg_bound)
            continue
        solve_leaf(fixed, -neg_bound)                                    # every chip fixed: solve
    if best is None:
        return None, float("inf" if unproven else "-inf"), False
    return best, max([best[0], ceiling] + unproven), not unproven


def _run_cbc(prob, seconds, gap, relax, integers):
    """The CBC fallback (no highspy): (objective, bound, values) or None."""
    constant = prob.objective.constant
    if relax:
        for v in integers:
            v.cat = pulp.LpContinuous
    try:
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "cbc.log"
            prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=seconds, gapRel=gap, mip=not relax,
                                         logPath=str(log)))
            text = log.read_text(errors="ignore") if log.exists() else ""
    finally:
        if relax:
            for v in integers:
                v.cat = pulp.LpInteger
    if prob.status == pulp.LpStatusInfeasible:
        return None
    if prob.sol_status not in (pulp.LpSolutionOptimal, pulp.LpSolutionIntegerFeasible):
        return OUT_OF_TIME
    objective = pulp.value(prob.objective)
    if relax:
        return objective, objective, None
    found = re.findall(r"best possible (-?[\d.eE+]+)", text)        # CBC minimises -objective
    bound = (objective if prob.sol_status == pulp.LpSolutionOptimal
             else constant - float(found[-1]) if found else float("inf"))
    return objective, max(bound, objective), {v.name: v.varValue for v in prob.variables()}


def _link_free_transfers(prob, ft_next, w, ft, counted, paid, chip_flag) -> None:
    """ft_next = min(5, max(ft - used, 0) + 1) in a normal week; = ft after a Wildcard/Free Hit.

    Exact (both bounds), so the reported count is the real one even when nothing
    later depends on it. ft - used + paid = max(ft - used, 0) because paid is exact.
    In a chip week no transfer counts and none is paid, so `rolled` = ft + 1 and its
    upper bound needs no chip term.
    """
    chip = chip_flag("wildcard", w) + chip_flag("freehit", w)
    rolled = ft[w] - counted[w] + paid[w] + 1
    capped = pulp.LpVariable(f"{ft_next.name}_capped", cat="Binary")   # 1: the cap of 5 applies
    prob += ft_next <= rolled
    prob += ft_next >= rolled - BIG * capped - BIG * chip
    prob += ft_next >= rules.MAX_FREE_TRANSFERS * capped - BIG * chip
    prob += ft_next <= ft[w] + BIG * (1 - chip)
    prob += ft_next >= ft[w] - BIG * (1 - chip)


def _on(var) -> bool:
    """True if a binary variable (or a 0/1 constant) is 1 in the solution."""
    return round(var.value() or 0) == 1 if hasattr(var, "value") else var == 1


def _read_plan(inp, m: _Model, settings, status) -> Plan:
    gws, P, xp, x, tin, tout, z, s, c, use, paid, ft = (m.gws, m.P, inp.xp, m.x, m.tin, m.tout, m.z, m.s,
                                                        m.c, m.use, m.paid, m.ft)
    weeks, bank, value = [], inp.bank, 0.0
    for i, w in enumerate(gws):
        chip = next((ch.name for ch in inp.chips if (ch.key, w) in use and _on(use[ch.key, w])), None)
        bought = [p for p in P if _on(tin[p, w])]
        sold = [p for p in P if _on(tout[p, w])]
        bank += sum(int(inp.pool.at[p, "sell"]) for p in sold) - sum(int(inp.pool.at[p, "price"]) for p in bought)
        squad = [p for p in P if _on(z[p, w])]
        starters = [p for p in P if _on(s[p, w])]
        captain = next(p for p in P if _on(c[p, w]))
        on_bench = [p for p in squad if p not in starters]
        bench = ([p for p in on_bench if inp.pool.at[p, "position"] == "GK"]
                 + sorted((p for p in on_bench if inp.pool.at[p, "position"] != "GK"),
                          key=lambda p: -xp.at[p, w]))
        vice = max((p for p in starters if p != captain), key=lambda p: xp.at[p, w])
        hits = int(round(paid[w].value() or 0))
        chip_extra = (xp.at[captain, w] if chip == "3xc"
                      else sum(xp.at[p, w] for p in on_bench) if chip == "bboost"
                      else 0.0)
        week_xp = (sum(xp.at[p, w] for p in starters) + xp.at[captain, w]
                   + (xp.at[captain, w] if chip == "3xc" else 0)
                   + (sum(xp.at[p, w] for p in on_bench) if chip == "bboost" else 0)
                   - rules.HIT_COST * hits)
        value += settings.discount ** i * (week_xp - chip_extra) + chip_extra
        weeks.append(WeekPlan(
            gw=w, chip=chip, transfers_in=bought, transfers_out=sold,
            free_transfers=None if ft[w] is None else int(round(ft[w].value())), paid=hits,
            owned=[p for p in P if _on(x[p, w])], squad=squad, starters=starters, bench=bench,
            captain=captain, vice=vice, xp=float(week_xp), bank=int(bank)))
    return Plan(weeks=weeks, value=float(value), status=status)


def best_squad(xp: pd.DataFrame, pool: pd.DataFrame, budget: int, season: str,
               time_limit: int = 20) -> Plan:
    """The best single squad (no transfers after week 1) for these weeks, from scratch.

    Used by the chip scan for Free Hit and Wildcard values.
    """
    inp = PlanInput(pool=pool, xp=xp, squad=[], bank=budget, free_transfers=0, chips=[], season=season)
    flat = PlanSettings(discount=1.0, hit_margin=0.0, ft_end_value=0.0, bench_weight=0.0)
    return solve(inp, flat, fixed_squad=True, time_limit=time_limit)
