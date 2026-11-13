"""Offline tests for fpl/optimizer.py on small made-up player pools."""
import dataclasses

import pandas as pd
import pytest

from fpl import optimizer, rules
from fpl.settings import PlanSettings

FLAT = PlanSettings(discount=1.0, hit_margin=1.0, ft_end_value=1.5, bench_weight=0.1)


def make_pool():
    """30 players: 4 GK, 10 DEF, 10 MID, 6 FWD over 10 clubs; prices 45-65."""
    rows, code = [], 1
    for position, count in [("GK", 4), ("DEF", 10), ("MID", 10), ("FWD", 6)]:
        for i in range(count):
            rows.append({"player_code": code, "name": f"{position}{i}", "position": position,
                         "team": f"C{code % 10}", "price": 45 + 5 * (i % 5)})
            code += 1
    pool = pd.DataFrame(rows).set_index("player_code")
    pool["sell"] = pool["price"]
    return pool


def owned_squad(pool):
    """A legal 15: the first 2 GK, 5 DEF, 5 MID, 3 FWD by code."""
    return [c for pos, k in rules.SQUAD_SHAPE.items() for c in pool[pool.position == pos].index[:k]]


def flat_xp(pool, gws, value=2.0):
    return pd.DataFrame(value, index=pool.index, columns=gws)


def plan_input(pool, xp, squad, bank=100, free=1, chips=(), season="2026-27", holds=None):
    return optimizer.PlanInput(pool=pool, xp=xp, squad=squad, bank=bank, free_transfers=free,
                               chips=list(chips), season=season, hold_values=holds or {})


def test_new_team_obeys_every_squad_rule():
    pool = make_pool()
    xp = flat_xp(pool, [1, 2])
    xp.loc[pool.index, 1] = range(1, len(pool) + 1)             # distinct values
    plan = optimizer.solve(plan_input(pool, xp, [], bank=rules.BUDGET, free=0), FLAT)
    week = plan.weeks[0]
    chosen = pool.loc[week.owned]
    assert chosen.position.value_counts().to_dict() == rules.SQUAD_SHAPE
    assert chosen.team.value_counts().max() <= rules.MAX_PER_CLUB
    assert chosen.price.sum() + week.bank == rules.BUDGET and week.bank >= 0
    assert rules.valid_xi(pool.loc[week.starters, "position"])
    assert week.captain == max(week.starters, key=lambda p: xp.at[p, 1])
    assert week.vice != week.captain and week.vice in week.starters
    assert pool.at[week.bench[0], "position"] == "GK" and len(week.bench) == 4
    assert plan.weeks[1].free_transfers == 1


@pytest.mark.parametrize("gain,expect_hit", [(4.5, False), (5.5, True)])
def test_hit_only_when_gain_beats_four_plus_margin(gain, expect_hit):
    pool = make_pool()
    squad = owned_squad(pool)
    xp = flat_xp(pool, [7])
    star = pool[(pool.position == "MID") & ~pool.index.isin(squad)].index[0]
    xp.loc[squad[-1], 7] = 10.0                                   # captain stays fixed
    xp.loc[star, 7] = 2.0 + gain
    plan = optimizer.solve(plan_input(pool, xp, squad, bank=100, free=0), FLAT)
    assert (star in plan.weeks[0].transfers_in) == expect_hit
    assert plan.weeks[0].paid == (1 if expect_hit else 0)


def test_rolls_a_free_transfer_when_nothing_is_worth_buying_and_caps_at_five():
    pool = make_pool()
    squad = owned_squad(pool)
    plan = optimizer.solve(plan_input(pool, flat_xp(pool, [7, 8]), squad, free=1), FLAT)
    assert plan.weeks[0].transfers_in == [] and plan.weeks[1].free_transfers == 2
    plan = optimizer.solve(plan_input(pool, flat_xp(pool, [7, 8]), squad, free=5), FLAT)
    assert plan.weeks[1].free_transfers == 5


def test_wildcard_keeps_free_transfers_without_adding_one():
    pool = make_pool()
    squad = owned_squad(pool)
    xp = flat_xp(pool, [7, 8], value=1.0)
    others = [p for p in pool.index if p not in squad]
    xp.loc[others, :] = 5.0                                      # everyone you don't own is better
    plan = optimizer.solve(plan_input(pool, xp, squad, bank=0, free=2,
                                      chips=[rules.Chip("wildcard", 2, 19)]), FLAT)
    assert plan.weeks[0].chip == "wildcard" and plan.weeks[0].paid == 0
    assert len(plan.weeks[0].transfers_in) > 2
    assert plan.weeks[1].free_transfers == 2


def test_free_hit_squad_reverts_the_next_week():
    pool = make_pool()
    squad = owned_squad(pool)
    xp = flat_xp(pool, [7, 8])
    others = [p for p in pool.index if p not in squad]
    xp.loc[others, 7] = 9.0                                      # only good for one week
    plan = optimizer.solve(plan_input(pool, xp, squad, free=1,
                                      chips=[rules.Chip("freehit", 2, 19)]), FLAT)
    week1, week2 = plan.weeks
    assert week1.chip == "freehit" and week1.transfers_in == []
    assert set(week1.squad) != set(squad) and set(week2.squad) == set(squad)
    assert week2.free_transfers == 1                             # kept, no +1


def test_chip_windows_and_hold_values_are_respected():
    pool = make_pool()
    squad = owned_squad(pool)
    xp = flat_xp(pool, [7, 8])
    late_tc = rules.Chip("3xc", 20, 38)
    plan = optimizer.solve(plan_input(pool, xp, squad, chips=[late_tc]), FLAT)
    assert all(w.chip is None for w in plan.weeks)               # outside its window
    tc = rules.Chip("3xc", 1, 19)
    kept = optimizer.solve(plan_input(pool, xp, squad, chips=[tc], holds={tc.key: 50.0}), FLAT)
    used = optimizer.solve(plan_input(pool, xp, squad, chips=[tc], holds={tc.key: 0.0}), FLAT)
    assert all(w.chip is None for w in kept.weeks)
    assert sum(w.chip == "3xc" for w in used.weeks) == 1


def test_selling_prices_limit_what_you_can_buy():
    pool = make_pool()
    squad = owned_squad(pool)
    seller = squad[2]                                            # a DEF
    pool.loc[squad, "sell"] = 45                                 # no other sale can fund the target
    pool.loc[seller, ["price", "sell"]] = [60, 50]
    target = pool[(pool.position == "DEF") & ~pool.index.isin(squad)].index[0]
    pool.loc[target, ["price", "sell"]] = [55, 55]
    xp = flat_xp(pool, [7])
    xp.loc[target, 7] = 20.0
    xp.loc[squad[2:7], 7] = [0.0, 3.0, 3.0, 3.0, 3.0]            # the seller is the one to replace
    poor = optimizer.solve(plan_input(pool, xp, squad, bank=4, free=1), FLAT)
    rich = optimizer.solve(plan_input(pool, xp, squad, bank=5, free=1), FLAT)
    assert target not in poor.weeks[0].transfers_in and target in rich.weeks[0].transfers_in


def test_afcon_top_up_gives_five_free_transfers_in_2025_26_gw16():
    pool = make_pool()
    squad = owned_squad(pool)
    xp = flat_xp(pool, [16], value=1.0)
    xp.loc[[p for p in pool.index if p not in squad], 16] = 4.0
    plan = optimizer.solve(plan_input(pool, xp, squad, bank=200, free=1, season="2025-26"), FLAT)
    assert plan.weeks[0].free_transfers == 5 and plan.weeks[0].paid == 0


def test_a_player_with_no_predictions_is_benched_or_sold():
    pool = make_pool()
    squad = owned_squad(pool)
    xp = flat_xp(pool, [7, 8])
    xp.loc[squad[3], :] = 0.0                                    # e.g. loaned out (McNally)
    week = optimizer.solve(plan_input(pool, xp, squad, free=1), FLAT).weeks[0]
    assert squad[3] not in week.starters


def test_forced_hit_is_a_genuine_extra_transfer():
    """With 1 free transfer, 'some' must make 2+ transfers and must not earn a free transfer next week."""
    pool = make_pool()
    squad = owned_squad(pool)
    xp = flat_xp(pool, [7, 8])
    chips = [rules.Chip("wildcard", 2, 19)]
    forced = optimizer.solve(plan_input(pool, xp, squad, free=1, chips=chips), FLAT, week1_hits="some")
    week1, week2 = forced.weeks
    assert week1.chip is None
    assert len(week1.transfers_in) >= 2
    assert week1.paid == len(week1.transfers_in) - 1
    assert week2.free_transfers == 1
    none = optimizer.solve(plan_input(pool, xp, squad, free=1), FLAT, week1_hits="none")
    assert none.weeks[0].paid == 0


def test_paid_hits_equal_transfers_beyond_the_free_ones():
    pool = make_pool()
    squad = owned_squad(pool)
    xp = flat_xp(pool, [7, 8])
    for week in optimizer.solve(plan_input(pool, xp, squad, free=2), FLAT, week1_hits="some").weeks:
        if week.free_transfers is not None and week.chip is None:
            assert week.paid == max(0, len(week.transfers_in) - week.free_transfers)


def test_impossible_budget_raises_a_clear_error():
    pool = make_pool()
    with pytest.raises(ValueError, match="no valid plan"):
        optimizer.solve(plan_input(pool, flat_xp(pool, [1]), [], bank=100, free=0), FLAT)


def test_candidate_pool_keeps_owned_and_cheap_players():
    pool = make_pool()
    xp = flat_xp(pool, [7])
    keep = optimizer.candidate_pool(xp, pool, [30], 0.9, per_position=1, value_per_position=1, cheapest=1)
    assert 30 in keep and len(keep) <= 1 + 4 * 3


def test_triple_captain_goes_to_the_best_week_not_the_soonest():
    pool = make_pool()
    squad = owned_squad(pool)
    xp = flat_xp(pool, [1, 2, 3, 4, 5, 6])
    star = squad[-1]
    xp.loc[star, 1], xp.loc[star, 4] = 6.0, 7.0
    tc = rules.Chip("3xc", 1, 19)
    settings = PlanSettings(discount=0.9, hit_margin=1.0, ft_end_value=1.5, bench_weight=0.1)
    plan = optimizer.solve(plan_input(pool, xp, squad, chips=[tc], holds={tc.key: 0.0}), settings)
    assert [w.gw for w in plan.weeks if w.chip == "3xc"] == [4]


def chip_case(seed, holds):
    """3 weeks, random xP, all four first-set chips: a small model the chip search must solve exactly."""
    import numpy as np
    pool = make_pool()
    squad = owned_squad(pool)
    rng = np.random.default_rng(seed)
    xp = pd.DataFrame(rng.uniform(0, 6, (len(pool), 3)), index=pool.index, columns=[7, 8, 9])
    chips = [rules.Chip("wildcard", 2, 19), rules.Chip("freehit", 2, 19),
             rules.Chip("bboost", 1, 19), rules.Chip("3xc", 1, 19)]
    return plan_input(pool, xp, squad, bank=20, free=1, chips=chips,
                      holds={ch.key: holds for ch in chips})


def best_by_enumeration(inp):
    """The best objective over every chip-week assignment, each solved as its own MIP."""
    import itertools
    model = optimizer._Model(inp, FLAT, "free", False)
    keys = sorted({k for k, _ in model.use})
    options = [[None] + [w for kk, w in model.use if kk == k] for k in keys]
    best = -float("inf")
    for choice in itertools.product(*options):
        weeks = [w for w in choice if w is not None]
        if len(weeks) == len(set(weeks)):                       # at most one chip a week
            model.fix(dict(zip(keys, choice)))
            result = model.run(60, 0.0)
            if result is not None and result is not optimizer.OUT_OF_TIME:
                best = max(best, result[0])
    return best


@pytest.mark.parametrize("seed,holds", [(0, 0.0), (1, 8.0)])
def test_chip_search_finds_the_best_chip_weeks(monkeypatch, seed, holds):
    inp = chip_case(seed, holds)
    searched = optimizer.solve(inp, FLAT, gap=0.0)
    assert searched.status == "Optimal"
    assert searched.objective == pytest.approx(best_by_enumeration(inp), abs=1e-5)
    monkeypatch.setattr(optimizer, "HIGHS", False)              # the CBC fallback agrees
    cbc = optimizer.solve(inp, FLAT, gap=0.0)
    assert cbc.status == "Optimal" and cbc.objective == pytest.approx(searched.objective, abs=1e-4)


@pytest.mark.parametrize("with_chips", [True, False])
def test_a_plan_stopped_at_the_time_limit_reports_its_gap(monkeypatch, with_chips):
    real_run = optimizer._Model.run

    def stopped_early(self, seconds, gap, relax=False, cutoff=None, scale=None):
        result = real_run(self, seconds, gap, relax, cutoff, scale)
        if relax or result is None or result is optimizer.OUT_OF_TIME:
            return result
        objective, _, values = result
        return objective, objective + 0.05 * abs(objective), values   # as if the MIP ran out of time

    monkeypatch.setattr(optimizer._Model, "run", stopped_early)
    inp = chip_case(0, 5.0)
    if not with_chips:
        inp = dataclasses.replace(inp, chips=[])
    plan = optimizer.solve(inp, FLAT)
    assert plan.status == f"time limit (gap {100 * plan.gap:.2f}%)"
    assert 0.0005 < plan.gap <= 0.05 + 1e-9          # capped by the LP bound when chips are searched


def four_from_one_club():
    """Owned squad where a mid-season move left 4 players at club C5 (codes 5, 6, 15, 25)."""
    pool = make_pool()
    squad = owned_squad(pool)
    pool.loc[6, "team"] = "C5"
    return pool, squad


def club_count(pool, players, team="C5"):
    return int((pool.loc[players, "team"] == team).sum())


def test_after_a_club_move_you_may_keep_four_from_one_club():
    pool, squad = four_from_one_club()
    plan = optimizer.solve(plan_input(pool, flat_xp(pool, [20, 21]), squad, free=1), FLAT)
    assert all(club_count(pool, w.owned) == 4 for w in plan.weeks)       # no forced sale, no error


def test_you_can_only_buy_from_that_club_if_you_end_up_with_three_or_fewer():
    pool, squad = four_from_one_club()
    star = 20                                                            # a MID you don't own...
    pool.loc[star, "team"] = "C5"                                        # ...at the same club
    xp = flat_xp(pool, [20, 21])
    xp.loc[star, :] = 20.0
    plan = optimizer.solve(plan_input(pool, xp, squad, bank=100, free=1), FLAT)
    bought_star = [w for w in plan.weeks if star in w.transfers_in]
    assert bought_star                                                   # worth 2 sales of C5 players
    for w in plan.weeks:
        assert club_count(pool, w.owned) <= (3 if any(pool.loc[w.transfers_in, "team"] == "C5") else 4)
    week = bought_star[0]
    assert club_count(pool, week.owned) <= 3 and len(week.transfers_in) >= 2


def test_a_free_hit_squad_still_has_at_most_three_from_a_club_after_a_move():
    pool, squad = four_from_one_club()
    xp = flat_xp(pool, [20, 21])
    xp.loc[[p for p in pool.index if p not in squad], 20] = 9.0      # a Free Hit is worth playing
    xp.loc[pool.index[pool.team == "C5"], 20] = 15.0                 # and it would love more C5 players
    plan = optimizer.solve(plan_input(pool, xp, squad, free=1, chips=[rules.Chip("freehit", 2, 38)]), FLAT)
    week1, week2 = plan.weeks
    assert week1.chip == "freehit"
    assert club_count(pool, week1.squad) <= 3                        # the Free Hit squad obeys the normal limit
    assert club_count(pool, week2.squad) == 4                        # your own squad comes back with its 4


@pytest.mark.parametrize("with_chips", [True, False])
def test_running_out_of_time_before_any_plan_says_so(monkeypatch, with_chips):
    """A time-out with no plan found is not a budget problem: the message must say to retry."""
    def out_of_time(self, seconds, gap, relax=False, cutoff=None, scale=None):
        return optimizer.OUT_OF_TIME
    monkeypatch.setattr(optimizer._Model, "run", out_of_time)
    inp = chip_case(0, 5.0)
    if not with_chips:
        inp = dataclasses.replace(inp, chips=[])
    with pytest.raises(optimizer.SolverTimeout,
                       match=r"no plan found within 7 s; try again or raise the time limit"):
        optimizer.solve(inp, FLAT, time_limit=7)
    assert issubclass(optimizer.SolverTimeout, ValueError)   # callers that retry still catch it
