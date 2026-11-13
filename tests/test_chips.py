"""Offline tests for fpl/chips.py."""
import pandas as pd
import pytest

from fpl import chips, rules
from test_optimizer import flat_xp, make_pool, owned_squad


def test_best_xi_is_a_legal_formation_of_the_best_players():
    pool = make_pool()
    squad = owned_squad(pool)
    xp = pd.Series(range(15), index=squad, dtype=float)
    xi = chips.best_xi(xp, pool.loc[squad, "position"])
    assert len(xi) == 11 and rules.valid_xi(pool.loc[xi, "position"])
    assert squad[-1] in xi                                        # the best player starts


def test_bench_boost_prefers_the_double_gameweek():
    pool = make_pool()
    squad = owned_squad(pool)
    xp = flat_xp(pool, [7, 8, 9, 10])
    xp.loc[squad, 9] = 4.0                                        # double gameweek for the squad
    found = chips.scan(xp, pool, squad, 0, [rules.Chip("bboost", 1, 19)], after_gw=7, season="2026-27")
    best = found.loc[found["value"].idxmax()]
    assert best["gw"] == 9 and best["value"] == 16.0


def test_hold_value_is_zero_when_the_window_ends_inside_the_plan():
    tc = rules.Chip("3xc", 1, 12)
    scan = pd.DataFrame(columns=["key", "name", "gw", "value"])
    assert chips.hold_values(scan, [tc]) == {tc.key: 0.0}
    scan = pd.DataFrame([{"key": tc.key, "name": "3xc", "gw": 9, "value": 10.0}])
    assert chips.hold_values(scan, [tc])[tc.key] == pytest.approx(10.0)


def test_wildcard_value_is_positive_for_a_bad_squad_and_zero_for_the_best():
    pool = make_pool()
    squad = owned_squad(pool)
    gws = [7, 8, 9, 10, 11, 12]
    xp = flat_xp(pool, gws, value=1.0)
    xp.loc[[p for p in pool.index if p not in squad], gws] = 6.0
    wc = rules.Chip("wildcard", 1, 19)
    found = chips.scan(xp, pool, squad, 0, [wc], after_gw=6, season="2026-27")
    assert found.loc[found["gw"] == 7, "value"].iloc[0] > 10
    good = flat_xp(pool, gws, value=1.0)
    good.loc[squad, gws] = 6.0
    found = chips.scan(good, pool, squad, 0, [wc], after_gw=6, season="2026-27")
    assert found.loc[found["gw"] == 7, "value"].iloc[0] == pytest.approx(0.0, abs=1e-6)
