"""Offline tests for fpl/rules.py: prices, free transfers, chips and gameweek scoring."""
import json
from pathlib import Path

import pandas as pd

from fpl import rules

DATA = Path(__file__).parent / "data"
XI = ["GK"] + ["DEF"] * 4 + ["MID"] * 4 + ["FWD"] * 2
BENCH = ["GK", "DEF", "MID", "FWD"]


def week(points=None, minutes=None, captain=10, vice=9, positions=None):
    """15 picks in pick order: 11 starters, bench GK, then the outfield bench in order."""
    p = pd.DataFrame({"position": positions or XI + BENCH,
                      "points": points or [2] * 15, "minutes": minutes or [90] * 15})
    p["is_captain"] = p.index == captain
    p["is_vice"] = p.index == vice
    return p


def test_selling_price_shares_rises_and_keeps_falls():
    assert rules.selling_price(50, 53) == 51
    assert rules.selling_price(50, 54) == 52
    assert rules.selling_price(50, 47) == 47
    assert rules.selling_price(50, 50) == 50


def test_free_transfers_roll_cap_and_chips():
    assert rules.next_free_transfers(1, 0, None) == 2
    assert rules.next_free_transfers(2, 2, None) == 1
    assert rules.next_free_transfers(1, 3, None) == 1
    assert rules.next_free_transfers(5, 0, None) == 5
    assert rules.next_free_transfers(2, 9, "wildcard") == 2
    assert rules.next_free_transfers(3, 0, "freehit") == 3
    assert rules.next_free_transfers(1, 0, "bboost") == 2
    assert rules.paid_transfers(1, 3, None) == 2
    assert rules.paid_transfers(1, 9, "wildcard") == 0


def test_afcon_top_up_only_in_2025_26_gw16():
    assert rules.free_transfers_at(16, 2, "2025-26") == 5
    assert rules.free_transfers_at(17, 2, "2025-26") == 2
    assert rules.free_transfers_at(16, 2, "2026-27") == 2


def test_rebuilt_free_transfers_match_every_hit_real_managers_paid():
    histories = json.loads((DATA / "manager_histories.json").read_text())
    checked = 0
    for h in histories:
        chip_at = {c["event"]: c["name"] for c in h["chips"]}
        free = None
        for row in h["current"]:
            gw, made = row["event"], row["event_transfers"]
            if free is None:
                free = 1            # first gameweek: unlimited changes, then 1 free transfer
                continue
            chip = chip_at.get(gw)
            assert row["event_transfers_cost"] == rules.HIT_COST * rules.paid_transfers(free, made, chip), (gw, h)
            checked += 1
            free = rules.next_free_transfers(free, made, chip)
    assert checked > 50


def test_chips_left_removes_the_copy_used():
    windows = [rules.Chip("wildcard", 2, 19), rules.Chip("wildcard", 20, 38), rules.Chip("3xc", 1, 19)]
    left = rules.chips_left(windows, [("wildcard", 5)])
    assert left == [rules.Chip("wildcard", 20, 38), rules.Chip("3xc", 1, 19)]
    assert rules.Chip("3xc", 1, 19).allows(19) and not rules.Chip("3xc", 1, 19).allows(20)


def test_score_week_everyone_plays():
    assert rules.score_week(week()) == 11 * 2 + 2           # captain doubled


def test_auto_sub_takes_first_bench_player_who_played():
    points = [2] * 11 + [1, 3, 5, 7]
    minutes = [90] * 15
    points[1], minutes[1] = 0, 0                            # a starting DEF didn't play
    assert rules.score_week(week(points, minutes)) == 10 * 2 + 3 + 2


def test_auto_sub_keeps_a_legal_formation():
    positions = ["GK"] + ["DEF"] * 3 + ["MID"] * 5 + ["FWD"] * 2 + ["GK", "MID", "DEF", "FWD"]
    points = [2] * 11 + [1, 4, 6, 8]
    minutes = [90] * 15
    points[1], minutes[1] = 0, 0          # a DEF out: bench MID would leave 2 DEF, so the bench DEF comes on
    assert rules.score_week(week(points, minutes, positions=positions)) == 10 * 2 + 6 + 2


def test_vice_captain_steps_in_and_triple_captain_triples():
    points = [2] * 15
    minutes = [90] * 15
    points[10], minutes[10] = 0, 0                          # captain didn't play
    points[9] = 6
    assert rules.score_week(week(points, minutes)) == 9 * 2 + 6 + 6 + 2   # vice doubled; bench DEF (2) subs on
    assert rules.score_week(week(), chip="3xc") == 11 * 2 + 4


def test_bench_boost_and_hits():
    assert rules.score_week(week(), chip="bboost") == 15 * 2 + 2
    assert rules.score_week(week(), hits=2) == 11 * 2 + 2 - 8
