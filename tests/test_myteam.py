"""Offline tests for fpl/myteam.py."""
import json
from pathlib import Path

import pandas as pd
import pytest

from fpl import myteam, rules

DATA = Path(__file__).parent / "data"
CHIPS = [{"name": n, "start_event": a, "stop_event": b} for n, a, b in
         [("wildcard", 2, 19), ("wildcard", 20, 38), ("freehit", 2, 19), ("bboost", 1, 19),
          ("3xc", 1, 19), ("freehit", 20, 38), ("bboost", 20, 38), ("3xc", 20, 38)]]
TYPES = [1, 1] + [2] * 5 + [3] * 5 + [4] * 3 + [3, 3]       # elements 1-15 squad, 16-17 extras


def players(now=None):
    now = now or {}
    return pd.DataFrame({"id": range(1, 18), "code": range(1001, 1018),
                         "web_name": [f"P{i}" for i in range(1, 18)],
                         "element_type": TYPES, "team": [i % 9 + 1 for i in range(1, 18)],
                         "now_cost": [now.get(i, 50) for i in range(1, 18)],
                         "status": "a", "news": "", "chance_of_playing_next_round": None})


def picks(elements):
    return {"picks": [{"element": e} for e in elements]}


def late_joiner():
    """Joined in GW3; one transfer in GW5; Free Hit in GW6 (its picks and transfer must be ignored)."""
    history = {"current": [
        {"event": 3, "bank": 0, "event_transfers": 0, "event_transfers_cost": 0},
        {"event": 4, "bank": 0, "event_transfers": 0, "event_transfers_cost": 0},
        {"event": 5, "bank": 0, "event_transfers": 1, "event_transfers_cost": 0},
        {"event": 6, "bank": 0, "event_transfers": 1, "event_transfers_cost": 0}],
        "chips": [{"name": "freehit", "event": 6}]}
    transfers = [
        {"element_in": 16, "element_in_cost": 50, "element_out": 15, "element_out_cost": 50,
         "event": 5, "time": "2026-09-20T10:00:00Z"},
        {"element_in": 17, "element_in_cost": 50, "element_out": 2, "element_out_cost": 50,
         "event": 6, "time": "2026-09-27T10:00:00Z"}]
    by_gw = {3: picks(range(1, 16)), 5: picks([*range(1, 15), 16]), 6: picks([1, 17, *range(3, 15), 16])}
    ph = pd.DataFrame({"element": list(range(1, 18)) * 2, "round": [1] * 17 + [3] * 17,
                       "value": [45] * 17 + [50] * 17})
    return history, transfers, by_gw, ph


def test_late_joiner_free_hit_and_selling_prices():
    history, transfers, by_gw, ph = late_joiner()
    state = myteam.load_team_state(1, {"name": "T"}, history, transfers, by_gw.__getitem__,
                                   players(now={1: 56}), ph, CHIPS)
    assert state.next_gw == 7
    assert sorted(state.squad["element"]) == [*range(1, 15), 16]          # Free Hit squad ignored
    p1 = state.squad.set_index("element").loc[1]
    assert (p1["purchase"], p1["now"], p1["sell"]) == (50, 56, 53)       # GW3 price, not GW1
    assert state.free_transfers == 2       # 1 for GW4, rolled to 2, used 1 -> 2, Free Hit keeps 2
    assert state.chips_played == [("freehit", 6)]
    assert rules.Chip("freehit", 2, 19) not in state.chips and len(state.chips) == 7


def test_pending_transfers_are_applied():
    history, transfers, by_gw, ph = late_joiner()
    transfers.append({"element_in": 17, "element_in_cost": 45, "element_out": 3,
                      "element_out_cost": 50, "event": 7, "time": "2026-10-05T10:00:00Z"})
    state = myteam.load_team_state(1, {"name": "T"}, history, transfers, by_gw.__getitem__,
                                   players(), ph, CHIPS)
    assert 17 in set(state.squad["element"]) and 3 not in set(state.squad["element"])
    assert state.bank == 5
    assert (state.pending_transfers, state.free_transfers, state.pending_hits) == (1, 1, 0)


def test_real_team_5234632():
    folder = DATA / "team_5234632"
    load = lambda name: json.loads((folder / f"{name}.json").read_text())
    by_gw = {int(p.stem.split("_")[1]): json.loads(p.read_text()) for p in folder.glob("picks_*.json")}
    state = myteam.load_team_state(
        5234632, load("entry"), load("history"), load("transfers"), by_gw.__getitem__,
        pd.read_csv(DATA / "players_sample.csv"), pd.read_csv(folder / "player_history.csv"),
        json.loads((DATA / "chips_2026-27.json").read_text()))
    assert (state.next_gw, state.free_transfers, state.bank) == (7, 1, 6)
    assert len(state.squad) == 15 and state.squad["purchase_known"].all()
    assert (state.squad["sell"] <= state.squad[["purchase", "now"]].max(axis=1)).all()
    assert len(state.chips) == 8 and state.chips_played == []
