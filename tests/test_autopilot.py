"""Autopilot: every safety check, the requests it would send, and the read-back (no network)."""
import json
from dataclasses import replace

import pandas as pd
import pytest
import requests

from fpl import api, autopilot, myteam
from fpl.autopilot import Blocked, Failed
from test_recommend import tiny_advice

NOW = pd.Timestamp("2026-10-17T07:00:00Z")


def players_for(a):
    """A players table matching an advice: id = code + 1000, everyone available."""
    return pd.DataFrame({"id": [int(c) + 1000 for c in a.info.index], "code": [int(c) for c in a.info.index],
                         "web_name": a.info["name"].to_numpy(), "now_cost": a.info["price"].to_numpy(),
                         "status": "a", "element_type": a.info["position"].map(
                             {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}).to_numpy(),
                         "team": 1})


def my_team(a, made=0, limit=2, bank=500, chips=()):
    picks = [{"element": int(c) + 1000, "position": i + 1, "selling_price": int(a.info.at[c, "sell"]),
              "purchase_price": int(a.info.at[c, "sell"]), "is_captain": i == 0, "is_vice_captain": i == 1,
              "multiplier": 2 if i == 0 else 1}
             for i, c in enumerate(a.state.codes)]
    return {"picks": picks, "chips": list(chips),
            "transfers": {"limit": limit, "made": made, "bank": bank, "cost": 4, "status": "cost"}}


def planned(paid=0, chip=None):
    """A valid one-transfer week on the tiny example: P8 out, X in, all different clubs."""
    a = tiny_advice([], chip=chip)
    a.info["team"] = [f"C{i:02d}" for i in range(len(a.info))]
    codes = list(range(1, 16))
    a.state = myteam.TeamState(team_id=1, name="T", next_gw=7, squad=pd.DataFrame({"player_code": codes}),
                               bank=6, free_transfers=2, pending_transfers=0, pending_hits=0, chips=[],
                               chips_played=[])
    owned = [101 if c == 8 else c for c in codes]
    week = replace(a.plan.weeks[0], transfers_out=[8], transfers_in=[101], owned=owned, squad=owned, paid=paid,
                   starters=[1, 3, 4, 5, 6, 101, 9, 10, 11, 13, 14])
    a.plan = replace(a.plan, weeks=[week])
    return a


@pytest.fixture
def ready():
    a = planned()
    return a, players_for(a)


def orders(a, players, mine=None, now=NOW):
    return autopilot.build_orders(a, a.plan, mine or my_team(a), players, now)


def test_good_plan_becomes_one_transfer_and_a_full_lineup(ready):
    a, players = ready
    built = orders(a, players)
    assert len(built.transfers) == 1 and len(built.picks) == 15
    item = built.transfers[0]
    assert set(item) == {"element_in", "element_out", "purchase_price", "selling_price"}
    assert [p["position"] for p in built.picks] == list(range(1, 16))
    assert sum(p["is_captain"] for p in built.picks) == 1 and sum(p["is_vice_captain"] for p in built.picks) == 1
    assert [p["multiplier"] for p in built.picks].count(0) == 4
    assert json.loads(json.dumps(built.transfers)) == built.transfers


def test_hits_and_chips_never_pass():
    a = planned(paid=1)
    with pytest.raises(Blocked, match="chip or a hit"):
        orders(a, players_for(a))


def test_chip_plan_never_passes():
    a = planned(chip="bboost")
    with pytest.raises(Blocked, match="chip or a hit"):
        orders(a, players_for(a))


def test_full_policy_allows_optimizer_hit_and_chip(monkeypatch):
    monkeypatch.setenv("FPL_AUTOPILOT_POLICY", "all_planned")
    assert orders(planned(paid=1), players_for(planned(paid=1))).transfers
    assert orders(planned(chip="bboost"), players_for(planned(chip="bboost"))).picks


def test_too_close_to_the_deadline(ready):
    a, players = ready
    with pytest.raises(Blocked, match="minutes to the deadline"):
        orders(a, players, now=a.deadline - pd.Timedelta(minutes=10))


def test_your_own_transfers_are_left_alone(ready):
    a, players = ready
    with pytest.raises(Blocked, match="already made"):
        orders(a, players, my_team(a, made=1))


def test_active_chip_is_left_alone(ready):
    a, players = ready
    mine = my_team(a, chips=[{"name": "wildcard", "status_for_entry": "active"}])
    with pytest.raises(Blocked, match="wildcard is active"):
        orders(a, players, mine)


def test_no_free_transfer_means_no_transfer(ready):
    a, players = ready
    with pytest.raises(Blocked, match="more than the free transfers"):
        orders(a, players, my_team(a, limit=0))


def test_unavailable_signing_is_refused(ready):
    a, players = ready
    players.loc[players["code"] == 101, "status"] = "d"
    with pytest.raises(Blocked, match="isn't fully available"):
        orders(a, players)


def test_cannot_overspend(ready):
    a, players = ready
    players.loc[players["code"] == 101, "now_cost"] = 900
    with pytest.raises(Blocked, match="more than you have"):
        orders(a, players, my_team(a, bank=0))


def test_squad_changed_since_planning(ready):
    a, players = ready
    mine = my_team(a)
    mine["picks"][0]["element"] = 999999
    with pytest.raises(Blocked, match="squad changed"):
        orders(a, players, mine)


def test_illegal_formation_is_refused(ready):
    a, players = ready
    week = a.plan.weeks[0]
    no_forward = [c for c in week.starters if c != 13 and c != 14] + [2, 7]    # two keepers, no forwards
    a.plan = replace(a.plan, weeks=[replace(week, starters=no_forward, bench=[12, 13, 14, 15],
                                            captain=9, vice=10)])
    with pytest.raises(Blocked, match="formation"):
        orders(a, players)


def test_too_many_from_one_club_is_refused(ready):
    a, players = ready
    a.info.loc[[101, 1, 2, 3], "team"] = "SAME"
    with pytest.raises(Blocked, match="too many players"):
        orders(a, players)


class FakeSession:
    def __init__(self, replies):
        self.team_id, self.access_token, self.replies = 1, "tok", list(replies)

    def my_team(self):
        return self.replies.pop(0)


def landed(built, mine):
    done = dict(mine)
    done["picks"] = built.picks
    return done


@pytest.fixture
def posts(monkeypatch):
    sent = []
    monkeypatch.setattr(api, "post_transfers", lambda payload, token: sent.append(("transfers", payload)))
    monkeypatch.setattr(api, "post_picks", lambda team, payload, token: sent.append(("picks", payload)))
    return sent


def test_dry_run_sends_nothing(ready, posts):
    a, players = ready
    autopilot.carry_out(a, a.plan, FakeSession([my_team(a)]), players, NOW, dry_run=True)
    assert posts == []


def test_transfers_go_first_then_picks_then_read_back(ready, posts):
    a, players = ready
    mine = my_team(a)
    built = orders(a, players, mine)
    autopilot.carry_out(a, a.plan, FakeSession([mine, landed(built, mine)]), players, NOW)
    assert [name for name, _ in posts] == ["transfers", "picks"]
    assert posts[0][1]["chip"] is None and posts[0][1]["confirmed"] is True and posts[0][1]["event"] == built.gw


def test_full_policy_sends_chip_to_both_fpl_requests(monkeypatch, posts):
    monkeypatch.setenv("FPL_AUTOPILOT_POLICY", "all_planned")
    a = planned(chip="bboost")
    players = players_for(a)
    mine = my_team(a)
    built = orders(a, players, mine)
    autopilot.carry_out(a, a.plan, FakeSession([mine, landed(built, mine)]), players, NOW)
    assert posts[0][0] == "transfers" and posts[0][1]["chip"] is None
    assert posts[1][0] == "picks" and posts[1][1]["chip"] == "bboost"


def test_read_back_mismatch_is_a_failure(ready, posts):
    a, players = ready
    mine = my_team(a)
    with pytest.raises(Failed, match="expected the squad"):
        autopilot.carry_out(a, a.plan, FakeSession([mine, mine]), players, NOW)


def test_lost_reply_is_judged_by_the_read_back(ready, monkeypatch):
    a, players = ready
    mine = my_team(a)
    built = orders(a, players, mine)

    def boom(*_):
        raise requests.ConnectionError("no reply")
    monkeypatch.setattr(api, "post_transfers", boom)
    monkeypatch.setattr(api, "post_picks", lambda *_: None)
    autopilot.carry_out(a, a.plan, FakeSession([mine, landed(built, mine)]), players, NOW)


def test_refresh_token_from_browser_json_or_plain():
    assert autopilot.parse_refresh_token('{"access_token": "a", "refresh_token": "r"}') == "r"
    assert autopilot.parse_refresh_token("  plain \n") == "plain"
    with pytest.raises(Failed):
        autopilot.parse_refresh_token('{"access_token": "a"}')


def test_new_token_is_stored_and_never_in_errors(monkeypatch):
    stored = []
    monkeypatch.setenv("FPL_REFRESH_TOKEN", "old-secret")
    monkeypatch.setattr(api, "refresh_access_token", lambda t: {"access_token": "a", "refresh_token": "new-secret"})
    session = autopilot.Session.connect(7, store=stored.append)
    assert session.access_token == "a" and stored == ["new-secret"]

    def refused(_):
        raise requests.HTTPError(response=type("R", (), {"status_code": 403})())
    monkeypatch.setattr(api, "refresh_access_token", refused)
    with pytest.raises(Failed) as caught:
        autopilot.Session.connect(7, store=stored.append)
    assert "403" in str(caught.value) and "secret" not in str(caught.value)


def test_autopilot_is_off_by_default(monkeypatch):
    monkeypatch.delenv("FPL_AUTOPILOT", raising=False)
    assert not autopilot.enabled()
    monkeypatch.setenv("FPL_AUTOPILOT", "on")
    assert autopilot.enabled()


def test_live_state_uses_the_account_numbers(ready):
    a, players = ready
    mine = my_team(a, made=0, limit=1, bank=37)
    mine["picks"][0]["selling_price"] = 61
    state = autopilot.live_state(a.state, mine, players)
    assert state.bank == 37 and state.free_transfers == 1 and state.squad["sell"].iloc[0] == 61
    assert list(state.squad["position"][:2]) == ["GK", "GK"]
    del mine["transfers"]
    with pytest.raises(Failed, match="changed shape"):
        autopilot.live_state(a.state, mine, players)
