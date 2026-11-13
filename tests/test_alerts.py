"""Deadline alerts: windows, the short phone message, and spotting changed advice (no network)."""
import json

import pandas as pd

from fpl import alerts
from test_recommend import hit_advice, tiny_advice

DEADLINE = pd.Timestamp("2026-10-17T10:00:00Z")          # Sat 17 Oct 13:00 in Nairobi


def hours_before(hours):
    return DEADLINE - pd.Timedelta(hours=hours)


def test_next_deadline_is_the_first_one_still_ahead():
    events = [{"id": 7, "deadline_time": "2026-10-03T10:00:00Z"},
              {"id": 9, "deadline_time": "2026-10-24T10:00:00Z"},
              {"id": 8, "deadline_time": "2026-10-17T10:00:00Z"}]
    assert alerts.next_deadline(events, hours_before(30)) == (8, DEADLINE)
    assert alerts.next_deadline(events, DEADLINE) == (9, pd.Timestamp("2026-10-24T10:00:00Z"))
    assert alerts.next_deadline(events, pd.Timestamp("2027-06-01T00:00:00Z")) is None  # season over


def test_alert_windows():
    assert alerts.stage_for(hours_before(25), DEADLINE) is None
    assert alerts.stage_for(hours_before(24.5), DEADLINE) == "early"
    assert alerts.stage_for(hours_before(20.5), DEADLINE) == "early"
    assert alerts.stage_for(hours_before(20), DEADLINE) is None
    assert alerts.stage_for(hours_before(10), DEADLINE) is None
    assert alerts.stage_for(hours_before(3.5), DEADLINE) == "late"
    assert alerts.stage_for(hours_before(1.1), DEADLINE) == "late"
    assert alerts.stage_for(hours_before(1), DEADLINE) is None
    assert alerts.stage_for(hours_before(-1), DEADLINE) is None


def test_decisions_are_the_week_one_calls_only():
    a = tiny_advice([])
    d = alerts.decisions(a)
    assert d == {"moves": [], "hit": 0, "captain": "P14 (AAA)", "vice": "P13 (AAA)", "chip": None}
    assert json.loads(json.dumps(d)) == d                    # saved as JSON between runs
    a.xp.loc[:, 7] = 9.9                                     # new numbers alone are not a change
    assert alerts.decisions(a) == d


def test_decisions_record_transfers_hits_and_chips():
    d = alerts.decisions(hit_advice(2.4))
    assert d["moves"] == ["P3 (AAA) -> InD1 (AAA)", "P4 (AAA) -> InD2 (AAA)", "P8 (AAA) -> InM (AAA)"]
    assert d["hit"] == 8
    assert alerts.decisions(tiny_advice([], chip="bboost"))["chip"] == "Bench Boost"
    a = tiny_advice([], chip="freehit")
    a.plan.weeks[0].squad = [101, 102, 103] + list(range(4, 16))
    free_hit = alerts.decisions(a)
    assert free_hit["chip"] == "Free Hit" and free_hit["moves"] == ["Free Hit squad adds: X, X, X"]


def test_changed_lists_each_difference_in_plain_english():
    old = alerts.decisions(tiny_advice([]))
    assert alerts.changed(old, old) == []
    new = dict(old, captain="P13 (AAA)", vice="P14 (AAA)", moves=["P3 (AAA) -> X (AAA)"], hit=4,
               chip="Triple Captain")
    assert alerts.changed(old, new) == [
        "Transfers: save your free transfer -> P3 (AAA) -> X (AAA)",
        "Hit: none -> -4",
        "Captain: P14 (AAA) -> P13 (AAA)",
        "Vice-captain: P13 (AAA) -> P14 (AAA)",
        "Chip: none -> Triple Captain"]


def test_late_action():
    d = alerts.decisions(tiny_advice([]))
    assert alerts.late_action(None, d) == "full"                       # the early message was missed
    assert alerts.late_action({"stage": "early", "decisions": d}, d) is None
    assert alerts.late_action({"stage": "early", "decisions": dict(d, captain="X")}, d) == "changed"


def test_short_message_when_saving_the_transfer():
    title, body = alerts.short_message(tiny_advice([]))
    assert title == "FPL GW7 advice"
    assert body.splitlines() == [
        "Deadline Sat 17 Oct 13:00 (Nairobi time)",
        "Transfers: save your free transfer",
        "Hit: no",
        "Captain: P14 (AAA) · Vice: P13 (AAA)",
        "Chip: none this week"]


def test_short_message_with_a_hit_and_a_chip():
    a = hit_advice(1.2)
    a.plan.weeks[0].chip = "bboost"
    _, body = alerts.short_message(a)
    assert "Transfers: P3 (AAA) -> InD1 (AAA), P4 (AAA) -> InD2 (AAA), P8 (AAA) -> InM (AAA)" in body
    assert "Hit: take the -8 (about 1.2 pts better than no hit over 4 weeks; a close call)" in body
    assert "Chip: play your Bench Boost this week" in body


def test_short_message_explains_a_skipped_hit():
    a = tiny_advice([])
    a.hit_gain = 0.6
    assert "Hit: no (a hit would gain only about 0.6 pts over 4 weeks)" in alerts.short_message(a)[1]
    a.hit_gain = -2.0
    assert "Hit: no (a hit would lose about 2.0 pts over 4 weeks)" in alerts.short_message(a)[1]


def test_short_message_flags_approximate_plans_and_pending_transfers():
    a = tiny_advice([])
    a.plan.status = "time limit (gap 0.40%)"
    a.state.pending_transfers = 1
    _, body = alerts.short_message(a)
    assert "You've already made 1 transfer(s) this week." in body
    assert body.endswith("(Approximate: the solver hit its time limit.)")


def test_changed_message_leads_with_what_changed():
    title, body = alerts.short_message(tiny_advice([]), changes=["Captain: A -> B"], test=True)
    assert title == "TEST FPL GW7: advice changed"
    assert body.splitlines()[:2] == ["What changed: Captain: A -> B",
                                     "Deadline Sat 17 Oct 13:00 (Nairobi time)"]


def test_a_wildcard_message_stays_short():
    a = tiny_advice([])
    week = a.plan.weeks[0]
    week.chip, week.transfers_out, week.transfers_in = "wildcard", [3, 8], [101, 102]
    a.info.loc[[101, 102], ["name", "position"]] = [["InDEF", "DEF"], ["InMID", "MID"]]
    _, body = alerts.short_message(a)
    assert "Transfers: Out: DEF P3; MID P8 / In: DEF InDEF; MID InMID" in body
    assert "Chip: play your Wildcard this week" in body
    assert len(body.encode("utf-8")) < 4096


def run_alert(monkeypatch, tmp_path, advice_obj, stage, **flags):
    """alert.run with the advice pipeline and ntfy faked; returns the messages 'sent'."""
    import alert
    import recommend
    from fpl import api, storage
    sent = []
    monkeypatch.setattr(storage, "DATA_DIR", tmp_path)
    monkeypatch.setattr(recommend, "run", lambda team, log, live=None: advice_obj)
    monkeypatch.setattr(api, "send_ntfy", lambda topic, title, body, **kw: sent.append((topic, title, body)))
    monkeypatch.setenv("NTFY_TOPIC", "t")
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    alert.run(stage, 1, **flags)
    return sent


def test_early_sends_once_late_only_when_changed(monkeypatch, tmp_path):
    from fpl import storage
    a = tiny_advice([])
    assert [s[1] for s in run_alert(monkeypatch, tmp_path, a, "early")] == ["FPL GW7 advice"]
    assert storage.load_alert(7)["stage"] == "early"
    assert run_alert(monkeypatch, tmp_path, a, "early") == []            # never twice
    assert run_alert(monkeypatch, tmp_path, a, "late") == []             # nothing changed
    assert storage.load_alert(7)["stage"] == "late"                      # ...but the re-check is recorded
    a.plan.weeks[0].captain = 13
    sent = run_alert(monkeypatch, tmp_path, a, "late")
    assert sent[0][1] == "FPL GW7: advice changed"
    assert sent[0][2].startswith("What changed: Captain: P14 (AAA) -> P13 (AAA)")


def test_late_sends_the_full_message_if_the_early_one_was_missed(monkeypatch, tmp_path):
    sent = run_alert(monkeypatch, tmp_path, tiny_advice([]), "late")
    assert [s[1] for s in sent] == ["FPL GW7 advice"]


def test_dry_runs_and_tests_save_nothing(monkeypatch, tmp_path):
    from fpl import storage
    assert run_alert(monkeypatch, tmp_path, tiny_advice([]), "early", dry_run=True) == []
    sent = run_alert(monkeypatch, tmp_path, tiny_advice([]), "early", test=True)
    assert [s[1] for s in sent] == ["TEST FPL GW7 advice"]
    assert storage.load_alert(7) is None


def test_a_wildcard_or_free_hit_week_needs_no_hit():
    a = tiny_advice([], chip="wildcard")
    a.hit_gain = -2.5
    assert "Hit: none needed (Wildcard: unlimited free transfers)" in alerts.short_message(a)[1]


def run_pilot(monkeypatch, tmp_path, advice_obj, carry_out):
    """A late run with autopilot on, FPL faked; returns the messages 'sent'."""
    import alert
    import pandas as pd
    from fpl import autopilot, storage
    monkeypatch.setenv("FPL_AUTOPILOT", "on")
    monkeypatch.setenv("ALERT_NOW", "2026-10-17T08:30:00Z")
    monkeypatch.setattr(alert, "connect", lambda team: (object(), None))
    monkeypatch.setattr(storage, "load_table", lambda name: pd.DataFrame())
    monkeypatch.setattr(autopilot, "carry_out", carry_out)
    return run_alert(monkeypatch, tmp_path, advice_obj, "late")


def test_autopilot_reports_what_it_did(monkeypatch, tmp_path):
    from fpl import autopilot
    a = tiny_advice([])
    a.safe_plan = a.plan
    done = autopilot.Orders(7, [], [], ["A -> B"])
    sent = run_pilot(monkeypatch, tmp_path, a, lambda *args: done)
    assert sent[0][1] == "FPL GW7: team updated" and "Done: A -> B" in sent[0][2]


def test_autopilot_standing_down_says_why(monkeypatch, tmp_path):
    from fpl import autopilot

    def refuse(*args):
        raise autopilot.Blocked("you've already made 1 transfer(s) this week")
    a = tiny_advice([])
    a.safe_plan = a.plan
    sent = run_pilot(monkeypatch, tmp_path, a, refuse)
    assert sent[0][1] == "FPL GW7: autopilot stood down" and "already made 1" in sent[0][2]


def test_a_preferred_hit_is_reported_but_not_planned(monkeypatch, tmp_path):
    from fpl import autopilot
    a, safe = hit_advice(2.4), tiny_advice([]).plan
    a.safe_plan = safe
    seen = []
    sent = run_pilot(monkeypatch, tmp_path, a, lambda _, plan, *rest: seen.append(plan) or
                     autopilot.Orders(7, [], [], []))
    assert seen == [safe] and "The model would rather" in sent[0][2] and "-12" not in sent[0][1]
