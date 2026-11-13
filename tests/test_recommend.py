"""End to end on a saved copy of team 5234632: state -> plan -> printed advice (no network, no training)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import recommend
from fpl import advice, myteam
from fpl.settings import PlanSettings

DATA = Path(__file__).parent / "data"


def saved_state():
    folder = DATA / "team_5234632"
    load = lambda name: json.loads((folder / f"{name}.json").read_text())
    by_gw = {int(p.stem.split("_")[1]): json.loads(p.read_text()) for p in folder.glob("picks_*.json")}
    players = pd.read_csv(DATA / "players_sample.csv")
    state = myteam.load_team_state(5234632, load("entry"), load("history"), load("transfers"),
                                   by_gw.__getitem__, players, pd.read_csv(folder / "player_history.csv"),
                                   json.loads((DATA / "chips_2026-27.json").read_text()))
    return state, players


def test_local_time_is_nairobi():
    assert advice.local_time(pd.Timestamp("2026-10-17T10:00:00Z")) == "Sat 17 Oct 13:00"
    assert advice.money(6) == "£0.6m" and advice.money(105) == "£10.5m"


def test_advice_end_to_end_on_the_saved_team():
    state, players = saved_state()
    teams = pd.DataFrame({"id": range(1, 21), "name": [f"Club {i}" for i in range(1, 21)],
                          "short_name": [f"C{i:02d}" for i in range(1, 21)]})
    gws = list(range(7, 20))
    rng = np.random.default_rng(0)
    xp = pd.DataFrame(rng.uniform(1, 6, (len(players), len(gws))), index=players["code"], columns=gws)
    loaned = players.loc[players["status"] == "u", "code"]
    xp = xp.drop(index=loaned)                                   # no predictions, like McNally
    avail = pd.DataFrame({"player_code": [], "gw": [], "availability": [], "news": []})
    fixtures = pd.DataFrame({"season": "2026-27", "gw": np.repeat(gws, 20), "team": [f"Club {i}" for i in range(1, 21)] * len(gws),
                             "fixture_id": range(20 * len(gws))})
    result = recommend.build_advice(state, players, teams, fixtures, xp, avail,
                                    pd.Timestamp("2026-10-17T10:00:00Z"),
                                    PlanSettings(horizon=4), time_limit=30)
    text = advice.render(result)
    for section in ["Sat 17 Oct 13:00", "THIS WEEK", "Captain:", "Bench (in order):",
                    "NEXT WEEKS", "CHIPS", "£0.6m"]:
        assert section in text, section
    for code in loaned:
        if code in state.codes:
            assert code not in result.plan.weeks[0].starters
    table = advice.plan_table(result)
    assert set(table["gw"]) == {7, 8, 9, 10}



def test_advice_after_a_club_move_leaves_four_at_one_club():
    """January-style move: a 4th squad player now plays for the same club. Advice must still work
    (FPL lets you keep 4) and must never buy from that club unless it ends up with 3 or fewer."""
    state, players = saved_state()
    squad_rows = players["code"].isin(state.codes)
    club = players.loc[squad_rows, "team"].value_counts().index[0]
    others = players.loc[squad_rows & (players["team"] != club), "code"]
    need = 4 - int((players.loc[squad_rows, "team"] == club).sum())
    players.loc[players["code"].isin(others.iloc[:need]), "team"] = club
    teams = pd.DataFrame({"id": range(1, 21), "name": [f"Club {i}" for i in range(1, 21)],
                          "short_name": [f"C{i:02d}" for i in range(1, 21)]})
    gws = list(range(7, 20))
    rng = np.random.default_rng(1)
    xp = pd.DataFrame(rng.uniform(1, 6, (len(players), len(gws))), index=players["code"], columns=gws)
    xp.loc[players.loc[players["team"] == club, "code"]] += 3.0          # tempt buys from that club
    avail = pd.DataFrame({"player_code": [], "gw": [], "availability": [], "news": []})
    fixtures = pd.DataFrame({"season": "2026-27", "gw": np.repeat(gws, 20),
                             "team": [f"Club {i}" for i in range(1, 21)] * len(gws),
                             "fixture_id": range(20 * len(gws))})
    result = recommend.build_advice(state, players, teams, fixtures, xp, avail,
                                    pd.Timestamp("2026-10-17T10:00:00Z"),
                                    PlanSettings(horizon=4), time_limit=60)
    short = f"C{int(club):02d}"
    for week in result.plan.weeks:
        count = int((result.info.loc[week.owned, "team"] == short).sum())
        bought_there = any(result.info.loc[week.transfers_in, "team"] == short)
        assert count <= (3 if bought_there else 4), (week.gw, count, bought_there)


def tiny_advice(avail_rows, chip=None, sold=()):
    """A hand-made one-week Advice: 15 players 1-15, players 1 and 2 are the ones with news."""
    from types import SimpleNamespace
    from fpl import optimizer, rules
    codes = list(range(1, 16))
    positions = ["GK"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    info = pd.DataFrame({"name": [f"P{c}" for c in codes], "position": positions,
                         "team": "AAA", "price": 50, "sell": 50}, index=pd.Index(codes, name="player_code"))
    starters = [1, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14]
    bench = [2, 7, 12, 15]
    owned = codes if not sold else [c for c in codes if c not in sold] + [99]
    week = optimizer.WeekPlan(gw=7, chip=chip, transfers_in=[], transfers_out=list(sold), free_transfers=1,
                              paid=0, owned=owned, squad=codes if chip != "freehit" else [c + 100 for c in range(3)] + codes[3:],
                              starters=starters, bench=bench, captain=14, vice=13, xp=50.0, bank=6)
    state = SimpleNamespace(name="T", bank=6, free_transfers=1, pending_transfers=0, pending_hits=0,
                            chips=[rules.Chip("freehit", 1, 19)], codes=codes)
    xp = pd.DataFrame(3.0, index=codes + [101, 102, 103, 99], columns=[7])
    info = info.reindex(info.index.union([101, 102, 103, 99]))
    info.loc[[101, 102, 103, 99]] = ["X", "MID", "AAA", 50, 50]
    avail = pd.DataFrame(avail_rows, columns=["player_code", "gw", "availability", "news"])
    return advice.Advice(state=state, plan=optimizer.Plan([week], 50.0, "Optimal"), hit_gain=None, info=info,
                         xp=xp, avail=avail, scan=pd.DataFrame(columns=["key", "name", "gw", "value"]),
                         counts=pd.DataFrame(columns=["team", "gw", "fixtures"]),
                         deadline=pd.Timestamp("2026-10-17T10:00:00Z"), horizon=4)


def test_injury_watch_lists_only_doubtful_players_and_never_prints_nan():
    rows = [(1, 7, 1.0, float("nan")), (2, 7, 0.5, float("nan")), (3, 7, 0.0, "Out for the season")]
    text = advice.render(tiny_advice(rows))
    assert "INJURY WATCH" in text and "nan" not in text
    assert "P1 (AAA)" not in text.split("INJURY WATCH")[1]       # 100% every week: not listed
    assert "P2 (AAA): Chance available: GW7 50%" in text
    assert "P3 (AAA): Out for the season. Chance available: GW7 0%" in text
    assert "P3 (AAA) won't play in the plan: replace him when you have a spare free transfer." in text
    assert "P2 (AAA) won't play" not in text


def test_sold_player_gets_no_wont_play_line():
    rows = [(3, 7, 0.0, "Out")]
    text = advice.render(tiny_advice(rows, sold=[3]))
    assert "won't play" not in text


def test_free_hit_wording_counts_changed_players():
    text = advice.render(tiny_advice([], chip="freehit"))
    assert "Play your Free Hit this week (3 players changed for this week only)." in text


def test_a_time_limited_plan_says_how_far_from_the_best_it_could_be():
    a = tiny_advice([])
    a.plan.status, a.plan.gap = "time limit (gap 0.40%)", 0.004
    assert "could score up to 0.4% more" in advice.render(a)
    assert "time limit" not in advice.render(tiny_advice([]))


def test_the_comparison_solve_wins_when_it_finds_a_better_plan():
    from types import SimpleNamespace
    week = lambda paid: SimpleNamespace(paid=paid)
    hit = SimpleNamespace(weeks=[week(1)], objective=100.0)
    no_hit = SimpleNamespace(weeks=[week(0)], objective=101.5)
    flat = PlanSettings(hit_margin=0.0)
    plan, other, gain, better = recommend.pick_plan(hit, no_hit, flat)  # main plan took a hit, the check beat it
    assert plan is no_hit and other is hit and better and gain == pytest.approx(-1.5)
    plan, other, gain, better = recommend.pick_plan(no_hit, hit, flat)  # main plan kept, hit not worth it
    assert plan is no_hit and other is hit and not better and gain == pytest.approx(-1.5)
    assert recommend.pick_plan(hit, None, flat) == (hit, None, None, False)
    a = tiny_advice([])
    a.comparison_better, a.hit_gain = True, -1.5
    text = advice.render(a)
    assert "found this plan" in text
    assert ("No hit: the best plan with a hit this week is expected to score about 1.5 pts less "
            "over 4 weeks, after paying for the hit.") in text
    a.hit_gain = -0.01                                           # a tie: no "-0.0"
    assert ("No hit: the best plan with a hit this week is expected to score about the same as this one "
            "after paying for the hit, so it isn't worth the risk.") in advice.render(a)
    a.hit_gain = 0.6
    assert ("No hit: the best plan with a hit this week is expected to score only about 0.6 pts more "
            "over 4 weeks after paying for the hit: too little to be worth the risk.") in advice.render(a)
    assert "+0" not in advice.render(a)


def test_hit_gain_is_points_after_the_hits_but_before_the_safety_margin():
    """The objective charges 4 + hit_margin per hit; the advice reports the gain after the 4 only."""
    from types import SimpleNamespace
    week = lambda paid: SimpleNamespace(paid=paid)
    hit = SimpleNamespace(weeks=[week(2), week(0), week(1)], objective=99.7)
    no_hit = SimpleNamespace(weeks=[week(0), week(0), week(0)], objective=100.0)
    settings = PlanSettings(discount=0.85, hit_margin=1.0)
    plan, other, gain, better = recommend.pick_plan(no_hit, hit, settings)
    assert plan is no_hit and gain == pytest.approx(-0.3 + 2 + 0.85 ** 2)


def hit_advice(gain, alt_moves=((3, 101),), alt_status="Optimal"):
    """tiny_advice whose week 1 takes a -8 hit; the best no-hit plan makes alt_moves."""
    import copy
    a = tiny_advice([])
    week = a.plan.weeks[0]
    week.paid, week.transfers_out, week.transfers_in = 2, [3, 4, 8], [101, 102, 103]
    a.info.loc[[101, 102, 103], ["name", "position"]] = [["InD1", "DEF"], ["InD2", "DEF"], ["InM", "MID"]]
    alt = copy.deepcopy(a.plan)
    alt.status = alt_status
    alt.weeks[0].paid = 0
    alt.weeks[0].transfers_out = [o for o, _ in alt_moves]
    alt.weeks[0].transfers_in = [i for _, i in alt_moves]
    a.hit_gain, a.alternative = gain, alt
    return a


def test_a_hit_is_explained_in_expected_points_after_paying_for_it():
    text = advice.render(hit_advice(2.4))
    assert ("Taking a -8 hit: over 4 weeks this plan is expected to score about 2.4 pts more than "
            "the best plan without a hit, after paying the 8.") in text
    assert "close call" not in text and "without a hit, this week" not in text


def test_a_small_hit_gain_is_a_close_call_and_shows_the_no_hit_moves():
    text = advice.render(hit_advice(1.2))
    assert "about 1.2 pts more" in text and "close call" in text
    assert "Best plan without a hit, this week: P3 -> InD1" in text
    text = advice.render(hit_advice(1.2, alt_moves=()))
    assert "Best plan without a hit, this week: save your free transfer" in text


def test_an_unproven_comparison_is_called_approximate():
    assert "approximate" not in advice.render(hit_advice(2.4))
    assert "approximate" in advice.render(hit_advice(2.4, alt_status="time limit (gap 0.40%)"))


def test_transfers_are_paired_within_a_position():
    a = tiny_advice([])
    a.info.loc[[101, 102], ["name", "position"]] = [["InMID", "MID"], ["InDEF", "DEF"]]
    week = a.plan.weeks[0]
    week.transfers_out, week.transfers_in = [3, 8], [101, 102]   # out: DEF, MID; in: MID, DEF
    text = "\n".join(advice._transfer_lines(a, week))
    assert text.index("Sell P3") < text.index("buy InDEF") < text.index("Sell P8") < text.index("buy InMID")
    assert advice._week_summary(a, week) == "P3 -> InDEF, P8 -> InMID"


def test_wildcard_week_prints_grouped_out_and_in_lines():
    a = tiny_advice([])
    a.info.loc[[101, 102], ["name", "position"]] = [["InMID", "MID"], ["InDEF", "DEF"]]
    week = a.plan.weeks[0]
    week.chip, week.transfers_out, week.transfers_in = "wildcard", [3, 8], [101, 102]
    lines = advice._transfer_lines(a, week)
    assert lines == ["  Out: DEF P3; MID P8", "  In : DEF InDEF; MID InMID"]
    assert "Out: DEF P3; MID P8" in advice._week_summary(a, week)


def test_wont_play_line_is_absent_when_the_plan_sells_him_later():
    import copy
    a = tiny_advice([(3, 7, 0.0, "Out"), (3, 8, 0.0, "Out")])
    later = copy.deepcopy(a.plan.weeks[0])
    later.gw, later.transfers_out, later.transfers_in = 8, [3], [101]
    a.plan.weeks.append(later)
    a.xp[8] = 3.0
    text = advice.render(a)
    assert "INJURY WATCH" in text and "won't play" not in text


def test_triple_captain_is_folded_into_the_captain_line():
    a = tiny_advice([], chip="3xc")
    text = advice.render(a)
    assert "Captain (Triple Captain): P14 (AAA) 3.0 x3" in text
    assert "Play your Triple Captain" not in text


def test_chip_weeks_without_transfers_still_say_save_your_free_transfer():
    for chip in ("bboost", "3xc"):
        text = advice.render(tiny_advice([], chip=chip))
        assert "Save your free transfer" in text, chip
    text = advice.render(tiny_advice([], chip="bboost"))
    this_week = text.split("THIS WEEK")[1]
    assert this_week.index("Play your Bench Boost this week.") < this_week.index("Save your free transfer")


def test_more_transfers_made_than_free_warns_about_an_invisible_chip():
    a = tiny_advice([])
    a.state.pending_transfers, a.state.pending_hits, a.state.free_transfers = 4, 12, 0
    text = advice.render(a)
    assert ("You've made 4 transfers this week; if that's because you activated a Wildcard or "
            "Free Hit, the plan doesn't know — it assumes no chip is active.") in text
    a.state.pending_transfers, a.state.pending_hits = 1, 0
    assert "activated a Wildcard" not in advice.render(a)


def test_a_solver_time_out_is_explained_plainly():
    from fpl import optimizer
    text = recommend.plan_error_message(optimizer.SolverTimeout(
        "no plan found within 180 s; try again or raise the time limit"))
    assert text == ("Couldn't make a plan: no plan found within 180 s; try again or raise the time limit "
                    "(SOLVER_TIME_LIMIT in fpl/settings.py).")
    assert "check the budget" not in text
    assert (recommend.plan_error_message(ValueError("no valid plan: check the budget and squad"))
            == "Couldn't make a plan: no valid plan: check the budget and squad")


def test_a_hit_with_no_net_gain_is_described_as_about_the_same():
    """The planner can take a hit that only wins on later-week hits saved: never print a negative gain."""
    from types import SimpleNamespace
    week = SimpleNamespace(paid=1)
    a = SimpleNamespace(hit_gain=-0.3, horizon=6, alternative=None,
                        plan=SimpleNamespace(status="Optimal"))
    lines = advice._hit_lines(a, week)
    assert "about the same as" in lines[0] and "-0.3" not in lines[0]
    assert any("close call" in line for line in lines)


def test_safe_plan_makes_the_best_moves_within_the_weekly_cap():
    state, players = saved_state()
    state.free_transfers = 5
    teams = pd.DataFrame({"id": range(1, 21), "name": [f"Club {i}" for i in range(1, 21)],
                          "short_name": [f"C{i:02d}" for i in range(1, 21)]})
    gws = list(range(7, 20))
    rng = np.random.default_rng(0)
    xp = pd.DataFrame(rng.uniform(1, 6, (len(players), len(gws))), index=players["code"], columns=gws)
    xp = xp.drop(index=players.loc[players["status"] == "u", "code"])
    xp.loc[xp.index.difference(state.codes)] += 3.0              # tempt a big reshuffle
    avail = pd.DataFrame({"player_code": [], "gw": [], "availability": [], "news": []})
    fixtures = pd.DataFrame({"season": "2026-27", "gw": np.repeat(gws, 20),
                             "team": [f"Club {i}" for i in range(1, 21)] * len(gws),
                             "fixture_id": range(20 * len(gws))})
    result = recommend.build_advice(state, players, teams, fixtures, xp, avail,
                                    pd.Timestamp("2026-10-17T10:00:00Z"),
                                    PlanSettings(horizon=4), time_limit=60, safe=True)
    assert len(result.plan.weeks[0].transfers_in) > 2
    week = result.safe_plan.weeks[0]
    assert 0 < len(week.transfers_in) <= 2 and week.paid == 0 and week.chip is None
