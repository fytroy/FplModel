"""Offline tests for fpl/simulate.py: no future data, and a short made-up season runs."""
import numpy as np
import pandas as pd
import pytest

from conftest import make_fixtures, make_matches
from fpl import rules, simulate
from fpl.settings import PlanSettings


def test_snapshot_for_picks_the_newest_before_the_deadline():
    snaps = [(pd.Timestamp("2025-08-01", tz="UTC"), "a"), (pd.Timestamp("2025-09-01", tz="UTC"), "b")]
    assert simulate.snapshot_for(pd.Timestamp("2025-08-20", tz="UTC"), snaps) == "a"
    assert simulate.snapshot_for(pd.Timestamp("2025-09-02", tz="UTC"), snaps) == "b"
    with pytest.raises(ValueError):
        simulate.snapshot_for(pd.Timestamp("2025-07-01", tz="UTC"), snaps)


def test_build_predictions_only_sees_the_past():
    history = make_matches(gws=6)
    fixtures_all = make_fixtures(history)
    season = "2025-26"
    f = fixtures_all[fixtures_all.season == season]
    deadlines = (f.groupby("gw")["kickoff_time"].min() - pd.Timedelta(hours=2)).to_dict()
    raw = pd.DataFrame({"id": f.fixture_id, "event": f.gw, "kickoff_time": f.kickoff_time.astype(str),
                        "finished": False, "team_h": 0, "team_a": 0, "team_h_score": None,
                        "team_a_score": None}).drop_duplicates("id")
    seen = []

    def fake_train(past, player_rows, team_rows, cutoffs, yardstick=True):
        seen.append((past["kickoff_time"].max(), player_rows["asof"].min()))
        return player_rows.assign(xp=1.0)

    teams = pd.DataFrame({"id": [0], "name": ["Alpha"]})
    simulate.build_predictions(history, fixtures_all, [(pd.Timestamp("2025-08-01", tz="UTC"), raw)],
                               teams, season, deadlines, gws=[3, 4], weeks=2, train=fake_train,
                               log=lambda m: None)
    assert seen and all(latest < asof for latest, asof in seen)


def made_up_season(gws=4, seed=0):
    """8 clubs × (1 GK, 3 DEF, 3 MID, 2 FWD); random prices, xP and real points."""
    rng = np.random.default_rng(seed)
    rows, code = [], 1
    for club in range(8):
        for position, count in [("GK", 1), ("DEF", 3), ("MID", 3), ("FWD", 2)]:
            for _ in range(count):
                rows.append({"player_code": code, "player_name": f"P{code}", "position": position,
                             "team": f"C{club}", "price": int(rng.integers(40, 80))})
                code += 1
    base = pd.DataFrame(rows)
    players = pd.concat([base.assign(gw=g) for g in range(1, gws + 1)], ignore_index=True)
    actual = players[["gw", "player_code"]].assign(points=rng.integers(0, 10, len(players)),
                                                   minutes=rng.choice([0, 90], len(players), p=[0.1, 0.9]))
    xp = pd.concat([base[["player_code"]].assign(origin_gw=o, gw=g, xp=rng.uniform(1, 6, len(base)))
                    for o in range(1, gws + 1) for g in range(o, gws + 1)], ignore_index=True)
    chips = [rules.Chip("bboost", 1, 19), rules.Chip("3xc", 1, 19)]
    deadlines = {g: pd.Timestamp("2025-08-15", tz="UTC") + pd.Timedelta(days=7 * (g - 1)) for g in range(1, gws + 1)}
    data = simulate.SimData(season="2025-26", gws=list(range(1, gws + 1)), deadlines=deadlines,
                            chips=chips, players=players, actual=actual)
    return data, xp


def test_a_short_made_up_season_runs_and_adds_up():
    data, xp = made_up_season()
    weeks = simulate.run_season(data, xp, PlanSettings(horizon=2), time_limit=20, log=lambda m: None)
    assert list(weeks["gw"]) == [1, 2, 3, 4]
    assert weeks["bank"].ge(0).all()
    assert weeks["chip"].isin(["", "bboost", "3xc"]).all()
    summary = simulate.summarize(weeks, pd.DataFrame({"gw": [1, 2, 3, 4], "average": [50] * 4}),
                                 pd.DataFrame({"points": [100, 200], "rank": [1000, 10]}), 2061)
    assert summary["total"] == weeks["points"].sum()
    assert summary["average_total"] == 200
    assert not summary["full_season"]                     # GW1-4 only: no rank, no 2,061 comparison
    assert summary["rank"] is None and summary["vs_user"] is None and "partial" in summary["rank_note"]


def test_a_season_survives_a_mid_season_club_move(monkeypatch):
    """Every C1 player joins C0 from GW3, so a squad holding 3 of each ends up with 6 at C0."""
    data, xp = made_up_season()
    moved = (data.players["team"] == "C1") & (data.players["gw"] >= 3)
    data.players.loc[moved, "team"] = "C0"
    stars = data.players.loc[data.players["team"].isin(["C0", "C1"]), "player_code"].unique()
    xp.loc[xp["player_code"].isin(stars), "xp"] = 10.0                # so GW1 picks 3 from each club
    most_at_one_club = []
    real_solve = simulate.optimizer.solve

    def spy(inp, *args, **kwargs):
        if inp.squad:
            most_at_one_club.append(int(inp.pool.loc[inp.squad, "team"].value_counts().max()))
        return real_solve(inp, *args, **kwargs)

    monkeypatch.setattr(simulate.optimizer, "solve", spy)
    weeks = simulate.run_season(data, xp, PlanSettings(horizon=2), time_limit=20, log=lambda m: None)
    assert list(weeks["gw"]) == [1, 2, 3, 4]
    assert max(most_at_one_club) > rules.MAX_PER_CLUB                # the move really left 4+ at one club


def test_summarize_ranks_only_a_full_season():
    weeks = pd.DataFrame({"gw": range(1, 39), "points": 55, "hits": 0, "transfers": 1,
                          "chip": "", "chip_gain": None, "captain_best": True})
    summary = simulate.summarize(weeks, pd.DataFrame({"gw": range(1, 39), "average": 50}),
                                 pd.DataFrame({"points": [1800, 2200], "rank": [5_000_000, 100_000]}), 2061)
    assert summary["full_season"] and summary["vs_user"] == 55 * 38 - 2061 and summary["rank"] > 0


def test_chip_weeks_names_each_copy_and_flags_the_unplayed():
    windows = [rules.Chip("wildcard", 2, 19), rules.Chip("wildcard", 20, 38),
               rules.Chip("bboost", 2, 19), rules.Chip("bboost", 20, 38)]
    weeks = pd.DataFrame({"gw": [5, 19, 25, 30],
                          "chip": ["wildcard", "bboost", "wildcard", ""]})
    played = simulate.chip_weeks(weeks, windows)
    assert played == {"wildcard_2": 5, "wildcard_20": 25, "bboost_2": 19, "bboost_20": None}
    text = simulate.describe_chip_weeks(played, windows)
    assert "Wildcard GW5" in text and "Wildcard GW25" in text
    assert "window ends GW19" in text and "window ends GW38" in text
    assert "NEVER PLAYED" in text and "Bench Boost" in text
    assert text.count("NEVER PLAYED") == 1


def test_summarize_includes_a_column_per_chip_copy():
    data, xp = made_up_season()
    weeks = simulate.run_season(data, xp, PlanSettings(horizon=2), time_limit=20, log=lambda m: None)
    summary = simulate.summarize(weeks, pd.DataFrame({"gw": [1, 2, 3, 4], "average": [50] * 4}),
                                 pd.DataFrame({"points": [100, 200], "rank": [1000, 10]}), 2061,
                                 chips=data.chips)
    assert set(summary) >= {"bboost_1", "3xc_1"}
    played = weeks.loc[weeks["chip"] != "", "gw"].tolist()
    assert sorted(v for k, v in summary.items() if k in ("bboost_1", "3xc_1") and v is not None) == played


def test_report_is_rebuilt_from_saved_run_files(tmp_path, monkeypatch):
    from fpl import storage
    monkeypatch.setattr(storage, "DATA_DIR", tmp_path)
    data, xp = made_up_season()
    weeks = simulate.run_season(data, xp, PlanSettings(horizon=2), time_limit=20, log=lambda m: None)
    storage.save_run("default", weeks)
    storage.save_run("default, final fixtures", weeks)
    average = pd.DataFrame({"gw": [1, 2, 3, 4], "average": [50] * 4})
    sample = pd.DataFrame({"points": [100, 200], "rank": [1000, 10]})
    order = ["default", "hit margin 0", "default, final fixtures"]
    report, _ = simulate.build_report(storage.load_runs(), order, average, sample, 2061, data.chips)
    assert list(report["run"]) == ["default", "default, final fixtures"]
    storage.save_run("hit margin 0", weeks)                 # a later run adds to the earlier ones
    report, stacked = simulate.build_report(storage.load_runs(), order, average, sample, 2061, data.chips)
    assert list(report["run"]) == ["default", "hit margin 0", "default, final fixtures"]
    assert set(stacked["run"]) == set(report["run"]) and len(stacked) == 12
    assert report["total"].eq(weeks["points"].sum()).all()


def test_season_players_uses_the_price_at_the_gameweeks_first_match():
    """In a double gameweek the price can rise between the two matches; the deadline price is the first."""
    kick = pd.to_datetime(["2025-08-16 14:00", "2025-08-20 19:00", "2025-08-23 14:00"], utc=True)
    h = pd.DataFrame({"season": "2025-26", "gw": [1, 1, 2], "kickoff_time": kick, "player_code": 7,
                      "player_name": "P7", "position": "MID", "team": "Alpha", "price": [6.0, 6.1, 6.2]})
    players = simulate.season_players(h, "2025-26").set_index("gw")
    assert players.loc[1, "price"] == 60 and players.loc[2, "price"] == 62
