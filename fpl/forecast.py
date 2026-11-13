"""Live predictions for the coming gameweeks, with injury news applied.

Shared by predict.py (ranked table) and recommend.py (planner).
"""
import pandas as pd

from fpl import availability, matches, models
from fpl.seasons import CURRENT_SEASON


def deadlines_from(gameweeks: pd.DataFrame) -> dict[int, pd.Timestamp]:
    """FPL's deadline per gameweek (UTC), from the gameweeks table."""
    return {int(i): pd.Timestamp(t) for i, t in zip(gameweeks["id"], gameweeks["deadline_time"])}


def week_ends_from(fixtures_all: pd.DataFrame, season: str,
                   deadlines: dict | None = None) -> dict[int, pd.Timestamp]:
    """Last kickoff of each gameweek in a season: when the gameweek ends.

    A gameweek with no fixtures (all postponed) ends at its deadline, if given.
    """
    last = fixtures_all[fixtures_all["season"] == season].groupby("gw")["kickoff_time"].max()
    return {**(deadlines or {}), **{int(g): pd.Timestamp(t) for g, t in last.items()}}


def live_forecast(history, fixtures_all, players, teams, gameweeks, gws, season=CURRENT_SEASON):
    """(per-fixture predictions with availability applied, availability table).

    Flagged players are judged on their form before their absence; their
    availability for each week comes from FPL's news (availability.py).
    """
    flagged = players.loc[players["status"] != "a", "code"].astype(int).tolist()
    avail = availability.weekly_availability(
        players, gws, week_ends_from(fixtures_all, season, deadlines_from(gameweeks)),
        availability.recovery_curve(history),
        availability.matches_missed(history, flagged))
    cutoffs = matches.gw_cutoffs(fixtures_all)
    asof = cutoffs[(cutoffs["season"] == season) & (cutoffs["gw"] == gws[0])]["cutoff"].iloc[0]
    player_rows = matches.upcoming_player_rows(fixtures_all, players, teams, season, gws)
    team_rows = matches.upcoming_team_rows(fixtures_all, season, gws)
    pred = models.train_and_predict(availability.trim_absences(history, flagged),
                                    player_rows.assign(asof=asof), team_rows.assign(asof=asof),
                                    cutoffs)
    return availability.apply(pred, avail), avail
