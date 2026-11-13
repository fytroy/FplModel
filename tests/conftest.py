"""A small made-up league shaped exactly like the real match table.

Alpha, Bravo and Charlie play both seasons; Delta is relegated after 2024-25
and Echo (all new players) is promoted for 2025-26. Player 999 ("Mover")
plays for Alpha in 2024-25 and Bravo in 2025-26. Fixture ids restart each
season, as in real FPL data.
"""
import numpy as np
import pandas as pd
import pytest

from fpl import scoring
from fpl.matches import DEFCON_COLUMNS, MATCH_COLUMNS

TEAMS = {"2024-25": ["Alpha", "Bravo", "Charlie", "Delta"],
         "2025-26": ["Alpha", "Bravo", "Charlie", "Echo"]}
POSITIONS = ["GK", "DEF", "MID", "FWD"]
TEAM_CODE = {"Alpha": 100, "Bravo": 200, "Charlie": 300, "Delta": 400, "Echo": 500}
MOVER = 999
# Each gameweek's pairings (indexes into the season's team list), home team first.
PAIRINGS = [[(0, 1), (2, 3)], [(2, 0), (3, 1)], [(0, 3), (1, 2)]]


def squad(team, season):
    """(player_code, position, price) for each player in a team that season."""
    players = [(TEAM_CODE[team] + i, pos, 4.5 + 1.5 * i) for i, pos in enumerate(POSITIONS)]
    if (team, season) in {("Alpha", "2024-25"), ("Bravo", "2025-26")}:
        players.append((MOVER, "MID", 8.0))
    return players


def make_matches(gws=6, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for season, teams in TEAMS.items():
        start = pd.Timestamp(f"{season[:4]}-08-16 14:00", tz="UTC")
        fixture_id = 0
        for gw in range(1, gws + 1):
            for slot, (h, a) in enumerate(PAIRINGS[(gw - 1) % len(PAIRINGS)]):
                fixture_id += 1
                kickoff = start + pd.Timedelta(days=7 * (gw - 1), hours=2 * slot)
                goals = {teams[h]: int(rng.poisson(1.4)), teams[a]: int(rng.poisson(1.1))}
                for team, opponent, home in [(teams[h], teams[a], True), (teams[a], teams[h], False)]:
                    for code, pos, price in squad(team, season):
                        minutes = int(rng.choice([0, 30, 90], p=[0.15, 0.15, 0.7]))
                        on = minutes > 0
                        attacker = on and pos != "GK"
                        rows.append({
                            "season": season, "gw": gw, "kickoff_time": kickoff,
                            "fixture_id": fixture_id, "player_code": code,
                            "player_name": f"P{code}", "position": pos, "team": team,
                            "opponent": opponent, "was_home": home, "price": price,
                            "team_goals": goals[team], "opp_goals": goals[opponent],
                            "minutes": minutes, "starts": int(minutes >= 60),
                            "goals_scored": int(rng.poisson(0.3)) if attacker else 0,
                            "assists": int(rng.poisson(0.2)) if attacker else 0,
                            "clean_sheets": int(minutes >= 60 and goals[opponent] == 0),
                            "goals_conceded": goals[opponent] if on else 0,
                            "saves": int(rng.poisson(3)) if on and pos == "GK" else 0,
                            "yellow_cards": int(on and rng.random() < 0.1), "red_cards": 0,
                            "penalties_saved": 0, "penalties_missed": 0, "own_goals": 0,
                            "bonus": int(rng.choice([0, 0, 0, 1, 2, 3])) if on else 0,
                            "bps": int(rng.integers(0, 40)) if on else 0,
                            "expected_goals": round(float(rng.gamma(1.0, 0.15)), 2) if attacker else 0.0,
                            "expected_assists": round(float(rng.gamma(1.0, 0.1)), 2) if attacker else 0.0,
                            "expected_goals_conceded": round(float(rng.gamma(2.0, 0.6)), 2) if on else 0.0,
                            "threat": round(float(rng.uniform(0, 50)), 1) if on else 0.0,
                            "clearances_blocks_interceptions": float(rng.integers(0, 9)) if on else 0.0,
                            "tackles": float(rng.integers(0, 5)) if on else 0.0,
                            "recoveries": float(rng.integers(0, 6)) if on else 0.0,
                            "total_points": 0, "rules_era": scoring.rules_era(season),
                        })
    m = pd.DataFrame(rows)
    m.loc[m["season"] < "2025-26", DEFCON_COLUMNS] = np.nan  # no DefCon data before 2025/26
    m["total_points"] = scoring.actual_points(m)
    return m[MATCH_COLUMNS]


def make_fixtures(matches):
    """The fixture list (as matches.load_all_fixtures returns it) for synthetic matches."""
    cols = ["season", "gw", "fixture_id", "kickoff_time", "team", "opponent", "was_home",
            "team_goals", "opp_goals"]
    return (matches[cols].drop_duplicates(["season", "fixture_id", "team"])
            .assign(finished=True).reset_index(drop=True))


@pytest.fixture
def history():
    return make_matches()


@pytest.fixture
def fixtures_all(history):
    return make_fixtures(history)
