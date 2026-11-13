"""One tidy table of every player in every finished fixture, across seasons.

Past seasons come from the vaastav files saved by pull_history.py; the
current season from our own FPL API pull (pull_data.py). Only the load_*
functions read the disk, so the rest is easy to test.
"""
import pandas as pd

from fpl import storage
from fpl.scoring import rules_era
from fpl.seasons import CURRENT_SEASON, PAST_SEASONS

# element_type 5 was the 2024/25 Assistant Manager; those rows are dropped.
POSITION_BY_TYPE = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
# The same club is spelled differently in some seasons; map to one spelling.
TEAM_ALIASES = {"Ipswich Town": "Ipswich"}

STAT_COLUMNS = ["minutes", "starts", "goals_scored", "assists", "clean_sheets",
                "goals_conceded", "saves", "yellow_cards", "red_cards", "penalties_saved",
                "penalties_missed", "own_goals", "bonus", "bps", "expected_goals",
                "expected_assists", "expected_goals_conceded", "threat", "total_points"]
# FPL only began publishing these in Nov 2022 (2022/23 GW16); earlier rows hold 0,
# which means "not published", not "no chances". blank_unpublished_expected finds
# the start from the data and turns the earlier zeros into NaN (missing).
EXPECTED_COLUMNS = ["expected_goals", "expected_assists", "expected_goals_conceded"]
DEFCON_COLUMNS = ["clearances_blocks_interceptions", "tackles", "recoveries"]  # 2025/26 on
MATCH_COLUMNS = ["season", "gw", "kickoff_time", "fixture_id", "player_code", "player_name",
                 "position", "team", "opponent", "was_home", "price", "team_goals",
                 "opp_goals", *STAT_COLUMNS, *DEFCON_COLUMNS, "rules_era"]
# What is known about a fixture before it is played (inputs for predictions).
PLAYER_ROW_COLUMNS = ["season", "gw", "fixture_id", "kickoff_time", "player_code",
                      "player_name", "position", "price", "team", "opponent", "was_home"]
TEAM_ROW_COLUMNS = ["season", "gw", "fixture_id", "kickoff_time", "team", "opponent",
                    "was_home"]


def canonical_team(name: str) -> str:
    return TEAM_ALIASES.get(name, name)


def _team_names(teams: pd.DataFrame) -> dict:
    return {tid: canonical_team(name) for tid, name in zip(teams["id"], teams["name"])}


def fixture_list(season: str, fixtures: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """Two rows per scheduled fixture (one per team), with kickoff and finished flag."""
    f = fixtures.dropna(subset=["event"])  # postponed fixtures without a new date have no GW
    names = _team_names(teams)
    base = pd.DataFrame({"season": season, "gw": f["event"].astype(int), "fixture_id": f["id"],
                         "kickoff_time": pd.to_datetime(f["kickoff_time"], utc=True),
                         "finished": f["finished"].astype(bool)})
    home = base.assign(team=f["team_h"].map(names), opponent=f["team_a"].map(names),
                       was_home=True, team_goals=f["team_h_score"], opp_goals=f["team_a_score"])
    away = base.assign(team=f["team_a"].map(names), opponent=f["team_h"].map(names),
                       was_home=False, team_goals=f["team_a_score"], opp_goals=f["team_h_score"])
    return (pd.concat([home, away], ignore_index=True)
            .sort_values(["kickoff_time", "fixture_id", "was_home"],
                         ascending=[True, True, False], ignore_index=True))


def build_season(season: str, gws: pd.DataFrame, players: pd.DataFrame,
                 teams: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    """One row per player per finished fixture for one season (MATCH_COLUMNS).

    gws:      player-by-fixture stats (vaastav merged_gw.csv or our player_history.csv)
    players:  id, code (permanent across seasons), element_type, web_name
    teams:    id, name
    fixtures: FPL fixtures (id, event, kickoff_time, team_h, team_a, scores, finished)
    """
    fx = fixture_list(season, fixtures, teams)
    fx = fx[fx["finished"]].drop(columns="finished")
    keep = ["element", "fixture", "was_home", "value", *STAT_COLUMNS,
            *[c for c in DEFCON_COLUMNS if c in gws.columns]]
    rows = gws[keep].drop_duplicates(subset=["element", "fixture"])
    # The team side of each row comes from the fixture plus home/away.
    rows = rows.merge(fx, left_on=["fixture", "was_home"], right_on=["fixture_id", "was_home"])
    people = players[["id", "code", "element_type", "web_name"]].rename(
        columns={"id": "element", "code": "player_code", "web_name": "player_name"})
    rows = rows.merge(people, on="element")
    rows["position"] = rows["element_type"].map(POSITION_BY_TYPE)
    rows = rows[rows["position"].notna()].copy()
    rows["price"] = rows["value"] / 10
    for col in DEFCON_COLUMNS:
        if col not in rows:
            rows[col] = float("nan")
    rows["rules_era"] = rules_era(season)
    rows = blank_unpublished_expected(rows)
    return rows[MATCH_COLUMNS].sort_values(["kickoff_time", "fixture_id", "player_code"],
                                           ignore_index=True)


def blank_unpublished_expected(table: pd.DataFrame) -> pd.DataFrame:
    """Set EXPECTED_COLUMNS to NaN for each season's matches before FPL first published them.

    Within a season, every row that kicked off before the first non-zero value of
    a column is set to NaN. A season with no non-zero value at all is left alone
    (nothing shows when publishing started).
    """
    out = table.copy()
    for col in EXPECTED_COLUMNS:
        out[col] = out[col].astype(float)
        published = out["kickoff_time"].where(out[col].fillna(0) != 0)
        start = published.groupby(out["season"]).transform("min")
        out.loc[start.notna() & (out["kickoff_time"] < start), col] = float("nan")
    return out


def _season_files(season: str):
    """(gws, players, teams, fixtures) for one season, from disk."""
    if season == CURRENT_SEASON:
        return tuple(storage.load_table(n) for n in
                     ("player_history", "players", "teams", "fixtures"))
    return tuple(storage.load_vaastav(season, p) for p in storage.VAASTAV_FILES)


def load_all_matches() -> pd.DataFrame:
    """Every finished player-fixture from 2022/23 to now. Run pull_history.py and pull_data.py first."""
    seasons = [build_season(s, *_season_files(s)) for s in PAST_SEASONS + [CURRENT_SEASON]]
    return pd.concat(seasons, ignore_index=True)


def load_all_fixtures() -> pd.DataFrame:
    """Every scheduled fixture (finished or not), two rows each, all seasons."""
    lists = []
    for s in PAST_SEASONS + [CURRENT_SEASON]:
        _, _, teams, fixtures = _season_files(s)
        lists.append(fixture_list(s, fixtures, teams))
    return pd.concat(lists, ignore_index=True)


def gw_cutoffs(fixtures_all: pd.DataFrame) -> pd.DataFrame:
    """First kickoff of each gameweek: predictions for it may only use earlier matches."""
    return (fixtures_all.groupby(["season", "gw"], as_index=False)["kickoff_time"].min()
            .rename(columns={"kickoff_time": "cutoff"}))


def with_asof(rows: pd.DataFrame, cutoffs: pd.DataFrame) -> pd.DataFrame:
    """Add `asof` = the cutoff of each row's own gameweek (keeps row order)."""
    return rows.merge(cutoffs.rename(columns={"cutoff": "asof"}), on=["season", "gw"], how="left")


def finished_gws(fixtures_all: pd.DataFrame, season: str) -> list[int]:
    """Gameweeks of a season whose fixtures have all finished."""
    done = fixtures_all[fixtures_all["season"] == season].groupby("gw")["finished"].all()
    return sorted(int(gw) for gw in done.index[done])


def upcoming_player_rows(fixtures_all, players, teams, season, gws) -> pd.DataFrame:
    """One row per current player per fixture of his team in `gws` (PLAYER_ROW_COLUMNS).

    players is our processed players.csv; status 'u' means he has left the club.
    """
    squad = players[players["status"] != "u"]
    squad = pd.DataFrame({"player_code": squad["code"], "player_name": squad["web_name"],
                          "position": squad["element_type"].map(POSITION_BY_TYPE),
                          "price": squad["now_cost"] / 10,
                          "team": squad["team"].map(_team_names(teams))})
    squad = squad[squad["position"].notna()]
    fx = upcoming_team_rows(fixtures_all, season, gws)
    rows = fx.merge(squad, on="team")
    return rows[PLAYER_ROW_COLUMNS].sort_values(["kickoff_time", "fixture_id", "player_code"],
                                                ignore_index=True)


def upcoming_team_rows(fixtures_all, season, gws) -> pd.DataFrame:
    fx = fixtures_all[(fixtures_all["season"] == season) & fixtures_all["gw"].isin(gws)]
    return fx[TEAM_ROW_COLUMNS].reset_index(drop=True)
