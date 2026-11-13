"""Turn raw FPL API JSON into tidy pandas tables.

Nothing here touches the internet or the disk, so it is easy to test.
"""
import pandas as pd


def _team_names(bootstrap: dict) -> dict:
    return {t["id"]: t["short_name"] for t in bootstrap["teams"]}


def players_table(bootstrap: dict) -> pd.DataFrame:
    """One row per player, with readable team, position and price columns."""
    players = pd.DataFrame(bootstrap["elements"])
    full_names = {t["id"]: t["name"] for t in bootstrap["teams"]}
    positions = {p["id"]: p["singular_name_short"] for p in bootstrap["element_types"]}

    players["team_name"] = players["team"].map(full_names)
    players["team_short"] = players["team"].map(_team_names(bootstrap))
    players["position"] = players["element_type"].map(positions)
    # The API stores prices in tenths of a million: 105 means £10.5m.
    players["price"] = players["now_cost"] / 10

    # Put the most useful columns first; keep everything else after them.
    first = ["id", "web_name", "team_name", "team_short", "position", "price",
             "total_points", "form", "selected_by_percent", "status", "news",
             "chance_of_playing_next_round"]
    first = [c for c in first if c in players.columns]
    rest = [c for c in players.columns if c not in first]
    return players[first + rest]


def teams_table(bootstrap: dict) -> pd.DataFrame:
    """One row per Premier League club, including FPL's strength ratings."""
    return pd.DataFrame(bootstrap["teams"])


def gameweeks_table(bootstrap: dict) -> pd.DataFrame:
    """One row per gameweek: deadline, finished flag, average score, etc."""
    gws = pd.DataFrame(bootstrap["events"])
    # chip_plays / top_element_info are nested and not needed in a CSV.
    return gws.drop(columns=["chip_plays", "top_element_info"], errors="ignore")


def current_gameweek(bootstrap: dict) -> int | None:
    """The gameweek FPL marks as current, or None before the season starts."""
    for gw in bootstrap["events"]:
        if gw.get("is_current"):
            return gw["id"]
    return None


def fixtures_table(fixtures: list, bootstrap: dict) -> pd.DataFrame:
    """One row per fixture, with home/away team short names added."""
    table = pd.DataFrame(fixtures)
    names = _team_names(bootstrap)
    table["home_team"] = table["team_h"].map(names)
    table["away_team"] = table["team_a"].map(names)
    # 'stats' is a big nested list of per-match events; drop it from the CSV.
    return table.drop(columns=["stats"], errors="ignore")


def history_table(summaries: dict) -> pd.DataFrame:
    """Combine every player's gameweek history into one long table.

    `summaries` maps player id -> the JSON from the element-summary endpoint.
    Each row is one player in one match (minutes, goals, xG, BPS, points, ...).
    """
    rows = [row for summary in summaries.values() for row in summary.get("history", [])]
    return pd.DataFrame(rows)
