"""Predict expected points (xP) for the next gameweek(s) and print a ranked table.

Usage:
    python predict.py               # next gameweek
    python predict.py --horizon 6   # next 6 gameweeks: one column per week plus a total
Run pull_data.py first so this season's data is fresh (and pull_history.py once).
"""
import argparse
import sys

import numpy as np
import pandas as pd

from fpl import forecast, matches, storage

NEXT_GW_COLUMNS = ["p_plays", "p_60", "xG", "xA", "p_cs", "p_defcon", "bonus", "xP", "baseline"]


def ranked_table(pred: pd.DataFrame, teams: pd.DataFrame, gws: list[int],
                 players: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per player: next-gameweek breakdown, plus per-week xP if gws has several weeks.

    With `players` (our players table) every current player (status != 'u') is listed, even
    one whose team has no fixture in `gws`; otherwise only players present in `pred`.
    """
    short = {matches.canonical_team(n): s for n, s in zip(teams["name"], teams["short_name"])}
    who = pred.groupby("player_code").agg(player=("player_name", "first"), team=("team", "first"),
                                          pos=("position", "first"), price=("price", "first"))
    who["team"] = who["team"].map(short)
    if players is not None:
        squad = players[players["status"] != "u"]
        everyone = pd.DataFrame({
            "player": squad["web_name"].to_numpy(),
            "team": squad["team"].map(dict(zip(teams["id"], teams["short_name"]))).to_numpy(),
            "pos": squad["element_type"].map(matches.POSITION_BY_TYPE).to_numpy(),
            "price": (squad["now_cost"] / 10).to_numpy()}, index=pd.Index(squad["code"], name="player_code"))
        who = who.combine_first(everyone).loc[everyone.index]
    nxt = pred[pred["gw"] == gws[0]].assign(
        fixture=lambda d: d["opponent"].map(short) + np.where(d["was_home"].astype(bool), " (H)", " (A)"),
        xG=lambda d: d["p_plays"] * d["xg"], xA=lambda d: d["p_plays"] * d["xa"],
        bonus=lambda d: d["p_plays"] * d["e_bonus"])
    stats = nxt.groupby("player_code").agg(
        next=("fixture", ", ".join), p_plays=("p_plays", "mean"), p_60=("p_60", "mean"),
        xG=("xG", "sum"), xA=("xA", "sum"), p_cs=("p_cs", "mean"), p_defcon=("p_defcon", "mean"),
        bonus=("bonus", "sum"), xP=("xp", "sum"), baseline=("baseline", "sum"))
    table = who.join(stats)
    table["next"] = table["next"].fillna("-")  # blank gameweek
    table[NEXT_GW_COLUMNS] = table[NEXT_GW_COLUMNS].fillna(0)
    sort_by = "xP"
    if len(gws) > 1:
        weekly = (pred.pivot_table(index="player_code", columns="gw", values="xp", aggfunc="sum")
                  .reindex(columns=gws).fillna(0))
        weekly.columns = [f"xP_gw{g}" for g in gws]
        table = table.join(weekly)
        table["xP_total"] = weekly.sum(axis=1)
        sort_by = "xP_total"
    return table.sort_values(sort_by, ascending=False).reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--horizon", type=int, default=1, choices=range(1, 7),
                        help="how many gameweeks to predict (1-6)")
    parser.add_argument("--top", type=int, default=30, help="rows in the overall table")
    args = parser.parse_args()
    # Player names contain non-ASCII characters; the Windows default (cp1252) can't print them.
    sys.stdout.reconfigure(encoding="utf-8")

    history = matches.load_all_matches()
    fixtures_all = matches.load_all_fixtures()
    players = storage.load_table("players")
    teams = storage.load_table("teams")
    gameweeks = storage.load_table("gameweeks")
    upcoming = gameweeks.loc[gameweeks["is_next"].astype(bool), "id"]
    if upcoming.empty:
        raise SystemExit("FPL lists no next gameweek (season over?). Re-run pull_data.py.")
    next_gw = int(upcoming.iloc[0])
    gws = [g for g in range(next_gw, next_gw + args.horizon) if g <= 38]

    print(f"Training on {len(history):,} player-matches and predicting GW{gws[0]}"
          f"{f'-{gws[-1]}' if len(gws) > 1 else ''}...", flush=True)
    pred, _ = forecast.live_forecast(history, fixtures_all, players, teams, gameweeks, gws)

    table = ranked_table(pred, teams, gws, players)
    storage.save_table("xp_next", table)
    show = ["player", "team", "pos", "price", "next", *NEXT_GW_COLUMNS]
    if len(gws) > 1:
        show += [f"xP_gw{g}" for g in gws] + ["xP_total"]
    fmt = {"float_format": lambda v: f"{v:.2f}", "index": False}
    span = f"GW{gws[0]}-{gws[-1]}, ranked by total xP" if len(gws) > 1 else f"GW{next_gw}"
    print(f"\nTop {args.top} players, {span}:")
    print(table.head(args.top)[show].to_string(**fmt))
    for pos in ["GK", "DEF", "MID", "FWD"]:
        print(f"\nTop 10 {pos}:")
        print(table[table["pos"] == pos].head(10)[show].to_string(**fmt))
    print(f"\nAll players saved to {storage.DATA_DIR / 'processed' / 'xp_next.csv'}")


if __name__ == "__main__":
    main()
