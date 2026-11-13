"""The user's FPL team from public API responses: squad, prices, bank, free transfers, chips.

FPL only shows selling prices and free transfers when you're logged in, so both
are rebuilt from the public transfer and gameweek history. Prices are FPL units
(tenths of £1m). Nothing here touches the internet or the disk.
"""
from dataclasses import dataclass

import pandas as pd

from fpl import rules
from fpl.matches import POSITION_BY_TYPE
from fpl.seasons import CURRENT_SEASON


@dataclass
class TeamState:
    team_id: int
    name: str
    next_gw: int
    squad: pd.DataFrame          # element, player_code, web_name, position, team, purchase, now, sell, purchase_known
    bank: int
    free_transfers: int          # free transfers left for next_gw
    pending_transfers: int       # transfers already made for next_gw
    pending_hits: int            # points already committed to hits for next_gw
    chips: list                  # rules.Chip copies still available
    chips_played: list           # [(name, gameweek)]

    @property
    def codes(self) -> list[int]:
        return [int(c) for c in self.squad["player_code"]]


def _price_at(player_history: pd.DataFrame, element: int, gw: int) -> int | None:
    """His price at gameweek gw, or at his next match if his team had none that week."""
    rows = player_history[(player_history["element"] == element) & (player_history["round"] >= gw)]
    return int(rows.sort_values("round")["value"].iloc[0]) if len(rows) else None


def load_team_state(team_id, entry, history, transfers, fetch_picks, players, player_history,
                    bootstrap_chips, season=CURRENT_SEASON) -> TeamState:
    """Rebuild the team as it stands for the next deadline.

    entry/history/transfers: FPL's /entry/{id}/, /history/, /transfers/ responses.
    fetch_picks(gw): /entry/{id}/event/{gw}/picks/ for a gameweek whose deadline has passed.
    players: our players.csv (bootstrap elements). player_history: part 1's history table.
    """
    current = history["current"]
    first_gw, last_gw = current[0]["event"], current[-1]["event"]
    played = [(c["name"], c["event"]) for c in history["chips"]]
    chip_at = dict((gw, name) for name, gw in played)
    freehit_gws = {gw for name, gw in played if name == "freehit"}
    base_gw = max(r["event"] for r in current if r["event"] not in freehit_gws)

    # Purchase prices: the starting squad at its first gameweek's prices, then every
    # transfer in order (Free Hit transfers revert, so they are skipped).
    purchase = {p["element"]: _price_at(player_history, p["element"], first_gw)
                for p in fetch_picks(first_gw)["picks"]}
    ordered = sorted(transfers, key=lambda t: t["time"])
    for t in ordered:
        if t["event"] not in freehit_gws:
            purchase[t["element_in"]] = t["element_in_cost"]

    squad = [p["element"] for p in fetch_picks(base_gw)["picks"]]
    bank = next(r["bank"] for r in current if r["event"] == base_gw)
    for t in ordered:   # made after the squad we read: Free-Hit-reverted weeks and the next deadline
        if t["event"] > base_gw and t["event"] not in freehit_gws:
            squad[squad.index(t["element_out"])] = t["element_in"]
            bank += t["element_out_cost"] - t["element_in_cost"]

    free = 1            # the first gameweek's changes are unlimited; one free transfer after it
    for r in current[1:]:
        free = rules.free_transfers_at(r["event"], free, season)
        free = rules.next_free_transfers(free, r["event_transfers"], chip_at.get(r["event"]))
    next_gw = last_gw + 1
    free = rules.free_transfers_at(next_gw, free, season)
    pending = sum(1 for t in transfers if t["event"] == next_gw)

    info = players.set_index("id").loc[squad]
    now = info["now_cost"].astype(int).to_numpy()
    bought = [purchase.get(e) for e in squad]
    table = pd.DataFrame({
        "element": squad, "player_code": info["code"].astype(int).to_numpy(),
        "web_name": info["web_name"].to_numpy(),
        "position": info["element_type"].map(POSITION_BY_TYPE).to_numpy(),
        "team": info["team"].to_numpy(), "now": now,
        "purchase": [b if b is not None else n for b, n in zip(bought, now)],
        "purchase_known": [b is not None for b in bought]})
    table["sell"] = [rules.selling_price(p, n) for p, n in zip(table["purchase"], table["now"])]
    return TeamState(
        team_id=team_id, name=entry.get("name", ""), next_gw=next_gw, squad=table, bank=int(bank),
        free_transfers=max(0, free - pending), pending_transfers=pending,
        pending_hits=rules.HIT_COST * max(0, pending - free),
        chips=rules.chips_left(rules.chip_windows(bootstrap_chips), played), chips_played=played)
