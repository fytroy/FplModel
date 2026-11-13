"""Autopilot: apply this week's optimizer plan on your real FPL team.

Off unless FPL_AUTOPILOT=on. Safe mode only carries out free transfers; all_planned mode
also carries out optimizer hits and chips. Every check below must pass first.
If FPL refuses the login or a request it stops: it never works around a block.
Tokens are never printed. Network calls live in api.py.
"""
import json
import os
import subprocess
from dataclasses import dataclass, replace

import pandas as pd
import requests

from fpl import advice, api, rules
from fpl.matches import POSITION_BY_TYPE
from fpl.settings import (AUTOPILOT_MAX_TRANSFERS, AUTOPILOT_MIN_MINUTES, AUTOPILOT_POLICY,
                          ENV_FILE)

SQUAD_SHAPE = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
POSITION_ORDER = ["GK", "DEF", "MID", "FWD"]
MIN_STARTERS = {"GK": 1, "DEF": 3, "MID": 2, "FWD": 1}


class Blocked(Exception):
    """A safety check said no: the team is left as it is, and retrying won't help."""


class Failed(Exception):
    """Something went wrong talking to FPL: worth a retry, or your attention."""


@dataclass
class Orders:
    gw: int
    transfers: list                  # the /transfers/ items
    picks: list                      # the /my-team/ items
    lines: list                      # plain-English summary
    updated: bool = True              # whether a POST is needed


def enabled() -> bool:
    return os.environ.get("FPL_AUTOPILOT", "off").strip().lower() == "on"


def full_policy() -> bool:
    """Whether the optimizer may execute hits and chips."""
    return os.environ.get("FPL_AUTOPILOT_POLICY", AUTOPILOT_POLICY).strip().lower() == "all_planned"


def parse_refresh_token(raw: str) -> str:
    """The token itself, or the whole oidc.user JSON copied from the browser."""
    text = raw.strip()
    if text.startswith("{"):
        try:
            return json.loads(text)["refresh_token"]
        except (ValueError, KeyError):
            raise Failed("FPL_REFRESH_TOKEN looks like JSON but has no refresh_token field.") from None
    return text


def store_refresh_token(token: str) -> None:
    """Keep the replacement token: as a GitHub secret on Actions, else in the local .env."""
    if os.environ.get("GITHUB_ACTIONS"):
        print(f"::add-mask::{token}", flush=True)
        done = subprocess.run(["gh", "secret", "set", "FPL_REFRESH_TOKEN", "--repo",
                               os.environ["GITHUB_REPOSITORY"]], input=token, text=True,
                              capture_output=True, env={**os.environ, "GH_TOKEN": os.environ.get("SECRET_WRITE_TOKEN", "")})
        if done.returncode:
            raise Failed("Couldn't save the new login token as a GitHub secret. Paste a fresh "
                         "FPL_REFRESH_TOKEN and check SECRET_WRITE_TOKEN.")
        return
    lines = ENV_FILE.read_text(encoding="utf-8").splitlines() if ENV_FILE.exists() else []
    kept = [line for line in lines if not line.startswith("FPL_REFRESH_TOKEN=")]
    ENV_FILE.write_text("\n".join(kept + [f"FPL_REFRESH_TOKEN={token}"]) + "\n", encoding="utf-8")


class Session:
    """A logged-in connection to one FPL team."""

    def __init__(self, team_id: int, access_token: str):
        self.team_id, self.access_token = team_id, access_token

    @classmethod
    def connect(cls, team_id: int, store=store_refresh_token) -> "Session":
        raw = os.environ.get("FPL_REFRESH_TOKEN", "")
        if not raw.strip():
            raise Failed("FPL_REFRESH_TOKEN is not set.")
        try:
            tokens = api.refresh_access_token(parse_refresh_token(raw))
        except requests.RequestException as error:
            raise Failed(f"FPL login failed ({_reason(error)}). Paste a fresh FPL_REFRESH_TOKEN.") from None
        if tokens.get("refresh_token"):
            store(tokens["refresh_token"])
            os.environ["FPL_REFRESH_TOKEN"] = tokens["refresh_token"]
        return cls(team_id, tokens["access_token"])

    def my_team(self) -> dict:
        try:
            return api.fetch_my_team(self.team_id, self.access_token)
        except requests.RequestException as error:
            raise Failed(f"Couldn't read your team ({_reason(error)}).") from None


def _reason(error: requests.RequestException) -> str:
    """Why a request failed, without the URL, headers or anything sensitive."""
    response = getattr(error, "response", None)
    return f"HTTP {response.status_code}" if response is not None else type(error).__name__


def live_state(state, mine: dict, players: pd.DataFrame):
    """The team as your account shows it: real selling prices, bank and free transfers."""
    try:
        picks, money = mine["picks"], mine["transfers"]
        limit, made, bank = money["limit"], money["made"], money["bank"]
        rows = players.set_index("id").loc[[p["element"] for p in picks]]
        table = pd.DataFrame({
            "element": [p["element"] for p in picks], "player_code": rows["code"].astype(int).to_numpy(),
            "web_name": rows["web_name"].to_numpy(),
            "position": rows["element_type"].map(POSITION_BY_TYPE).to_numpy(),
            "team": rows["team"].to_numpy(), "now": rows["now_cost"].astype(int).to_numpy(),
            "purchase": [p["purchase_price"] for p in picks], "purchase_known": True,
            "sell": [p["selling_price"] for p in picks]})
    except (KeyError, TypeError) as error:
        raise Failed(f"FPL's my-team reply has changed shape ({error}).") from None
    free = state.free_transfers if limit is None else min(state.free_transfers + state.pending_transfers,
                                                          max(0, limit - made))
    return replace(state, squad=table, bank=int(bank), free_transfers=int(free), pending_transfers=int(made))


def _active_chip(mine: dict) -> str | None:
    return next((c.get("name", "a chip") for c in mine.get("chips", [])
                 if c.get("status_for_entry") == "active"), None)


def build_orders(a: advice.Advice, plan, mine: dict, players: pd.DataFrame, now: pd.Timestamp) -> Orders:
    """Turn week 1 of a plan into FPL requests, or raise Blocked. Touches nothing."""
    week = plan.weeks[0]
    left = (a.deadline - now) / pd.Timedelta(minutes=1)
    if left < AUTOPILOT_MIN_MINUTES:
        raise Blocked(f"only {left:.0f} minutes to the deadline")
    if a.state.next_gw != week.gw:
        raise Blocked(f"the plan is for GW{week.gw} but your team is on GW{a.state.next_gw}")
    if not full_policy() and (week.chip or week.paid):
        raise Blocked("the plan needs a chip or a hit")
    chip = _active_chip(mine)
    if chip:
        raise Blocked(f"{chip} is active on your team")
    made = mine["transfers"]["made"]
    if made:
        raise Blocked(f"you've already made {made} transfer(s) this week")

    by_code = players.set_index("code")
    element = lambda code: int(by_code.at[code, "id"])
    current = {p["element"]: p for p in mine["picks"]}
    if set(current) != {element(c) for c in a.state.codes}:
        raise Blocked("the squad changed while the plan was being made")
    if set(a.state.codes) != (set(week.owned) - set(week.transfers_in)) | set(week.transfers_out):
        raise Blocked("the plan doesn't start from your current squad")

    pairs = advice._pairs(a, week)
    if len(pairs) != len(week.transfers_in) or len(pairs) != len(week.transfers_out):
        raise Blocked("the transfers don't pair up by position")
    limit = mine["transfers"]["limit"]
    max_transfers = (min(AUTOPILOT_MAX_TRANSFERS, 0 if limit is None else limit, a.state.free_transfers)
                     if not full_policy() else len(pairs))
    if len(pairs) > max_transfers:
        raise Blocked(f"{len(pairs)} transfers is more than the free transfers or the weekly limit")

    transfers, money = [], int(mine["transfers"]["bank"])
    for out, inn in pairs:
        status = by_code.at[inn, "status"]
        if status != "a":
            raise Blocked(f"{a.info.at[inn, 'name']} isn't fully available (status {status})")
        sold, price = current[element(out)]["selling_price"], int(by_code.at[inn, "now_cost"])
        money += sold - price
        transfers.append({"element_in": element(inn), "element_out": element(out),
                          "purchase_price": price, "selling_price": sold})
    if money < 0:
        raise Blocked("the transfers cost more than you have in the bank")

    positions, clubs = a.info.loc[week.owned, "position"], a.info.loc[week.owned, "team"]
    if positions.value_counts().to_dict() != SQUAD_SHAPE:
        raise Blocked("the new squad isn't 2 GK, 5 DEF, 5 MID, 3 FWD")
    for inn in week.transfers_in:
        if (clubs == a.info.at[inn, "team"]).sum() > rules.MAX_PER_CLUB:
            raise Blocked(f"too many players from {a.info.at[inn, 'team']}")

    lines = [f"{advice._name(a, o)} -> {advice._name(a, i)}" for o, i in pairs]
    return Orders(week.gw, transfers, _picks(a, week, element), lines)


def _picks(a: advice.Advice, week, element) -> list:
    """Starters by position, then the bench in order; FPL's positions 1-15."""
    playing_squad = week.squad if week.chip == "freehit" else week.owned
    if set(week.starters) | set(week.bench) != set(playing_squad) or len(week.starters) != 11:
        raise Blocked("the lineup doesn't match the squad")
    if week.captain not in week.starters or week.vice not in week.starters or week.captain == week.vice:
        raise Blocked("the captain or vice-captain isn't a different starter")
    position = a.info["position"]
    counts = position[list(week.starters)].value_counts()
    if counts.get("GK", 0) != 1 or any(counts.get(p, 0) < n for p, n in MIN_STARTERS.items()):
        raise Blocked("the starting XI isn't a legal formation")
    starters = sorted(week.starters, key=lambda c: POSITION_ORDER.index(position[c]))
    picks = []
    for slot, code in enumerate(starters + list(week.bench), start=1):
        captain = code == week.captain
        picks.append({"element": element(code), "position": slot, "is_captain": captain,
                      "is_vice_captain": code == week.vice,
                      "multiplier": 0 if slot > 11 else 2 if captain else 1})
    return picks


def _lineup(picks: list) -> list:
    return sorted((p["position"], p["element"], bool(p["is_captain"]), bool(p["is_vice_captain"])) for p in picks)


def mismatches(orders: Orders, mine: dict) -> list[str]:
    """What FPL shows that differs from what was ordered (empty: all done)."""
    wanted = _lineup(orders.picks)
    have = _lineup(mine["picks"])
    if {w[1] for w in wanted} != {h[1] for h in have}:
        return ["the squad"]
    return [] if wanted == have else ["the lineup or captain"]


def carry_out(a: advice.Advice, plan, session: Session, players: pd.DataFrame, now: pd.Timestamp,
              dry_run: bool = False) -> Orders:
    """Send the orders, then read the team back to prove they landed."""
    mine = session.my_team()
    orders = build_orders(a, plan, mine, players, now)
    chip = plan.weeks[0].chip
    orders.updated = bool(orders.transfers or _lineup(orders.picks) != _lineup(mine["picks"]))
    if dry_run:
        return orders
    try:
        if orders.transfers:
            transfer_chip = chip if chip in rules.TRANSFER_CHIPS else None
            api.post_transfers({"chip": transfer_chip, "entry": session.team_id, "event": orders.gw,
                                "transfers": orders.transfers, "confirmed": True}, session.access_token)
        if _lineup(orders.picks) != _lineup(mine["picks"]):
            api.post_picks(session.team_id, {"chip": chip, "picks": orders.picks}, session.access_token)
    except requests.RequestException as error:      # a lost reply may still have worked: the read-back decides
        print(f"FPL request failed ({_reason(error)}); checking what landed.", flush=True)
    wrong = mismatches(orders, session.my_team())
    if wrong:
        raise Failed(f"FPL doesn't show the expected {' and '.join(wrong)}.")
    return orders
