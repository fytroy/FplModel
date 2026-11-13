"""Turn a plan into plain-English advice and a saved table. No internet, no disk."""
from dataclasses import dataclass

import pandas as pd

from fpl import matches, rules
from fpl.settings import TIMEZONE


def money(tenths) -> str:
    return f"£{int(tenths) / 10:.1f}m"


def local_time(ts: pd.Timestamp, tz: str = TIMEZONE) -> str:
    """FPL deadline (UTC) in the user's time zone, e.g. 'Sat 17 Oct 13:00'."""
    return pd.Timestamp(ts).tz_convert(tz).strftime("%a %d %b %H:%M")


def pool_info(players: pd.DataFrame, teams: pd.DataFrame, state) -> pd.DataFrame:
    """Everyone who can be in the plan: buyable players plus the user's squad, keyed by code.

    Players who have left the club (status 'u') are only included if the user owns them.
    """
    short = dict(zip(teams["id"], teams["short_name"]))
    owned = set(state.codes)
    p = players[(players["status"] != "u") | players["code"].isin(owned)]
    info = pd.DataFrame({
        "name": p["web_name"].to_numpy(), "position": p["element_type"].map(matches.POSITION_BY_TYPE).to_numpy(),
        "team": p["team"].map(short).to_numpy(), "price": p["now_cost"].astype(int).to_numpy()},
        index=pd.Index(p["code"].astype(int), name="player_code"))
    info = info[info["position"].notna()]
    info["sell"] = info["price"]
    sells = state.squad.set_index("player_code")["sell"]
    info.loc[sells.index, "sell"] = sells.astype(int)
    return info


def fixture_counts(fixtures_all: pd.DataFrame, teams: pd.DataFrame, season: str) -> pd.DataFrame:
    """Fixtures per team (short name) per gameweek: 2 = double gameweek, missing = blank."""
    short = {matches.canonical_team(n): s for n, s in zip(teams["name"], teams["short_name"])}
    f = fixtures_all[fixtures_all["season"] == season]
    return (f.groupby(["team", "gw"]).size().rename("fixtures").reset_index()
            .assign(team=lambda d: d["team"].map(short)))


@dataclass
class Advice:
    state: object                 # myteam.TeamState
    plan: object                  # optimizer.Plan
    hit_gain: float | None        # expected points of the best plan with a week-1 hit minus the best without,
                                  # after paying the hits but before the planner's safety margin (chips kept
                                  # and free transfers saved included; None if not checked)
    info: pd.DataFrame
    xp: pd.DataFrame
    avail: pd.DataFrame
    scan: pd.DataFrame            # chip scan from the plan's final squad
    counts: pd.DataFrame          # fixture_counts
    deadline: pd.Timestamp
    horizon: int
    comparison_better: bool = False   # the hit-check solve found this plan (it beat the first solve's)
    alternative: object = None        # the best plan with week 1's hit decision flipped (optimizer.Plan)
    safe_plan: object = None          # what autopilot may do: no week-1 hit, no chip (same as plan if it qualifies)


def _name(a: Advice, code) -> str:
    return f"{a.info.at[code, 'name']} ({a.info.at[code, 'team']})"


def _plan_xp(a: Advice, code) -> float:
    return float(a.xp.loc[code, [w.gw for w in a.plan.weeks]].sum()) if code in a.xp.index else 0.0


def _news(a: Advice, code) -> str:
    rows = a.avail[a.avail["player_code"] == code]
    if rows.empty or pd.isna(rows["news"].iloc[0]):
        return ""
    return str(rows["news"].iloc[0])


def _pairs(a: Advice, week) -> list[tuple]:
    """(sold, bought) pairs within the same position (FPL only allows like-for-like transfers)."""
    pos = a.info["position"]
    pairs = []
    for position in ("GK", "DEF", "MID", "FWD"):
        outs = [p for p in week.transfers_out if pos[p] == position]
        ins = [p for p in week.transfers_in if pos[p] == position]
        pairs += list(zip(outs, ins))
    return pairs


def _grouped(a: Advice, week) -> list[str]:
    """Wildcard week: who goes out and who comes in, by position, not as pairs."""
    pos = a.info["position"]
    lines = []
    for label, codes in (("Out", week.transfers_out), ("In ", week.transfers_in)):
        parts = []
        for position in ("GK", "DEF", "MID", "FWD"):
            names = ", ".join(str(a.info.at[p, "name"]) for p in codes if pos[p] == position)
            if names:
                parts.append(f"{position} {names}")
        lines.append(f"  {label}: " + "; ".join(parts))
    return lines


def _transfer_lines(a: Advice, week) -> list[str]:
    if week.chip == "wildcard":
        return _grouped(a, week)
    lines = []
    for out, inn in _pairs(a, week):
        why = _news(a, out) or f"{_plan_xp(a, out):.1f} pts expected over {a.horizon} weeks"
        lines.append(f"  Sell {_name(a, out)} for {money(a.info.at[out, 'sell'])} ({why})")
        lines.append(f"   buy {_name(a, inn)} for {money(a.info.at[inn, 'price'])}, "
                     f"{a.info.at[inn, 'position']}, {_plan_xp(a, inn):.1f} pts expected over {a.horizon} weeks")
    return lines


def _freehit_changes(week) -> int:
    return len(set(week.squad) - set(week.owned))


def _week_summary(a: Advice, week) -> str:
    if week.chip in rules.TRANSFER_CHIPS:
        if week.chip == "freehit":
            return f"play your Free Hit ({_freehit_changes(week)} players changed for this week only)"
        return (f"play your {rules.CHIP_LABELS[week.chip]} ({len(week.transfers_in)} changes)\n"
                + "\n".join("  " + line for line in _grouped(a, week)))
    if not week.transfers_in:
        return f"save your free transfer ({week.free_transfers} available)"
    moves = ", ".join(f"{a.info.at[o, 'name']} -> {a.info.at[i, 'name']}"
                      for o, i in _pairs(a, week))
    hit = f" (-{rules.HIT_COST * week.paid} hit)" if week.paid else ""
    chip = f"; {rules.CHIP_LABELS[week.chip]}" if week.chip else ""
    return moves + hit + chip


def _chip_lines(a: Advice) -> list[str]:
    lines, first = [], a.plan.weeks[0].gw
    squad_teams = set(a.info.loc[a.plan.weeks[-1].owned, "team"])
    for ch in a.state.chips:
        if ch.start > first:
            continue                                          # second set: not open yet
        label = rules.CHIP_LABELS[ch.name]
        planned = next((w.gw for w in a.plan.weeks if w.chip == ch.name), None)
        expiry = f" Expires after GW{ch.stop}." if ch.stop - first < 4 else ""
        if planned is not None:
            when = "this week" if planned == first else f"in GW{planned}"
            lines.append(f"  {label}: play it {when} (in the plan).{expiry}")
            continue
        mine = a.scan[a.scan["key"] == ch.key]
        if mine.empty or mine["value"].max() <= 0:
            lines.append(f"  {label}: hold; no clearly better week found yet (window ends GW{ch.stop}).{expiry}")
            continue
        best = mine.loc[mine["value"].idxmax()]
        week_counts = a.counts[(a.counts["gw"] == best["gw"]) & a.counts["team"].isin(squad_teams)]
        doubles = int((week_counts["fixtures"] >= 2).sum())
        blanks = len(squad_teams) - len(week_counts)
        reason = (f", double gameweek for {doubles} of your clubs" if doubles
                  else f", {blanks} of your clubs have no match" if blanks else "")
        lines.append(f"  {label}: hold. Best week GW{int(best['gw'])} (+{best['value']:.1f} pts{reason}).{expiry}")
    return lines


CLOSE_CALL = 2.0          # a hit gaining less than this many points is a close call


def _hit_lines(a: Advice, week) -> list[str]:
    """Why this week does (or doesn't) take a hit, in expected points after paying for it."""
    if a.hit_gain is None:
        return []
    gain, weeks = a.hit_gain, a.horizon
    if week.paid:
        cost = rules.HIT_COST * week.paid
        verdict = (f"about {gain:.1f} pts more than" if round(gain, 1) > 0 else "about the same as")
        lines = [f"  Taking a -{cost} hit: over {weeks} weeks this plan is expected to score {verdict} "
                 f"the best plan without a hit, after paying the {cost}."]
        if gain < CLOSE_CALL:
            lines.append("  That's a close call; you could skip the hit instead.")
            if a.alternative is not None:
                lines.append("  Best plan without a hit, this week: "
                             + _week_summary(a, a.alternative.weeks[0]))
    elif round(gain, 1) == 0:
        lines = ["  No hit: the best plan with a hit this week is expected to score about the same as "
                 "this one after paying for the hit, so it isn't worth the risk."]
    elif gain > 0:
        lines = [f"  No hit: the best plan with a hit this week is expected to score only about {gain:.1f} "
                 f"pts more over {weeks} weeks after paying for the hit: too little to be worth the risk."]
    else:
        lines = [f"  No hit: the best plan with a hit this week is expected to score about {-gain:.1f} "
                 f"pts less over {weeks} weeks, after paying for the hit."]
    unproven = a.plan.status != "Optimal" or (a.alternative is not None
                                              and a.alternative.status != "Optimal")
    if unproven:
        lines.append("  (This comparison is approximate: a solve hit its time limit before proving "
                     "its plan best.)")
    return lines


def render(a: Advice) -> str:
    st, week = a.state, a.plan.weeks[0]
    chips_left = ", ".join(rules.CHIP_LABELS[c.name] for c in st.chips if c.start <= week.gw) or "none"
    first_set_end = min((c.stop for c in st.chips if c.start <= week.gw), default=None)
    out = [f"FPL advice for {st.name}: GW{week.gw}, deadline {local_time(a.deadline)} (Nairobi time)",
           f"Bank {money(st.bank)} · {st.free_transfers} free transfer{'s' if st.free_transfers != 1 else ''}"
           f" · Chips left: {chips_left}"
           + (f" (use by GW{first_set_end})" if first_set_end else "")]
    if st.pending_transfers:
        out.append(f"You've already made {st.pending_transfers} transfer(s) for GW{week.gw}"
                   + (f" (-{st.pending_hits} in hits)" if st.pending_hits else "") + ".")
    if st.pending_hits:          # more transfers than free ones: maybe a Wildcard/Free Hit we can't see
        out.append(f"You've made {st.pending_transfers} transfers this week; if that's because you activated "
                   "a Wildcard or Free Hit, the plan doesn't know — it assumes no chip is active.")
    if a.plan.status != "Optimal":
        out.append(f"(Solver stopped at its time limit: this is the best plan it found; the best possible "
                   f"plan could score up to {100 * a.plan.gap:.1f}% more.)")
    if a.comparison_better:
        out.append("(The second solve, made to check the hit decision, found this plan; "
                   "it scores higher than the first solve's.)")
    out += ["", "THIS WEEK"]
    if week.chip in rules.TRANSFER_CHIPS:
        extra = (f" ({_freehit_changes(week)} players changed for this week only)"
                 if week.chip == "freehit" else "")
        out.append(f"  Play your {rules.CHIP_LABELS[week.chip]} this week{extra}.")
    if week.chip == "bboost":    # Triple Captain is shown on the captain line
        out.append(f"  Play your {rules.CHIP_LABELS[week.chip]} this week.")
    out += _transfer_lines(a, week) if week.chip != "freehit" else []
    if not week.transfers_in and week.chip not in rules.TRANSFER_CHIPS:
        nxt = a.plan.weeks[1].free_transfers if len(a.plan.weeks) > 1 else None
        out.append("  Save your free transfer" + (f": you'll have {nxt} next week." if nxt else "."))
    out += _hit_lines(a, week)
    vice = f"   Vice-captain: {_name(a, week.vice)} ({a.xp.at[week.vice, week.gw]:.1f})"
    if week.chip == "3xc":
        out.append(f"  Captain ({rules.CHIP_LABELS['3xc']}): {_name(a, week.captain)} "
                   f"{a.xp.at[week.captain, week.gw]:.1f} x3" + vice)
    else:
        out.append(f"  Captain: {_name(a, week.captain)} ({a.xp.at[week.captain, week.gw]:.1f} pts expected)"
                   + vice)
    out.append("  Starting XI:")
    for position in ("GK", "DEF", "MID", "FWD"):
        names = [f"{_name(a, p)} {a.xp.at[p, week.gw]:.1f}" for p in week.starters
                 if a.info.at[p, "position"] == position]
        out.append(f"    {position}: " + " | ".join(names))
    gk, *outfield = week.bench
    out.append("  Bench (in order): " + "  ".join(f"{i}. {_name(a, p)}" for i, p in enumerate(outfield, 1))
               + f"  GK {_name(a, gk)}")
    plan_gws = [w.gw for w in a.plan.weeks]
    in_plan = a.avail[a.avail["gw"].isin(plan_gws)]
    flagged = [c for c in st.codes
               if (in_plan.loc[in_plan["player_code"] == c, "availability"] < 1).any()]
    if flagged:
        out += ["", "INJURY WATCH"]
        for code in flagged:
            rows = in_plan[in_plan["player_code"] == code].sort_values("gw").head(a.horizon)
            weeks = ", ".join(f"GW{int(r.gw)} {r.availability:.0%}" for r in rows.itertuples())
            news = _news(a, code)
            out.append(f"  {_name(a, code)}: " + (f"{news}. " if news else "") + f"Chance available: {weeks}")
        for code in flagged:
            rows = in_plan[in_plan["player_code"] == code]
            if (set(rows["gw"]) >= set(plan_gws) and (rows["availability"] == 0).all()
                    and not any(code in w.transfers_out for w in a.plan.weeks)):
                out.append(f"  {_name(a, code)} won't play in the plan: "
                           "replace him when you have a spare free transfer.")
    out += ["", "NEXT WEEKS (a suggested plan; re-run next week with fresh news)"]
    out += [f"  GW{w.gw}: {_week_summary(a, w)}" for w in a.plan.weeks[1:]]
    out += ["", "CHIPS"] + _chip_lines(a)
    return "\n".join(out)


def plan_table(a: Advice) -> pd.DataFrame:
    """One row per player per planned week: role, transfers, chip, xP."""
    rows = []
    for w in a.plan.weeks:
        for p in w.squad:
            role = ("captain" if p == w.captain else "vice-captain" if p == w.vice
                    else "starter" if p in w.starters
                    else "bench GK" if w.bench.index(p) == 0 else f"bench {w.bench.index(p)}")
            rows.append({"gw": w.gw, "player": a.info.at[p, "name"], "team": a.info.at[p, "team"],
                         "position": a.info.at[p, "position"], "role": role,
                         "bought_this_week": p in w.transfers_in, "chip": w.chip or "",
                         "xp": round(float(a.xp.at[p, w.gw]) if p in a.xp.index else 0.0, 2)})
    return pd.DataFrame(rows)
