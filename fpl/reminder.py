"""Team switch reminder: compares current team vs recommended moves and alerts before deadline.

Highlights exact action items: transfers to execute, captain switch, starting lineup & bench substitutions,
deadline countdown in Nairobi time, and notification sending.
"""
from dataclasses import dataclass
import time

import pandas as pd

from fpl import advice, api, rules
from fpl.settings import TIMEZONE


@dataclass
class TransferSwitch:
    player_out: str
    player_in: str
    out_price: int
    in_price: int
    out_xp: float
    in_xp: float
    is_hit: bool


@dataclass
class CaptainSwitch:
    current_captain: str
    recommended_captain: str
    current_xp: float
    recommended_xp: float
    gain: float
    needs_switch: bool
    recommended_vice: str


@dataclass
class LineupSwitch:
    move_to_starters: list[str]   # currently on bench, should start
    move_to_bench: list[str]      # currently starting, should be benched
    bench_order: list[str]        # recommended bench order [GK, Sub 1, Sub 2, Sub 3]


@dataclass
class DeadlineStatus:
    deadline_utc: pd.Timestamp
    deadline_local: str
    hours_remaining: float
    time_str: str
    urgency: str                  # "URGENT (<3h)", "WARNING (<24h)", "UPCOMING", "PASSED"


@dataclass
class SwitchReminder:
    team_id: int
    team_name: str
    gw: int
    deadline: DeadlineStatus
    transfers: list[TransferSwitch]
    hit_cost: int
    captain_switch: CaptainSwitch
    lineup_switch: LineupSwitch
    chip_advice: str | None
    current_squad_xp: float
    recommended_squad_xp: float
    xp_gain: float
    dream_team_xp: float | None = None


def money(tenths: int) -> str:
    return f"£{tenths / 10:.1f}m"


def compute_deadline_status(deadline_utc: pd.Timestamp,
                            now_utc: pd.Timestamp | None = None,
                            tz: str = TIMEZONE) -> DeadlineStatus:
    """Calculate deadline countdown and urgency in the local timezone."""
    if now_utc is None:
        now_utc = pd.Timestamp.now(tz="UTC")
    else:
        now_utc = pd.Timestamp(now_utc).tz_convert("UTC")

    deadline = pd.Timestamp(deadline_utc).tz_convert("UTC")
    diff = deadline - now_utc
    hours = diff.total_seconds() / 3600.0

    if hours < 0:
        urgency = "PASSED"
        time_str = "Deadline has passed"
    elif hours <= 3.0:
        urgency = "URGENT (<3h)"
        time_str = f"{int(diff.total_seconds() // 60)} minutes remaining!"
    elif hours <= 24.0:
        urgency = "WARNING (<24h)"
        time_str = f"{hours:.1f} hours remaining"
    else:
        days = int(hours // 24)
        rem_hrs = int(hours % 24)
        urgency = "UPCOMING"
        time_str = f"{days}d {rem_hrs}h remaining"

    local_str = advice.local_time(deadline, tz=tz)
    return DeadlineStatus(
        deadline_utc=deadline,
        deadline_local=local_str,
        hours_remaining=round(hours, 1),
        time_str=time_str,
        urgency=urgency,
    )


def generate_switch_reminder(current_state, plan, info: pd.DataFrame, xp: pd.DataFrame,
                             deadline_utc: pd.Timestamp, dream_team_xp: float | None = None,
                             now_utc: pd.Timestamp | None = None) -> SwitchReminder:
    """Compare current squad state against week 1 of the recommended plan."""
    week = plan.weeks[0]
    gw = week.gw
    deadline = compute_deadline_status(deadline_utc, now_utc=now_utc)

    # 1. Transfers
    pos = info["position"]
    transfers = []
    pairs = advice._pairs(advice.Advice(
        state=current_state, plan=plan, hit_gain=None, info=info, xp=xp,
        avail=None, scan=None, counts=None, deadline=deadline_utc, horizon=1
    ), week)

    hit_cost = rules.HIT_COST * int(week.paid)
    gw_col = gw if gw in xp.columns else xp.columns[0]

    for out_code, in_code in pairs:
        out_name = f"{info.at[out_code, 'name']} ({info.at[out_code, 'team']})"
        in_name = f"{info.at[in_code, 'name']} ({info.at[in_code, 'team']})"
        out_pr = int(info.at[out_code, "price"])
        in_pr = int(info.at[in_code, "price"])
        out_p = float(xp.at[out_code, gw_col]) if out_code in xp.index else 0.0
        in_p = float(xp.at[in_code, gw_col]) if in_code in xp.index else 0.0

        transfers.append(
            TransferSwitch(
                player_out=out_name,
                player_in=in_name,
                out_price=out_pr,
                in_price=in_pr,
                out_xp=round(out_p, 1),
                in_xp=round(in_p, 1),
                is_hit=(hit_cost > 0),
            )
        )

    # 2. Current Captain vs Recommended Captain
    curr_squad = current_state.squad
    # In myteam.TeamState, squad has element / player_code
    # If picks info exists with is_captain, read it; else default to highest value
    rec_cap_code = week.captain
    rec_cap_name = f"{info.at[rec_cap_code, 'name']} ({info.at[rec_cap_code, 'team']})"
    rec_cap_xp = float(xp.at[rec_cap_code, gw_col]) if rec_cap_code in xp.index else 0.0

    rec_vice_code = week.vice
    rec_vice_name = f"{info.at[rec_vice_code, 'name']} ({info.at[rec_vice_code, 'team']})"

    # Current captain guess or pick
    if "is_captain" in curr_squad.columns and curr_squad["is_captain"].any():
        curr_cap_row = curr_squad[curr_squad["is_captain"]].iloc[0]
        curr_cap_code = int(curr_cap_row["player_code"])
    else:
        # If not tracked, default to first star player or Haaland/Salah in current squad
        curr_cap_code = int(curr_squad["player_code"].iloc[0])

    curr_cap_name = f"{info.at[curr_cap_code, 'name']} ({info.at[curr_cap_code, 'team']})" if curr_cap_code in info.index else "Current Captain"
    curr_cap_xp = float(xp.at[curr_cap_code, gw_col]) if curr_cap_code in xp.index else 0.0

    needs_cap_switch = (curr_cap_code != rec_cap_code)
    cap_gain = round(rec_cap_xp - curr_cap_xp, 1)

    captain_switch = CaptainSwitch(
        current_captain=curr_cap_name,
        recommended_captain=rec_cap_name,
        current_xp=round(curr_cap_xp, 1),
        recommended_xp=round(rec_cap_xp, 1),
        gain=cap_gain,
        needs_switch=needs_cap_switch,
        recommended_vice=rec_vice_name,
    )

    # 3. Lineup / Bench switches
    # Current starters vs Recommended starters
    current_owned = set(current_state.codes)
    rec_starters = set(week.starters)
    rec_bench = week.bench

    # Bench order names
    bench_order_names = [f"{info.at[c, 'name']} ({info.at[c, 'team']})" for c in rec_bench if c in info.index]

    # Starters to bench, bench to starters
    if "position_num" in curr_squad.columns:
        curr_starters = set(curr_squad[curr_squad["position_num"] <= 11]["player_code"].astype(int))
    else:
        # Default first 11
        curr_starters = set(curr_squad["player_code"].iloc[:11].astype(int))

    to_start = [f"{info.at[c, 'name']} ({info.at[c, 'team']})" for c in (rec_starters - curr_starters) if c in info.index and c in current_owned]
    to_bench = [f"{info.at[c, 'name']} ({info.at[c, 'team']})" for c in (curr_starters - rec_starters) if c in info.index and c in current_owned]

    lineup_switch = LineupSwitch(
        move_to_starters=to_start,
        move_to_bench=to_bench,
        bench_order=bench_order_names,
    )

    # 4. Squad xP comparison
    curr_pts = sum(float(xp.at[c, gw_col]) for c in curr_starters if c in xp.index) + curr_cap_xp
    rec_pts = sum(float(xp.at[c, gw_col]) for c in rec_starters if c in xp.index) + rec_cap_xp - hit_cost

    chip_label = rules.CHIP_LABELS.get(week.chip, week.chip) if week.chip else None

    return SwitchReminder(
        team_id=current_state.team_id,
        team_name=current_state.name,
        gw=gw,
        deadline=deadline,
        transfers=transfers,
        hit_cost=hit_cost,
        captain_switch=captain_switch,
        lineup_switch=lineup_switch,
        chip_advice=chip_label,
        current_squad_xp=round(curr_pts, 1),
        recommended_squad_xp=round(rec_pts, 1),
        xp_gain=round(rec_pts - curr_pts, 1),
        dream_team_xp=round(dream_team_xp, 1) if dream_team_xp else None,
    )


def render_switch_reminder(rem: SwitchReminder) -> str:
    """Format an actionable, high-visibility deadline switch reminder."""
    lines = [
        "=" * 74,
        f"       FPL GW{rem.gw} TEAM SWITCH REMINDER: {rem.team_name.upper()}",
        f"       Deadline: {rem.deadline.deadline_local} ({rem.deadline.time_str})",
        "=" * 74,
    ]

    if rem.deadline.urgency in ("URGENT (<3h)", "WARNING (<24h)"):
        lines += [f"*** {rem.deadline.urgency}: REVIEW AND CONFIRM YOUR TEAM NOW! ***", "-" * 74]

    # Section 1: Transfers
    lines.append("1. TRANSFERS TO MAKE:")
    if rem.transfers:
        for t in rem.transfers:
            hit_tag = " (-4 pt hit)" if t.is_hit else " (Free Transfer)"
            lines.append(f"   [SELL] {t.player_out:<24} ({money(t.out_price)}, {t.out_xp} xP)")
            lines.append(f"   [BUY]  {t.player_in:<24} ({money(t.in_price)}, {t.in_xp} xP){hit_tag}")
            gain = round(t.in_xp - t.out_xp, 1)
            lines.append(f"   --> Projected GW gain: {'+' if gain >= 0 else ''}{gain} pts")
    else:
        lines.append("   Save your free transfer (No transfers recommended this week).")
    lines.append("")

    # Section 2: Captain Switch
    lines.append("2. CAPTAIN & ARMBAND SWITCH:")
    cs = rem.captain_switch
    if cs.needs_switch:
        lines.append(f"   [!] SWITCH CAPTAIN: {cs.current_captain} -> {cs.recommended_captain}")
        lines.append(f"       Expected Points: {cs.recommended_xp:.1f} xP vs {cs.current_xp:.1f} xP (Gain: +{cs.gain:.1f} pts)")
    else:
        lines.append(f"   KEEP CAPTAIN: {cs.recommended_captain} ({cs.recommended_xp:.1f} xP)")
    lines.append(f"   VICE-CAPTAIN: {cs.recommended_vice}")
    lines.append("")

    # Section 3: Lineup & Bench Switches
    lines.append("3. STARTING LINEUP & BENCH ADJUSTMENTS:")
    ls = rem.lineup_switch
    if ls.move_to_starters:
        lines.append(f"   [+] PROMOTE TO STARTERS: {', '.join(ls.move_to_starters)}")
    if ls.move_to_bench:
        lines.append(f"   [-] MOVE TO BENCH:       {', '.join(ls.move_to_bench)}")
    if not ls.move_to_starters and not ls.move_to_bench:
        lines.append("   Starting XI lineup is already optimal.")

    if ls.bench_order:
        sub_labels = ["GK Sub", "1st Sub", "2nd Sub", "3rd Sub"]
        bench_str = " | ".join(f"{sub_labels[i]}: {name}" for i, name in enumerate(ls.bench_order[:4]))
        lines.append(f"   BENCH ORDER: {bench_str}")
    lines.append("")

    # Section 4: Chip Advice
    if rem.chip_advice:
        lines += [f"4. CHIP RECOMMENDATION: PLAY {rem.chip_advice.upper()} THIS WEEK!", ""]
    else:
        lines += ["4. CHIP: Do not use any chip this week.", ""]

    # Section 5: Score Comparison
    lines += [
        "-" * 74,
        "SQUAD PROJECTION COMPARISON:",
        f"  Current Squad Projected xP:    {rem.current_squad_xp:>6.1f} pts",
        f"  After Recommended Switches:   {rem.recommended_squad_xp:>6.1f} pts (Net gain: {'+' if rem.xp_gain >= 0 else ''}{rem.xp_gain:.1f} pts)",
    ]
    if rem.dream_team_xp:
        gap = rem.dream_team_xp - rem.recommended_squad_xp
        lines.append(f"  Theoretical Optimal Dream Team: {rem.dream_team_xp:>6.1f} pts (Difference: {gap:.1f} pts)")

    lines.append("=" * 74)
    return "\n".join(lines)


def send_switch_notification(rem: SwitchReminder, topic: str | None = None) -> bool:
    """Send short phone notification via ntfy."""
    if not topic:
        return False

    title = f"FPL GW{rem.gw} Switch Reminder: {rem.deadline.time_str}"
    body_lines = [
        f"Deadline: {rem.deadline.deadline_local}",
    ]
    if rem.transfers:
        body_lines.append("Moves: " + "; ".join(f"{t.player_out} -> {t.player_in}" for t in rem.transfers))
    else:
        body_lines.append("Moves: Save free transfer")

    if rem.captain_switch.needs_switch:
        body_lines.append(f"Captain: Switch to {rem.captain_switch.recommended_captain}")
    else:
        body_lines.append(f"Captain: {rem.captain_switch.recommended_captain}")

    if rem.lineup_switch.move_to_starters:
        body_lines.append(f"Start: {', '.join(rem.lineup_switch.move_to_starters)}")

    body = "\n".join(body_lines)
    try:
        api.send_ntfy(topic, title, body, priority=4 if rem.deadline.urgency.startswith("URGENT") else 3)
        return True
    except Exception as e:
        print(f"Failed to send notification: {e}")
        return False
