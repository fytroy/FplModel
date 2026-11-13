"""Deadline alerts: when to send, the short phone message, and whether the advice changed.

No internet, no disk: alert.py fetches, sends and saves.
"""
import pandas as pd

from fpl import advice, rules
from fpl.settings import EARLY_ALERT_HOURS, LATE_ALERT_HOURS

LABELS = {"moves": "Transfers", "hit": "Hit", "captain": "Captain", "vice": "Vice-captain", "chip": "Chip"}


def next_deadline(events: list[dict], now: pd.Timestamp) -> tuple[int, pd.Timestamp] | None:
    """(gameweek, deadline in UTC) of the next deadline after now, from FPL's bootstrap events."""
    ahead = [(int(e["id"]), pd.Timestamp(e["deadline_time"])) for e in events
             if e.get("deadline_time") and pd.Timestamp(e["deadline_time"]) > now]
    return min(ahead, key=lambda e: e[1]) if ahead else None


def stage_for(now: pd.Timestamp, deadline: pd.Timestamp) -> str | None:
    """'early' (about a day before), 'late' (a few hours before) or None (no alert now)."""
    left = (deadline - now) / pd.Timedelta(hours=1)
    for stage, (start, until) in (("early", EARLY_ALERT_HOURS), ("late", LATE_ALERT_HOURS)):
        if until < left <= start:
            return stage
    return None


def _moves(a: advice.Advice, week) -> list[str]:
    if week.chip == "freehit":
        added = sorted(str(a.info.at[p, "name"]) for p in set(week.squad) - set(week.owned))
        return ["Free Hit squad adds: " + ", ".join(added)]
    if week.chip == "wildcard":
        return [" ".join(line.split()).replace(" :", ":") for line in advice._grouped(a, week)]
    return [f"{advice._name(a, out)} -> {advice._name(a, inn)}" for out, inn in advice._pairs(a, week)]


def decisions(a: advice.Advice) -> dict:
    """This week's calls only (what the alert compares): transfers, hit, captain, vice, chip."""
    week = a.plan.weeks[0]
    return {"moves": _moves(a, week), "hit": rules.HIT_COST * int(week.paid),
            "captain": advice._name(a, week.captain), "vice": advice._name(a, week.vice),
            "chip": rules.CHIP_LABELS[week.chip] if week.chip else None}


def _show(key: str, value) -> str:
    if key == "moves":
        return ", ".join(value) if value else "save your free transfer"
    if key == "hit":
        return f"-{value}" if value else "none"
    return value or "none"


def changed(old: dict, new: dict) -> list[str]:
    """Each decision that differs, e.g. 'Captain: Salah (LIV) -> Haaland (MCI)'."""
    return [f"{label}: {_show(key, old.get(key))} -> {_show(key, new[key])}"
            for key, label in LABELS.items() if old.get(key) != new[key]]


def late_action(record: dict | None, new: dict) -> str | None:
    """The late re-check: 'full' if no message went out earlier, 'changed', or None (nothing new)."""
    if record is None:
        return "full"
    return "changed" if changed(record["decisions"], new) else None


def summary(d: dict) -> str:
    """One line for a set of decisions: chip, transfers and hit."""
    parts = [f"play {d['chip']}"] if d["chip"] else []
    parts += d["moves"]
    if d["hit"]:
        parts.append(f"take a -{d['hit']}")
    return ", ".join(parts)


def _hit_line(a: advice.Advice, week) -> str:
    gain = a.hit_gain
    if week.chip in rules.TRANSFER_CHIPS:
        return f"Hit: none needed ({rules.CHIP_LABELS[week.chip]}: unlimited free transfers)"
    if week.paid:
        cost = rules.HIT_COST * week.paid
        if gain is None:
            return f"Hit: take the -{cost}"
        better = f"about {gain:.1f} pts better than" if round(gain, 1) > 0 else "about the same as"
        close = "; a close call" if gain < advice.CLOSE_CALL else ""
        return f"Hit: take the -{cost} ({better} no hit over {a.horizon} weeks{close})"
    if gain is None or round(gain, 1) == 0:
        return "Hit: no"
    if gain > 0:
        return f"Hit: no (a hit would gain only about {gain:.1f} pts over {a.horizon} weeks)"
    return f"Hit: no (a hit would lose about {-gain:.1f} pts over {a.horizon} weeks)"


def short_message(a: advice.Advice, changes: list[str] | None = None, test: bool = False) -> tuple[str, str]:
    """(title, body) for the phone: this week's transfers, hit, captain, vice and chip."""
    week, d = a.plan.weeks[0], decisions(a)
    title = f"FPL GW{week.gw}: advice changed" if changes else f"FPL GW{week.gw} advice"
    lines = ["What changed: " + "; ".join(changes)] if changes else []
    lines.append(f"Deadline {advice.local_time(a.deadline)} (Nairobi time)")
    if a.state.pending_transfers:
        lines.append(f"You've already made {a.state.pending_transfers} transfer(s) this week.")
    joined = (" / " if week.chip == "wildcard" else ", ").join(d["moves"])
    lines += [f"Transfers: {joined or 'save your free transfer'}",
              _hit_line(a, week),
              f"Captain: {d['captain']} · Vice: {d['vice']}",
              f"Chip: play your {d['chip']} this week" if d["chip"] else "Chip: none this week"]
    if a.plan.status != "Optimal":
        lines.append("(Approximate: the solver hit its time limit.)")
    return ("TEST " if test else "") + title, "\n".join(lines)
