"""Deadline alerts: this week's FPL advice as a short phone message (ntfy).

GitHub Actions runs this every hour (.github/workflows/deadline-alerts.yml):
    python alert.py gate                         # is an alert due now? (one FPL request)
    python alert.py run --stage early            # ~24 h before: full advice, send the message
    python alert.py run --stage late             # ~3 h before: message only if the advice changed
    python alert.py run --stage early --dry-run  # print the message; send and save nothing
    python alert.py run --test                   # send now, titled TEST; save nothing
The ntfy topic comes from the NTFY_TOPIC environment variable (a GitHub secret), never
from the code. ALERT_NOW (an ISO time) pretends it's another time, to test the windows.

With FPL_AUTOPILOT=on the late run applies the plan permitted by FPL_AUTOPILOT_POLICY and
sets the captain and bench on your real team, then tells you what it did. --dry-run logs
in and shows what it would do, changing nothing.
"""
import argparse
import os
import sys
import time

import pandas as pd

from fpl import advice, alerts, api, autopilot, storage
from fpl.settings import MESSAGE_FREQUENCY, NTFY_TOPIC, TEAM_ID, require_team

RETRY_UNTIL_HOURS = 2.0                 # autopilot errors retry hourly until this close to the deadline


def now() -> pd.Timestamp:
    return pd.Timestamp(os.environ.get("ALERT_NOW") or pd.Timestamp.now(tz="UTC")).tz_convert("UTC")


def output(**values) -> None:
    """Hand values to later workflow steps (GitHub's $GITHUB_OUTPUT file); also print them."""
    path = os.environ.get("GITHUB_OUTPUT")
    for key, value in values.items():
        print(f"  {key}={value}")
        if path:
            with open(path, "a", encoding="utf-8") as f:
                f.write(f"{key}={value}\n")


def topic(override: str | None = None) -> str:
    name = (override or os.environ.get("NTFY_TOPIC") or NTFY_TOPIC).strip()
    if not name:
        raise SystemExit("NTFY_TOPIC is not set: add it as a GitHub secret (see CLAUDE.md, part 4).")
    return name


def gate(mode: str, topic_override: str | None = None) -> None:
    """Find the next deadline and whether an alert is due. Manual modes always go ahead."""
    events = api.fetch_bootstrap()["events"]
    print("FPL's API answered.")
    current = now()
    found = alerts.next_deadline(events, current)
    if found is None:
        print("No deadline ahead: the season is over.")
        output(run="false")
        return
    gw, deadline = found
    stage = alerts.stage_for(current, deadline)
    hours = (deadline - current) / pd.Timedelta(hours=1)
    when = advice.local_time(deadline)
    print(f"Next deadline: GW{gw}, {when} Nairobi time ({hours:.1f} h away). Alert due now: {stage or 'none'}.")
    if mode == "check":
        api.send_ntfy(topic(topic_override), "FPL alerts: test ping",
                      f"GitHub can reach FPL. Next deadline GW{gw}: {when} (Nairobi time). "
                      "Advice will arrive about 24 h and 3 h before each deadline.", tags=["white_check_mark"])
        print("Test ping sent.")
    frequency = os.environ.get("FPL_MESSAGE_FREQUENCY", MESSAGE_FREQUENCY).strip().lower()
    if mode == "auto" and frequency == "updated_only" and stage == "early":
        stage = None
    due = (mode == "auto" and stage is not None) or mode in ("dry-run", "test-send")
    output(gw=gw, stage=stage or "early", deadline=when, run=str(due).lower())


def run(stage: str, team: int, dry_run: bool = False, test: bool = False, topic_override: str | None = None) -> None:
    """Make this week's advice and send (or skip) the phone message for this stage."""
    start = time.monotonic()
    log = lambda line: print(f"[{time.monotonic() - start:4.0f}s] {line}", flush=True)
    import recommend                                  # heavy imports (LightGBM, solver) only here
    session, problem, live = None, None, None
    if autopilot.enabled() and not test:
        session, problem = connect(team)
        if session:
            live = lambda state, players: live_view(session, state, players, log)
    a = recommend.run(team, log=log, live=live)
    log(f"Plan: {a.plan.status}" + (f"; hit check: {a.alternative.status}" if a.alternative else ""))
    preferred = None
    if (a.safe_plan is not None and a.safe_plan is not a.plan
            and not autopilot.full_policy()):                  # safe mode leaves hits/chips to you
        preferred = alerts.decisions(a)
        a.plan, a.alternative, a.hit_gain = a.safe_plan, None, None
    gw, new = a.plan.weeks[0].gw, alerts.decisions(a)
    record = storage.load_alert(gw)
    if autopilot.enabled() and (stage == "late" or dry_run) and not test:   # a dry run is a rehearsal, any time
        report = pilot(a, session, problem, dry_run)
    else:
        report = None
    if test:
        action = "full"
    elif stage == "early":
        action = None if record else "full"            # the early message goes out once
    else:
        action = alerts.late_action(record, new)
    frequency = os.environ.get("FPL_MESSAGE_FREQUENCY", MESSAGE_FREQUENCY).strip().lower()
    if report:
        action = "full"
    elif frequency == "updated_only":
        action = None
    changes = alerts.changed(record["decisions"], new) if action == "changed" else None
    title, body = alerts.short_message(a, changes, test=test)
    if preferred:
        body += f"\nThe model would rather: {alerts.summary(preferred)}. Autopilot won't do that; it's your call."
    if problem and stage == "early":
        action = "full"
        body += f"\nAutopilot problem: {problem}"
    if report:
        title, body = report[0], report[1] + "\n\n" + body
    print(f"\n{title}\n{body}\n")
    if dry_run:
        print("Dry run: nothing sent or saved.")
        return
    if action is None:
        print("Already sent, or nothing changed: no message.")
    else:
        urgent = bool(report and report[2])
        api.send_ntfy(topic(topic_override), title, body, priority=5 if urgent else 4 if changes else 3,
                      tags=["rotating_light"] if urgent else ["warning"] if changes else ["soccer"])
        log("Message sent.")
    if test or (stage == "early" and record):
        return
    storage.save_alert(gw, {"gw": gw, "stage": stage, "sent": action,
                            "at": now().isoformat(), "decisions": new})
    output(saved="true", gw=gw)


def connect(team: int) -> tuple:
    """(session, None), or (None, why not) if the login failed."""
    try:
        return autopilot.Session.connect(team), None
    except autopilot.Failed as error:
        return None, str(error)


def live_view(session, state, players, log):
    """The logged-in team if it can be read; the public rebuild is a fine stand-in for planning."""
    try:
        return autopilot.live_state(state, session.my_team(), players)
    except autopilot.Failed as error:
        log(f"Using the public view of your team: {error}")
        return state


def pilot(a, session, problem, dry_run: bool) -> tuple[str, str, bool] | None:
    """Carry out the plan: (title, what happened, urgent). Errors retry while there's time."""
    gw = a.plan.weeks[0].gw
    manual = "Nothing was changed, so make any moves yourself."
    try:
        if problem:
            raise autopilot.Failed(problem)
        orders = autopilot.carry_out(a, a.plan, session, storage.load_table("players"), now(), dry_run)
    except autopilot.Blocked as error:
        return f"FPL GW{gw}: autopilot stood down", f"Left your team alone: {error}. {manual}", True
    except autopilot.Failed as error:
        if (a.deadline - now()) / pd.Timedelta(hours=1) > RETRY_UNTIL_HOURS and not dry_run:
            print(f"Autopilot failed, will retry: {error}", flush=True)
            sys.exit(1)
        return f"FPL GW{gw}: autopilot FAILED", f"{error} Open FPL and check your team.", True
    moves = "; ".join(orders.lines) if orders.lines else f"no transfers, {a.state.free_transfers} free kept"
    if dry_run:
        return f"FPL GW{gw}: autopilot rehearsal", f"Would do: {moves}, and set captain and bench.", False
    if not orders.updated:
        return None
    return f"FPL GW{gw}: team updated", f"Done: {moves}. Captain and bench set.", False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    g = sub.add_parser("gate", help="is an alert due now?")
    g.add_argument("--mode", default="auto", choices=["auto", "check", "dry-run", "test-send"])
    g.add_argument("--topic", default=None, help="ntfy topic")
    r = sub.add_parser("run", help="make the advice and send the message")
    r.add_argument("--stage", default="early", choices=["early", "late"])
    r.add_argument("--team", type=int, default=TEAM_ID, help="FPL team ID")
    r.add_argument("--dry-run", action="store_true", help="print only; send and save nothing")
    r.add_argument("--test", action="store_true", help="send now, titled TEST; save nothing")
    r.add_argument("--topic", default=None, help="ntfy topic")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if args.command == "gate":
        gate(args.mode or "auto", topic_override=args.topic)
    else:
        run(args.stage, require_team(args.team), args.dry_run, args.test, topic_override=args.topic)


if __name__ == "__main__":
    main()
