import pandas as pd
import pytest

from fpl import reminder


def test_deadline_status():
    deadline = pd.Timestamp("2026-10-31 11:00:00", tz="UTC")

    # 48 hours before
    now_early = pd.Timestamp("2026-10-29 11:00:00", tz="UTC")
    st = reminder.compute_deadline_status(deadline, now_utc=now_early)
    assert st.urgency == "UPCOMING"
    assert "2d 0h" in st.time_str

    # 12 hours before
    now_warn = pd.Timestamp("2026-10-30 23:00:00", tz="UTC")
    st = reminder.compute_deadline_status(deadline, now_utc=now_warn)
    assert st.urgency == "WARNING (<24h)"

    # 1 hour before
    now_urgent = pd.Timestamp("2026-10-31 10:00:00", tz="UTC")
    st = reminder.compute_deadline_status(deadline, now_utc=now_urgent)
    assert st.urgency == "URGENT (<3h)"

    # After deadline
    now_passed = pd.Timestamp("2026-10-31 12:00:00", tz="UTC")
    st = reminder.compute_deadline_status(deadline, now_utc=now_passed)
    assert st.urgency == "PASSED"


def test_render_switch_reminder():
    deadline_status = reminder.DeadlineStatus(
        deadline_utc=pd.Timestamp("2026-10-31 11:00:00", tz="UTC"),
        deadline_local="Sat 31 Oct 14:00",
        hours_remaining=2.5,
        time_str="2 hours 30 minutes remaining!",
        urgency="URGENT (<3h)"
    )

    cap_switch = reminder.CaptainSwitch(
        current_captain="Palmer (CHE)",
        recommended_captain="Haaland (MCI)",
        current_xp=6.5,
        recommended_xp=9.2,
        gain=2.7,
        needs_switch=True,
        recommended_vice="Salah (LIV)"
    )

    lineup = reminder.LineupSwitch(
        move_to_starters=["Rogers (AVL)"],
        move_to_bench=["Winks (LEI)"],
        bench_order=["Fabianski (WHU)", "Winks (LEI)", "Bednarek (SOU)", "Stewart (SOU)"]
    )

    transfers = [
        reminder.TransferSwitch(
            player_out="Wood (NFO)",
            player_in="Isak (LIV)",
            out_price=60,
            in_price=85,
            out_xp=3.5,
            in_xp=6.8,
            is_hit=False
        )
    ]

    rem = reminder.SwitchReminder(
        team_id=12345,
        team_name="Test FC",
        gw=9,
        deadline=deadline_status,
        transfers=transfers,
        hit_cost=0,
        captain_switch=cap_switch,
        lineup_switch=lineup,
        chip_advice=None,
        current_squad_xp=55.0,
        recommended_squad_xp=62.5,
        xp_gain=7.5,
        dream_team_xp=78.0
    )

    rendered = reminder.render_switch_reminder(rem)
    assert "FPL GW9 TEAM SWITCH REMINDER" in rendered
    assert "URGENT (<3h)" in rendered
    assert "SWITCH CAPTAIN: Palmer (CHE) -> Haaland (MCI)" in rendered
    assert "PROMOTE TO STARTERS: Rogers (AVL)" in rendered
    assert "MOVE TO BENCH:       Winks (LEI)" in rendered
    assert "Wood (NFO)" in rendered
    assert "Isak (LIV)" in rendered
    assert "Optimal Dream Team:   78.0 pts" in rendered
