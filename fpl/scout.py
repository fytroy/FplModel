"""Upcoming gameweek scout: checks and gives top players for the upcoming week.

Provides ranked recommendations by position (GK, DEF, MID, FWD), top captaincy candidates,
budget value enablers, and differentials for the upcoming gameweek.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from fpl import rules
from fpl.matches import POSITION_BY_TYPE


@dataclass
class ScoutPlayer:
    code: int
    name: str
    position: str
    team: str
    price: int          # tenths of £1m
    next_xp: float      # expected points next week
    horizon_xp: float   # average xP over horizon
    fixture: str        # e.g. "SOU (H)"
    difficulty: int     # FDR 1-5
    ownership: float    # selected_by_percent
    status: str         # 'a', 'd', 'i', etc.
    news: str           # injury / suspension news


@dataclass
class CaptainPick:
    player: ScoutPlayer
    rank: int
    reason: str


@dataclass
class ScoutReport:
    gw: int
    top_by_position: dict[str, list[ScoutPlayer]]   # 'GK', 'DEF', 'MID', 'FWD'
    captains: list[CaptainPick]
    value_picks: list[ScoutPlayer]
    differentials: list[ScoutPlayer]


def money(tenths: int) -> str:
    return f"£{tenths / 10:.1f}m"


def build_scout_report(players: pd.DataFrame, teams: pd.DataFrame,
                       xp: pd.DataFrame, gw: int,
                       fixtures_all: pd.DataFrame | None = None,
                       avail: pd.DataFrame | None = None,
                       top_n: int = 5) -> ScoutReport:
    """Generate comprehensive scout recommendations for gameweek gw."""
    short = dict(zip(teams["id"], teams["short_name"]))
    p = players[players["status"] != "u"].copy()

    # Match player code to element / details
    p["code"] = p["code"].astype(int)
    p["price"] = p["now_cost"].astype(int)
    p["pos"] = p["element_type"].map(POSITION_BY_TYPE)
    p["club"] = p["team"].map(short)
    p["selected"] = pd.to_numeric(p["selected_by_percent"], errors="coerce").fillna(0.0)

    # Fixtures lookup for this gameweek
    fix_map = {}
    fdr_map = {}
    if fixtures_all is not None and not fixtures_all.empty:
        gw_col_name = "gw" if "gw" in fixtures_all.columns else ("event" if "event" in fixtures_all.columns else None)
        if gw_col_name:
            gw_fix = fixtures_all[fixtures_all[gw_col_name] == gw]
            for _, row in gw_fix.iterrows():
                if "team" in row and "opponent" in row:
                    club_name = str(row["team"])
                    opp_name = str(row["opponent"])
                    venue = "H" if bool(row.get("was_home", True)) else "A"
                    fix_map[club_name] = f"{opp_name} ({venue})"
                    fdr_map[club_name] = int(row.get("difficulty", 3))
                elif "team_h" in row and "team_a" in row:
                    h_id = int(row["team_h"])
                    a_id = int(row["team_a"])
                    h_club = short.get(h_id, str(h_id))
                    a_club = short.get(a_id, str(a_id))
                    h_diff = int(row.get("team_h_difficulty", 3))
                    a_diff = int(row.get("team_a_difficulty", 3))
                    fix_map[h_club] = f"{a_club} (H)"
                    fdr_map[h_club] = h_diff
                    fix_map[a_club] = f"{h_club} (A)"
                    fdr_map[a_club] = a_diff

    # News / availability lookup
    news_map = {}
    if avail is not None and not avail.empty:
        for _, row in avail.iterrows():
            c = int(row.get("player_code", 0))
            n = row.get("news")
            if c and pd.notna(n):
                news_map[c] = str(n)

    # XP series
    gw_col = gw if gw in xp.columns else (str(gw) if str(gw) in xp.columns else xp.columns[0])
    next_pts = xp[gw_col].fillna(0.0)
    horizon_avg = xp.mean(axis=1).fillna(0.0)

    scout_list = []
    for _, row in p.iterrows():
        c = int(row["code"])
        if c not in next_pts.index:
            continue
        pt = float(next_pts.loc[c])
        avg_pt = float(horizon_avg.loc[c])
        club_short = str(row["club"])
        fixture_str = fix_map.get(club_short, "-")
        diff = fdr_map.get(club_short, 3)
        news_text = news_map.get(c, str(row.get("news", "")) if pd.notna(row.get("news")) else "")

        scout_list.append(
            ScoutPlayer(
                code=c,
                name=str(row["web_name"]),
                position=str(row["pos"]),
                team=club_short,
                price=int(row["price"]),
                next_xp=round(pt, 2),
                horizon_xp=round(avg_pt, 2),
                fixture=fixture_str,
                difficulty=diff,
                ownership=float(row["selected"]),
                status=str(row.get("status", "a")),
                news=news_text,
            )
        )

    # Group by position
    by_pos = {}
    for pos_name in ("GK", "DEF", "MID", "FWD"):
        pos_players = [p for p in scout_list if p.position == pos_name]
        pos_sorted = sorted(pos_players, key=lambda x: -x.next_xp)[:top_n]
        by_pos[pos_name] = pos_sorted

    # Top captain picks (attackers / high xP players)
    cap_candidates = sorted([p for p in scout_list if p.position in ("MID", "FWD")], key=lambda x: -x.next_xp)[:5]
    captains = []
    for rank, cp in enumerate(cap_candidates, 1):
        reason = f"Top projection ({cp.next_xp:.1f} xP) vs {cp.fixture}"
        if cp.difficulty <= 2:
            reason += " (Favorable fixture)"
        captains.append(CaptainPick(player=cp, rank=rank, reason=reason))

    # Value enablers (high xP per £m, price <= £6.0m)
    budget_players = [p for p in scout_list if p.price <= 60 and p.next_xp >= 3.0]
    value_picks = sorted(budget_players, key=lambda x: -(x.next_xp / (x.price / 10.0)))[:top_n]

    # Differentials (ownership < 10.0%, xP >= 4.0)
    diff_players = [p for p in scout_list if p.ownership < 10.0 and p.next_xp >= 3.5]
    differentials = sorted(diff_players, key=lambda x: -x.next_xp)[:top_n]

    return ScoutReport(
        gw=gw,
        top_by_position=by_pos,
        captains=captains,
        value_picks=value_picks,
        differentials=differentials,
    )


def render_scout_report(report: ScoutReport) -> str:
    """Render plain-text scout report for the upcoming gameweek."""
    lines = [
        "=" * 76,
        f"               GAMEWEEK {report.gw} PLAYER SCOUT & RECOMMENDATIONS",
        "=" * 76,
    ]

    # Captaincy Section
    lines += ["CAPTAINCY PICKS:", "-" * 76]
    for cp in report.captains:
        p = cp.player
        lines.append(
            f"  #{cp.rank} {p.name:<18} ({p.team:<3}) {p.position:<4} {money(p.price):>6} | "
            f"Next: {p.next_xp:>4.1f} xP (2x={p.next_xp*2:>4.1f}) | {p.fixture:<10} | {cp.reason}"
        )
    lines.append("")

    # Position Breakdown
    for pos in ("GK", "DEF", "MID", "FWD"):
        pos_label = {"GK": "GOALKEEPERS", "DEF": "DEFENDERS", "MID": "MIDFIELDERS", "FWD": "FORWARDS"}[pos]
        lines += [f"TOP {pos_label}:", "-" * 76]
        lines.append(f"  {'Player':<20} {'Club':<6} {'Price':<8} {'Next xP':<10} {'3-GW Avg':<10} {'Fixture':<10} {'Status':<10}")
        for p in report.top_by_position.get(pos, []):
            news_tag = f" [{p.news[:18]}...]" if p.news else ""
            status_tag = "OK" if p.status == "a" else f"WARN{news_tag}"
            lines.append(
                f"  {p.name:<20} {p.team:<6} {money(p.price):<8} {p.next_xp:<10.1f} {p.horizon_xp:<10.1f} {p.fixture:<10} {status_tag:<10}"
            )
        lines.append("")

    # Value Picks
    if report.value_picks:
        lines += ["BUDGET & VALUE ENABLERS (High xP / £m, Price <= £6.0m):", "-" * 76]
        for p in report.value_picks:
            ppm = p.next_xp / (p.price / 10.0)
            lines.append(
                f"  {p.name:<20} ({p.team:<3}) {p.position:<4} {money(p.price):>6} | "
                f"Next: {p.next_xp:>4.1f} xP | Value: {ppm:>4.2f} pts/£m | {p.fixture}"
            )
        lines.append("")

    # Differentials
    if report.differentials:
        lines += ["TOP DIFFERENTIALS (Ownership < 10%):", "-" * 76]
        for p in report.differentials:
            lines.append(
                f"  {p.name:<20} ({p.team:<3}) {p.position:<4} {money(p.price):>6} | "
                f"Ownership: {p.ownership:>4.1f}% | Next: {p.next_xp:>4.1f} xP | {p.fixture}"
            )
        lines.append("")

    lines.append("=" * 76)
    return "\n".join(lines)
