"""Gameweek tracker: knows and analyzes the points gotten by your team at the end of each gameweek.

Shows gameweek-by-gameweek scores, transfer hits, rank, points left on the bench,
player-by-player score breakdown, captain returns, auto-subs, and comparisons to the Gameweek Dream Team.
"""
from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd

from fpl import api, rules, storage
from fpl.matches import POSITION_BY_TYPE


@dataclass
class GameweekSummary:
    gw: int
    points: int             # gross points
    hits: int               # transfer costs (points deducted)
    net_points: int         # points - hits
    total_points: int       # overall total
    gw_rank: int | None
    overall_rank: int | None
    points_on_bench: int
    chip: str | None
    transfers: int
    bank: int


@dataclass
class PlayerPickScore:
    element: int
    code: int
    name: str
    team: str
    position: str
    points: int
    multiplier: int
    effective_points: int
    is_captain: bool
    is_vice_captain: bool
    is_starter: bool
    bench_order: int | None = None


@dataclass
class GameweekBreakdown:
    gw: int
    summary: GameweekSummary | None
    starters: list[PlayerPickScore]
    bench: list[PlayerPickScore]
    captain: PlayerPickScore | None
    vice_captain: PlayerPickScore | None
    auto_subs: list[dict]
    chip: str | None
    dream_team_points: float | None = None


@dataclass
class TeamSeasonPerformance:
    team_id: int
    name: str
    gameweeks: list[GameweekSummary]
    total_points: int
    overall_rank: int | None
    avg_points: float
    best_gw: tuple[int, int]       # (gw, net_points)
    worst_gw: tuple[int, int]      # (gw, net_points)
    total_hits_cost: int
    total_bench_points: int
    chips_played: list[tuple[str, int]]


def parse_team_performance(team_id: int, entry_data: dict, history_data: dict) -> TeamSeasonPerformance:
    """Parse raw entry and history responses into structured TeamSeasonPerformance."""
    current = history_data.get("current", [])
    chips_played = [(c["name"], int(c["event"])) for c in history_data.get("chips", [])]
    chip_map = dict((gw, name) for name, gw in chips_played)

    summaries = []
    for row in current:
        gw = int(row["event"])
        pts = int(row.get("points", 0))
        hits = int(row.get("event_transfers_cost", 0))
        net = pts - hits
        tot = int(row.get("total_points", 0))
        gw_rank = row.get("rank")
        ov_rank = row.get("overall_rank")
        bench_pts = int(row.get("points_on_bench", 0))
        chip = chip_map.get(gw)
        transfers = int(row.get("event_transfers", 0))
        bank = int(row.get("bank", 0))

        summaries.append(
            GameweekSummary(
                gw=gw,
                points=pts,
                hits=hits,
                net_points=net,
                total_points=tot,
                gw_rank=int(gw_rank) if gw_rank is not None else None,
                overall_rank=int(ov_rank) if ov_rank is not None else None,
                points_on_bench=bench_pts,
                chip=chip,
                transfers=transfers,
                bank=bank,
            )
        )

    if summaries:
        best_gw = max(((s.gw, s.net_points) for s in summaries), key=lambda x: x[1])
        worst_gw = min(((s.gw, s.net_points) for s in summaries), key=lambda x: x[1])
        avg_pts = round(sum(s.net_points for s in summaries) / len(summaries), 1)
        tot_hits = sum(s.hits for s in summaries)
        tot_bench = sum(s.points_on_bench for s in summaries)
        latest_tot = summaries[-1].total_points
        latest_rank = summaries[-1].overall_rank
    else:
        best_gw = (0, 0)
        worst_gw = (0, 0)
        avg_pts = 0.0
        tot_hits = 0
        tot_bench = 0
        latest_tot = entry_data.get("summary_overall_points", 0)
        latest_rank = entry_data.get("summary_overall_rank")

    name = entry_data.get("name", f"Team {team_id}")

    return TeamSeasonPerformance(
        team_id=team_id,
        name=name,
        gameweeks=summaries,
        total_points=latest_tot,
        overall_rank=latest_rank,
        avg_points=avg_pts,
        best_gw=best_gw,
        worst_gw=worst_gw,
        total_hits_cost=tot_hits,
        total_bench_points=tot_bench,
        chips_played=chips_played,
    )


def resolve_player_points_for_gw(gw: int, player_history: pd.DataFrame | None = None,
                                 live_data: dict | None = None) -> dict[int, int]:
    """Map element_id -> actual points scored in gameweek gw."""
    res = {}
    if live_data and "elements" in live_data:
        for item in live_data["elements"]:
            el = int(item["id"])
            stats = item.get("stats", {})
            res[el] = int(stats.get("total_points", 0))
        return res

    if player_history is not None and not player_history.empty:
        gw_rows = player_history[player_history["round"] == gw]
        if not gw_rows.empty:
            grouped = gw_rows.groupby("element")["total_points"].sum()
            for el, pts in grouped.items():
                res[int(el)] = int(pts)
    return res


def parse_gameweek_breakdown(gw: int, picks_data: dict, players: pd.DataFrame,
                             teams: pd.DataFrame, player_history: pd.DataFrame | None = None,
                             live_data: dict | None = None,
                             summary: GameweekSummary | None = None,
                             dream_team_points: float | None = None) -> GameweekBreakdown:
    """Parse picks for one gameweek with individual player points and returns."""
    short = dict(zip(teams["id"], teams["short_name"]))
    p_by_id = players.set_index("id")

    pts_map = resolve_player_points_for_gw(gw, player_history=player_history, live_data=live_data)

    picks = picks_data.get("picks", [])
    chip = picks_data.get("active_chip")
    auto_subs = picks_data.get("automatic_subs", [])

    starters = []
    bench = []
    captain = None
    vice = None

    for p in picks:
        el = int(p["element"])
        multiplier = int(p.get("multiplier", 1))
        is_cap = bool(p.get("is_captain", False))
        is_vc = bool(p.get("is_vice_captain", False))
        pos_num = int(p.get("position", 1))

        if el in p_by_id.index:
            row = p_by_id.loc[el]
            name = str(row["web_name"])
            code = int(row["code"])
            team_short = short.get(int(row["team"]), "UNK")
            pos_str = POSITION_BY_TYPE.get(int(row["element_type"]), "MID")
        else:
            name = f"Player {el}"
            code = el
            team_short = "UNK"
            pos_str = "MID"

        pts = pts_map.get(el, 0)
        eff_pts = pts * multiplier
        is_starter = (pos_num <= 11)
        bench_ord = (pos_num - 11) if not is_starter else None

        item = PlayerPickScore(
            element=el,
            code=code,
            name=name,
            team=team_short,
            position=pos_str,
            points=pts,
            multiplier=multiplier,
            effective_points=eff_pts,
            is_captain=is_cap,
            is_vice_captain=is_vc,
            is_starter=is_starter,
            bench_order=bench_ord,
        )

        if is_cap:
            captain = item
        if is_vc:
            vice = item

        if is_starter:
            starters.append(item)
        else:
            bench.append(item)

    return GameweekBreakdown(
        gw=gw,
        summary=summary,
        starters=starters,
        bench=bench,
        captain=captain,
        vice_captain=vice,
        auto_subs=auto_subs,
        chip=chip,
        dream_team_points=dream_team_points,
    )


def render_history_table(perf: TeamSeasonPerformance) -> str:
    """Render full season gameweek table with points, hits, rank and bench."""
    lines = [
        "=" * 78,
        f"       SEASON HISTORY & GAMEWEEK PERFORMANCE: {perf.name.upper()} ({perf.team_id})",
        "=" * 78,
        f"{'GW':<4} {'Points':<8} {'Hits':<6} {'Net':<8} {'Overall Rank':<15} {'GW Rank':<12} {'Bench Pts':<10} {'Chip':<10}",
        "-" * 78,
    ]

    for s in perf.gameweeks:
        chip_str = rules.CHIP_LABELS.get(s.chip, s.chip) if s.chip else "-"
        hits_str = f"-{s.hits}" if s.hits else "0"
        ov_rank_str = f"{s.overall_rank:,}" if s.overall_rank else "-"
        gw_rank_str = f"{s.gw_rank:,}" if s.gw_rank else "-"

        lines.append(
            f"GW{s.gw:<2} {s.points:<8} {hits_str:<6} {s.net_points:<8} {ov_rank_str:<15} {gw_rank_str:<12} {s.points_on_bench:<10} {chip_str:<10}"
        )

    ov_rank_display = f"{perf.overall_rank:,}" if perf.overall_rank else "N/A"
    lines += [
        "-" * 78,
        f"TOTAL POINTS: {perf.total_points} | CURRENT OVERALL RANK: {ov_rank_display}",
        f"AVERAGE / GW: {perf.avg_points:.1f} pts | BEST GW: GW{perf.best_gw[0]} ({perf.best_gw[1]} pts) | WORST GW: GW{perf.worst_gw[0]} ({perf.worst_gw[1]} pts)",
        f"TOTAL HITS: -{perf.total_hits_cost} pts | BENCH POINTS UNPLAYED: {perf.total_bench_points} pts",
        "=" * 78,
    ]
    return "\n".join(lines)


def render_gameweek_breakdown(bd: GameweekBreakdown) -> str:
    """Render detailed starting XI and bench breakdown for one gameweek."""
    lines = [
        "=" * 70,
        f"                   GAMEWEEK {bd.gw} DETAILED SQUAD BREAKDOWN",
        "=" * 70,
    ]

    if bd.summary:
        s = bd.summary
        chip_label = rules.CHIP_LABELS.get(bd.chip, bd.chip) if bd.chip else "None"
        gw_rk_str = f"{s.gw_rank:,}" if s.gw_rank else "N/A"
        ov_rk_str = f"{s.overall_rank:,}" if s.overall_rank else "N/A"
        lines += [
            f"Gross Score: {s.points} pts | Hits: -{s.hits} pts | Net Score: {s.net_points} pts",
            f"GW Rank: {gw_rk_str} | Overall Rank: {ov_rk_str} | Chip: {chip_label}",
        ]
        if bd.dream_team_points:
            diff = bd.dream_team_points - s.net_points
            lines.append(f"Dream Team: {bd.dream_team_points:.1f} pts (Difference: -{diff:.1f} pts)")
        lines.append("-" * 70)

    lines.append("STARTING XI:")
    tot_starter_pts = 0
    for p in bd.starters:
        role = ""
        if p.is_captain:
            role = " (C) "
        elif p.is_vice_captain:
            role = " (V) "
        else:
            role = "     "

        pts_display = f"{p.effective_points} pts"
        if p.multiplier > 1:
            pts_display += f" ({p.multiplier}x {p.points})"

        lines.append(f"  {p.position:<4} {p.name:<20} ({p.team:<3}){role}  {pts_display:>16}")
        tot_starter_pts += p.effective_points

    lines += [
        f"  {'Total Starting XI Points:':<32} {tot_starter_pts:>16} pts",
        "",
        "BENCH:",
    ]

    tot_bench_pts = 0
    for p in bd.bench:
        prefix = "GK  " if p.bench_order == 1 and p.position == "GK" else f"Sub{p.bench_order}"
        lines.append(f"  {prefix:<4} {p.name:<20} ({p.team:<3})        {p.points:>6} pts")
        tot_bench_pts += p.points

    lines.append(f"  {'Total Bench Points Left:':<32} {tot_bench_pts:>14} pts")

    if bd.auto_subs:
        lines += ["", "AUTOMATIC SUBSTITUTIONS:"]
        for sub in bd.auto_subs:
            lines.append(f"  Sub In: Element {sub.get('element_in')} for Element {sub.get('element_out')}")

    lines.append("=" * 70)
    return "\n".join(lines)


def save_team_history_cache(team_id: int, entry: dict, history: dict) -> Path:
    """Cache team entry and history data to data/processed for offline access."""
    folder = storage.DATA_DIR / "processed"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"team_{team_id}_history.json"
    path.write_text(json.dumps({"entry": entry, "history": history}, indent=2), encoding="utf-8")
    return path


def load_cached_team_history(team_id: int) -> tuple[dict, dict] | None:
    """Load cached team entry and history data if available."""
    path = storage.DATA_DIR / "processed" / f"team_{team_id}_history.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("entry", {}), data.get("history", {})
    return None
