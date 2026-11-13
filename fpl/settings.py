"""Every tunable number for the squad planner, in one place."""
import os
from dataclasses import dataclass
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def _load_env_file() -> None:
    """Local secrets and settings from .env (never committed); real env vars win."""
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        if sep and not key.strip().startswith("#"):
            os.environ.setdefault(key.strip(), value.strip())


_load_env_file()

TEAM_ID = int(os.environ.get("FPL_TEAM_ID") or 0)   # your team: set FPL_TEAM_ID in .env or GitHub secrets
TIMEZONE = "Africa/Nairobi"
NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "")       # anyone with the name can read it: never a shared default
AUTOPILOT_MIN_MINUTES = 15          # never touch the team closer to the deadline than this
AUTOPILOT_MAX_TRANSFERS = 2         # most transfers made in one week (all free; a bigger plan is left to you)
AUTOPILOT_POLICY = os.environ.get("FPL_AUTOPILOT_POLICY", "safe").strip().lower()
MESSAGE_FREQUENCY = os.environ.get("FPL_MESSAGE_FREQUENCY", "early_and_late").strip().lower()
HORIZON = 6                         # weeks planned (recommend.py --horizon 4-6)
SOLVER_TIME_LIMIT = 180             # seconds per solve in recommend.py; run weekly, so a proven plan is worth the wait
SIM_TIME_LIMIT = 180                # seconds per solve in the season simulation (60 s left GW1-19 plans unproven)
SOLVER_GAP = 0.001                  # a plan within 0.1% of the best possible counts as optimal
DATA_MAX_AGE_HOURS = 6              # recommend.py refreshes FPL data if older
FIRST_WEEK_BACK = 0.8               # availability in a player's first week back from a dated absence
CHIP_SCAN_WEEKS = 13                # weeks predicted ahead (plan + chip scan); GW7 + 12 = GW19
CANDIDATES_PER_POSITION = 40        # planner pool: best by discounted xP...
VALUE_CANDIDATES_PER_POSITION = 15  # ...plus best xP per £m...
CHEAPEST_PER_POSITION = 5           # ...plus the cheapest, so a valid squad always fits the budget
RATING_HALF_LIFE_DAYS = 180         # team ratings: a match's weight halves every 180 days
# Deadline alerts (alert.py, run hourly on GitHub Actions): hours before the deadline, (from, until).
EARLY_ALERT_HOURS = (24.5, 20.0)    # the first message; later hourly runs retry if one fails
LATE_ALERT_HOURS = (3.5, 1.0)       # the re-check after press conferences; sent only if advice changed


def require_team(team_id: int) -> int:
    """The team to work on, or a clear stop if none has been set."""
    if not team_id:
        raise SystemExit("No FPL team set: add FPL_TEAM_ID=<your id> to .env, or pass --team <id>.")
    return team_id


@dataclass(frozen=True)
class PlanSettings:
    """The planner's judgement calls; simulate.py tests a few values of each."""
    discount: float = 0.85          # weight of plan week k is discount ** (k - 1); 0.85 beat 0.9 in both
                                    # 2025/26 runs at 180 s (+35, +92); see spec §7
    hit_margin: float = 1.0         # a hit must gain more than 4 + this (after discounting)
    ft_end_value: float = 1.5       # points per free transfer still saved after the last plan week
    bench_weight: float = 0.1       # share of bench players' xP counted (value of cover)
    horizon: int = HORIZON
