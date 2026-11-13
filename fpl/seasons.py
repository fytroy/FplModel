"""Which seasons the model uses, and helpers for season labels like '2025-26'."""

# Past seasons come from the vaastav GitHub repo. FPL's own xG starts in 2022/23.
PAST_SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]
# The current season comes from our own FPL API pull (pull_data.py).
CURRENT_SEASON = "2026-27"


def previous_season(season: str) -> str:
    """'2026-27' -> '2025-26'."""
    year = int(season[:4]) - 1
    return f"{year}-{str(year + 1)[-2:]}"
