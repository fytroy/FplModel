# Part 3: Squad Optimizer and Transfer Planner — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `python recommend.py` prints plain-English transfer, captain, bench and chip advice for FPL team 5234632 over a 6-week plan. `python simulate.py` replays 2025/26 with the whole system and compares the result with the average manager (1,895), the user's 2,061 and an estimated overall rank.

**Architecture:**
- Pure modules for the FPL rules (`rules`), the user's team state (`myteam`) and injury news (`availability`).
- Team ratings for the clean-sheet fix (`team_ratings`), wired into the existing part 2 LightGBM models.
- A PuLP integer program plans 6 weeks at once (`optimizer`), with a chip scan for weeks beyond the plan (`chips`).
- Front ends: `recommend.py`, which uses `forecast` and `advice`, and `simulate.py`, which uses `simulate` and `benchmarks`.
- All internet access stays in `fpl/api.py`, and all disk access in `fpl/storage.py`.

**Tech Stack:** Python 3.13 (system interpreter), pandas, numpy, LightGBM, scikit-learn (PoissonRegressor), PuLP 2.x with its bundled CBC solver, tzdata/zoneinfo, pytest.

**Spec:** `docs/superpowers/specs/2026-10-11-squad-planner-design.md` (approved 2026-10-11; Task 0 applies the user's one fix plus the clarifications listed there).

## Context

Parts 1 (FPL API pull) and 2 (xP model, walk-forward backtest) are merged to `master`. Part 3 turns the predictions into decisions for the user's real team:
- this week's transfers, or saving the free transfer;
- whether a −4 hit is worth it;
- captain, vice-captain and bench order;
- a suggested plan for the following weeks;
- chip timing. The first chip set ends at GW19 (`stop_event: 19` in FPL's data).

It also fixes two part 2 gaps:
- **Injury flags.** At the moment they only affect next week. The user has three players flagged red (Isak, Tzolis, McNally) and one yellow (Robinson).
- **Clean sheets.** Predicted chances only span 17–33% between easy and hard fixtures, while real rates differ about twice as much.

The season simulation shows whether the system actually beats real managers, and lists both its optimistic and pessimistic biases.

## Global Constraints

- Run everything with `python` (system Python 3.13, where parts 1–2 were built and all 61 tests pass). The `.venv` folder is stale (it has no lightgbm); don't use it.
- Prices inside the code are FPL units (tenths of £1m: 55 = £5.5m). Divide by 10 only when displaying.
- Deadlines and chip windows always come from FPL's API fields (`events[].deadline_time`, `chips[].start_event/stop_event`), never from news sites. First-set chips can be played up to and including GW19.
- Every printed time is in `Africa/Nairobi` (UTC+3). GW7 `2026-10-17T10:00:00Z` must print as `Sat 17 Oct 13:00`.
- Only `fpl/api.py` uses the internet, and only `fpl/storage.py` reads or writes `data/`. Tests run offline, using small files in `tests/data/`.
- Advice only: never log in, never make transfers.
- Free transfers:
  - one per gameweek, up to 5 saved;
  - each extra transfer costs 4 points;
  - after a Wildcard or Free Hit you keep the count but get no +1 that week;
  - 2025/26 only: topped up to 5 in GW16 (AFCON, premierleague.com/en/news/4362102).
- Selling price = purchase + floor((now − purchase)/2) on a rise, = now on a fall.
- Settings defaults: discount 0.9, hit margin 1.0, end-of-plan free-transfer value 1.5, bench weight 0.1, horizon 6, first week back 0.8, solver limit 60 s.
- All existing tests keep passing. Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **After every commit, run `git push`** (branch `part3-squad-planner`, remote `origin` = github.com/savirmediratta123/fpl-model, private).

## Review Focus

1. **A squad player with no predictions** (McNally: status `u`, loaned out; `upcoming_player_rows` drops him). He must get 0 xP every week, stay sellable at his selling price, and never start. Tests: `test_optimizer.py::test_a_player_with_no_predictions_is_benched_or_sold` and `test_recommend.py`.
2. **Transfers already made for the next gameweek** before the user runs the tool. Squad, bank and free transfers must reflect them. Test: `test_myteam.py::test_pending_transfers_are_applied`.
3. **Double and blank gameweeks in multi-week xP.** Fixtures are summed per gameweek, and a blank gives 0 (not missing). Test: `test_forecast.py::test_weekly_xp_sums_doubles_and_fills_blanks`.
4. **No valid plan** (budget impossible) or solver failure. The user should get a clear one-line error, not a stack trace. Test: `test_optimizer.py::test_impossible_budget_raises_a_clear_error`.
5. **Odd news text** (an invalid date such as "Expected back 31 Feb", or no date). It must fall back to the chance/unknown-return rules, not crash. Test: `test_availability.py::test_return_date_handles_odd_text`.

## File Structure

| File | Responsibility |
|---|---|
| `fpl/settings.py` (new) | Every tunable number; `PlanSettings` dataclass |
| `fpl/rules.py` (new) | Selling price, free-transfer update, chip windows, one-gameweek scoring with auto-subs |
| `fpl/api.py` (extend) | Entry endpoints, vaastav file at a commit, Internet Archive index and fetch |
| `fpl/storage.py` (extend) | Benchmarks, fixture snapshots, table age |
| `fpl/myteam.py` (new) | `TeamState` from public API responses |
| `fpl/availability.py` (new) | News → availability per gameweek; recovery curve; trimming missed matches |
| `fpl/team_ratings.py` (new) | Time-weighted Poisson attack/defence ratings |
| `fpl/features.py`, `fpl/models.py` (modify) | Rating features for the clean-sheet and goals-conceded models; `yardstick` flag; `weekly_xp`; remove `apply_availability` |
| `fpl/backtest.py`, `backtest.py` (modify) | `clean_sheet_report` |
| `fpl/forecast.py` (new) | Live per-fixture predictions with availability (shared by `predict.py` and `recommend.py`) |
| `fpl/optimizer.py` (new) | `PlanInput`, `WeekPlan`, `Plan`, `solve`, `best_squad`, `candidate_pool` |
| `fpl/chips.py` (new) | `best_xi`, `scan`, `hold_values` |
| `fpl/advice.py` (new) | Plain-English rendering, `plan_table`, `local_time`, `money` |
| `fpl/benchmarks.py` (new) | Average manager, points-to-rank estimate |
| `fpl/simulate.py` (new) | Season data, prediction cache builder, `run_season`, `summarize` |
| `pull_data.py` (modify) | Expose `pull(skip_history)` for `recommend.py` |
| `pull_benchmarks.py` (new) | One-off download: archived 2025/26 bootstrap, rank sample, fixture snapshots |
| `recommend.py` (new) | The user's command |
| `simulate.py` (new) | Season simulation and settings test |
| `predict.py` (modify) | Use `forecast.live_forecast` |
| `tests/data/…` (new) | Saved team 5234632 responses, player/gameweek subsets, sampled manager histories |

---

### Task 0: Spec fix, plan document, dependencies

**Files:**
- Modify: `docs/superpowers/specs/2026-10-11-squad-planner-design.md`
- Create: `docs/superpowers/plans/2026-10-11-squad-planner.md` (a copy of this plan)
- Modify: `requirements.txt`

- [ ] **Step 1: Apply the user's fix in spec §1.** Replace the bullet starting `- in the GW1 squad and never re-bought:` and its continuation lines with:

```markdown
  - in the team's starting squad (picks for the first gameweek in
    `history.current`, so managers who joined after GW1 work too) and never
    re-bought: his price at that gameweek, from part 1's `player_history`
    (`value` at that round, or at his next match if his team had none),
    else his current price (flagged as unknown).
```

- [ ] **Step 2: Record the clarifications made while planning (spec).** Make these edits:
  - In §5, replace the Wildcard bullet with:

    ```markdown
    - **Wildcard:** gain = the best fixed squad's xP over the 6 weeks from that
      week minus the current squad's; the free transfers you'd get anyway
      could make some of those changes, so value = gain × max(0, k − F) / k,
      where k = players changed and F = weeks in the window (one free
      transfer a week). A rough estimate.
    ```

  - In §5, after "It uses the squad at the end of the plan:", add: "(The hold values fed to the planner use the current squad, because the plan isn't known yet. The printed chip advice re-scans with the plan's final squad.)"
  - In §7 under **2025/26 rules**, replace the AFCON bullet with: "the AFCON top-up: free transfers set to 5 in GW16 (premierleague.com/en/news/4362102); chip windows in the archived 2025/26 data match this season's (set 1: GW1/2–19, set 2: GW20–38)."
  - In **Files**, add these rows:
    - `fpl/forecast.py`: live predictions with availability, shared by `predict.py` and `recommend.py`;
    - `fpl/advice.py`: plain-English rendering for `recommend.py`;
    - `fpl/settings.py` already listed.

- [ ] **Step 3: Copy this plan** to `docs/superpowers/plans/2026-10-11-squad-planner.md`.

- [ ] **Step 4: Add dependencies.** Append to `requirements.txt`:

```
pulp>=2.8,<3
tzdata
```

Run: `python -m pip install "pulp>=2.8,<3" tzdata`
Then: `python -c "import pulp; print(pulp.__version__, pulp.PULP_CBC_CMD(msg=False).available())"`
Expected: a 2.x version and `True`.

- [ ] **Step 5: Run the existing tests.** Run `python -m pytest -q`. Expected: `61 passed`.

- [ ] **Step 6: Commit and push**

```bash
git add docs/superpowers requirements.txt
git commit -m "Spec fix for late joiners; add part 3 plan and dependencies

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push
```

---

### Task 1: API and storage extensions

**Files:**
- Modify: `fpl/api.py`, `fpl/storage.py`
- Test: `tests/test_api.py` (new), `tests/test_storage.py`

**Interfaces:**
- Produces in `fpl/api.py`:
  - `fetch_entry(team_id) -> dict`
  - `fetch_entry_history(team_id) -> dict`
  - `fetch_entry_transfers(team_id) -> list`
  - `fetch_entry_picks(team_id, gw) -> dict`
  - `fetch_vaastav_commits(path) -> list[tuple[str, str]]`, giving (sha, ISO date), newest first
  - `fetch_vaastav_file_at(sha, path) -> bytes`
  - `fetch_wayback_index(url, start, end) -> list[str]` (timestamps)
  - `fetch_wayback(url, timestamp) -> bytes`
  - `maybe_gunzip(data: bytes) -> bytes`
- Produces in `fpl/storage.py`:
  - `save_benchmark(name, content: bytes) -> Path`
  - `benchmark_path(name) -> Path`
  - `load_benchmark_json(name) -> dict`
  - `save_snapshot(season, stamp, source, table) -> Path`
  - `load_snapshots(season) -> list[tuple[pd.Timestamp, pd.DataFrame]]` (oldest first)
  - `table_age_hours(name) -> float` (inf if missing)

- [ ] **Step 1: Write failing tests.** Create `tests/test_api.py`:

```python
"""Offline tests for fpl/api.py helpers (no network)."""
import gzip

from fpl import api


def test_maybe_gunzip_unpacks_gzip_and_leaves_plain_bytes():
    assert api.maybe_gunzip(gzip.compress(b'{"a": 1}')) == b'{"a": 1}'
    assert api.maybe_gunzip(b'{"a": 1}') == b'{"a": 1}'
```

Append to `tests/test_storage.py`:

```python
import time

import pandas as pd


def test_snapshots_load_oldest_first(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DATA_DIR", tmp_path)
    storage.save_snapshot("2025-26", "20250901T120000", "vaastav", pd.DataFrame({"id": [2]}))
    storage.save_snapshot("2025-26", "20250801T120000", "wayback", pd.DataFrame({"id": [1]}))
    snaps = storage.load_snapshots("2025-26")
    assert [s[0] for s in snaps] == [pd.Timestamp("2025-08-01 12:00", tz="UTC"),
                                     pd.Timestamp("2025-09-01 12:00", tz="UTC")]
    assert list(snaps[0][1]["id"]) == [1]


def test_benchmark_round_trip_and_table_age(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DATA_DIR", tmp_path)
    storage.save_benchmark("x.json", b'{"a": 1}')
    assert storage.load_benchmark_json("x.json") == {"a": 1}
    assert storage.table_age_hours("players") == float("inf")
    storage.save_table("players", pd.DataFrame({"a": [1]}))
    assert 0 <= storage.table_age_hours("players") < 0.1
```

- [ ] **Step 2: Run to verify they fail.** Run `python -m pytest tests/test_api.py tests/test_storage.py -v`. Expected: FAIL (`AttributeError: module 'fpl.api' has no attribute 'maybe_gunzip'`).

- [ ] **Step 3: Implement.** Append to `fpl/api.py` (add `import gzip` at the top):

```python
GITHUB_API = "https://api.github.com/repos/vaastav/Fantasy-Premier-League"
GITHUB_RAW = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League"
WAYBACK = "https://web.archive.org"


def fetch_entry(team_id: int) -> dict:
    """A manager's team: name, current gameweek, bank. Public; no login."""
    return get_json(f"{BASE_URL}/entry/{team_id}/")


def fetch_entry_history(team_id: int) -> dict:
    """Gameweek-by-gameweek points, bank, transfers and hits, chips played, past seasons."""
    return get_json(f"{BASE_URL}/entry/{team_id}/history/")


def fetch_entry_transfers(team_id: int) -> list:
    """Every transfer this season with buy (element_in_cost) and sale (element_out_cost) prices."""
    return get_json(f"{BASE_URL}/entry/{team_id}/transfers/")


def fetch_entry_picks(team_id: int, gw: int) -> dict:
    """The 15 picks, captain and chip for one gameweek whose deadline has passed."""
    return get_json(f"{BASE_URL}/entry/{team_id}/event/{gw}/picks/")


def fetch_vaastav_commits(path: str) -> list[tuple[str, str]]:
    """(sha, ISO commit date) of every vaastav commit touching `path`, newest first."""
    commits = get_json(f"{GITHUB_API}/commits?path={path}&per_page=100")
    return [(c["sha"], c["commit"]["committer"]["date"]) for c in commits]


def fetch_vaastav_file_at(sha: str, path: str) -> bytes:
    """A vaastav file as it was at one commit, e.g. path 'data/2025-26/fixtures.csv'."""
    return get_bytes(f"{GITHUB_RAW}/{sha}/{path}")


def fetch_wayback_index(url: str, start: str, end: str) -> list[str]:
    """Internet Archive timestamps (YYYYmmddHHMMSS) of good copies of `url`, one per day."""
    rows = get_json(f"{WAYBACK}/cdx/search/cdx?url={url}&from={start}&to={end}"
                    "&output=json&filter=statuscode:200&collapse=timestamp:8")
    return [r[1] for r in rows[1:]]


def fetch_wayback(url: str, timestamp: str) -> bytes:
    """The archived copy of `url` at `timestamp`, exactly as originally served."""
    return maybe_gunzip(get_bytes(f"{WAYBACK}/web/{timestamp}id_/{url}"))


def maybe_gunzip(data: bytes) -> bytes:
    """The archive sometimes returns the original gzip-compressed body."""
    return gzip.decompress(data) if data[:2] == b"\x1f\x8b" else data
```

Append to `fpl/storage.py` (add `import time` at the top):

```python
def benchmark_path(name: str) -> Path:
    """data/raw/benchmarks/<name>: downloaded once by pull_benchmarks.py."""
    return DATA_DIR / "raw" / "benchmarks" / name


def save_benchmark(name: str, content: bytes) -> Path:
    path = benchmark_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def load_benchmark_json(name: str) -> dict:
    return json.loads(benchmark_path(name).read_text(encoding="utf-8"))


def save_snapshot(season: str, stamp: str, source: str, table: pd.DataFrame) -> Path:
    """A dated copy of a season's fixture list: data/raw/fixture_snapshots/<season>/<stamp>_<source>.csv.

    stamp is UTC, 'YYYYmmddTHHMMSS'.
    """
    folder = DATA_DIR / "raw" / "fixture_snapshots" / season
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{stamp}_{source}.csv"
    table.to_csv(path, index=False, encoding="utf-8")
    return path


def load_snapshots(season: str) -> list[tuple[pd.Timestamp, pd.DataFrame]]:
    """Every saved fixture-list copy for a season, oldest first."""
    folder = DATA_DIR / "raw" / "fixture_snapshots" / season
    snaps = [(pd.Timestamp(p.stem.split("_")[0], tz="UTC"), read_csv(p))
             for p in folder.glob("*.csv")]
    return sorted(snaps, key=lambda s: s[0])


def table_age_hours(name: str) -> float:
    """Hours since data/processed/<name>.csv was written (infinite if it doesn't exist)."""
    path = DATA_DIR / "processed" / f"{name}.csv"
    return (time.time() - path.stat().st_mtime) / 3600 if path.exists() else float("inf")
```

- [ ] **Step 4: Run the tests.** Run `python -m pytest tests/test_api.py tests/test_storage.py -v`. Expected: all PASS.

- [ ] **Step 5: Commit and push**

```bash
git add fpl/api.py fpl/storage.py tests/test_api.py tests/test_storage.py
git commit -m "Add FPL entry, vaastav-commit and archive fetches; snapshot storage

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push
```

---

### Task 2: Settings and game rules

**Files:**
- Create: `fpl/settings.py`, `fpl/rules.py`, `tests/test_rules.py`, `tests/data/manager_histories.json`

**Interfaces:**
- Produces in `fpl/settings.py`:
  - constants `TEAM_ID`, `TIMEZONE`, `HORIZON`, `SOLVER_TIME_LIMIT`, `SIM_TIME_LIMIT`, `DATA_MAX_AGE_HOURS`, `FIRST_WEEK_BACK`, `CHIP_SCAN_WEEKS`, `CANDIDATES_PER_POSITION`, `VALUE_CANDIDATES_PER_POSITION`, `CHEAPEST_PER_POSITION`, `RATING_HALF_LIFE_DAYS`;
  - frozen dataclass `PlanSettings(discount=0.9, hit_margin=1.0, ft_end_value=1.5, bench_weight=0.1, horizon=6)`.
- Produces in `fpl/rules.py`:
  - constants `SQUAD_SHAPE`, `XI_LIMITS`, `MAX_PER_CLUB`, `BUDGET`, `HIT_COST`, `MAX_FREE_TRANSFERS`, `TRANSFER_CHIPS`, `CHIP_LABELS`, `FREE_TRANSFER_TOP_UPS`;
  - `selling_price(purchase, now) -> int`
  - `paid_transfers(free, transfers, chip) -> int`
  - `next_free_transfers(free, transfers, chip) -> int`
  - `free_transfers_at(gw, rolled, season) -> int`
  - frozen dataclass `Chip(name, start, stop)` with `.key` and `.allows(gw)`
  - `chip_windows(bootstrap_chips) -> list[Chip]`
  - `chips_left(windows, played) -> list[Chip]`
  - `valid_xi(positions) -> bool`
  - `score_week(picks, chip=None, hits=0) -> int`

- [ ] **Step 1: Save sampled real manager histories** (other managers' public data, with IDs stripped). Run:

```bash
python - <<'EOF'
import json, random
from pathlib import Path
from fpl import api
random.seed(7)
keep = []
for team_id in random.sample(range(1, 9_000_000), 400):
    try:
        h = api.fetch_entry_history(team_id)
    except Exception:
        continue
    cur = h.get("current", [])
    if cur and cur[0]["event"] == 1 and any(c["name"] in ("wildcard", "freehit") for c in h["chips"]):
        keep.append({"current": [{k: r[k] for k in ("event", "event_transfers", "event_transfers_cost")}
                                 for r in cur],
                     "chips": [{"name": c["name"], "event": c["event"]} for c in h["chips"]]})
    if len(keep) == 25:
        break
Path("tests/data").mkdir(exist_ok=True)
Path("tests/data/manager_histories.json").write_text(json.dumps(keep, indent=1))
print(len(keep), "histories saved")
EOF
```

Expected: `25 histories saved`, or close to it. (About 38% of sampled managers have played a Wildcard or Free Hit.)

- [ ] **Step 2: Write failing tests.** Create `tests/test_rules.py`:

```python
"""Offline tests for fpl/rules.py: prices, free transfers, chips and gameweek scoring."""
import json
from pathlib import Path

import pandas as pd

from fpl import rules

DATA = Path(__file__).parent / "data"
XI = ["GK"] + ["DEF"] * 4 + ["MID"] * 4 + ["FWD"] * 2
BENCH = ["GK", "DEF", "MID", "FWD"]


def week(points=None, minutes=None, captain=10, vice=9, positions=None):
    """15 picks in pick order: 11 starters, bench GK, then the outfield bench in order."""
    p = pd.DataFrame({"position": positions or XI + BENCH,
                      "points": points or [2] * 15, "minutes": minutes or [90] * 15})
    p["is_captain"] = p.index == captain
    p["is_vice"] = p.index == vice
    return p


def test_selling_price_shares_rises_and_keeps_falls():
    assert rules.selling_price(50, 53) == 51
    assert rules.selling_price(50, 54) == 52
    assert rules.selling_price(50, 47) == 47
    assert rules.selling_price(50, 50) == 50


def test_free_transfers_roll_cap_and_chips():
    assert rules.next_free_transfers(1, 0, None) == 2
    assert rules.next_free_transfers(2, 2, None) == 1
    assert rules.next_free_transfers(1, 3, None) == 1
    assert rules.next_free_transfers(5, 0, None) == 5
    assert rules.next_free_transfers(2, 9, "wildcard") == 2
    assert rules.next_free_transfers(3, 0, "freehit") == 3
    assert rules.next_free_transfers(1, 0, "bboost") == 2
    assert rules.paid_transfers(1, 3, None) == 2
    assert rules.paid_transfers(1, 9, "wildcard") == 0


def test_afcon_top_up_only_in_2025_26_gw16():
    assert rules.free_transfers_at(16, 2, "2025-26") == 5
    assert rules.free_transfers_at(17, 2, "2025-26") == 2
    assert rules.free_transfers_at(16, 2, "2026-27") == 2


def test_rebuilt_free_transfers_match_every_hit_real_managers_paid():
    histories = json.loads((DATA / "manager_histories.json").read_text())
    checked = 0
    for h in histories:
        chip_at = {c["event"]: c["name"] for c in h["chips"]}
        free = None
        for row in h["current"]:
            gw, made = row["event"], row["event_transfers"]
            if free is None:
                free = 1            # first gameweek: unlimited changes, then 1 free transfer
                continue
            chip = chip_at.get(gw)
            assert row["event_transfers_cost"] == rules.HIT_COST * rules.paid_transfers(free, made, chip), (gw, h)
            checked += 1
            free = rules.next_free_transfers(free, made, chip)
    assert checked > 50


def test_chips_left_removes_the_copy_used():
    windows = [rules.Chip("wildcard", 2, 19), rules.Chip("wildcard", 20, 38), rules.Chip("3xc", 1, 19)]
    left = rules.chips_left(windows, [("wildcard", 5)])
    assert left == [rules.Chip("wildcard", 20, 38), rules.Chip("3xc", 1, 19)]
    assert rules.Chip("3xc", 1, 19).allows(19) and not rules.Chip("3xc", 1, 19).allows(20)


def test_score_week_everyone_plays():
    assert rules.score_week(week()) == 11 * 2 + 2           # captain doubled


def test_auto_sub_takes_first_bench_player_who_played():
    points = [2] * 11 + [1, 3, 5, 7]
    minutes = [90] * 15
    points[1], minutes[1] = 0, 0                            # a starting DEF didn't play
    assert rules.score_week(week(points, minutes)) == 10 * 2 + 3 + 2


def test_auto_sub_keeps_a_legal_formation():
    positions = ["GK"] + ["DEF"] * 3 + ["MID"] * 5 + ["FWD"] * 2 + ["GK", "MID", "DEF", "FWD"]
    points = [2] * 11 + [1, 4, 6, 8]
    minutes = [90] * 15
    points[1], minutes[1] = 0, 0          # a DEF out: bench MID would leave 2 DEF, so the bench DEF comes on
    assert rules.score_week(week(points, minutes, positions=positions)) == 10 * 2 + 6 + 2


def test_vice_captain_steps_in_and_triple_captain_triples():
    points = [2] * 15
    minutes = [90] * 15
    points[10], minutes[10] = 0, 0                          # captain didn't play
    points[9] = 6
    assert rules.score_week(week(points, minutes)) == 9 * 2 + 6 + 6 + 2   # vice doubled; bench DEF (2) subs on
    assert rules.score_week(week(), chip="3xc") == 11 * 2 + 4


def test_bench_boost_and_hits():
    assert rules.score_week(week(), chip="bboost") == 15 * 2 + 2
    assert rules.score_week(week(), hits=2) == 11 * 2 + 2 - 8
```

- [ ] **Step 3: Run to verify they fail.** Run `python -m pytest tests/test_rules.py -v`. Expected: FAIL with `ImportError: cannot import name 'rules'`.

- [ ] **Step 4: Implement `fpl/settings.py`:**

```python
"""Every tunable number for the squad planner, in one place."""
from dataclasses import dataclass

TEAM_ID = 5234632
TIMEZONE = "Africa/Nairobi"
HORIZON = 6                         # weeks planned (recommend.py --horizon 4-6)
SOLVER_TIME_LIMIT = 60              # seconds per solve in recommend.py
SIM_TIME_LIMIT = 30                 # seconds per solve in the season simulation
DATA_MAX_AGE_HOURS = 6              # recommend.py refreshes FPL data if older
FIRST_WEEK_BACK = 0.8               # availability in a player's first week back from a dated absence
CHIP_SCAN_WEEKS = 13                # weeks predicted ahead (plan + chip scan); GW7 + 12 = GW19
CANDIDATES_PER_POSITION = 40        # planner pool: best by discounted xP...
VALUE_CANDIDATES_PER_POSITION = 15  # ...plus best xP per £m...
CHEAPEST_PER_POSITION = 5           # ...plus the cheapest, so a valid squad always fits the budget
RATING_HALF_LIFE_DAYS = 180         # team ratings: a match's weight halves every 180 days


@dataclass(frozen=True)
class PlanSettings:
    """The planner's judgement calls; simulate.py tests a few values of each."""
    discount: float = 0.9           # weight of plan week k is discount ** (k - 1)
    hit_margin: float = 1.0         # a hit must gain more than 4 + this (after discounting)
    ft_end_value: float = 1.5       # points per free transfer still saved after the last plan week
    bench_weight: float = 0.1       # share of bench players' xP counted (value of cover)
    horizon: int = HORIZON
```

- [ ] **Step 5: Implement `fpl/rules.py`:**

```python
"""FPL game rules shared by the planner and the season simulation.

Prices are FPL units (tenths of £1m: 55 = £5.5m) so rounding matches FPL.
Nothing here touches the internet or the disk.
"""
from dataclasses import dataclass

import pandas as pd

SQUAD_SHAPE = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
XI_LIMITS = {"GK": (1, 1), "DEF": (3, 5), "MID": (2, 5), "FWD": (1, 3)}
MAX_PER_CLUB = 3
BUDGET = 1000                       # £100.0m
HIT_COST = 4
MAX_FREE_TRANSFERS = 5
TRANSFER_CHIPS = ("wildcard", "freehit")
CHIP_LABELS = {"wildcard": "Wildcard", "freehit": "Free Hit", "bboost": "Bench Boost",
               "3xc": "Triple Captain"}
# Season-wide top-ups: {season: {gameweek: free transfers}}. AFCON 2025/26, per
# premierleague.com/en/news/4362102 ("topped up to the maximum possible number of five in GW16").
FREE_TRANSFER_TOP_UPS = {"2025-26": {16: 5}}


def selling_price(purchase: int, now: int) -> int:
    """A rise is shared 50/50 with FPL (your half rounded down); a fall is all yours."""
    return now if now <= purchase else purchase + (now - purchase) // 2


def paid_transfers(free: int, transfers: int, chip: str | None) -> int:
    """How many of this gameweek's transfers cost a hit."""
    return 0 if chip in TRANSFER_CHIPS else max(0, transfers - free)


def next_free_transfers(free: int, transfers: int, chip: str | None) -> int:
    """Free transfers for the next gameweek.

    Normal week: unused ones roll over (max 5) and one is added. Wildcard or
    Free Hit week: the count is kept but none is added. Checked against 478
    real manager-weeks in 2026/27 (see the part 3 spec).
    """
    if chip in TRANSFER_CHIPS:
        return free
    return min(MAX_FREE_TRANSFERS, max(free - transfers, 0) + 1)


def free_transfers_at(gw: int, rolled: int, season: str) -> int:
    """The rolled-over count, unless FPL topped everyone up that gameweek."""
    return FREE_TRANSFER_TOP_UPS.get(season, {}).get(gw, rolled)


@dataclass(frozen=True)
class Chip:
    """One copy of a chip, playable from `start` to `stop` inclusive (FPL's chips data)."""
    name: str
    start: int
    stop: int

    @property
    def key(self) -> str:
        return f"{self.name}_{self.start}"

    def allows(self, gw: int) -> bool:
        return self.start <= gw <= self.stop


def chip_windows(bootstrap_chips: list) -> list[Chip]:
    """bootstrap-static['chips'] -> one Chip per copy."""
    return [Chip(c["name"], int(c["start_event"]), int(c["stop_event"])) for c in bootstrap_chips]


def chips_left(windows: list[Chip], played: list[tuple[str, int]]) -> list[Chip]:
    """Drop the copy each played (name, gameweek) used up."""
    left = list(windows)
    for name, gw in played:
        used = next((c for c in left if c.name == name and c.allows(gw)), None)
        if used is not None:
            left.remove(used)
    return left


def valid_xi(positions) -> bool:
    counts = pd.Series(list(positions)).value_counts()
    return all(lo <= counts.get(p, 0) <= hi for p, (lo, hi) in XI_LIMITS.items())


def score_week(picks: pd.DataFrame, chip: str | None = None, hits: int = 0) -> int:
    """Score one gameweek as FPL does.

    picks: 15 rows in pick order (11 starters, the bench goalkeeper, then the
    outfield bench in order) with position, points, minutes (his total that
    gameweek; 0 if he had no match), is_captain, is_vice.
    Auto-subs: a starter with 0 minutes is replaced by the first bench player
    who played and keeps a legal formation (goalkeeper only by goalkeeper).
    The captain's points are doubled (tripled with Triple Captain), or the
    vice-captain's if the captain didn't play. Bench Boost counts all 15.
    """
    p = picks.reset_index(drop=True)
    played = p["minutes"] > 0
    if chip == "bboost":
        counted = list(range(15))
    else:
        counted = list(range(11))
        bench = [11, 12, 13, 14]
        for i in range(11):
            if played[counted[i]]:
                continue
            out_gk = p.at[counted[i], "position"] == "GK"
            for b in bench:
                if not played[b] or (p.at[b, "position"] == "GK") != out_gk:
                    continue
                trial = [p.at[j, "position"] for j in counted if j != counted[i]] + [p.at[b, "position"]]
                if valid_xi(trial):
                    bench.remove(b)
                    counted[i] = b
                    break
    total = int(p.loc[counted, "points"].sum())
    captain = int(p.index[p["is_captain"]][0])
    vice = int(p.index[p["is_vice"]][0])
    armband = captain if played[captain] else (vice if played[vice] else None)
    if armband is not None and armband in counted:
        total += (2 if chip == "3xc" else 1) * int(p.at[armband, "points"])
    return total - HIT_COST * hits
```

- [ ] **Step 6: Run the tests.** Run `python -m pytest tests/test_rules.py -v`. Expected: all PASS.

  If `test_rebuilt_free_transfers_match_every_hit_real_managers_paid` fails for a saved manager, **stop**: the printed (gw, history) shows a week the rule doesn't explain. Investigate before changing the rule or the data.

- [ ] **Step 7: Commit and push**

```bash
git add fpl/settings.py fpl/rules.py tests/test_rules.py tests/data/manager_histories.json
git commit -m "Add planner settings and FPL game rules with auto-sub scoring

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push
```

---

### Task 3: Load the user's team (`fpl/myteam.py`)

**Files:**
- Create: `fpl/myteam.py`, `tests/test_myteam.py`, `tests/data/team_5234632/*`, `tests/data/players_sample.csv`, `tests/data/gameweeks_2026-27.csv`, `tests/data/chips_2026-27.json`

**Interfaces:**
- Consumes: `rules.selling_price`, `rules.next_free_transfers`, `rules.free_transfers_at`, `rules.chip_windows`, `rules.chips_left`, `rules.HIT_COST`, `matches.POSITION_BY_TYPE`.
- Produces:
  - dataclass `TeamState(team_id, name, next_gw, squad: pd.DataFrame, bank: int, free_transfers: int, pending_transfers: int, pending_hits: int, chips: list[rules.Chip], chips_played: list[tuple[str,int]])`;
  - `squad` columns: `element, player_code, web_name, position, team, purchase, now, sell, purchase_known`;
  - `load_team_state(team_id, entry, history, transfers, fetch_picks, players, player_history, bootstrap_chips, season=CURRENT_SEASON) -> TeamState`, where `fetch_picks(gw) -> dict`;
  - `TeamState.codes -> list[int]` (player codes).

- [ ] **Step 1: Save test data** (run before the GW7 deadline, Sat 17 Oct 13:00 Nairobi; it records the state the assertions below expect). Run `python pull_data.py` first, then:

```bash
python - <<'EOF'
import json
from pathlib import Path
import pandas as pd
from fpl import api, storage
team = 5234632
out = Path("tests/data/team_5234632"); out.mkdir(parents=True, exist_ok=True)
entry = api.fetch_entry(team)
entry = {k: entry[k] for k in ("id", "name", "current_event")}        # drop personal details
history = api.fetch_entry_history(team)
transfers = api.fetch_entry_transfers(team)
gws = sorted({history["current"][0]["event"], history["current"][-1]["event"]})
picks = {gw: api.fetch_entry_picks(team, gw) for gw in gws}
for name, data in [("entry", entry), ("history", history), ("transfers", transfers)]:
    (out / f"{name}.json").write_text(json.dumps(data, indent=1))
for gw, data in picks.items():
    (out / f"picks_{gw}.json").write_text(json.dumps(data, indent=1))
players = storage.load_table("players")
owned = {p["element"] for d in picks.values() for p in d["picks"]} | {t["element_in"] for t in transfers}
top = players.sort_values("total_points", ascending=False).groupby("element_type").head(25)
cols = ["id", "code", "web_name", "element_type", "team", "now_cost", "status", "news", "news_added",
        "chance_of_playing_next_round", "total_points"]
sample = pd.concat([players[players["id"].isin(owned)], top]).drop_duplicates("id")[cols]
sample.to_csv("tests/data/players_sample.csv", index=False)
ph = storage.load_table("player_history")
ph[ph["element"].isin(owned)][["element", "round", "value"]].to_csv(out / "player_history.csv", index=False)
gw = storage.load_table("gameweeks")[["id", "deadline_time", "is_next", "finished"]]
gw.to_csv("tests/data/gameweeks_2026-27.csv", index=False)
bootstrap = json.loads(sorted(Path("data/raw").glob("2026-*/bootstrap_static.json"))[-1].read_text())
Path("tests/data/chips_2026-27.json").write_text(json.dumps(bootstrap["chips"], indent=1))
print("saved; free transfers check needs history:", [(r["event"], r["event_transfers"]) for r in history["current"]])
EOF
```

  If this runs after the user has made GW7 transfers, work the expected values out by hand from `history.json`/`transfers.json` and update the three numbers in `test_real_team_5234632`.

- [ ] **Step 2: Write failing tests.** Create `tests/test_myteam.py`:

```python
"""Offline tests for fpl/myteam.py."""
import json
from pathlib import Path

import pandas as pd
import pytest

from fpl import myteam, rules

DATA = Path(__file__).parent / "data"
CHIPS = [{"name": n, "start_event": a, "stop_event": b} for n, a, b in
         [("wildcard", 2, 19), ("wildcard", 20, 38), ("freehit", 2, 19), ("bboost", 1, 19),
          ("3xc", 1, 19), ("freehit", 20, 38), ("bboost", 20, 38), ("3xc", 20, 38)]]
TYPES = [1, 1] + [2] * 5 + [3] * 5 + [4] * 3 + [3, 3]       # elements 1-15 squad, 16-17 extras


def players(now=None):
    now = now or {}
    return pd.DataFrame({"id": range(1, 18), "code": range(1001, 1018),
                         "web_name": [f"P{i}" for i in range(1, 18)],
                         "element_type": TYPES, "team": [i % 9 + 1 for i in range(1, 18)],
                         "now_cost": [now.get(i, 50) for i in range(1, 18)],
                         "status": "a", "news": "", "chance_of_playing_next_round": None})


def picks(elements):
    return {"picks": [{"element": e} for e in elements]}


def late_joiner():
    """Joined in GW3; one transfer in GW5; Free Hit in GW6 (its picks and transfer must be ignored)."""
    history = {"current": [
        {"event": 3, "bank": 0, "event_transfers": 0, "event_transfers_cost": 0},
        {"event": 4, "bank": 0, "event_transfers": 0, "event_transfers_cost": 0},
        {"event": 5, "bank": 0, "event_transfers": 1, "event_transfers_cost": 0},
        {"event": 6, "bank": 0, "event_transfers": 1, "event_transfers_cost": 0}],
        "chips": [{"name": "freehit", "event": 6}]}
    transfers = [
        {"element_in": 16, "element_in_cost": 50, "element_out": 15, "element_out_cost": 50,
         "event": 5, "time": "2026-09-20T10:00:00Z"},
        {"element_in": 17, "element_in_cost": 50, "element_out": 2, "element_out_cost": 50,
         "event": 6, "time": "2026-09-27T10:00:00Z"}]
    by_gw = {3: picks(range(1, 16)), 5: picks([*range(1, 15), 16]), 6: picks([1, 17, *range(3, 15), 16])}
    ph = pd.DataFrame({"element": list(range(1, 18)) * 2, "round": [1] * 17 + [3] * 17,
                       "value": [45] * 17 + [50] * 17})
    return history, transfers, by_gw, ph


def test_late_joiner_free_hit_and_selling_prices():
    history, transfers, by_gw, ph = late_joiner()
    state = myteam.load_team_state(1, {"name": "T"}, history, transfers, by_gw.__getitem__,
                                   players(now={1: 56}), ph, CHIPS)
    assert state.next_gw == 7
    assert sorted(state.squad["element"]) == [*range(1, 15), 16]          # Free Hit squad ignored
    p1 = state.squad.set_index("element").loc[1]
    assert (p1["purchase"], p1["now"], p1["sell"]) == (50, 56, 53)       # GW3 price, not GW1
    assert state.free_transfers == 2       # 1 for GW4, rolled to 2, used 1 -> 2, Free Hit keeps 2
    assert state.chips_played == [("freehit", 6)]
    assert rules.Chip("freehit", 2, 19) not in state.chips and len(state.chips) == 7


def test_pending_transfers_are_applied():
    history, transfers, by_gw, ph = late_joiner()
    transfers.append({"element_in": 17, "element_in_cost": 45, "element_out": 3,
                      "element_out_cost": 50, "event": 7, "time": "2026-10-05T10:00:00Z"})
    state = myteam.load_team_state(1, {"name": "T"}, history, transfers, by_gw.__getitem__,
                                   players(), ph, CHIPS)
    assert 17 in set(state.squad["element"]) and 3 not in set(state.squad["element"])
    assert state.bank == 5
    assert (state.pending_transfers, state.free_transfers, state.pending_hits) == (1, 1, 0)


def test_real_team_5234632():
    folder = DATA / "team_5234632"
    load = lambda name: json.loads((folder / f"{name}.json").read_text())
    by_gw = {int(p.stem.split("_")[1]): json.loads(p.read_text()) for p in folder.glob("picks_*.json")}
    state = myteam.load_team_state(
        5234632, load("entry"), load("history"), load("transfers"), by_gw.__getitem__,
        pd.read_csv(DATA / "players_sample.csv"), pd.read_csv(folder / "player_history.csv"),
        json.loads((DATA / "chips_2026-27.json").read_text()))
    assert (state.next_gw, state.free_transfers, state.bank) == (7, 1, 6)
    assert len(state.squad) == 15 and state.squad["purchase_known"].all()
    assert (state.squad["sell"] <= state.squad[["purchase", "now"]].max(axis=1)).all()
    assert len(state.chips) == 8 and state.chips_played == []
```

- [ ] **Step 3: Run to verify they fail.** Run `python -m pytest tests/test_myteam.py -v`. Expected: FAIL (`cannot import name 'myteam'`).

- [ ] **Step 4: Implement `fpl/myteam.py`:**

```python
"""The user's FPL team from public API responses: squad, prices, bank, free transfers, chips.

FPL only shows selling prices and free transfers when you're logged in, so both
are rebuilt from the public transfer and gameweek history. Prices are FPL units
(tenths of £1m). Nothing here touches the internet or the disk.
"""
from dataclasses import dataclass

import pandas as pd

from fpl import rules
from fpl.matches import POSITION_BY_TYPE
from fpl.seasons import CURRENT_SEASON


@dataclass
class TeamState:
    team_id: int
    name: str
    next_gw: int
    squad: pd.DataFrame          # element, player_code, web_name, position, team, purchase, now, sell, purchase_known
    bank: int
    free_transfers: int          # free transfers left for next_gw
    pending_transfers: int       # transfers already made for next_gw
    pending_hits: int            # points already committed to hits for next_gw
    chips: list                  # rules.Chip copies still available
    chips_played: list           # [(name, gameweek)]

    @property
    def codes(self) -> list[int]:
        return [int(c) for c in self.squad["player_code"]]


def _price_at(player_history: pd.DataFrame, element: int, gw: int) -> int | None:
    """His price at gameweek gw, or at his next match if his team had none that week."""
    rows = player_history[(player_history["element"] == element) & (player_history["round"] >= gw)]
    return int(rows.sort_values("round")["value"].iloc[0]) if len(rows) else None


def load_team_state(team_id, entry, history, transfers, fetch_picks, players, player_history,
                    bootstrap_chips, season=CURRENT_SEASON) -> TeamState:
    """Rebuild the team as it stands for the next deadline.

    entry/history/transfers: FPL's /entry/{id}/, /history/, /transfers/ responses.
    fetch_picks(gw): /entry/{id}/event/{gw}/picks/ for a gameweek whose deadline has passed.
    players: our players.csv (bootstrap elements). player_history: part 1's history table.
    """
    current = history["current"]
    first_gw, last_gw = current[0]["event"], current[-1]["event"]
    played = [(c["name"], c["event"]) for c in history["chips"]]
    chip_at = dict((gw, name) for name, gw in played)
    freehit_gws = {gw for name, gw in played if name == "freehit"}
    base_gw = max(r["event"] for r in current if r["event"] not in freehit_gws)

    # Purchase prices: the starting squad at its first gameweek's prices, then every
    # transfer in order (Free Hit transfers revert, so they are skipped).
    purchase = {p["element"]: _price_at(player_history, p["element"], first_gw)
                for p in fetch_picks(first_gw)["picks"]}
    ordered = sorted(transfers, key=lambda t: t["time"])
    for t in ordered:
        if t["event"] not in freehit_gws:
            purchase[t["element_in"]] = t["element_in_cost"]

    squad = [p["element"] for p in fetch_picks(base_gw)["picks"]]
    bank = next(r["bank"] for r in current if r["event"] == base_gw)
    for t in ordered:   # made after the squad we read: Free-Hit-reverted weeks and the next deadline
        if t["event"] > base_gw and t["event"] not in freehit_gws:
            squad[squad.index(t["element_out"])] = t["element_in"]
            bank += t["element_out_cost"] - t["element_in_cost"]

    free = 1            # the first gameweek's changes are unlimited; one free transfer after it
    for r in current[1:]:
        free = rules.free_transfers_at(r["event"], free, season)
        free = rules.next_free_transfers(free, r["event_transfers"], chip_at.get(r["event"]))
    next_gw = last_gw + 1
    free = rules.free_transfers_at(next_gw, free, season)
    pending = sum(1 for t in transfers if t["event"] == next_gw)

    info = players.set_index("id").loc[squad]
    now = info["now_cost"].astype(int).to_numpy()
    bought = [purchase.get(e) for e in squad]
    table = pd.DataFrame({
        "element": squad, "player_code": info["code"].astype(int).to_numpy(),
        "web_name": info["web_name"].to_numpy(),
        "position": info["element_type"].map(POSITION_BY_TYPE).to_numpy(),
        "team": info["team"].to_numpy(), "now": now,
        "purchase": [b if b is not None else n for b, n in zip(bought, now)],
        "purchase_known": [b is not None for b in bought]})
    table["sell"] = [rules.selling_price(p, n) for p, n in zip(table["purchase"], table["now"])]
    return TeamState(
        team_id=team_id, name=entry.get("name", ""), next_gw=next_gw, squad=table, bank=int(bank),
        free_transfers=max(0, free - pending), pending_transfers=pending,
        pending_hits=rules.HIT_COST * max(0, pending - free),
        chips=rules.chips_left(rules.chip_windows(bootstrap_chips), played), chips_played=played)
```

- [ ] **Step 5: Run the tests.** Run `python -m pytest tests/test_myteam.py -v`. Expected: all PASS.

- [ ] **Step 6: Commit and push**

```bash
git add fpl/myteam.py tests/test_myteam.py tests/data
git commit -m "Rebuild the user's team, selling prices and free transfers from public data

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push
```

---

### Task 4: Injury flags across the whole plan (`fpl/availability.py`)

**Files:**
- Create: `fpl/availability.py`, `fpl/forecast.py`, `tests/test_availability.py`, `tests/test_forecast.py`
- Modify:
  - `fpl/models.py`: remove `apply_availability`; add a `yardstick` flag to `train_and_predict`; add `weekly_xp`.
  - `tests/test_models.py`: delete `test_apply_availability_caps_next_gameweek_only`.
  - `predict.py`: use `forecast.live_forecast`.

**Interfaces:**
- Consumes: `scoring.expected_points`, `settings.FIRST_WEEK_BACK`, `matches.*`, `models.train_and_predict`.
- Produces in `fpl/availability.py`:
  - `return_date(news, added) -> date | None`
  - `recovery_curve(history) -> pd.Series` (index 1..8)
  - `matches_missed(history, codes) -> pd.Series`
  - `weekly_availability(players, gws, deadlines, curve, missed) -> DataFrame[player_code, gw, availability, news]` (flagged players only)
  - `apply(pred, avail) -> DataFrame`
  - `trim_absences(history, codes) -> DataFrame`
- Produces in `fpl/models.py`:
  - `train_and_predict(..., yardstick=True)`
  - `weekly_xp(pred, gws) -> DataFrame` (index player_code, one column per gw)
- Produces in `fpl/forecast.py`:
  - `deadlines_from(gameweeks) -> dict[int, pd.Timestamp]`
  - `live_forecast(history, fixtures_all, players, teams, gameweeks, gws, season=CURRENT_SEASON) -> (pred, avail)`

- [ ] **Step 1: Write failing tests.** Create `tests/test_availability.py`:

```python
"""Offline tests for fpl/availability.py, using the user's four flagged players' real news."""
from datetime import date

import pandas as pd
import pytest

from fpl import availability

DEADLINES = {gw: pd.Timestamp(t, tz="UTC") for gw, t in {
    7: "2026-10-17 10:00", 8: "2026-10-24 10:00", 9: "2026-10-31 11:00",
    10: "2026-11-07 11:00", 11: "2026-11-21 11:00", 12: "2026-11-28 11:00"}.items()}
GWS = list(DEADLINES)
CURVE = pd.Series([0.5, 0.4, 0.3, 0.3, 0.2, 0.2, 0.2, 0.1], index=range(1, 9))


def player(code, status, news, chance, added="2026-10-02T15:00:11Z"):
    return {"code": code, "status": status, "news": news, "news_added": added,
            "chance_of_playing_next_round": chance}


PLAYERS = pd.DataFrame([
    player(1, "u", "Has joined York City on loan for the rest of the season", 0.0),      # McNally
    player(2, "i", "Hamstring injury - Expected back 24 Oct", 0.0),                       # Tzolis
    player(3, "i", "Thigh injury - Unknown return date", 0.0, "2026-09-27T10:30:09Z"),    # Isak
    player(4, "d", "Ankle injury - 75% chance of playing", 75.0),                         # Robinson
    player(5, "s", "Suspended until 31 Oct", 0.0),
    player(6, "a", "", None),
])


def table():
    avail = availability.weekly_availability(PLAYERS, GWS, DEADLINES, CURVE, pd.Series({3: 2}))
    return avail.pivot(index="player_code", columns="gw", values="availability")


def test_each_news_form_gives_the_right_weekly_chance():
    t = table()
    assert list(t.loc[1]) == [0, 0, 0, 0, 0, 0]
    assert list(t.loc[2]) == [0, 0.8, 1, 1, 1, 1]
    assert list(t.loc[4]) == [0.75, 0.875, 1, 1, 1, 1]
    assert list(t.loc[5]) == [0, 0, 0.8, 1, 1, 1]
    assert 6 not in t.index                                     # fit players aren't listed


def test_unknown_return_uses_fpls_chance_then_the_recovery_curve():
    isak = table().loc[3]
    assert isak[7] == 0                                         # FPL says 0% for next week
    assert isak[8] == pytest.approx(0.3)                        # missed 2 + GW7 = 3: curve[3]
    assert isak[9] == pytest.approx(1 - 0.7 * 0.7)              # then curve[4]
    assert isak.is_monotonic_increasing and isak[12] < 1


def test_unknown_return_without_a_chance_uses_the_curve_from_next_week():
    nameless = PLAYERS.iloc[[2]].assign(chance_of_playing_next_round=None)
    avail = availability.weekly_availability(nameless, GWS, DEADLINES, CURVE, pd.Series({3: 2}))
    assert avail["availability"].iloc[0] == pytest.approx(0.4)  # curve[2]


def test_return_date_handles_odd_text():
    added = pd.Timestamp("2026-12-20", tz="UTC")
    assert availability.return_date("Knee injury - Expected back 05 Jan", added) == date(2027, 1, 5)
    assert availability.return_date("Knee injury - Expected back 31 Feb", added) is None
    assert availability.return_date("Knee injury - Unknown return date", added) is None
    assert availability.return_date(None, added) is None


def history(minutes_by_player):
    rows = []
    for code, minutes in minutes_by_player.items():
        for i, m in enumerate(minutes):
            rows.append({"player_code": code, "kickoff_time": pd.Timestamp("2025-08-01", tz="UTC")
                         + pd.Timedelta(days=7 * i), "minutes": m, "fixture_id": i, "season": "2025-26"})
    return pd.DataFrame(rows)


def test_recovery_curve_from_regular_starters():
    h = history({1: [90] * 12 + [0, 0, 90], 2: [90] * 12 + [0, 90], 3: [10] * 12 + [0, 0, 0, 90]})
    curve = availability.recovery_curve(h)
    assert curve[1] == pytest.approx(0.5)                       # player 3 isn't a regular starter
    assert curve.is_monotonic_decreasing and curve.between(0, 1).all()


def test_matches_missed_and_trim_absences():
    h = history({1: [90, 90, 0, 0], 2: [90, 0, 90, 90]})
    assert availability.matches_missed(h, [1, 2]).to_dict() == {1: 2, 2: 0}
    trimmed = availability.trim_absences(h, [1])
    assert len(trimmed[trimmed.player_code == 1]) == 2 and len(trimmed[trimmed.player_code == 2]) == 4


def test_apply_scales_plays_and_sixty_per_week():
    base = {"season": "2026-27", "position": "MID", "p_plays": 0.9, "p_60": 0.6, "xg": 0.5,
            "xa": 0.2, "p_cs": 0.3, "gc_lambda": 1.0, "saves_lambda": 0.0, "p_yellow": 0.1,
            "p_red": 0.0, "p_defcon": 0.2, "e_bonus": 0.4}
    pred = pd.DataFrame([{**base, "player_code": 4, "gw": 7}, {**base, "player_code": 4, "gw": 9},
                         {**base, "player_code": 6, "gw": 7}])
    out = availability.apply(pred, availability.weekly_availability(
        PLAYERS, GWS, DEADLINES, CURVE, pd.Series(dtype=float)))
    assert list(out.p_plays) == pytest.approx([0.675, 0.9, 0.9])
    assert out.p_60.iloc[0] == pytest.approx(0.45) and out.xp.iloc[0] < out.xp.iloc[2]
```

Create `tests/test_forecast.py`:

```python
"""Offline tests for weekly xP and live-forecast helpers."""
import pandas as pd

from fpl import forecast, models


def test_weekly_xp_sums_doubles_and_fills_blanks():
    pred = pd.DataFrame({"player_code": [1, 1, 1, 2], "gw": [7, 8, 8, 7], "xp": [2.0, 3.0, 1.5, 4.0]})
    xp = models.weekly_xp(pred, [7, 8, 9])
    assert xp.loc[1].tolist() == [2.0, 4.5, 0.0]
    assert xp.loc[2].tolist() == [4.0, 0.0, 0.0]


def test_deadlines_from_gameweeks_are_utc():
    gws = pd.DataFrame({"id": [7], "deadline_time": ["2026-10-17T10:00:00Z"]})
    assert forecast.deadlines_from(gws)[7] == pd.Timestamp("2026-10-17 10:00", tz="UTC")
```

- [ ] **Step 2: Run to verify they fail.** Run `python -m pytest tests/test_availability.py tests/test_forecast.py -v`. Expected: FAIL (`cannot import name 'availability'`).

- [ ] **Step 3: Implement `fpl/availability.py`:**

```python
"""Turn FPL's injury and suspension news into a chance of being available each gameweek.

Part 2 only capped next week, so a 6-week plan treated an injured player as fit
from week 2. Each flagged player now gets availability a(w) for every planned
gameweek; P(plays) and P(60+) are multiplied by it. Live predictions only:
past news wasn't stored.
"""
import re
from datetime import date

import numpy as np
import pandas as pd

from fpl import scoring
from fpl.settings import FIRST_WEEK_BACK

MAX_MISSED = 8                       # recovery curve: 8+ missed matches pooled
REGULAR_WINDOW, REGULAR_MIN_60 = 10, 6
_DATE = re.compile(r"(?:Expected back|Suspended until)\s+(\d{1,2})\s+([A-Z][a-z]{2})")
_MONTHS = {m: i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], start=1)}
_DEFAULT_CURVE = pd.Series(np.linspace(0.5, 0.15, MAX_MISSED), index=range(1, MAX_MISSED + 1))


def return_date(news, added: pd.Timestamp) -> date | None:
    """'Hamstring injury - Expected back 24 Oct' -> 2026-10-24; None if there's no valid date.

    The year comes from when the news was added (a January date in December news is next year).
    """
    m = _DATE.search(news) if isinstance(news, str) else None
    if not m or m.group(2) not in _MONTHS or pd.isna(added):
        return None
    month = _MONTHS[m.group(2)]
    year = added.year + (1 if month < added.month else 0)
    try:
        return date(year, month, int(m.group(1)))
    except ValueError:
        return None


def recovery_curve(history: pd.DataFrame) -> pd.Series:
    """Chance a regular starter plays his next fixture after missing m in a row (m = 1..8).

    Regular = 60+ minutes in at least 6 of the 10 fixtures before the absence.
    Made non-increasing in m. It includes non-injury absences (dropped,
    rested), so it is a rough guide; recommend.py prints what it implies.
    """
    records = []
    h = history.sort_values(["player_code", "kickoff_time"])
    for _, minutes in h.groupby("player_code", sort=False)["minutes"]:
        mins = minutes.to_numpy()
        missed, regular = 0, False
        for i in range(len(mins) - 1):
            if mins[i] > 0:
                missed = 0
                continue
            if missed == 0:
                before = mins[max(0, i - REGULAR_WINDOW):i]
                regular = len(before) == REGULAR_WINDOW and (before >= 60).sum() >= REGULAR_MIN_60
            missed += 1
            if regular:
                records.append((min(missed, MAX_MISSED), mins[i + 1] > 0))
    if not records:
        return _DEFAULT_CURVE.copy()
    rates = pd.DataFrame(records, columns=["m", "back"]).groupby("m")["back"].mean()
    curve = rates.reindex(range(1, MAX_MISSED + 1)).ffill().fillna(rates.iloc[0])
    return pd.Series(np.minimum.accumulate(curve.to_numpy()), index=curve.index)


def matches_missed(history: pd.DataFrame, codes) -> pd.Series:
    """Fixtures each player has missed in a row since his last appearance (0 if he played last)."""
    h = history[history["player_code"].isin(list(codes))].sort_values("kickoff_time")

    def trailing_zeros(minutes: pd.Series) -> int:
        played = np.flatnonzero(minutes.to_numpy() > 0)
        return len(minutes) - (played[-1] + 1 if len(played) else 0)
    return h.groupby("player_code")["minutes"].apply(trailing_zeros).astype(int)


def trim_absences(history: pd.DataFrame, codes) -> pd.DataFrame:
    """Drop listed players' 0-minute matches after their last appearance.

    So a returning player is judged on his form before the injury, not on the
    empty weeks he missed.
    """
    h = history
    flagged = h["player_code"].isin(list(codes))
    last_on = h[flagged & (h["minutes"] > 0)].groupby("player_code")["kickoff_time"].max()
    after = h["kickoff_time"] > h["player_code"].map(last_on)
    return h[~(flagged & after.fillna(False))]


def _chance(row) -> float | None:
    c = row["chance_of_playing_next_round"]
    return None if pd.isna(c) else float(c) / 100


def _weeks(row, gws, deadlines, curve, missed) -> list[float]:
    n = len(gws)
    chance = _chance(row)
    if row["status"] == "u":
        return [0.0] * n
    back = return_date(row["news"], pd.Timestamp(row["news_added"]))
    if back is not None:
        out, returned = [], False
        for gw in gws:
            if deadlines[gw].date() < back:
                out.append(0.0)
            else:
                out.append(1.0 if returned else FIRST_WEEK_BACK)
                returned = True
        return out
    if chance is not None and 0 < chance < 1:
        return [chance, (1 + chance) / 2] + [1.0] * (n - 2) if n >= 2 else [chance]
    if row["status"] in ("i", "s"):
        # Next week: FPL's own chance when given (e.g. Isak 0%); after that, the recovery curve.
        m = max(int(missed.get(row["code"], 0)), 1)
        still_out, out = 1.0, []
        for k in range(n):
            if k == 0 and chance is not None:
                still_out = 1 - chance
            else:
                still_out *= 1 - curve[min(m + k, MAX_MISSED)]
            out.append(1 - still_out)
        return out
    if chance is not None:
        return [chance] + [1.0] * (n - 1)
    return [1.0] * n


def weekly_availability(players, gws, deadlines, curve, missed) -> pd.DataFrame:
    """One row per flagged player per gameweek: player_code, gw, availability, news.

    players: players.csv rows (code, status, news, news_added, chance_of_playing_next_round).
    A player who isn't listed is fully available. See the part 3 spec §2 for the rules.
    """
    flagged = players[(players["status"] != "a") | players["chance_of_playing_next_round"].notna()]
    rows = []
    for _, row in flagged.iterrows():
        for gw, a in zip(gws, _weeks(row, gws, deadlines, curve, missed)):
            rows.append({"player_code": int(row["code"]), "gw": gw, "availability": a,
                         "news": row["news"]})
    return pd.DataFrame(rows, columns=["player_code", "gw", "availability", "news"])


def apply(pred: pd.DataFrame, avail: pd.DataFrame) -> pd.DataFrame:
    """Multiply per-fixture P(plays) and P(60+) by that week's availability; recompute xP."""
    out = pred.merge(avail[["player_code", "gw", "availability"]], on=["player_code", "gw"],
                     how="left")
    factor = out.pop("availability").fillna(1.0)
    out["p_plays"] = out["p_plays"] * factor
    out["p_60"] = out["p_60"] * factor
    return scoring.expected_points(out)
```

- [ ] **Step 4: Edit `fpl/models.py`.**
  - Delete the whole `apply_availability` function.
  - Change the `train_and_predict` signature and its yardstick lines as below.
  - Append `weekly_xp`.

```python
def train_and_predict(history: pd.DataFrame, player_rows: pd.DataFrame,
                      team_rows: pd.DataFrame, cutoffs: pd.DataFrame,
                      yardstick: bool = True) -> pd.DataFrame:
```

Docstring: add the line `yardstick: also fit model B (xp_b); the planner and simulation skip it to save time`. In the body, replace `model_b = TotalPointsModel().fit(train_players)` and `pred["xp_b"] = model_b.predict(players)` with:

```python
    if yardstick:
        pred["xp_b"] = TotalPointsModel().fit(train_players).predict(players)
```

placed right after `pred = model_a.predict(players, teams)`.

```python
def weekly_xp(pred: pd.DataFrame, gws) -> pd.DataFrame:
    """xP per player per gameweek: fixtures summed (double gameweeks), 0 for a blank."""
    return (pred.pivot_table(index="player_code", columns="gw", values="xp", aggfunc="sum")
            .reindex(columns=list(gws)).fillna(0.0))
```

Delete `test_apply_availability_caps_next_gameweek_only` from `tests/test_models.py`. `tests/test_availability.py::test_apply_scales_plays_and_sixty_per_week` replaces it.

- [ ] **Step 5: Create `fpl/forecast.py`:**

```python
"""Live predictions for the coming gameweeks, with injury news applied.

Shared by predict.py (ranked table) and recommend.py (planner).
"""
import pandas as pd

from fpl import availability, matches, models
from fpl.seasons import CURRENT_SEASON


def deadlines_from(gameweeks: pd.DataFrame) -> dict[int, pd.Timestamp]:
    """FPL's deadline per gameweek (UTC), from the gameweeks table."""
    return {int(i): pd.Timestamp(t) for i, t in zip(gameweeks["id"], gameweeks["deadline_time"])}


def live_forecast(history, fixtures_all, players, teams, gameweeks, gws, season=CURRENT_SEASON):
    """(per-fixture predictions with availability applied, availability table).

    Flagged players are judged on their form before their absence; their
    availability for each week comes from FPL's news (availability.py).
    """
    flagged = players.loc[players["status"] != "a", "code"].astype(int).tolist()
    avail = availability.weekly_availability(
        players, gws, deadlines_from(gameweeks), availability.recovery_curve(history),
        availability.matches_missed(history, flagged))
    cutoffs = matches.gw_cutoffs(fixtures_all)
    asof = cutoffs[(cutoffs["season"] == season) & (cutoffs["gw"] == gws[0])]["cutoff"].iloc[0]
    player_rows = matches.upcoming_player_rows(fixtures_all, players, teams, season, gws)
    team_rows = matches.upcoming_team_rows(fixtures_all, season, gws)
    pred = models.train_and_predict(availability.trim_absences(history, flagged),
                                    player_rows.assign(asof=asof), team_rows.assign(asof=asof),
                                    cutoffs)
    return availability.apply(pred, avail), avail
```

- [ ] **Step 6: Update `predict.py`.** Replace the block from `cutoffs = matches.gw_cutoffs(fixtures_all)` through `pred = models.apply_availability(pred, players, next_gw)` with:

```python
    print(f"Training on {len(history):,} player-matches and predicting GW{gws[0]}"
          f"{f'-{gws[-1]}' if len(gws) > 1 else ''}...", flush=True)
    pred, _ = forecast.live_forecast(history, fixtures_all, players, teams, gameweeks, gws)
```

Change the import line to `from fpl import forecast, storage`, and drop the `CURRENT_SEASON` and `models` imports if they become unused. Keep `matches` (`ranked_table` uses it).

- [ ] **Step 7: Run the tests.** Run `python -m pytest -q`. Expected: all PASS, including part 2's (minus the one deleted availability test).

- [ ] **Step 8: Smoke-test live.** Run `python predict.py --horizon 6`. Expected:
  - the table prints;
  - Isak, Tzolis and McNally have xP_gw7 = 0 (McNally isn't listed, since status `u`);
  - Tzolis has non-zero xP from GW8;
  - Robinson's P(plays) is at most 0.75.

- [ ] **Step 9: Commit and push**

```bash
git add fpl/availability.py fpl/forecast.py fpl/models.py predict.py tests
git commit -m "Apply injury news to every planned week; judge returning players on pre-injury form

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push
```

---

### Task 5: Clean-sheet fix (`fpl/team_ratings.py`)

**Files:**
- Create: `fpl/team_ratings.py`, `tests/test_team_ratings.py`
- Modify: `fpl/features.py` (`_team_values`, `_add_team_features`, `team_features`), `fpl/models.py` (`XPModel` uses `CS_FEATURES`), `fpl/backtest.py` (`clean_sheet_report`), `backtest.py` (print it)

**Interfaces:**
- Produces in `fpl/team_ratings.py`:
  - `RATING_FEATURES = ["rating_xg_for", "rating_xg_against", "rating_cs"]`
  - `fit_ratings(team_matches, asof) -> dict | None`
  - `rating_values(rows, team_matches) -> DataFrame[rating_xg_for, rating_xg_against]`
- Produces in `fpl/models.py`: `CS_FEATURES = TEAM_FEATURES + RATING_FEATURES`.
- Produces in `fpl/backtest.py`: `clean_sheet_report(preds) -> dict(p5, p95, log_loss, actual_by_fifth)`.

- [ ] **Step 1: Record part 2's numbers before changing anything.** Run:

```bash
python - <<'EOF'
import shutil
shutil.copy("data/processed/backtest_predictions.csv", "data/processed/backtest_predictions_part2.csv")
EOF
```

- [ ] **Step 2: Write failing tests.** Create `tests/test_team_ratings.py`:

```python
"""Offline tests for fpl/team_ratings.py and the rating features."""
import numpy as np
import pandas as pd
import pytest

from conftest import make_matches
from fpl import team_ratings
from fpl.features import FeatureBuilder
from fpl.matches import TEAM_ROW_COLUMNS


def league(seed=0, n_rounds=20):
    """6 teams; 'Wall' concedes little xG, 'Sieve' a lot."""
    rng = np.random.default_rng(seed)
    teams = ["Wall", "Sieve", "A", "B", "C", "D"]
    concede = {"Wall": 0.5, "Sieve": 2.5, "A": 1.3, "B": 1.3, "C": 1.3, "D": 1.3}
    rows, start = [], pd.Timestamp("2025-08-16", tz="UTC")
    fid = 0
    for r in range(n_rounds):
        order = rng.permutation(teams)
        for h, a in zip(order[::2], order[1::2]):
            fid += 1
            when = start + pd.Timedelta(days=7 * r)
            for team, opp, home in [(h, a, True), (a, h, False)]:
                rows.append({"season": "2025-26", "gw": r + 1, "fixture_id": fid, "kickoff_time": when,
                             "team": team, "opponent": opp, "was_home": home,
                             "xg_for": rng.gamma(4, concede[opp] / 4)})
    return pd.DataFrame(rows)


def ask(team, opponent, asof):
    return pd.DataFrame({"team": [team], "opponent": [opponent], "was_home": [True], "asof": [asof]})


def test_ratings_see_strong_and_weak_defences():
    tm = league()
    asof = pd.Timestamp("2026-06-01", tz="UTC")
    vs_wall = team_ratings.rating_values(ask("A", "Wall", asof), tm).iloc[0]
    vs_sieve = team_ratings.rating_values(ask("A", "Sieve", asof), tm).iloc[0]
    assert vs_wall["rating_xg_for"] < 0.8 < 1.8 < vs_sieve["rating_xg_for"]


def test_ratings_ignore_future_matches():
    tm = league()
    asof = pd.Timestamp("2025-12-01", tz="UTC")
    before = team_ratings.rating_values(ask("A", "Wall", asof), tm).iloc[0]
    changed = tm.copy()
    changed.loc[changed["kickoff_time"] >= asof, "xg_for"] = 9.0
    team_ratings._cache.clear()
    after = team_ratings.rating_values(ask("A", "Wall", asof), changed).iloc[0]
    assert after["rating_xg_for"] == pytest.approx(before["rating_xg_for"])


def test_too_little_history_gives_nan():
    tm = league(n_rounds=2)
    out = team_ratings.rating_values(ask("A", "Wall", pd.Timestamp("2026-01-01", tz="UTC")), tm)
    assert out.isna().all(axis=None)


def test_team_features_include_ratings_and_promoted_prior():
    """Echo is promoted for 2025-26: before its first match its rating is the promoted prior."""
    history = make_matches(gws=30)
    start = pd.Timestamp("2025-08-16", tz="UTC")           # 2025-26 opening kickoff in conftest
    builder = FeatureBuilder(history[history.kickoff_time < start])
    rows = pd.DataFrame([{"season": "2025-26", "gw": 1, "fixture_id": 1, "kickoff_time": start,
                          "team": team, "opponent": opp, "was_home": home, "asof": start}
                         for team, opp, home in [("Echo", "Alpha", True), ("Alpha", "Echo", False)]])
    out = builder.team_features(rows[TEAM_ROW_COLUMNS + ["asof"]])
    for col in team_ratings.RATING_FEATURES:
        assert col in out and out[col].notna().all()
    assert out["rating_cs"].between(0, 1).all()
    promoted = builder.team_priors["promoted"]
    assert out.loc[0, "rating_xg_against"] == pytest.approx(promoted["xg_against"])
```

- [ ] **Step 3: Run to verify they fail.** Run `python -m pytest tests/test_team_ratings.py -v`. Expected: FAIL (`cannot import name 'team_ratings'`).

- [ ] **Step 4: Implement `fpl/team_ratings.py`:**

```python
"""Time-weighted attack and defence ratings for every team (the clean-sheet fix).

Part 2's team features average only each team's last 6 matches, which is too
noisy: on 2025/26 predicted clean-sheet chances spanned 17-33% while real rates
differed about twice as much. Here: log(xG) = base + attack[team] +
defence[opponent] + home, fitted on every team-match before the cutoff, with a
match's weight halving every RATING_HALF_LIFE_DAYS. xG is used rather than
goals because it is less noisy. Fits are cached by cutoff.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor

from fpl.settings import RATING_HALF_LIFE_DAYS

RATING_FEATURES = ["rating_xg_for", "rating_xg_against", "rating_cs"]
MIN_MATCHES = 40          # team-matches needed before a fit is trusted
_cache: dict = {}


def fit_ratings(team_matches: pd.DataFrame, asof: pd.Timestamp) -> dict | None:
    """Ratings from matches strictly before `asof`; None if there are too few."""
    past = team_matches[(team_matches["kickoff_time"] < asof) & team_matches["xg_for"].notna()]
    if len(past) < MIN_MATCHES:
        return None
    key = (asof, len(past), round(float(past["xg_for"].sum()), 6))
    if key in _cache:
        return _cache[key]
    teams = sorted(set(past["team"]) | set(past["opponent"]))
    index = {t: i for i, t in enumerate(teams)}
    n, rows = len(teams), np.arange(len(past))
    X = np.zeros((len(past), 2 * n + 1))
    X[rows, past["team"].map(index).to_numpy()] = 1
    X[rows, n + past["opponent"].map(index).to_numpy()] = 1
    X[:, 2 * n] = past["was_home"].astype(float).to_numpy()
    age_days = (asof - past["kickoff_time"]).dt.total_seconds().to_numpy() / 86400
    model = PoissonRegressor(alpha=1e-3, max_iter=300).fit(
        X, past["xg_for"].to_numpy(), sample_weight=0.5 ** (age_days / RATING_HALF_LIFE_DAYS))
    coef = model.coef_
    fit = {"base": float(model.intercept_), "home": float(coef[2 * n]),
           "attack": dict(zip(teams, coef[:n])), "defence": dict(zip(teams, coef[n:2 * n]))}
    _cache[key] = fit
    return fit


def rating_values(rows: pd.DataFrame, team_matches: pd.DataFrame) -> pd.DataFrame:
    """Ratings' expected goals for and against each row (team, opponent, was_home, asof).

    NaN where a team is unrated (no fit yet, or a team with no past matches).
    """
    out = np.full((len(rows), 2), np.nan)
    for asof, idx in rows.groupby("asof").indices.items():
        fit = fit_ratings(team_matches, asof)
        if fit is None:
            continue
        r = rows.iloc[idx]
        home = r["was_home"].astype(float).to_numpy()
        att_t = r["team"].map(fit["attack"]).to_numpy(dtype=float)
        att_o = r["opponent"].map(fit["attack"]).to_numpy(dtype=float)
        def_t = r["team"].map(fit["defence"]).to_numpy(dtype=float)
        def_o = r["opponent"].map(fit["defence"]).to_numpy(dtype=float)
        out[idx, 0] = np.exp(fit["base"] + att_t + def_o + fit["home"] * home)
        out[idx, 1] = np.exp(fit["base"] + att_o + def_t + fit["home"] * (1 - home))
    return pd.DataFrame(out, columns=["rating_xg_for", "rating_xg_against"], index=rows.index)
```

- [ ] **Step 5: Wire the ratings into `fpl/features.py`.** Add `from fpl import scoring, team_ratings` (replacing `from fpl import scoring`).

In `_team_values`, inside the `for stat in TEAM_STATS:` loop, after computing `prior`, add `out[f"prior_{stat}"] = prior.to_numpy()`. Before `return out`, add `out["n"] = n.to_numpy()`.

Replace `_add_team_features` and `team_features` with:

```python
    def _add_team_features(self, rows: pd.DataFrame, with_ratings: bool = False) -> pd.DataFrame:
        own = self._team_values(rows["team"], rows["season"], rows["asof"])
        opp = self._team_values(rows["opponent"], rows["season"], rows["asof"])
        for stat in TEAM_STATS:
            rows[f"team_{stat}"] = own[stat]
            rows[f"opp_{stat}"] = opp[stat]
        if with_ratings:
            # Promoted teams have no rating yet: blend towards the same prior as the rolling stats.
            raw = team_ratings.rating_values(rows, self.team_matches)
            n = pd.Series(own["n"], index=rows.index)
            for col, stat in [("rating_xg_for", "xg_for"), ("rating_xg_against", "xg_against")]:
                prior = pd.Series(own[f"prior_{stat}"], index=rows.index)
                rows[col] = blend(raw[col], n, prior, TEAM_K).to_numpy()
            rows["rating_cs"] = np.exp(-rows["rating_xg_against"])
        rows["was_home"] = rows["was_home"].astype(int)
        rows["rules_era_code"] = rows["season"].map(
            lambda s: scoring.RULES_ERA_CODE[scoring.rules_era(s)])
        return rows

    def team_features(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Add TEAM_FEATURES and the rating features to team rows (TEAM_ROW_COLUMNS + asof)."""
        _check_asof(rows)
        return self._add_team_features(rows.copy(), with_ratings=True)
```

`player_features` keeps calling `self._add_team_features(out)` (no ratings), so `PLAYER_FEATURES` is unchanged.

- [ ] **Step 6: Use the ratings in `fpl/models.py`.** Add `from fpl.team_ratings import RATING_FEATURES` and, below the imports, `CS_FEATURES = TEAM_FEATURES + RATING_FEATURES  # clean-sheet and goals-conceded models only`. In `XPModel.fit` and `XPModel.predict`, change `T = teams[TEAM_FEATURES]` to `T = teams[CS_FEATURES]`.

- [ ] **Step 7: Add `clean_sheet_report` to `fpl/backtest.py`:**

```python
def clean_sheet_report(preds: pd.DataFrame) -> dict:
    """Spread and accuracy of P(clean sheet) one week ahead, one row per team-fixture."""
    h1 = preds[preds["horizon"] == 1].drop_duplicates(["season", "fixture_id", "team"])
    kept = (h1["opp_goals"] == 0).astype(int)
    p = h1["p_cs"].clip(1e-6, 1 - 1e-6)
    fifths = pd.qcut(p, 5, labels=False, duplicates="drop")
    return {"p5": float(p.quantile(0.05)), "p95": float(p.quantile(0.95)),
            "log_loss": float(-(kept * np.log(p) + (1 - kept) * np.log(1 - p)).mean()),
            "actual_by_fifth": kept.groupby(fifths).mean().round(3).tolist()}
```

In `backtest.py`, after printing the summary, add for each season:

```python
    for p in preds:
        season = p["season"].iloc[0]
        cs = backtest.clean_sheet_report(p)
        print(f"\n{season} clean sheets: P(CS) 5th-95th percentile {cs['p5']:.2f}-{cs['p95']:.2f}, "
              f"log loss {cs['log_loss']:.4f}, actual rate by predicted fifth {cs['actual_by_fifth']}")
```

Add a test to `tests/test_backtest.py`:

```python
def test_clean_sheet_report_on_hand_made_numbers():
    preds = pd.DataFrame({"horizon": 1, "season": "s", "fixture_id": range(10), "team": "T",
                          "p_cs": [0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5],
                          "opp_goals": [1, 1, 0, 0, 0, 1, 1, 1, 0, 0]})
    cs = backtest.clean_sheet_report(preds)
    assert cs["p5"] == pytest.approx(0.1) and cs["p95"] == pytest.approx(0.5)
    assert cs["actual_by_fifth"] == [0.0, 0.0, 0.5, 1.0, 1.0]
```

(Import `pytest` and `pandas as pd` if `tests/test_backtest.py` doesn't already.)

- [ ] **Step 8: Run the tests.** Run `python -m pytest -q`. Expected: all PASS.

- [ ] **Step 9: Acceptance check on 2025/26 (about 10–20 minutes).** Run in the background:
  - `python backtest.py --season 2025-26 --gws 1-38 --horizon 1`
  - then: `python -c "import pandas as pd; from fpl import backtest; print(backtest.clean_sheet_report(pd.read_csv('data/processed/backtest_predictions_part2.csv')))"`

  Accept only if all of these hold:
  - the new 5th–95th percentile range is wider than part 2's (about 0.17–0.33);
  - the new log loss is ≤ part 2's;
  - the actual rate rises across the fifths;
  - "A beats baseline on MAE: yes" and "on top-20 points: yes" both still print.

  If the spread widens but log loss gets worse, **stop and report to the user** (spec §3). Record the before/after numbers in the commit message.

- [ ] **Step 10: Full backtest so part 2's outputs are current.** Run `python backtest.py` in the background (20–45 min). Check that A still beats the baseline for 2025/26 at horizon 1.

- [ ] **Step 11: Commit and push**

```bash
git add fpl/team_ratings.py fpl/features.py fpl/models.py fpl/backtest.py backtest.py tests
git commit -m "Add time-weighted team ratings to the clean-sheet model

2025/26, one week ahead: P(CS) 5-95% <before> -> <after>; log loss <before> -> <after>.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push
```

---

### Task 6: The planner (`fpl/optimizer.py`)

**Files:**
- Create: `fpl/optimizer.py`, `tests/test_optimizer.py`

**Interfaces:**
- Consumes: `rules.*`, `settings.PlanSettings`, `settings.SOLVER_TIME_LIMIT`, `settings.CANDIDATES_PER_POSITION`, `settings.VALUE_CANDIDATES_PER_POSITION`, `settings.CHEAPEST_PER_POSITION`.
- Produces:
  - dataclass `PlanInput(pool, xp, squad, bank, free_transfers, chips, season, hold_values={})`:
    - `pool`: index player_code; columns name, position, team, price, sell (ints for prices);
    - `xp`: index = pool index; one column per gameweek, in order.
  - dataclass `WeekPlan(gw, chip, transfers_in, transfers_out, free_transfers, paid, owned, squad, starters, bench, captain, vice, xp, bank)`:
    - `owned` = the squad you own after this week's transfers;
    - `squad` = the 15 who play (the Free Hit squad in a Free Hit week);
    - `free_transfers` is None in a new team's first week.
  - dataclass `Plan(weeks, value, status)`. `value` = Σ discount^i · week xp (hits included; margin, end value and holds excluded).
  - `solve(inp, settings=PlanSettings(), week1_hits="free"|"none"|"some", fixed_squad=False, time_limit=SOLVER_TIME_LIMIT) -> Plan`. Raises `ValueError` with a plain message if there is no valid plan.
  - `best_squad(xp, pool, budget, season, time_limit=20) -> Plan`
  - `candidate_pool(xp, pool, keep, discount, per_position=..., value_per_position=..., cheapest=...) -> list[int]`

- [ ] **Step 1: Write failing tests.** Create `tests/test_optimizer.py`:

```python
"""Offline tests for fpl/optimizer.py on small made-up player pools."""
import pandas as pd
import pytest

from fpl import optimizer, rules
from fpl.settings import PlanSettings

FLAT = PlanSettings(discount=1.0, hit_margin=1.0, ft_end_value=1.5, bench_weight=0.1)


def make_pool():
    """30 players: 4 GK, 10 DEF, 10 MID, 6 FWD over 10 clubs; prices 45-65."""
    rows, code = [], 1
    for position, count in [("GK", 4), ("DEF", 10), ("MID", 10), ("FWD", 6)]:
        for i in range(count):
            rows.append({"player_code": code, "name": f"{position}{i}", "position": position,
                         "team": f"C{code % 10}", "price": 45 + 5 * (i % 5)})
            code += 1
    pool = pd.DataFrame(rows).set_index("player_code")
    pool["sell"] = pool["price"]
    return pool


def owned_squad(pool):
    """A legal 15: the first 2 GK, 5 DEF, 5 MID, 3 FWD by code."""
    return [c for pos, k in rules.SQUAD_SHAPE.items() for c in pool[pool.position == pos].index[:k]]


def flat_xp(pool, gws, value=2.0):
    return pd.DataFrame(value, index=pool.index, columns=gws)


def plan_input(pool, xp, squad, bank=100, free=1, chips=(), season="2026-27", holds=None):
    return optimizer.PlanInput(pool=pool, xp=xp, squad=squad, bank=bank, free_transfers=free,
                               chips=list(chips), season=season, hold_values=holds or {})


def test_new_team_obeys_every_squad_rule():
    pool = make_pool()
    xp = flat_xp(pool, [1, 2])
    xp.loc[pool.index, 1] = range(1, len(pool) + 1)             # distinct values
    plan = optimizer.solve(plan_input(pool, xp, [], bank=rules.BUDGET, free=0), FLAT)
    week = plan.weeks[0]
    chosen = pool.loc[week.owned]
    assert chosen.position.value_counts().to_dict() == rules.SQUAD_SHAPE
    assert chosen.team.value_counts().max() <= rules.MAX_PER_CLUB
    assert chosen.price.sum() + week.bank == rules.BUDGET and week.bank >= 0
    assert rules.valid_xi(pool.loc[week.starters, "position"])
    assert week.captain == max(week.starters, key=lambda p: xp.at[p, 1])
    assert week.vice != week.captain and week.vice in week.starters
    assert pool.at[week.bench[0], "position"] == "GK" and len(week.bench) == 4
    assert plan.weeks[1].free_transfers == 1


@pytest.mark.parametrize("gain,expect_hit", [(4.5, False), (5.5, True)])
def test_hit_only_when_gain_beats_four_plus_margin(gain, expect_hit):
    pool = make_pool()
    squad = owned_squad(pool)
    xp = flat_xp(pool, [7])
    star = pool[(pool.position == "MID") & ~pool.index.isin(squad)].index[0]
    xp.loc[squad[-1], 7] = 10.0                                   # captain stays fixed
    xp.loc[star, 7] = 2.0 + gain
    plan = optimizer.solve(plan_input(pool, xp, squad, bank=100, free=0), FLAT)
    assert (star in plan.weeks[0].transfers_in) == expect_hit
    assert plan.weeks[0].paid == (1 if expect_hit else 0)


def test_rolls_a_free_transfer_when_nothing_is_worth_buying_and_caps_at_five():
    pool = make_pool()
    squad = owned_squad(pool)
    plan = optimizer.solve(plan_input(pool, flat_xp(pool, [7, 8]), squad, free=1), FLAT)
    assert plan.weeks[0].transfers_in == [] and plan.weeks[1].free_transfers == 2
    plan = optimizer.solve(plan_input(pool, flat_xp(pool, [7, 8]), squad, free=5), FLAT)
    assert plan.weeks[1].free_transfers == 5


def test_wildcard_keeps_free_transfers_without_adding_one():
    pool = make_pool()
    squad = owned_squad(pool)
    xp = flat_xp(pool, [7, 8], value=1.0)
    others = [p for p in pool.index if p not in squad]
    xp.loc[others, :] = 5.0                                      # everyone you don't own is better
    plan = optimizer.solve(plan_input(pool, xp, squad, bank=0, free=2,
                                      chips=[rules.Chip("wildcard", 2, 19)]), FLAT)
    assert plan.weeks[0].chip == "wildcard" and plan.weeks[0].paid == 0
    assert len(plan.weeks[0].transfers_in) > 2
    assert plan.weeks[1].free_transfers == 2


def test_free_hit_squad_reverts_the_next_week():
    pool = make_pool()
    squad = owned_squad(pool)
    xp = flat_xp(pool, [7, 8])
    others = [p for p in pool.index if p not in squad]
    xp.loc[others, 7] = 9.0                                      # only good for one week
    plan = optimizer.solve(plan_input(pool, xp, squad, free=1,
                                      chips=[rules.Chip("freehit", 2, 19)]), FLAT)
    week1, week2 = plan.weeks
    assert week1.chip == "freehit" and week1.transfers_in == []
    assert set(week1.squad) != set(squad) and set(week2.squad) == set(squad)
    assert week2.free_transfers == 1                             # kept, no +1


def test_chip_windows_and_hold_values_are_respected():
    pool = make_pool()
    squad = owned_squad(pool)
    xp = flat_xp(pool, [7, 8])
    late_tc = rules.Chip("3xc", 20, 38)
    plan = optimizer.solve(plan_input(pool, xp, squad, chips=[late_tc]), FLAT)
    assert all(w.chip is None for w in plan.weeks)               # outside its window
    tc = rules.Chip("3xc", 1, 19)
    kept = optimizer.solve(plan_input(pool, xp, squad, chips=[tc], holds={tc.key: 50.0}), FLAT)
    used = optimizer.solve(plan_input(pool, xp, squad, chips=[tc], holds={tc.key: 0.0}), FLAT)
    assert all(w.chip is None for w in kept.weeks)
    assert sum(w.chip == "3xc" for w in used.weeks) == 1


def test_selling_prices_limit_what_you_can_buy():
    pool = make_pool()
    squad = owned_squad(pool)
    seller = squad[2]                                            # a DEF
    pool.loc[squad, "sell"] = 45                                 # no other sale can fund the target
    pool.loc[seller, ["price", "sell"]] = [60, 50]
    target = pool[(pool.position == "DEF") & ~pool.index.isin(squad)].index[0]
    pool.loc[target, ["price", "sell"]] = [55, 55]
    xp = flat_xp(pool, [7])
    xp.loc[target, 7] = 20.0
    xp.loc[squad[2:7], 7] = [0.0, 3.0, 3.0, 3.0, 3.0]            # the seller is the one to replace
    poor = optimizer.solve(plan_input(pool, xp, squad, bank=4, free=1), FLAT)
    rich = optimizer.solve(plan_input(pool, xp, squad, bank=5, free=1), FLAT)
    assert target not in poor.weeks[0].transfers_in and target in rich.weeks[0].transfers_in


def test_afcon_top_up_gives_five_free_transfers_in_2025_26_gw16():
    pool = make_pool()
    squad = owned_squad(pool)
    xp = flat_xp(pool, [16], value=1.0)
    xp.loc[[p for p in pool.index if p not in squad], 16] = 4.0
    plan = optimizer.solve(plan_input(pool, xp, squad, bank=200, free=1, season="2025-26"), FLAT)
    assert plan.weeks[0].free_transfers == 5 and plan.weeks[0].paid == 0


def test_a_player_with_no_predictions_is_benched_or_sold():
    pool = make_pool()
    squad = owned_squad(pool)
    xp = flat_xp(pool, [7, 8])
    xp.loc[squad[3], :] = 0.0                                    # e.g. loaned out (McNally)
    week = optimizer.solve(plan_input(pool, xp, squad, free=1), FLAT).weeks[0]
    assert squad[3] not in week.starters


def test_forced_hit_is_a_genuine_extra_transfer():
    """With 1 free transfer, 'some' must make 2+ transfers and must not earn a free transfer next week."""
    pool = make_pool()
    squad = owned_squad(pool)
    xp = flat_xp(pool, [7, 8])
    chips = [rules.Chip("wildcard", 2, 19)]
    forced = optimizer.solve(plan_input(pool, xp, squad, free=1, chips=chips), FLAT, week1_hits="some")
    week1, week2 = forced.weeks
    assert week1.chip is None
    assert len(week1.transfers_in) >= 2
    assert week1.paid == len(week1.transfers_in) - 1
    assert week2.free_transfers == 1
    none = optimizer.solve(plan_input(pool, xp, squad, free=1), FLAT, week1_hits="none")
    assert none.weeks[0].paid == 0


def test_paid_hits_equal_transfers_beyond_the_free_ones():
    pool = make_pool()
    squad = owned_squad(pool)
    xp = flat_xp(pool, [7, 8])
    for week in optimizer.solve(plan_input(pool, xp, squad, free=2), FLAT, week1_hits="some").weeks:
        if week.free_transfers is not None and week.chip is None:
            assert week.paid == max(0, len(week.transfers_in) - week.free_transfers)


def test_impossible_budget_raises_a_clear_error():
    pool = make_pool()
    with pytest.raises(ValueError, match="no valid plan"):
        optimizer.solve(plan_input(pool, flat_xp(pool, [1]), [], bank=100, free=0), FLAT)


def test_candidate_pool_keeps_owned_and_cheap_players():
    pool = make_pool()
    xp = flat_xp(pool, [7])
    keep = optimizer.candidate_pool(xp, pool, [30], 0.9, per_position=1, value_per_position=1, cheapest=1)
    assert 30 in keep and len(keep) <= 1 + 4 * 3
```

- [ ] **Step 2: Run to verify they fail.** Run `python -m pytest tests/test_optimizer.py -v`. Expected: FAIL (`cannot import name 'optimizer'`).

- [ ] **Step 3: Implement `fpl/optimizer.py`:**

```python
"""The multi-week planner: squad, transfers, captain and chips over the next few gameweeks.

One integer linear program (PuLP + CBC) chooses every planned week at once,
so it can weigh "transfer now" against "save it for a double move later".
Prices are FPL units (tenths of £1m) and assumed constant over the plan.
The rules and the objective are in the part 3 spec, section 4.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pulp

from fpl import rules
from fpl.settings import (CANDIDATES_PER_POSITION, CHEAPEST_PER_POSITION, SOLVER_TIME_LIMIT,
                          VALUE_CANDIDATES_PER_POSITION, PlanSettings)

BIG = 15                                  # more than any week's transfer count


@dataclass
class PlanInput:
    pool: pd.DataFrame                    # index player_code: name, position, team, price, sell
    xp: pd.DataFrame                      # index = pool index; one column per gameweek
    squad: list[int]                      # current 15 player codes ([] = pick a new team)
    bank: int
    free_transfers: int                   # for the first planned gameweek
    chips: list                           # rules.Chip copies still available
    season: str
    hold_values: dict = field(default_factory=dict)   # chip key -> points if kept past the plan


@dataclass
class WeekPlan:
    gw: int
    chip: str | None
    transfers_in: list[int]
    transfers_out: list[int]
    free_transfers: int | None            # None: a new team's first week (unlimited)
    paid: int
    owned: list[int]                      # your squad after this week's transfers
    squad: list[int]                      # the 15 who play (Free Hit squad in a Free Hit week)
    starters: list[int]
    bench: list[int]                      # [backup GK, first outfield sub, second, third]
    captain: int
    vice: int
    xp: float                             # expected points incl. captain and chip, minus hits
    bank: int


@dataclass
class Plan:
    weeks: list[WeekPlan]
    value: float                          # discounted xP minus hits (no margin, end value or holds)
    status: str


def candidate_pool(xp: pd.DataFrame, pool: pd.DataFrame, keep, discount: float,
                   per_position: int = CANDIDATES_PER_POSITION,
                   value_per_position: int = VALUE_CANDIDATES_PER_POSITION,
                   cheapest: int = CHEAPEST_PER_POSITION) -> list[int]:
    """Players worth giving the solver: best by discounted xP, by xP per £m, the cheapest, and `keep`."""
    weights = discount ** np.arange(xp.shape[1])
    score = (xp * weights).sum(axis=1)
    chosen = set(int(k) for k in keep)
    for _, group in pool.groupby("position"):
        s = score.reindex(group.index).fillna(0)
        chosen |= set(s.nlargest(per_position).index)
        chosen |= set((s / group["price"]).nlargest(value_per_position).index)
        chosen |= set(group["price"].nsmallest(cheapest).index)
    return sorted(int(c) for c in chosen)


def solve(inp: PlanInput, settings: PlanSettings = PlanSettings(), week1_hits: str = "free",
          fixed_squad: bool = False, time_limit: int = SOLVER_TIME_LIMIT) -> Plan:
    """Best plan for the gameweeks in inp.xp's columns.

    week1_hits: 'free' (the solver decides), 'none' (no hit in week 1) or 'some'
    (at least one transfer beyond the free ones, no Wildcard/Free Hit), used to
    explain the hit verdict. fixed_squad: no transfers
    after week 1 (chip scan's best squad). Raises ValueError if no plan is valid.
    """
    gws = [int(g) for g in inp.xp.columns]
    P = [int(p) for p in inp.pool.index]
    pos, club = inp.pool["position"], inp.pool["team"]
    price = inp.pool["price"].astype(int)
    sale = inp.pool["sell"].astype(int)
    xp = inp.xp
    new_team = not inp.squad
    owned0 = set(inp.squad)
    prob = pulp.LpProblem("fpl_plan", pulp.LpMaximize)

    def binary(name):
        return pulp.LpVariable(name, cat="Binary")

    use = {(ch.key, w): binary(f"use_{ch.key}_{w}") for ch in inp.chips for w in gws if ch.allows(w)}

    def chip_flag(name, w):
        return pulp.lpSum(v for (k, ww), v in use.items() if ww == w and k.startswith(name + "_"))

    def chip_possible(name, w):
        return any(k.startswith(name + "_") and ww == w for k, ww in use)

    x, tin, tout, z, s, c, y, tcp, bbp = ({} for _ in range(9))
    paid, ft, n_in = {}, {}, {}
    objective, value_terms = [], []
    prev_x = {p: int(p in owned0) for p in P}
    bank_prev = inp.bank
    for i, w in enumerate(gws):
        for p in P:
            for d, tag in ((x, "x"), (tin, "in"), (tout, "out"), (s, "s"), (c, "c")):
                d[p, w] = binary(f"{tag}_{p}_{w}")
            prob += x[p, w] == prev_x[p] + tin[p, w] - tout[p, w]
            prob += tin[p, w] + tout[p, w] <= 1
        for position, k in rules.SQUAD_SHAPE.items():
            prob += pulp.lpSum(x[p, w] for p in P if pos[p] == position) == k
        for team in club.unique():
            prob += pulp.lpSum(x[p, w] for p in P if club[p] == team) <= rules.MAX_PER_CLUB
        bank_w = bank_prev + pulp.lpSum(sale[p] * tout[p, w] - price[p] * tin[p, w] for p in P)
        prob += bank_w >= 0
        n_in[w] = pulp.lpSum(tin[p, w] for p in P)
        paid[w] = pulp.LpVariable(f"paid_{w}", lowBound=0, cat="Integer")
        prob += paid[w] <= n_in[w]
        wc, fh = chip_flag("wildcard", w), chip_flag("freehit", w)
        if fixed_squad and i > 0:
            prob += n_in[w] == 0

        # Free transfers: see rules.next_free_transfers. ft[w] is the count for week w.
        if new_team and i == 0:
            prob += paid[w] == 0
            ft[w] = None
        else:
            ft[w] = pulp.LpVariable(f"ft_{w}", 0, rules.MAX_FREE_TRANSFERS, cat="Integer")
            top_up = rules.FREE_TRANSFER_TOP_UPS.get(inp.season, {}).get(w)
            if top_up is not None:
                prob += ft[w] == top_up
            elif i == 0:
                prob += ft[w] == inp.free_transfers
            elif ft[gws[i - 1]] is None:
                prob += ft[w] == 1
            else:
                _link_free_transfers(prob, ft[w], gws[i - 1], ft, n_in, paid, chip_flag)
            # paid is exactly max(0, transfers - free) in a normal week and 0 in a Wildcard week,
            # so _link_free_transfers can't be gamed by "paying" for a transfer that was free.
            hit = binary(f"hit_{w}")
            prob += paid[w] >= n_in[w] - ft[w] - BIG * wc
            prob += paid[w] <= n_in[w] - ft[w] + BIG * (1 - hit) + BIG * wc
            prob += paid[w] <= BIG * hit
            prob += paid[w] <= BIG * (1 - wc)

        # The 15 who play: your squad, or the Free Hit squad in a Free Hit week.
        if chip_possible("freehit", w):
            prob += n_in[w] <= BIG * (1 - fh)
            for p in P:
                y[p, w] = binary(f"y_{p}_{w}")
                z[p, w] = binary(f"z_{p}_{w}")
                prob += z[p, w] <= x[p, w] + fh
                prob += z[p, w] <= y[p, w] + 1 - fh
                prob += z[p, w] >= x[p, w] - fh
                prob += z[p, w] >= y[p, w] - (1 - fh)
            for position, k in rules.SQUAD_SHAPE.items():
                prob += pulp.lpSum(y[p, w] for p in P if pos[p] == position) == k * fh
            for team in club.unique():
                prob += pulp.lpSum(y[p, w] for p in P if club[p] == team) <= rules.MAX_PER_CLUB
            # Budget = what the squad is worth plus the bank (y is all 0 when no Free Hit is played).
            prob += (pulp.lpSum(sale[p] * y[p, w] for p in P)
                     <= bank_prev + pulp.lpSum(sale[p] * prev_x[p] for p in P))
        else:
            for p in P:
                z[p, w] = x[p, w]

        prob += pulp.lpSum(s[p, w] for p in P) == 11
        prob += pulp.lpSum(c[p, w] for p in P) == 1
        for p in P:
            prob += s[p, w] <= z[p, w]
            prob += c[p, w] <= s[p, w]
        for position, (lo, hi) in rules.XI_LIMITS.items():
            starters = pulp.lpSum(s[p, w] for p in P if pos[p] == position)
            prob += starters >= lo
            prob += starters <= hi
        if chip_possible("3xc", w):
            for p in P:
                tcp[p, w] = binary(f"tc_{p}_{w}")
                prob += tcp[p, w] <= c[p, w]
                prob += tcp[p, w] <= chip_flag("3xc", w)
        if chip_possible("bboost", w):
            for p in P:
                bbp[p, w] = binary(f"bb_{p}_{w}")
                prob += bbp[p, w] <= z[p, w] - s[p, w]
                prob += bbp[p, w] <= chip_flag("bboost", w)
        prob += pulp.lpSum(v for (k, ww), v in use.items() if ww == w) <= 1

        weight = settings.discount ** i
        points = pulp.lpSum(
            float(xp.at[p, w]) * (s[p, w] + c[p, w] + tcp.get((p, w), 0)
                                  + settings.bench_weight * (z[p, w] - s[p, w])
                                  + (1 - settings.bench_weight) * bbp.get((p, w), 0))
            for p in P)
        objective.append(weight * (points - (rules.HIT_COST + settings.hit_margin) * paid[w]))
        value_terms.append(weight * (points - rules.HIT_COST * paid[w]))
        prev_x = {p: x[p, w] for p in P}
        bank_prev = bank_w

    first = gws[0]
    if week1_hits == "none":
        prob += paid[first] == 0
    elif week1_hits == "some":
        # A genuine extra transfer: more transfers than free ones, with no chip making them free.
        prob += n_in[first] >= ft[first] + 1
        prob += chip_flag("wildcard", first) + chip_flag("freehit", first) == 0
    for ch in inp.chips:
        uses = [v for (k, _), v in use.items() if k == ch.key]
        if uses:
            prob += pulp.lpSum(uses) <= 1
        objective.append(inp.hold_values.get(ch.key, 0.0) * (1 - pulp.lpSum(uses)))
    last = gws[-1]
    ft_end = pulp.LpVariable("ft_end", 0, rules.MAX_FREE_TRANSFERS, cat="Integer")
    if ft[last] is None:
        prob += ft_end == 1
    else:
        _link_free_transfers(prob, ft_end, last, ft, n_in, paid, chip_flag)
    objective.append(settings.ft_end_value * ft_end)
    prob += pulp.lpSum(objective)

    prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit, gapRel=0.001))
    # sol_status 1 = proven best; 2 = a valid plan found before the time limit; else none.
    if prob.sol_status not in (pulp.LpSolutionOptimal, pulp.LpSolutionIntegerFeasible):
        raise ValueError(f"no valid plan ({pulp.LpStatus[prob.status]}): check the budget and squad")
    status = "Optimal" if prob.sol_status == pulp.LpSolutionOptimal else "time limit"
    return _read_plan(inp, gws, P, xp, x, tin, tout, z, s, c, use, paid, ft, settings, status)


def _link_free_transfers(prob, ft_next, w, ft, n_in, paid, chip_flag) -> None:
    """ft_next = min(5, max(ft - used, 0) + 1) in a normal week; = ft after a Wildcard/Free Hit."""
    chip = chip_flag("wildcard", w) + chip_flag("freehit", w)
    prob += ft_next <= ft[w] - n_in[w] + paid[w] + 1 + BIG * chip
    prob += ft_next <= ft[w] + BIG * (1 - chip)


def _on(var) -> bool:
    """True if a binary variable (or a 0/1 constant) is 1 in the solution."""
    return round(var.value() or 0) == 1 if hasattr(var, "value") else var == 1


def _read_plan(inp, gws, P, xp, x, tin, tout, z, s, c, use, paid, ft, settings, status) -> Plan:
    weeks, bank, value = [], inp.bank, 0.0
    for i, w in enumerate(gws):
        chip = next((ch.name for ch in inp.chips if (ch.key, w) in use and _on(use[ch.key, w])), None)
        bought = [p for p in P if _on(tin[p, w])]
        sold = [p for p in P if _on(tout[p, w])]
        bank += sum(int(inp.pool.at[p, "sell"]) for p in sold) - sum(int(inp.pool.at[p, "price"]) for p in bought)
        squad = [p for p in P if _on(z[p, w])]
        starters = [p for p in P if _on(s[p, w])]
        captain = next(p for p in P if _on(c[p, w]))
        on_bench = [p for p in squad if p not in starters]
        bench = ([p for p in on_bench if inp.pool.at[p, "position"] == "GK"]
                 + sorted((p for p in on_bench if inp.pool.at[p, "position"] != "GK"),
                          key=lambda p: -xp.at[p, w]))
        vice = max((p for p in starters if p != captain), key=lambda p: xp.at[p, w])
        hits = int(round(paid[w].value() or 0))
        week_xp = (sum(xp.at[p, w] for p in starters) + xp.at[captain, w] * (2 if chip == "3xc" else 1)
                   + (sum(xp.at[p, w] for p in on_bench) if chip == "bboost" else 0)
                   - rules.HIT_COST * hits)
        value += settings.discount ** i * week_xp
        weeks.append(WeekPlan(
            gw=w, chip=chip, transfers_in=bought, transfers_out=sold,
            free_transfers=None if ft[w] is None else int(round(ft[w].value())), paid=hits,
            owned=[p for p in P if _on(x[p, w])], squad=squad, starters=starters, bench=bench,
            captain=captain, vice=vice, xp=float(week_xp), bank=int(bank)))
    return Plan(weeks=weeks, value=float(value), status=status)


def best_squad(xp: pd.DataFrame, pool: pd.DataFrame, budget: int, season: str,
               time_limit: int = 20) -> Plan:
    """The best single squad (no transfers after week 1) for these weeks, from scratch.

    Used by the chip scan for Free Hit and Wildcard values.
    """
    inp = PlanInput(pool=pool, xp=xp, squad=[], bank=budget, free_transfers=0, chips=[], season=season)
    flat = PlanSettings(discount=1.0, hit_margin=0.0, ft_end_value=0.0, bench_weight=0.0)
    return solve(inp, flat, fixed_squad=True, time_limit=time_limit)
```

If the installed PuLP lacks `LpSolutionIntegerFeasible`, use the literal values 1 and 2 with the same meaning.

- [ ] **Step 4: Run the tests.** Run `python -m pytest tests/test_optimizer.py -v`. Expected: all PASS, in under about 60 s total.

- [ ] **Step 5: Commit and push**

```bash
git add fpl/optimizer.py tests/test_optimizer.py
git commit -m "Add the multi-week squad, transfer, captain and chip planner

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push
```

---

### Task 7: Chip scan (`fpl/chips.py`)

**Files:**
- Create: `fpl/chips.py`, `tests/test_chips.py`

**Interfaces:**
- Consumes: `optimizer.best_squad`, `optimizer.candidate_pool`, `rules.XI_LIMITS`, `rules.Chip`.
- Produces:
  - `best_xi(xp_week: pd.Series, positions: pd.Series) -> list[int]` (the 11 best legal starters from the given players)
  - `scan(xp, pool, squad, bank, chips, after_gw, season, wildcard_weeks=6) -> DataFrame[key, name, gw, value]`
  - `hold_values(scan_df, chips, first_gw, discount) -> dict[str, float]`

- [ ] **Step 1: Write failing tests.** Create `tests/test_chips.py`:

```python
"""Offline tests for fpl/chips.py."""
import pandas as pd
import pytest

from fpl import chips, rules
from test_optimizer import flat_xp, make_pool, owned_squad


def test_best_xi_is_a_legal_formation_of_the_best_players():
    pool = make_pool()
    squad = owned_squad(pool)
    xp = pd.Series(range(15), index=squad, dtype=float)
    xi = chips.best_xi(xp, pool.loc[squad, "position"])
    assert len(xi) == 11 and rules.valid_xi(pool.loc[xi, "position"])
    assert squad[-1] in xi                                        # the best player starts


def test_bench_boost_prefers_the_double_gameweek():
    pool = make_pool()
    squad = owned_squad(pool)
    xp = flat_xp(pool, [7, 8, 9, 10])
    xp.loc[squad, 9] = 4.0                                        # double gameweek for the squad
    found = chips.scan(xp, pool, squad, 0, [rules.Chip("bboost", 1, 19)], after_gw=7, season="2026-27")
    best = found.loc[found["value"].idxmax()]
    assert best["gw"] == 9 and best["value"] == 16.0


def test_hold_value_is_zero_when_the_window_ends_inside_the_plan():
    tc = rules.Chip("3xc", 1, 12)
    scan = pd.DataFrame(columns=["key", "name", "gw", "value"])
    assert chips.hold_values(scan, [tc], first_gw=7, discount=0.9) == {tc.key: 0.0}
    scan = pd.DataFrame([{"key": tc.key, "name": "3xc", "gw": 9, "value": 10.0}])
    assert chips.hold_values(scan, [tc], first_gw=7, discount=0.9)[tc.key] == pytest.approx(8.1)
```

- [ ] **Step 2: Run to verify they fail.** Run `python -m pytest tests/test_chips.py -v`. Expected: FAIL (`cannot import name 'chips'`).

- [ ] **Step 3: Implement `fpl/chips.py`:**

```python
"""Chip scan: what each unused chip is worth in each gameweek after the plan.

Rough by design (spec §5). It holds today's squad fixed and uses the
predictions as they are; predictions far ahead are flatter than near ones, so
these values run a little low. hold value = the best later week's value,
discounted to today; it is 0 when no week is left in the chip's window, so a
chip about to expire gets played.
"""
import pandas as pd

from fpl import optimizer, rules

SCAN_POOL = {"per_position": 15, "value_per_position": 5, "cheapest": 3}


def best_xi(xp_week: pd.Series, positions: pd.Series) -> list[int]:
    """The best legal starting 11 from these players for one week."""
    order = list(xp_week.sort_values(ascending=False).index)
    picked = [p for p in order if positions[p] == "GK"][:1]
    for position, (lo, _) in rules.XI_LIMITS.items():
        if position != "GK":
            picked += [p for p in order if positions[p] == position][:lo]
    rest = [p for p in order if p not in picked and positions[p] != "GK"]
    return picked + rest[:11 - len(picked)]


def _xi_points(xp_week: pd.Series, positions: pd.Series) -> float:
    xi = best_xi(xp_week, positions)
    return float(xp_week[xi].sum() + xp_week[xi].max())         # captain counted twice


def scan(xp, pool, squad, bank, chips, after_gw, season, wildcard_weeks=6) -> pd.DataFrame:
    """Value of each chip copy in each week after `after_gw` that its window and xp cover."""
    positions = pool["position"]
    budget = int(bank + pool.loc[squad, "sell"].sum())
    weeks = [int(w) for w in xp.columns if int(w) > after_gw]
    rows, best_cache = [], {}

    def best(window):
        if window not in best_cache:
            sub = xp[list(window)]
            keep = optimizer.candidate_pool(sub, pool, [], 1.0, **SCAN_POOL)
            best_cache[window] = optimizer.best_squad(sub.loc[keep], pool.loc[keep], budget, season)
        return best_cache[window]

    for ch in chips:
        for w in weeks:
            if not ch.allows(w):
                continue
            mine = xp.loc[squad, w]
            if ch.name == "bboost":
                xi = best_xi(mine, positions)
                value = float(mine.drop(xi).sum())
            elif ch.name == "3xc":
                value = float(mine[best_xi(mine, positions)].max())
            elif ch.name == "freehit":
                value = best((w,)).value - _xi_points(mine, positions)
            else:                                               # wildcard
                window = tuple(v for v in weeks if w <= v < w + wildcard_weeks)
                plan = best(window)
                gain = plan.value - sum(_xi_points(xp.loc[squad, v], positions) for v in window)
                changed = len(set(plan.weeks[0].owned) - set(squad))
                value = gain * max(0, changed - len(window)) / changed if changed else 0.0
            rows.append({"key": ch.key, "name": ch.name, "gw": w, "value": max(0.0, float(value))})
    return pd.DataFrame(rows, columns=["key", "name", "gw", "value"])


def hold_values(scan_df: pd.DataFrame, chips, first_gw: int, discount: float) -> dict[str, float]:
    """Points for keeping each chip past the plan: its best scanned week, discounted to first_gw."""
    out = {}
    for ch in chips:
        mine = scan_df[scan_df["key"] == ch.key]
        out[ch.key] = float((mine["value"] * discount ** (mine["gw"] - first_gw)).max()) if len(mine) else 0.0
    return out
```

- [ ] **Step 4: Run the tests.** Run `python -m pytest tests/test_chips.py -v`. Expected: all PASS.

- [ ] **Step 5: Commit and push**

```bash
git add fpl/chips.py tests/test_chips.py
git commit -m "Add chip scan and hold values for weeks beyond the plan

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push
```

---

### Task 8: `recommend.py` and plain-English advice

**Files:**
- Create: `fpl/advice.py`, `recommend.py`, `tests/test_recommend.py`
- Modify: `pull_data.py` (extract `pull(skip_history=False)`; `main()` calls it)

**Interfaces:**
- Consumes: `myteam.load_team_state`, `forecast.live_forecast`, `models.weekly_xp`, `optimizer.*`, `chips.*`, `storage.*`, `settings.*`.
- Produces in `fpl/advice.py`:
  - `money(tenths) -> str`
  - `local_time(ts, tz=TIMEZONE) -> str`
  - `pool_info(players, teams, state) -> DataFrame` (index code: name, position, team, price, sell)
  - `fixture_counts(fixtures_all, teams, season) -> DataFrame[team, gw, fixtures]`
  - dataclass `Advice`
  - `render(advice) -> str`
  - `plan_table(advice) -> DataFrame`
- Produces in `recommend.py`: `build_advice(state, players, teams, fixtures_all, xp, avail, deadline, settings, time_limit) -> Advice` (no network, no training).

- [ ] **Step 1: Refactor `pull_data.py`.** Move the body of `main()` after argument parsing into `def pull(skip_history: bool = False) -> None:`. `main()` becomes:

```python
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-history", action="store_true",
                        help="skip the slow per-player match history download")
    pull(parser.parse_args().skip_history)
```

Inside `pull`, replace `args.skip_history` with `skip_history`.

- [ ] **Step 2: Write failing tests.** Create `tests/test_recommend.py`:

```python
"""End to end on a saved copy of team 5234632: state -> plan -> printed advice (no network, no training)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import recommend
from fpl import advice, myteam
from fpl.settings import PlanSettings

DATA = Path(__file__).parent / "data"


def saved_state():
    folder = DATA / "team_5234632"
    load = lambda name: json.loads((folder / f"{name}.json").read_text())
    by_gw = {int(p.stem.split("_")[1]): json.loads(p.read_text()) for p in folder.glob("picks_*.json")}
    players = pd.read_csv(DATA / "players_sample.csv")
    state = myteam.load_team_state(5234632, load("entry"), load("history"), load("transfers"),
                                   by_gw.__getitem__, players, pd.read_csv(folder / "player_history.csv"),
                                   json.loads((DATA / "chips_2026-27.json").read_text()))
    return state, players


def test_local_time_is_nairobi():
    assert advice.local_time(pd.Timestamp("2026-10-17T10:00:00Z")) == "Sat 17 Oct 13:00"
    assert advice.money(6) == "£0.6m" and advice.money(105) == "£10.5m"


def test_advice_end_to_end_on_the_saved_team():
    state, players = saved_state()
    teams = pd.DataFrame({"id": range(1, 21), "name": [f"Club {i}" for i in range(1, 21)],
                          "short_name": [f"C{i:02d}" for i in range(1, 21)]})
    gws = list(range(7, 20))
    rng = np.random.default_rng(0)
    xp = pd.DataFrame(rng.uniform(1, 6, (len(players), len(gws))), index=players["code"], columns=gws)
    loaned = players.loc[players["status"] == "u", "code"]
    xp = xp.drop(index=loaned)                                   # no predictions, like McNally
    avail = pd.DataFrame({"player_code": [], "gw": [], "availability": [], "news": []})
    fixtures = pd.DataFrame({"season": "2026-27", "gw": np.repeat(gws, 20), "team": [f"Club {i}" for i in range(1, 21)] * len(gws),
                             "fixture_id": range(20 * len(gws))})
    result = recommend.build_advice(state, players, teams, fixtures, xp, avail,
                                    pd.Timestamp("2026-10-17T10:00:00Z"),
                                    PlanSettings(horizon=4), time_limit=30)
    text = advice.render(result)
    for section in ["Sat 17 Oct 13:00", "THIS WEEK", "Captain:", "Bench (in order):",
                    "NEXT WEEKS", "CHIPS", "£0.6m"]:
        assert section in text, section
    for code in loaned:
        if code in state.codes:
            assert code not in result.plan.weeks[0].starters
    table = advice.plan_table(result)
    assert set(table["gw"]) == {7, 8, 9, 10}
```

- [ ] **Step 3: Run to verify they fail.** Run `python -m pytest tests/test_recommend.py -v`. Expected: FAIL (`No module named 'recommend'`).

- [ ] **Step 4: Implement `fpl/advice.py`:**

```python
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
    hit_gain: float | None        # xP difference with vs without a week-1 hit (None if not checked)
    info: pd.DataFrame
    xp: pd.DataFrame
    avail: pd.DataFrame
    scan: pd.DataFrame            # chip scan from the plan's final squad
    counts: pd.DataFrame          # fixture_counts
    deadline: pd.Timestamp
    horizon: int


def _name(a: Advice, code) -> str:
    return f"{a.info.at[code, 'name']} ({a.info.at[code, 'team']})"


def _plan_xp(a: Advice, code) -> float:
    return float(a.xp.loc[code, [w.gw for w in a.plan.weeks]].sum()) if code in a.xp.index else 0.0


def _news(a: Advice, code) -> str | None:
    rows = a.avail[a.avail["player_code"] == code]
    return None if rows.empty else str(rows["news"].iloc[0])


def _transfer_lines(a: Advice, week) -> list[str]:
    lines = []
    for out, inn in zip(week.transfers_out, week.transfers_in):
        why = _news(a, out) or f"{_plan_xp(a, out):.1f} pts expected over {a.horizon} weeks"
        lines.append(f"  Sell {_name(a, out)} for {money(a.info.at[out, 'sell'])} ({why})")
        lines.append(f"   buy {_name(a, inn)} for {money(a.info.at[inn, 'price'])}, "
                     f"{a.info.at[inn, 'position']}, {_plan_xp(a, inn):.1f} pts expected over {a.horizon} weeks")
    return lines


def _week_summary(a: Advice, week) -> str:
    if week.chip in rules.TRANSFER_CHIPS:
        return f"play your {rules.CHIP_LABELS[week.chip]} ({len(week.transfers_in)} changes)"
    if not week.transfers_in:
        return f"save your free transfer ({week.free_transfers} available)"
    moves = ", ".join(f"{a.info.at[o, 'name']} -> {a.info.at[i, 'name']}"
                      for o, i in zip(week.transfers_out, week.transfers_in))
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
    if a.plan.status != "Optimal":
        out.append(f"(Solver stopped at its time limit: this is the best plan it found.)")
    out += ["", "THIS WEEK"]
    if week.chip in rules.TRANSFER_CHIPS:
        out.append(f"  Play your {rules.CHIP_LABELS[week.chip]} this week.")
    out += _transfer_lines(a, week) if week.chip != "freehit" else []
    if not week.transfers_in and week.chip is None:
        nxt = a.plan.weeks[1].free_transfers if len(a.plan.weeks) > 1 else None
        out.append("  Save your free transfer" + (f": you'll have {nxt} next week." if nxt else "."))
    if a.hit_gain is not None:
        if week.paid:
            out.append(f"  Taking a -{rules.HIT_COST * week.paid} hit: worth +{a.hit_gain:.1f} pts over the plan.")
        else:
            out.append(f"  No hit: the best move with a -4 hit changes expected points by "
                       f"{a.hit_gain:+.1f} over the plan, which isn't worth it.")
    if week.chip in ("bboost", "3xc"):
        out.append(f"  Play your {rules.CHIP_LABELS[week.chip]} this week.")
    out.append(f"  Captain: {_name(a, week.captain)} ({a.xp.at[week.captain, week.gw]:.1f} pts expected)"
               f"   Vice-captain: {_name(a, week.vice)} ({a.xp.at[week.vice, week.gw]:.1f})")
    out.append("  Starting XI:")
    for position in ("GK", "DEF", "MID", "FWD"):
        names = [f"{_name(a, p)} {a.xp.at[p, week.gw]:.1f}" for p in week.starters
                 if a.info.at[p, "position"] == position]
        out.append(f"    {position}: " + " | ".join(names))
    gk, *outfield = week.bench
    out.append("  Bench (in order): " + "  ".join(f"{i}. {_name(a, p)}" for i, p in enumerate(outfield, 1))
               + f"  GK {_name(a, gk)}")
    flagged = [c for c in st.codes if c in set(a.avail["player_code"])]
    if flagged:
        out += ["", "INJURY WATCH"]
        for code in flagged:
            rows = a.avail[a.avail["player_code"] == code].sort_values("gw").head(a.horizon)
            weeks = ", ".join(f"GW{int(r.gw)} {r.availability:.0%}" for r in rows.itertuples())
            out.append(f"  {_name(a, code)}: {_news(a, code)}. Chance available: {weeks}")
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
```

- [ ] **Step 5: Implement `recommend.py`:**

```python
"""Your best FPL moves this week, in plain English.

Usage:
    python recommend.py                  # team 5234632, 6-week plan
    python recommend.py --horizon 4      # shorter plan
    python recommend.py --team 123456    # another team
Advice only: it never logs in or makes transfers. Refreshes FPL data if it is
more than 6 hours old, then trains the model (a few minutes).
"""
import argparse
import json
import sys
from functools import partial
from pathlib import Path

import pandas as pd

import pull_data
from fpl import advice, api, chips, forecast, matches, models, myteam, optimizer, storage
from fpl.seasons import CURRENT_SEASON
from fpl.settings import (CHIP_SCAN_WEEKS, DATA_MAX_AGE_HOURS, SOLVER_TIME_LIMIT, TEAM_ID,
                          PlanSettings)


def build_advice(state, players, teams, fixtures_all, xp, avail, deadline,
                 settings: PlanSettings, time_limit: int = SOLVER_TIME_LIMIT) -> advice.Advice:
    """Plan and chip scan from predictions already made (no network, no training)."""
    info = advice.pool_info(players, teams, state)
    xp = xp.reindex(info.index).fillna(0.0)                 # no prediction (left the club): 0
    plan_gws = list(xp.columns[:settings.horizon])
    squad = state.codes
    holds = chips.hold_values(
        chips.scan(xp, info, squad, state.bank, state.chips, plan_gws[-1], CURRENT_SEASON),
        state.chips, plan_gws[0], settings.discount)
    keep = optimizer.candidate_pool(xp[plan_gws], info, squad, settings.discount)
    inp = optimizer.PlanInput(pool=info.loc[keep], xp=xp.loc[keep, plan_gws], squad=squad,
                              bank=state.bank, free_transfers=state.free_transfers,
                              chips=state.chips, season=CURRENT_SEASON, hold_values=holds)
    plan = optimizer.solve(inp, settings, time_limit=time_limit)
    try:
        if plan.weeks[0].paid:
            other = optimizer.solve(inp, settings, week1_hits="none", time_limit=time_limit)
            hit_gain = plan.value - other.value
        else:
            other = optimizer.solve(inp, settings, week1_hits="some", time_limit=time_limit)
            hit_gain = other.value - plan.value
    except ValueError:
        hit_gain = None
    end = plan.weeks[-1]
    final_scan = chips.scan(xp, info, end.owned, end.bank, state.chips, plan_gws[-1], CURRENT_SEASON)
    return advice.Advice(state=state, plan=plan, hit_gain=hit_gain, info=info, xp=xp, avail=avail,
                         scan=final_scan, counts=advice.fixture_counts(fixtures_all, teams, CURRENT_SEASON),
                         deadline=deadline, horizon=settings.horizon)


def load_team(team_id: int, players: pd.DataFrame) -> myteam.TeamState:
    bootstrap = json.loads(sorted(Path(storage.DATA_DIR / "raw").glob("*/bootstrap_static.json"))[-1]
                           .read_text(encoding="utf-8"))
    return myteam.load_team_state(
        team_id, api.fetch_entry(team_id), api.fetch_entry_history(team_id),
        api.fetch_entry_transfers(team_id), partial(api.fetch_entry_picks, team_id), players,
        storage.load_table("player_history"), bootstrap["chips"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--team", type=int, default=TEAM_ID, help="FPL team ID")
    parser.add_argument("--horizon", type=int, default=6, choices=range(4, 7), help="weeks to plan (4-6)")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    if storage.table_age_hours("player_history") > DATA_MAX_AGE_HOURS:
        print("FPL data is more than 6 hours old: refreshing (a few minutes)...", flush=True)
        pull_data.pull()
    players = storage.load_table("players")
    teams = storage.load_table("teams")
    gameweeks = storage.load_table("gameweeks")
    upcoming = gameweeks.loc[gameweeks["is_next"].astype(bool), "id"]
    if upcoming.empty:
        raise SystemExit("FPL lists no next gameweek (season over?).")
    next_gw = int(upcoming.iloc[0])

    print(f"Loading team {args.team}...", flush=True)
    state = load_team(args.team, players)
    gws = [g for g in range(next_gw, next_gw + CHIP_SCAN_WEEKS) if g <= 38]
    history, fixtures_all = matches.load_all_matches(), matches.load_all_fixtures()
    print("Training the model and predicting the next weeks (a few minutes)...", flush=True)
    pred, avail = forecast.live_forecast(history, fixtures_all, players, teams, gameweeks, gws)
    xp = models.weekly_xp(pred, gws)
    print("Planning...", flush=True)
    deadline = forecast.deadlines_from(gameweeks)[next_gw]
    try:
        result = build_advice(state, players, teams, fixtures_all, xp, avail, deadline,
                              PlanSettings(horizon=args.horizon))
    except ValueError as error:
        raise SystemExit(f"Couldn't make a plan: {error}")
    print()
    print(advice.render(result))
    path = storage.save_table("plan", advice.plan_table(result))
    print(f"\nFull week-by-week plan saved to {path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the tests.** Run `python -m pytest -q`. Expected: all PASS.

- [ ] **Step 7: Run it for real.** Run `python recommend.py` (allow 5–10 min). Check the output by hand:
  - the header shows `Sat 17 Oct 13:00`, `£0.6m` and `1 free transfer`;
  - McNally is never a starter and is (or will be) sold;
  - Tzolis shows 0% for GW7;
  - Isak's chances rise week by week;
  - a captain is named, and the bench has 4 players;
  - the chips section covers the first set with GW19 as its end.

  Paste the output to the user for their sanity check (spec success criterion).

- [ ] **Step 8: Commit and push**

```bash
git add fpl/advice.py recommend.py pull_data.py tests/test_recommend.py
git commit -m "Add recommend.py: plain-English transfers, captaincy, bench and chip advice

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push
```

---

### Task 9: Benchmarks and season data download

**Files:**
- Create: `fpl/benchmarks.py`, `pull_benchmarks.py`, `tests/test_benchmarks.py`

**Interfaces:**
- Produces in `fpl/benchmarks.py`:
  - `average_manager(events) -> pd.DataFrame[gw, average]`
  - `estimate_rank(points, sample) -> tuple[int, str]` (rank, note)
  - `USER_TOTALS = {"2025-26": 2061}`
- Produces (`pull_benchmarks.py`), under `data/raw/benchmarks/`:
  - `bootstrap_2025-26.json`
  - `rank_sample_2025-26.csv` (points, rank)
- Also produces fixture snapshots under `data/raw/fixture_snapshots/2025-26/`.

- [ ] **Step 1: Write failing tests.** Create `tests/test_benchmarks.py`:

```python
"""Offline tests for fpl/benchmarks.py."""
import pandas as pd

from fpl import benchmarks


def test_average_manager_from_events():
    events = [{"id": 1, "average_entry_score": 54}, {"id": 2, "average_entry_score": 51}]
    avg = benchmarks.average_manager(events)
    assert avg["average"].sum() == 105 and list(avg["gw"]) == [1, 2]


def test_estimate_rank_interpolates_and_is_monotonic():
    sample = pd.DataFrame({"points": [1800, 1900, 2000, 2100, 2061],
                           "rank": [8_000_000, 5_000_000, 3_000_000, 1_500_000, 2_927_134]})
    r2061, _ = benchmarks.estimate_rank(2061, sample)
    assert 1_500_000 < r2061 < 3_000_000
    assert benchmarks.estimate_rank(1950, sample)[0] > benchmarks.estimate_rank(2050, sample)[0]
    top, note = benchmarks.estimate_rank(2500, sample)
    assert top == 1_500_000 and "better than" in note
```

- [ ] **Step 2: Run to verify they fail.** Run `python -m pytest tests/test_benchmarks.py -v`. Expected: FAIL.

- [ ] **Step 3: Implement `fpl/benchmarks.py`:**

```python
"""2025/26 benchmarks for the season simulation: the average manager and a points-to-rank curve.

Rank curve: every sampled manager gives one exact (season points, final rank)
pair, so interpolating between them is unbiased; precision is limited by
sample size, worst at very high ranks. Nothing here touches the internet.
"""
import numpy as np
import pandas as pd

USER_TOTALS = {"2025-26": 2061}     # team 5234632, from /entry/5234632/history/ 'past'


def average_manager(events: list) -> pd.DataFrame:
    """FPL's average score per gameweek (bootstrap-static 'events')."""
    return pd.DataFrame({"gw": [e["id"] for e in events],
                         "average": [e["average_entry_score"] for e in events]})


def estimate_rank(points: float, sample: pd.DataFrame) -> tuple[int, str]:
    """Estimated overall rank for a season total, interpolated on log rank between sampled managers."""
    curve = (sample.assign(log_rank=np.log(sample["rank"])).groupby("points")["log_rank"].median()
             .sort_index())
    xs, ys = curve.index.to_numpy(dtype=float), curve.to_numpy()
    ys = np.minimum.accumulate(ys)                       # more points never means a worse rank
    note = "estimated, roughly ±10-20%"
    if points > xs[-1]:
        note = f"better than every sampled manager ({int(xs[-1])} pts); rank is at most this"
    elif points < xs[0]:
        note = f"below every sampled manager ({int(xs[0])} pts)"
    return int(round(float(np.exp(np.interp(points, xs, ys))))), note
```

- [ ] **Step 4: Implement `pull_benchmarks.py`:**

```python
"""One-off download of what the 2025/26 season simulation needs.

Usage:
    python pull_benchmarks.py           # about 10-20 minutes (polite pace)
Saves under data/raw/benchmarks/ and data/raw/fixture_snapshots/2025-26/.
"""
import io
import json
import random
import time

import pandas as pd

from fpl import api, storage

SEASON = "2025-26"
ARCHIVED_BOOTSTRAP = "20260526000012"     # Internet Archive copy taken after GW38
RANK_SAMPLE_SIZE = 3000
FIXTURE_COLUMNS = ["id", "code", "event", "kickoff_time", "finished", "team_h", "team_a",
                   "team_h_score", "team_a_score"]


def pull_bootstrap() -> None:
    raw = api.fetch_wayback("https://fantasy.premierleague.com/api/bootstrap-static/", ARCHIVED_BOOTSTRAP)
    data = json.loads(raw.decode("utf-8"))
    assert data["events"][-1]["finished"], "archived copy is from before the season ended"
    storage.save_benchmark(f"bootstrap_{SEASON}.json", raw)
    print(f"Average manager total: {sum(e['average_entry_score'] for e in data['events'])}")


def pull_rank_sample() -> None:
    total = api.fetch_bootstrap()["total_players"]
    rows, tried = [], 0
    random.seed(2526)
    while len(rows) < RANK_SAMPLE_SIZE:
        tried += 1
        try:
            past = api.fetch_entry_history(random.randint(1, total)).get("past", [])
        except Exception:
            continue
        season = next((p for p in past if p["season_name"] == "2025/26"), None)
        if season:
            rows.append({"points": season["total_points"], "rank": season["rank"]})
        if tried % 250 == 0:
            print(f"  tried {tried}, kept {len(rows)}", flush=True)
        time.sleep(0.1)
    buffer = io.StringIO()
    pd.DataFrame(rows).to_csv(buffer, index=False)
    storage.save_benchmark(f"rank_sample_{SEASON}.csv", buffer.getvalue().encode())


def pull_fixture_snapshots() -> None:
    path = f"data/{SEASON}/fixtures.csv"
    for sha, when in api.fetch_vaastav_commits(path):
        table = pd.read_csv(io.BytesIO(api.fetch_vaastav_file_at(sha, path)))
        stamp = pd.Timestamp(when).strftime("%Y%m%dT%H%M%S")
        storage.save_snapshot(SEASON, stamp, "vaastav", table[[c for c in FIXTURE_COLUMNS if c in table]])
    url = "https://fantasy.premierleague.com/api/fixtures/"
    for ts in api.fetch_wayback_index(url, "20250701", "20260601"):
        table = pd.DataFrame(json.loads(api.fetch_wayback(url, ts).decode("utf-8")))
        stamp = pd.Timestamp(ts, tz="UTC").strftime("%Y%m%dT%H%M%S")
        storage.save_snapshot(SEASON, stamp, "wayback", table[[c for c in FIXTURE_COLUMNS if c in table]])
    print(f"Saved {len(storage.load_snapshots(SEASON))} fixture snapshots")


if __name__ == "__main__":
    pull_bootstrap()
    pull_fixture_snapshots()
    pull_rank_sample()
```

- [ ] **Step 5: Run the tests, then the download.** Run `python -m pytest tests/test_benchmarks.py -v` (PASS). Then run `python pull_benchmarks.py` in the background. Expected:
  - `Average manager total: 1895`;
  - `Saved 17 fixture snapshots` (or close: archive copies can change);
  - a 3,000-row rank sample.
  - Check `estimate_rank(2061, sample)` lands near 2.9M (the user's real rank 2,927,134).

- [ ] **Step 6: Commit and push**

```bash
git add fpl/benchmarks.py pull_benchmarks.py tests/test_benchmarks.py
git commit -m "Add 2025/26 benchmarks: average manager, rank curve, dated fixture lists

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push
```

---

### Task 10: Season simulation (`fpl/simulate.py`, `simulate.py`)

**Files:**
- Create: `fpl/simulate.py`, `simulate.py`, `tests/test_simulate.py`

**Interfaces:**
- Consumes: `models.train_and_predict(..., yardstick=False)`, `matches.fixture_list`, `matches.gw_cutoffs`, `matches.PLAYER_ROW_COLUMNS`, `matches.TEAM_ROW_COLUMNS`, `optimizer.*`, `chips.*`, `rules.*`, `benchmarks.*`, `storage.*`.
- Produces in `fpl/simulate.py`:
  - `season_players(history, season) -> DataFrame[gw, player_code, player_name, position, team, price]` (price in tenths)
  - `weekly_actual(history, season) -> DataFrame[gw, player_code, points, minutes]`
  - `snapshot_for(deadline, snapshots) -> DataFrame`
  - `build_predictions(history, fixtures_all, snapshots, teams, season, deadlines, gws, weeks, train, log) -> DataFrame[variant, origin_gw, gw, player_code, xp]`
  - dataclass `SimData(season, gws, deadlines, chips, players, actual)`
  - `run_season(data, xp, settings, time_limit, log) -> DataFrame` (one row per week)
  - `summarize(weeks, average, rank_sample, user_total) -> dict`

- [ ] **Step 1: Write failing tests.** Create `tests/test_simulate.py`:

```python
"""Offline tests for fpl/simulate.py: no future data, and a short made-up season runs."""
import numpy as np
import pandas as pd
import pytest

from conftest import make_fixtures, make_matches
from fpl import rules, simulate
from fpl.settings import PlanSettings


def test_snapshot_for_picks_the_newest_before_the_deadline():
    snaps = [(pd.Timestamp("2025-08-01", tz="UTC"), "a"), (pd.Timestamp("2025-09-01", tz="UTC"), "b")]
    assert simulate.snapshot_for(pd.Timestamp("2025-08-20", tz="UTC"), snaps) == "a"
    assert simulate.snapshot_for(pd.Timestamp("2025-09-02", tz="UTC"), snaps) == "b"
    with pytest.raises(ValueError):
        simulate.snapshot_for(pd.Timestamp("2025-07-01", tz="UTC"), snaps)


def test_build_predictions_only_sees_the_past():
    history = make_matches(gws=6)
    fixtures_all = make_fixtures(history)
    season = "2025-26"
    f = fixtures_all[fixtures_all.season == season]
    deadlines = (f.groupby("gw")["kickoff_time"].min() - pd.Timedelta(hours=2)).to_dict()
    raw = pd.DataFrame({"id": f.fixture_id, "event": f.gw, "kickoff_time": f.kickoff_time.astype(str),
                        "finished": False, "team_h": 0, "team_a": 0, "team_h_score": None,
                        "team_a_score": None}).drop_duplicates("id")
    seen = []

    def fake_train(past, player_rows, team_rows, cutoffs, yardstick=True):
        seen.append((past["kickoff_time"].max(), player_rows["asof"].min()))
        return player_rows.assign(xp=1.0)

    teams = pd.DataFrame({"id": [0], "name": ["Alpha"]})
    simulate.build_predictions(history, fixtures_all, [(pd.Timestamp("2025-08-01", tz="UTC"), raw)],
                               teams, season, deadlines, gws=[3, 4], weeks=2, train=fake_train,
                               log=lambda m: None)
    assert seen and all(latest < asof for latest, asof in seen)


def made_up_season(gws=4, seed=0):
    """8 clubs × (1 GK, 3 DEF, 3 MID, 2 FWD); random prices, xP and real points."""
    rng = np.random.default_rng(seed)
    rows, code = [], 1
    for club in range(8):
        for position, count in [("GK", 1), ("DEF", 3), ("MID", 3), ("FWD", 2)]:
            for _ in range(count):
                rows.append({"player_code": code, "player_name": f"P{code}", "position": position,
                             "team": f"C{club}", "price": int(rng.integers(40, 80))})
                code += 1
    base = pd.DataFrame(rows)
    players = pd.concat([base.assign(gw=g) for g in range(1, gws + 1)], ignore_index=True)
    actual = players[["gw", "player_code"]].assign(points=rng.integers(0, 10, len(players)),
                                                   minutes=rng.choice([0, 90], len(players), p=[0.1, 0.9]))
    xp = pd.concat([base[["player_code"]].assign(origin_gw=o, gw=g, xp=rng.uniform(1, 6, len(base)))
                    for o in range(1, gws + 1) for g in range(o, gws + 1)], ignore_index=True)
    chips = [rules.Chip("bboost", 1, 19), rules.Chip("3xc", 1, 19)]
    deadlines = {g: pd.Timestamp("2025-08-15", tz="UTC") + pd.Timedelta(days=7 * (g - 1)) for g in range(1, gws + 1)}
    data = simulate.SimData(season="2025-26", gws=list(range(1, gws + 1)), deadlines=deadlines,
                            chips=chips, players=players, actual=actual)
    return data, xp


def test_a_short_made_up_season_runs_and_adds_up():
    data, xp = made_up_season()
    weeks = simulate.run_season(data, xp, PlanSettings(horizon=2), time_limit=20, log=lambda m: None)
    assert list(weeks["gw"]) == [1, 2, 3, 4]
    assert weeks["bank"].ge(0).all()
    assert weeks["chip"].isin(["", "bboost", "3xc"]).all()
    summary = simulate.summarize(weeks, pd.DataFrame({"gw": [1, 2, 3, 4], "average": [50] * 4}),
                                 pd.DataFrame({"points": [100, 200], "rank": [1000, 10]}), 2061)
    assert summary["total"] == weeks["points"].sum()
    assert summary["average_total"] == 200
```

- [ ] **Step 2: Run to verify they fail.** Run `python -m pytest tests/test_simulate.py -v`. Expected: FAIL.

- [ ] **Step 3: Implement `fpl/simulate.py`:**

```python
"""Replay a past season with the whole system: predict -> plan -> play -> score, week by week.

At each deadline the simulated manager knows only: matches already played,
prices that week, the newest fixture-list copy saved before the deadline, and
no injury news (none was stored). See the part 3 spec §7.
"""
import time
from dataclasses import dataclass

import pandas as pd

from fpl import benchmarks, chips, matches, models, optimizer, rules
from fpl.settings import CHIP_SCAN_WEEKS, SIM_TIME_LIMIT, PlanSettings


@dataclass
class SimData:
    season: str
    gws: list[int]
    deadlines: dict                 # gw -> UTC deadline
    chips: list                     # rules.Chip windows for the season
    players: pd.DataFrame           # season_players()
    actual: pd.DataFrame            # weekly_actual()


def season_players(history: pd.DataFrame, season: str) -> pd.DataFrame:
    """Who could be bought each gameweek, and at what price (tenths of £1m).

    A player counts from his first match row; in a gameweek his team blanks,
    his previous gameweek's row is used.
    """
    h = history[history["season"] == season].sort_values("kickoff_time")
    last = h.groupby(["gw", "player_code"], as_index=False).last()
    frames = []
    for gw in range(1, int(h["gw"].max()) + 1):
        now = last[last["gw"] == gw]
        prev = last[(last["gw"] == gw - 1) & ~last["player_code"].isin(now["player_code"])]
        frames.append(pd.concat([now, prev]).assign(gw=gw))
    out = pd.concat(frames, ignore_index=True)
    out["price"] = (out["price"] * 10).round().astype(int)
    return out[["gw", "player_code", "player_name", "position", "team", "price"]]


def weekly_actual(history: pd.DataFrame, season: str) -> pd.DataFrame:
    """Real points and minutes per player per gameweek (double gameweeks summed)."""
    h = history[history["season"] == season]
    return h.groupby(["gw", "player_code"], as_index=False).agg(
        points=("total_points", "sum"), minutes=("minutes", "sum"))


def snapshot_for(deadline: pd.Timestamp, snapshots: list):
    """The newest saved fixture list from before `deadline`."""
    before = [table for stamp, table in snapshots if stamp < deadline]
    if not before:
        raise ValueError(f"no fixture snapshot from before {deadline}")
    return before[-1]


def build_predictions(history, fixtures_all, snapshots, teams, season, deadlines,
                      gws=range(1, 39), weeks=CHIP_SCAN_WEEKS, train=models.train_and_predict,
                      log=print) -> pd.DataFrame:
    """xP per player per gameweek as seen at each deadline, for both fixture versions.

    'dated': gameweeks from the newest fixture copy before the deadline;
    'final': gameweeks from the final fixture list. Each fixture is predicted
    once (a fixture's prediction doesn't depend on which gameweek it's in).
    """
    players = season_players(history, season)
    cutoffs = matches.gw_cutoffs(fixtures_all)
    final = fixtures_all[fixtures_all["season"] == season]
    out = []
    for g in gws:
        deadline = deadlines[g]
        past = history[history["kickoff_time"] < deadline]
        targets = [t for t in range(g, g + weeks) if t <= 38]
        dated = matches.fixture_list(season, snapshot_for(deadline, snapshots), teams)
        dated = dated[dated["gw"].isin(targets)]
        fin = final[final["gw"].isin(targets)]
        fx = pd.concat([dated, fin], ignore_index=True).drop_duplicates(["fixture_id", "team"])
        team_rows = fx[matches.TEAM_ROW_COLUMNS].assign(asof=deadline)
        reg = players[players["gw"] == g].drop(columns="gw").assign(price=lambda d: d["price"] / 10)
        player_rows = (fx[matches.TEAM_ROW_COLUMNS].merge(reg, on="team")
                       [matches.PLAYER_ROW_COLUMNS].assign(asof=deadline))
        pred = train(past, player_rows, team_rows, cutoffs, yardstick=False)
        for variant, table in (("dated", dated), ("final", fin)):
            gw_of = table.drop_duplicates("fixture_id").set_index("fixture_id")["gw"]
            p = pred.assign(gw=pred["fixture_id"].map(gw_of)).dropna(subset=["gw"])
            out.append(p.groupby(["gw", "player_code"], as_index=False)["xp"].sum()
                       .assign(variant=variant, origin_gw=g))
        log(f"  predicted from GW{g} ({len(past):,} past rows)")
    result = pd.concat(out, ignore_index=True)
    result["gw"] = result["gw"].astype(int)
    return result


def _picks(week, positions, actual_g) -> pd.DataFrame:
    order = list(week.starters) + list(week.bench)
    return pd.DataFrame({
        "position": positions.reindex(order).to_numpy(),
        "points": actual_g["points"].reindex(order).fillna(0).to_numpy(),
        "minutes": actual_g["minutes"].reindex(order).fillna(0).to_numpy(),
        "is_captain": [p == week.captain for p in order],
        "is_vice": [p == week.vice for p in order]})


def _normal_week(squad, xp_g, positions):
    """What the owned squad would have fielded (best XI by xP) – the Free Hit comparison."""
    xi = chips.best_xi(xp_g.reindex(squad).fillna(0), positions)
    bench = [p for p in squad if p not in xi]
    bench = sorted(bench, key=lambda p: (positions[p] != "GK", -xp_g.get(p, 0)))
    ranked = sorted(xi, key=lambda p: -xp_g.get(p, 0))
    return optimizer.WeekPlan(gw=0, chip=None, transfers_in=[], transfers_out=[], free_transfers=0,
                              paid=0, owned=squad, squad=squad, starters=xi, bench=bench,
                              captain=ranked[0], vice=ranked[1], xp=0.0, bank=0)


def run_season(data: SimData, xp: pd.DataFrame, settings: PlanSettings = PlanSettings(),
               time_limit: int = SIM_TIME_LIMIT, log=print) -> pd.DataFrame:
    """Manage a team from the first to the last gameweek; one result row per week."""
    squad, purchase, bank, free = [], {}, rules.BUDGET, 0
    chips_left = list(data.chips)
    rows = []
    for g in data.gws:
        known = data.players[data.players["gw"] <= g].groupby("player_code").last()
        registered = set(data.players.loc[data.players["gw"] == g, "player_code"])
        info = known.loc[sorted(registered | set(squad)), ["player_name", "position", "team", "price"]]
        info = info.rename(columns={"player_name": "name"})
        info["sell"] = [rules.selling_price(purchase[p], info.at[p, "price"]) if p in purchase
                        else info.at[p, "price"] for p in info.index]
        horizon_end = min(max(data.gws), g + CHIP_SCAN_WEEKS - 1)
        week_xp = (xp[xp["origin_gw"] == g]
                   .pivot_table(index="player_code", columns="gw", values="xp", aggfunc="sum")
                   .reindex(index=info.index, columns=range(g, horizon_end + 1)).fillna(0.0))
        plan_gws = list(week_xp.columns[:settings.horizon])
        keep = optimizer.candidate_pool(week_xp[plan_gws], info, squad, settings.discount)
        scan_squad = squad
        if not squad:
            first = optimizer.solve(optimizer.PlanInput(info.loc[keep], week_xp.loc[keep, plan_gws], [],
                                                        bank, 0, [], data.season), settings,
                                    time_limit=time_limit)
            scan_squad = first.weeks[-1].owned
        started = time.perf_counter()
        holds = chips.hold_values(
            chips.scan(week_xp, info, scan_squad, bank, chips_left, plan_gws[-1], data.season),
            chips_left, plan_gws[0], settings.discount)
        scanned = time.perf_counter()
        plan = optimizer.solve(optimizer.PlanInput(info.loc[keep], week_xp.loc[keep, plan_gws], squad,
                                                   bank, free, chips_left, data.season, holds),
                               settings, time_limit=time_limit)
        solved = time.perf_counter()
        week = plan.weeks[0]
        for p in week.transfers_out:
            bank += int(info.at[p, "sell"])
            purchase.pop(p, None)
        for p in week.transfers_in:
            bank -= int(info.at[p, "price"])
            purchase[p] = int(info.at[p, "price"])
        before = squad
        squad = week.owned
        if week.chip:
            chips_left = rules.chips_left(chips_left, [(week.chip, g)])
        actual_g = data.actual[data.actual["gw"] == g].set_index("player_code")
        positions = info["position"]
        points = rules.score_week(_picks(week, positions, actual_g), week.chip, week.paid)
        chip_gain = None
        if week.chip in ("bboost", "3xc"):
            chip_gain = points - rules.score_week(_picks(week, positions, actual_g), None, week.paid)
        elif week.chip == "freehit":
            normal = _normal_week(before, week_xp[g], positions)
            chip_gain = points - rules.score_week(_picks(normal, positions, actual_g), None, 0)
        starters_pts = actual_g["points"].reindex(week.starters).fillna(0)
        rows.append({
            "gw": g, "points": points, "xp": round(week.xp, 2), "transfers": len(week.transfers_in),
            "hits": week.paid, "chip": week.chip or "", "chip_gain": chip_gain,
            "captain": info.at[week.captain, "name"],
            "captain_points": int(actual_g["points"].get(week.captain, 0)),
            "captain_best": bool(actual_g["points"].get(week.captain, 0) >= starters_pts.max()),
            "bank": bank, "status": plan.status,
            "scan_seconds": round(scanned - started, 1), "solve_seconds": round(solved - scanned, 1)})
        free = 1 if week.free_transfers is None else rules.next_free_transfers(
            week.free_transfers, len(week.transfers_in), week.chip)
        log(f"  GW{g}: {points} pts ({week.chip or 'no chip'}, {len(week.transfers_in)} transfers, "
            f"{week.paid} hits); chip scan {scanned - started:.0f}s, main solve {solved - scanned:.0f}s")
    return pd.DataFrame(rows)


def summarize(weeks: pd.DataFrame, average: pd.DataFrame, rank_sample: pd.DataFrame,
              user_total: int) -> dict:
    total = int(weeks["points"].sum())
    average_total = int(average[average["gw"].isin(weeks["gw"])]["average"].sum())
    chip_points = float(weeks["chip_gain"].fillna(0).sum())
    rank, note = benchmarks.estimate_rank(total, rank_sample)
    return {"total": total, "average_total": average_total, "vs_average": total - average_total,
            "user_total": user_total, "vs_user": total - user_total, "rank": rank, "rank_note": note,
            "chip_points": chip_points, "total_without_chips": total - chip_points,
            "hits": int(weeks["hits"].sum()), "transfers": int(weeks["transfers"].sum()),
            "captain_best_rate": float(weeks["captain_best"].mean())}
```

- [ ] **Step 4: Run the tests.** Run `python -m pytest tests/test_simulate.py -v`. Expected: all PASS.

- [ ] **Step 5: Implement `simulate.py`:**

```python
"""Replay 2025/26 with the whole system and compare it with real managers.

Usage:
    python simulate.py                  # full run: predictions (~20 min, cached) + 8 planner runs
    python simulate.py --quick          # GW1-6, default settings only (smoke test)
    python simulate.py --workers 4      # planner runs in parallel
Run pull_history.py and pull_benchmarks.py first.
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from fpl import benchmarks, matches, rules, simulate, storage
from fpl.settings import PlanSettings

SEASON = "2025-26"
DEFAULT = PlanSettings()
VARIANTS = [("default", DEFAULT),
            ("discount 0.85", PlanSettings(discount=0.85)), ("discount 0.95", PlanSettings(discount=0.95)),
            ("hit margin 0", PlanSettings(hit_margin=0.0)), ("hit margin 2", PlanSettings(hit_margin=2.0)),
            ("FT value 1", PlanSettings(ft_end_value=1.0)), ("FT value 2", PlanSettings(ft_end_value=2.0))]
OPTIMISTIC = [
    "Fixture copies may miss changes made between copies.",
    "Player positions and the player list are end-of-season versions (mid-season signings "
    "are only buyable from their first match).",
    "The xP model's settings were chosen in part 2 partly on 2025/26 data.",
    "The planner settings are tested on the same season they are reported for (7 runs only).",
]
PESSIMISTIC = [
    "No injury news: the simulated manager can't see flags real managers saw.",
    "No press-conference news.",
    "Fixture copies (gaps up to 2-3 months) may lag what managers knew.",
]


def _run(job):
    label, data, xp, settings = job
    return label, simulate.run_season(data, xp, settings, log=lambda m: print(f"[{label}]{m}", flush=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="GW1-6, default settings only")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rebuild", action="store_true", help="recompute the cached predictions")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    bootstrap = storage.load_benchmark_json(f"bootstrap_{SEASON}.json")
    deadlines = {e["id"]: pd.Timestamp(e["deadline_time"]) for e in bootstrap["events"]}
    gws = list(range(1, 7)) if args.quick else list(range(1, 39))
    history, fixtures_all = matches.load_all_matches(), matches.load_all_fixtures()
    cache = storage.DATA_DIR / "processed" / "simulation_xp.csv"
    if cache.exists() and not args.rebuild:
        xp = storage.load_table("simulation_xp")
    else:
        print("Predicting every gameweek from its own deadline (about 20 minutes)...", flush=True)
        xp = simulate.build_predictions(history, fixtures_all, storage.load_snapshots(SEASON),
                                        storage.load_vaastav(SEASON, "teams.csv"), SEASON, deadlines,
                                        gws=range(1, 39))
        storage.save_table("simulation_xp", xp)
    data = simulate.SimData(season=SEASON, gws=gws, deadlines=deadlines,
                            chips=rules.chip_windows(bootstrap["chips"]),
                            players=simulate.season_players(history, SEASON),
                            actual=simulate.weekly_actual(history, SEASON))
    dated, final = xp[xp["variant"] == "dated"], xp[xp["variant"] == "final"]
    jobs = [(label, data, dated, s) for label, s in (VARIANTS[:1] if args.quick else VARIANTS)]
    jobs.append(("default, final fixtures", data, final, DEFAULT))
    with ProcessPoolExecutor(args.workers) as pool:
        results = dict(pool.map(_run, jobs))

    if args.quick:
        weeks = results["default"]
        per_week = (weeks["scan_seconds"] + weeks["solve_seconds"]).mean()
        full_hours = per_week * 38 * (len(VARIANTS) + 1) / min(args.workers, len(VARIANTS) + 1) / 3600
        print(f"\nTiming: chip scan {weeks['scan_seconds'].mean():.0f}s, main solve "
              f"{weeks['solve_seconds'].mean():.0f}s per week on average.")
        print(f"Estimated full run ({len(VARIANTS) + 1} runs x 38 weeks, {args.workers} workers): "
              f"about {full_hours:.1f} hours. (GW1-6 are early-season weeks; later weeks can be slower.)")

    average = benchmarks.average_manager(bootstrap["events"])
    sample = pd.read_csv(storage.benchmark_path(f"rank_sample_{SEASON}.csv"))
    user = benchmarks.USER_TOTALS[SEASON]
    rows = [{"run": label, **simulate.summarize(weeks, average, sample, user)}
            for label, weeks in results.items()]
    report = pd.DataFrame(rows)
    storage.save_table("simulation_report", report)
    storage.save_table("simulation_weeks", pd.concat(
        [w.assign(run=label) for label, w in results.items()], ignore_index=True))

    main_run = report.iloc[0]
    print(f"\n2025/26 simulation (default settings, fixtures as known at each deadline):")
    print(f"  Season total:   {main_run.total}")
    print(f"  Average manager {main_run.average_total}  ({main_run.vs_average:+d})")
    print(f"  Your 2025/26    {main_run.user_total}  ({main_run.vs_user:+d})")
    print(f"  Estimated overall rank: {main_run['rank']:,} ({main_run.rank_note})")
    print(f"  Chips added {main_run.chip_points:.0f} pts; without them: {main_run.total_without_chips:.0f}")
    print(f"  Hits: {main_run.hits}, transfers: {main_run.transfers}, "
          f"captain was the best starter {main_run.captain_best_rate:.0%} of weeks")
    print("\nAll runs:")
    print(report[["run", "total", "vs_average", "vs_user", "rank", "chip_points", "hits"]]
          .to_string(index=False))
    print("\nWeek by week (default):")
    print(results["default"].to_string(index=False))
    print("\nWhat may make this result too GOOD:")
    print("\n".join(f"  - {line}" for line in OPTIMISTIC))
    print("What may make it too BAD:")
    print("\n".join(f"  - {line}" for line in PESSIMISTIC))
    print("\nSettings rule: keep the defaults unless another value wins by more than ~30 points "
          "AND its neighbouring values also do well.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Smoke run.** Run `python simulate.py --quick` in the background. It builds the full prediction cache (about 20 min) and then runs GW1–6. Expected:
  - six weekly lines;
  - a summary with an average-manager comparison and a rank;
  - no errors.

  - each week's log line shows chip-scan and main-solve seconds;
  - an estimated full-run time.

  Fix anything found before the full run.

- [ ] **Step 6b: Decide whether the full run is affordable.** If the estimate is far beyond 3 hours (say more than 4.5), **don't start it**. Report the timing to the user and suggest speed-ups, in this order of impact:
  1. scan Free Hit/Wildcard only every other week, or only within 6 weeks of the window end (the scan is usually the slower part);
  2. lower `SIM_TIME_LIMIT` to 15 s;
  3. shrink the planner pool for the simulation (`CANDIDATES_PER_POSITION` 40 → 25);
  4. run fewer settings variants (default plus the hit-margin pair first);
  5. use more workers, if the machine has the cores.

  Wait for the user's choice before running.

- [ ] **Step 7: Full run.** Run `python simulate.py --workers 4` in the background. That's 8 planner runs × 38 weeks at up to 30 s per solve; expect about 1–3 hours on 4 cores. Wait for the completion notification; don't poll.

  Record the headline numbers:
  - total vs 1,895 and 2,061;
  - estimated rank;
  - chip points;
  - final-fixtures total vs dated total;
  - the 7-run settings table.

  Apply the settings rule. Change a default in `fpl/settings.py` only if it clearly wins (by more than about 30) and its neighbours do too. If you change one, re-run `python -m pytest -q`.

- [ ] **Step 8: Commit and push**

```bash
git add fpl/simulate.py simulate.py tests/test_simulate.py fpl/settings.py
git commit -m "Add 2025/26 season simulation with dated fixtures, chip dependence and rank estimate

Result (default settings): <total> pts vs average 1895 (<+/->), vs user 2061 (<+/->), est. rank <rank>.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push
```

---

### Task 11: Final verification and wrap-up

- [ ] **Step 1: Run the full test suite.** Run `python -m pytest -q`. Expected: all pass, and the count is reported.
- [ ] **Step 2: Run `python recommend.py` once more** on fresh data, and save the output for the user.
- [ ] **Step 3:** Use superpowers:requesting-code-review on the branch, fix findings, commit and push each fix.
- [ ] **Step 4:** Use superpowers:finishing-a-development-branch (PR into `master`; PR body ends with the Claude Code line).

## Execution choice (for the user)

- **Subagent-driven** (recommended): a fresh agent builds each task and a fresh reviewer checks it before the next one starts. The tasks depend on each other's exact interfaces (`PlanInput`/`WeekPlan`, `TeamState`, the scan), so catching a mismatch at each task boundary is worth the extra cost.
- **Native**: I build every task in this session, with one review of the whole branch at the end. Cheaper and faster.

**Chosen by the user: subagent-driven.**

## Verification (end to end)

1. `python -m pytest -q`: every test passes offline, including part 2's.
2. `python backtest.py --season 2025-26 --gws 1-38 --horizon 1`: the clean-sheet spread is wider than 0.17–0.33, log loss is no worse, and A still beats the baseline.
3. `python recommend.py`:
   - the deadline shows as `Sat 17 Oct 13:00` (Nairobi);
   - `£0.6m`, 1 free transfer;
   - sensible handling of Isak, Tzolis, McNally and Robinson;
   - captain, vice, bench order and chip advice to GW19.
4. `python simulate.py`: the report shows totals vs 1,895 and 2,061, an estimated rank, chip dependence, the final-fixture comparison, the settings table, and both lists of biases.
5. `git log origin/part3-squad-planner` matches the local log after every task.
