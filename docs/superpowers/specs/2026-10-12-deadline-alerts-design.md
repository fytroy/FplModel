# Part 4: deadline alerts on GitHub Actions

## Context
Parts 1–3 are working, but `recommend.py` only runs when the laptop is on and someone starts it. Part 4 runs it on GitHub's servers about 24 h and 3 h before each FPL deadline and sends a short ntfy phone message. The second message is sent only if the advice changed. Each message is sent once per gameweek. Times are shown in Nairobi time, and if a run fails, a failure message is sent.

Facts that shape the design:
- The repo is **private** (the GitHub API returns 404 when not logged in). That means 2,000 free minutes a month and 2-core runners, slower than the laptop.
- `data/` is gitignored. On a fresh runner, `recommend.py` sees no `player_history.csv` (`storage.table_age_hours` → inf) and calls `pull_data.pull()`, about 700 player requests at 0.3 s each. Past seasons come from `data/raw/vaastav/` (24 MB, `pull_history.py`). Nothing else on disk is needed: no benchmarks and no fixture snapshots.
- GitHub's `schedule` cron is UTC, best-effort, and can start 10–30+ min late. It also can't follow FPL's changing deadlines. So the workflow runs a cheap **hourly check** that reads the deadline from the FPL API and only starts the full run inside an alert window.
- Project rules: only `fpl/api.py` uses the internet and only `fpl/storage.py` touches `data/`. Tests run offline. Use Python 3.13.

## Design

### Two jobs in one workflow, `.github/workflows/deadline-alerts.yml`
Triggers: `schedule: cron "23 * * * *"` (off the busy top of the hour) and `workflow_dispatch` with an input `mode`: `auto` | `check` | `dry-run` | `test-send`.

**Job 1: `gate`** (~30 s, billed as 1 min). Checkout, setup-python 3.13, `pip install requests pandas tzdata`, then `python alert.py gate`:
- fetches bootstrap (one request) and takes the first event whose `deadline_time` is in the future;
- picks a stage with `fpl.alerts.stage_for(now, deadline)`: **early** if 24.5 h ≥ time left > 20 h, **late** if 3.5 h ≥ time left > 1 h, otherwise none. The bounds live in `fpl/settings.py`. With hourly checks there are about 4 tries in the early window and 2–3 in the late one, so a slow or failed start is retried;
- writes `gw`, `stage`, `deadline_nairobi` to `$GITHUB_OUTPUT`.
- Then `actions/cache/restore` with `lookup-only: true` checks whether this stage has already been sent. For early, key `alert-gw{N}-early` with restore-key `alert-gw{N}-`, so a sent late message also counts. For late, the exact key `alert-gw{N}-late`. If it has been sent, the job stops here, so most hours cost one minute.

**Job 2: `advise`** (runs only when the gate says so; `timeout-minutes: 45`; `concurrency: fpl-alerts`, no cancel).
1. Checkout, setup-python 3.13 with pip cache, `pip install -r requirements.txt`.
2. `actions/cache` for `data/raw/vaastav`, key `vaastav-${{ hashFiles('fpl/seasons.py') }}`. Then `python pull_history.py`, which only downloads what is missing. Caches unused for 7 days are evicted, and a miss just re-downloads 24 MB in seconds.
3. `actions/cache/restore` for `data/alerts/`, key prefix `alert-gw{N}-`, to get the early record if there is one.
4. `python alert.py run --stage early|late` with env `NTFY_TOPIC` from secrets. It prints timings for each step (pull, train, solve) to the log.
5. On success with a message actually sent, `actions/cache/save` for `data/alerts/` under key `alert-gw{N}-{stage}`.
6. `if: failure()`: send a `curl` ntfy message (no Python, so it works even if setup broke) with a high-priority tag: "FPL alert run failed for GW{N} (deadline … Nairobi). Check your team manually: <run URL>". It is sent once per gw+stage using a `fail-gw{N}-{stage}` cache marker. Later retries in the window still run silently.

The gate itself sends a failure ntfy at most **once per UTC day** if it fails, for example because FPL blocks GitHub's servers (marker `gate-fail-YYYY-MM-DD`).

### Python code (TDD, offline tests)
- **`recommend.py`**: move the body of `main()` into `run(team_id, horizon) -> advice.Advice`, which does the refresh, load, train and plan. `main()` calls it and prints, so behaviour is unchanged. `alert.py` reuses it, so the phone advice is exactly the laptop advice.
- **`fpl/alerts.py`** (new, pure, no I/O):
  - `next_deadline(events, now) -> (gw, deadline) | None` (season over → None, exit quietly).
  - `stage_for(now, deadline) -> "early" | "late" | None`.
  - `decisions(a: Advice) -> dict`: just the week-1 calls: transfers (out/in codes), hits paid, captain, vice, chip, and save vs transfer. This is what gets compared, so a change in bench order or xP numbers alone doesn't send a message.
  - `changed(old, new) -> list[str]`: plain-English differences, e.g. "Captain: Salah → Haaland".
  - `short_message(a, stage, changes=None) -> (title, body)`: about 6 lines, under ntfy's 4 KB, reusing `advice.money`, `advice.local_time` and the `_pairs` / `_week_summary` / `_hit_lines` logic. For example:
    ```
    FPL GW8 · deadline Sat 17 Oct 13:30 (Nairobi)
    Transfer: Saka -> Palmer (free)          | Save your free transfer (2 next week)
    Hit: none (a -4 would gain ~0.8 pts, not worth it)
    Captain: Haaland · Vice: Salah
    Chip: none                                | Play Bench Boost this week
    ```
    The late-change message gets the title "GW8: advice changed" plus a "What changed:" line.
- **`fpl/api.py`**: `send_ntfy(topic, title, body, priority="default", tags=())`, which POSTs to `https://ntfy.sh/{topic}` through the shared session. `send` takes a list of channels, so WhatsApp (CallMeBot) can be added later as one more function plus two secrets. Not built now.
- **`fpl/storage.py`**: `save_alert(gw, record)` / `load_alert(gw)` → `data/alerts/gw{N}.json` (stage sent, time, decisions).
- **`alert.py`** (new entry script):
  - `gate`: run the gate logic and write outputs.
  - `run --stage`:
    - **early**: send the full short message, then save the record.
    - **late**: if there is no early record (it was missed), send the full message; if the decisions changed, send the "changed" message; otherwise send nothing. Save the record in every case, so late never runs twice.
  - `--dry-run` prints without sending or saving. `--test` sends with a "TEST" title and saves nothing.
  - The topic comes from env `NTFY_TOPIC` only and is never written in the code.
- **`fpl/settings.py`**: `EARLY_ALERT_HOURS = (24.5, 20)`, `LATE_ALERT_HOURS = (3.5, 1.0)`, `NTFY_URL`.

Tests (new `tests/test_alerts.py`, plus additions to test_api / test_storage / test_recommend):
- window edges;
- next deadline across the season end;
- `decisions` and `changed` for each of: transfer, captain, chip, roll vs transfer, hit;
- message content and length, with the deadline shown in Nairobi time;
- the late logic, with or without an early record and changed or unchanged;
- `send_ntfy` with a mocked session (no network);
- `run()`/`main()` give the same output as before.

### Time and minutes budget (checked in the test run, see Verification)
- Full run estimate on a 2-core runner: pip install ~1–2 min, history pull ~5–7 min, training a few minutes (possibly 2× the laptop), and two solves up to 180 s each (≤6 min). That comes to about **15–20 min**, inside the 45-min timeout and well before a deadline 1 h+ away.
- The solver limit is a wall-clock cap, so a slower runner can't overrun it. The cost is that the plan may stop unproven. `advice.render` already says this, and the phone message adds "(approximate)". If the test runs show unproven plans as a matter of course, the fallback is to raise `SOLVER_TIME_LIMIT` for CI only, through an env override, and re-check the timing.
- Monthly minutes: hourly gate 24 × 30 = **~720 min**, plus about 2 full runs per GW × ~4.5 GWs × ~20 min = **~180 min**, total **~900 of 2,000 (45%)**. Even with a few failed retries it stays under 1,300. If you want a bigger margin, a two-hourly gate halves the 720 but makes the timing looser (alerts arrive 2–4 h and 23–25 h before the deadline).

### Docs
- Short spec at `docs/superpowers/specs/2026-10-12-deadline-alerts-design.md` and a plan at `docs/superpowers/plans/2026-10-12-deadline-alerts.md`, following the part 1–3 pattern.
- `CLAUDE.md`: a part 4 checklist item, the `alert.py` command, the workflow, and setup steps.
- Branch `part4-deadline-alerts`, pushed after every commit, then a PR into master.

## Steps you do yourself on GitHub (written out in CLAUDE.md as well)
1. **Phone**: install the ntfy app, then add a subscription to topic `savir-fpl-7k3q9` on ntfy.sh. Anyone who knows the name can read the topic, which is why it is kept random and secret.
2. **Secret**: on github.com/savirmediratta123/fpl-model go to Settings → Secrets and variables → Actions → New repository secret. Name `NTFY_TOPIC`, value `savir-fpl-7k3q9`.
3. **Default branch**: scheduled workflows run only from the repo's default branch. Check under Settings → General that it is `master`. Schedules start working after the PR is merged.
4. **Manual test**: Actions tab → "FPL deadline alerts" → Run workflow → pick the mode:
   - `check`: confirms that FPL's API answers from GitHub, shows the next deadline in Nairobi time and which stage would run now, and sends a test ping.
   - `dry-run`: the full recommendation, with the message and timings printed in the log.
   - `test-send`: the real message, marked TEST.
5. **Usage**: after a week, check Settings → Billing → Usage, where Actions minutes should be ~200/week.
6. **Later, for WhatsApp**: get a CallMeBot key and give it to me. I'll add `CALLMEBOT_PHONE` / `CALLMEBOT_APIKEY` secrets and a second channel.

## Verification
1. `python -m pytest -q`: all 155 existing tests plus the new ones pass, offline.
2. Local checks:
   - `python alert.py gate` prints the right GW and deadline in Nairobi time, with stage none today;
   - `python alert.py run --stage early --dry-run` prints the short message, which matches `recommend.py`'s THIS WEEK section;
   - `--test` with `NTFY_TOPIC` set delivers to the phone.
3. On GitHub: the "Run workflow" button only appears once the workflow file is on the default branch, so merge after the local checks pass, then:
   - `check` mode shows the FPL API returning 200 from the runner;
   - `dry-run` records the total time and each step's time, and whether both solves were proven optimal within 180 s;
   - `test-send` delivers to the phone.
4. Failure path: a manual run with a deliberately bad input (e.g. `mode: test-send` and team 0) gives the failure ntfy with a run link.
5. Once-only: run `auto` twice inside a window (or simulate with an env override of "now"). The second run should stop at the gate's cache lookup.
