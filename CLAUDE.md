# Building an FPL Machine Learning Model

Oct 10, 2026 · @Savir Mediratta

The strongest approach is three pieces chained together, not one big model. The third piece is what handles a team you already own.

1. **Predict** each player's expected points (xP) for the next 3–6 gameweeks.
2. **Optimize** a squad against FPL's rules using those predictions.
3. **Plan** transfers, captaincy and chips for your existing team across several weeks.

## 1. Predict expected points per player, per gameweek

Don't train a model to pick a team directly. Predict each player's expected points, broken into the separate events FPL scores:

| Component | What drives it |
| --- | --- |
| Minutes probability | Will he start and play 60+? News matters most here. |
| Attacking returns | xG, xA, shots and box touches, adjusted for opponent strength |
| Clean sheet probability | His team's defensive numbers vs the opponent's attack |
| DefCon and bonus | Defensive actions; bonus model must reflect the 2026/27 BPS rework |

DefCon points remain this season. The bonus system was reworked to reduce overlap with DefCon and to help goalkeepers, full-backs and attackers, so last season's bonus data will be slightly off.

A gradient-boosted model (LightGBM or XGBoost) on rolling averages of these stats works well and is far simpler than deep learning.

**Data sources**

- **Official FPL API** (free): prices, ownership, fixtures, points history, plus `news` and `chance_of_playing_next_round` per player. This covers most injury and suspension news.
- **Understat or FBref** for xG and xA.
- **Press-conference news**: use an LLM to read articles and turn them into a "likely to start" score that feeds the minutes model.

## 2. Encode the rules in an optimizer

Feed the predictions into an integer linear programming solver (PuLP in Python), not a greedy "pick the top scorers" loop. Constraints to encode:

- £100m budget; 2 GK, 5 DEF, 5 MID, 3 FWD; max 3 players per club
- Captain scores double
- Each extra transfer costs 4 points; up to five free transfers can be banked
- Selling price = purchase price + half the profit, rounded down (matters for an existing team)
- Two sets of Wildcard, Free Hit, Triple Captain and Bench Boost. The first set must be used before the GW19 deadline on 2 January 2027 and does not roll over.
- No Assistant Manager chip this season, and no extra AFCON transfers

## 3. Handling your existing team

Give the system your FPL team ID. The API returns your squad, bank and the prices you paid. The optimizer then plans across several gameweeks and answers:

- **Roll or transfer?** Saving a free transfer is often worth more than a small upgrade now.
- **Is a -4 hit worth it?** Only if the incoming player's xP gain over the next few weeks clearly beats 4 points.
- **Captain and bench order** each week.
- **Chip timing.** The first chip set expires in January, so flag the best weeks for Bench Boost and Triple Captain, such as double gameweeks or strong fixture runs.

The biggest mistake is optimizing one week at a time. Planning over 4–6 weeks, with a small discount on later weeks, consistently does better.

## Claude Code setup and next steps

No FPL-specific skill is worth installing. The Python code-intelligence plugin and Superpowers (for planning and test-driven building) are the useful ones here.

- [x] Build a script that pulls the FPL API
- [x] Train the xP model on rolling stats
- [x] Run the transfer optimizer on the existing squad
- [ ] Deadline alerts on GitHub Actions (part 4: built, waiting for the first live run)
- [ ] Autopilot (part 5: tested against simulated FPL replies; the real FPL calls have never run)

## Project status (parts 1-3, merged to master 2026-10-12)

Run everything with the system `python` (3.13), not `.venv`. Only `fpl/api.py` uses the internet and only `fpl/storage.py` touches `data/`. About 200 offline tests: `python -m pytest -q`.

**Commands**

- `python pull_data.py` (part 1): downloads players, teams, gameweeks, fixtures and per-player match history from the FPL API into `data/`. `--skip-history` fetches everything except match history, in seconds. `pull_history.py` downloads past seasons from vaastav, once.
- `python predict.py` (part 2): trains the LightGBM xP model and prints a ranked xP table for the next gameweek. `--horizon 6` gives one column per week. Injury news is applied to every week.
- `python backtest.py` (part 2): walk-forward backtest of model A against model B and a last-5 baseline, plus a clean-sheet report. `--season`, `--gws`, `--horizon` give quicker runs.
- `python recommend.py` (part 3): plain-English advice for the `FPL_TEAM_ID` team (`--team`, `--horizon 4-6`). It covers transfers or rolling, whether a hit is worth it, captain and vice-captain, bench order and chip timing. Deadlines are shown in Nairobi time, and chip windows come from the FPL API (first set through GW19). It refreshes data more than 6 hours old.
- `python simulate.py` (part 3): replays 2025/26 with the whole system (fixture lists as known at each deadline) against the average manager (1,895), the original author's 2,061 and an estimated rank (full 38-GW runs only). It runs a settings table. `--quick`, `--only`, `--discounts`, `--time-limit`, `--workers`, `--rebuild`. Needs `pull_benchmarks.py` once.

- `python scout.py`: checks and gives top players for the upcoming gameweek by position (GK, DEF, MID, FWD), top captain candidates with projected returns, budget/value enablers, and differentials (<10% ownership). Supports `--gw <N>`, `--top <N>`, and `--fast`.
- `python remind.py`: compares your current team against the optimal setup and reminds you of exact switches to execute before the deadline: transfers to make, captain/vice armband switch, and starting XI/bench adjustments, with a deadline countdown in Nairobi time. Supports `--team <ID>`, `--fast`, and `--notify` (push to phone via ntfy).
- `python tracker.py`: tracks and analyzes points scored by your team at the end of each gameweek: gross score, transfer hits, net score, overall/GW ranks, points left on the bench, captain returns, and auto-subs, with comparisons to the Gameweek Dream Team. Supports `--team <ID>`, `--gw <N>`, and `--offline`.
- `python dream_team.py`: solves the integer linear program (PuLP) to find the theoretical optimal 15-player squad and Starting XI that achieves the highest gameweek points under £100m budget and 3-per-club rules. Shows formation, captain, vice-captain, bench, and compares against your current team. Supports `--gw <N>`, `--budget <tenths>`, `--team <ID>`, and `--fast`.

- `fpl/autopilot.py` + `alert.py` (part 5): with `FPL_AUTOPILOT=on`, the late run applies the plan and sets captain and bench on the real team. `FPL_AUTOPILOT_POLICY=safe` uses only free transfers and no chips; `all_planned` also executes optimizer hits and chips. `FPL_MESSAGE_FREQUENCY=updated_only` sends only after an actual update or a blocked/failed run. Live numbers come from the logged-in my-team call, and every action is read back. Blocked = a safety check said no (no retry); Failed = FPL trouble (hourly retry until 2 h before the deadline, then an urgent message). The refresh token rotates on every use, so `store_refresh_token` must save it. Never add retries to the POST calls, never log tokens, and never work around an FPL block.
- `python alert.py` (part 4): deadline alerts by ntfy. `gate` checks whether an alert is due (one FPL request). `run --stage early|late` makes the advice and sends a short message: transfers or save, hit, captain and vice, chip. Early goes out once, about 24 h before; late, about 3 h before, only if those calls changed. `--dry-run` prints only and `--test` sends a TEST message; neither saves. The topic comes from the `NTFY_TOPIC` env var, never the code. `ALERT_NOW=<ISO time>` fakes the clock.

**Part 4: GitHub Actions** (`.github/workflows/deadline-alerts.yml`). An hourly gate job (~1 min) reads the next deadline from the FPL API. Inside a window (24.5–20 h or 3.5–1 h before, see `EARLY_ALERT_HOURS`/`LATE_ALERT_HOURS`) it starts the advise job, which pulls fresh FPL data, restores past seasons from the cache, trains, plans and sends. "Already sent" is tracked by cache keys `alert-gw<N>-<stage>`, which hold `data/alerts/gw<N>.json`. A failed run sends an ntfy failure message, once per gameweek and stage; a failed gate sends one at most once a day. Minutes: about 720/month for the gate plus ~20 per advise run, out of 2,000 (private repo).

Setup is in `README.md`. Modes: `check` tests that GitHub can reach FPL and sends a test ping. `dry-run` puts the full advice and timings in the log, and with autopilot on also what it would do. `test-send` sends the real message, marked TEST. `auto` is what the hourly schedule does.

**Current defaults** (`fpl/settings.py`): discount 0.85, hit margin 1.0, end-of-plan free-transfer value 1.5, bench weight 0.1, horizon 6. Solver limit 180 s (both recommend and simulate). First week back from injury 0.8, team-rating half-life 180 days. The planner follows the club-move rule: you may keep 4+ from a club after a mid-season move, but can't buy from it until you're back to 3. Free Hit squads stay at 3 per club.

**Latest results:** 2025/26 simulation 2,249 pts (+354 vs average, +188 vs the original author's 2,061, est. rank ~461k). Settings runs moved by up to ~100 pts, so smaller gaps are noise.

**Open follow-ups**

- Clean-sheet top end: the best predicted fifth of fixtures kept clean sheets ~37% of the time vs ~34% predicted. Shorter half-lives did not help (2025/26 GW8-38, one week ahead, log loss: 180 days 0.5449, 120 days 0.5465, 90 days 0.5456; MAE 0.972 for all), so 180 stays. Try more weight on recent form, and measure with `backtest.clean_sheet_report`.
- Test discount 0.75/0.8 on another season (0.8 swung 107 pts between 2025/26 runs).
- Advice display:
  - near the season's end, "over 6 weeks" can overstate the weeks actually planned;
  - the no-hit alternative line reads oddly when that plan plays a Wildcard;
  - in a Wildcard week the "No hit" line compares against the best plan without a chip.
- Simulation: two run labels could in theory save to the same file name (`save_run`).

## Sources

- [Fantasy Football Scout: FPL 2026/27 rule changes](https://www.fantasyfootballscout.co.uk/2026/07/20/fpl-2026-27-5-rule-changes-new-features-announced)
- [Flashscore: FPL 2026/27 rule changes](https://www.flashscore.com/news/soccer-premier-league-fantasy-premier-league-2026-27-all-rule-changes-and-new-features/KGmq7ts1/)
- [BeatFPL: FPL 2026/27 rule changes](https://www.beatfpl.com/guides/fpl-26-27-rules-changes)
- [Onside Arena: FPL rule changes 2026/27](https://onsidearena.com/tips/fpl-rule-changes-2026-27)
