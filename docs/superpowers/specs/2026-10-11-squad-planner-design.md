# Part 3: Squad Optimizer and Transfer Planner — Design

Date: 2026-10-11 · Status: draft for review

## Goal

`python recommend.py` loads the user's FPL team (ID 5234632) from FPL's public
API and prints, in plain English: this week's transfers (or "save your free
transfer"), whether a −4 hit is worth it, captain, vice-captain, starting XI and
bench order, a suggested plan for the following weeks, and when to play each
unused chip. Advice only: it never logs in and never makes transfers.

Also in this part:
- Fix two part 2 gaps: injury flags only affect next week, and the predicted
  clean-sheet chance is too narrow between easy and hard fixtures.
- A full-season simulation on 2025/26 that runs the whole system (predict →
  plan → play) from GW1 to GW38 and compares it with the average manager, the
  user's own 2025/26 total and an estimated overall rank.

Out of scope: press-conference / LLM news (a later part), predicting price
changes, logging in or making transfers.

## Verified facts this design relies on

All checked on 2026-10-11 against FPL's public API unless stated.

- **The user's team is fully readable without a login.** `/entry/{id}/`,
  `/entry/{id}/history/`, `/entry/{id}/transfers/` and
  `/entry/{id}/event/{gw}/picks/` give squad, bank, transfers with buy prices,
  hits charged and chips played. Selling prices and free transfers are not
  given directly and are rebuilt (below). Rebuilt for team 5234632: **1 free
  transfer, £0.6m bank for GW7, no chips used**, which matches the FPL app.
- **Chip windows come from FPL's data**, `bootstrap-static` → `chips`: first
  set `stop_event: 19`, second set `start_event: 20`, `stop_event: 38`. So
  first-set chips can be played up to and including GW19. Deadlines and
  chip windows are always read from FPL's API, never from news sites.
- **Deadlines:** `events[].deadline_time` (UTC). Shown in the user's time zone,
  Africa/Nairobi (UTC+3): GW7 `2026-10-17T10:00:00Z` → Sat 17 Oct 13:00, which
  matches the user's app. GW19 → Fri 1 Jan 21:30.
- **Free transfers:** one per gameweek, up to 5 saved (`max_extra_free_transfers:
  4` in `game_settings`); each extra transfer costs 4 points. GW1 transfers are
  unlimited; you start GW2 with 1.
- **Wildcard / Free Hit and free transfers:** the official Premier League
  article (21 Sep 2026) says "When playing a Wildcard, you will KEEP your
  banked transfers." It says nothing about Free Hit or the chip week's +1.
  Checked against 1,500 randomly sampled managers this season (572 played a
  Wildcard or Free Hit): the only rule consistent with every hit FPL charged
  (478 weeks, 0 contradictions; four other candidate rules each contradicted
  52–85 times) is **after a Wildcard or Free Hit you keep the free transfers you
  had, but get no +1 for the chip week.** Example: 2 saved, Wildcard in GW10
  → 2 free transfers in GW11.
- **2025/26 benchmarks:**
  - Average manager: **1,895 points.** This is the sum of `average_entry_score`
    over GW1–38, from an Internet Archive copy of `bootstrap-static` dated
    26 May 2026. That copy lists 13,107,732 managers.
  - User's 2025/26: **2,061 points, rank 2,927,134** (`/entry/5234632/history/` → `past`).
  - Every manager's `past` list gives their 2025/26 total and final rank, so a
    sample of managers gives a points-to-rank curve.
- **2025/26 fixture lists as known during the season:** 17 dated copies
  exist. 12 are commits of `data/2025-26/fixtures.csv` in vaastav's GitHub
  repo (Aug ×3, Sep ×4, Oct, Nov, Feb, Mar, Jun). 5 are Internet Archive copies
  of `/api/fixtures/` (Aug, Sep, Dec, Jan ×2).
- **Current clean-sheet spread (part 2 backtest, 2025/26, 1 week ahead):**
  - predicted P(CS) ranges 0.17–0.33 (5th–95th percentile);
  - actual clean-sheet rate by predicted fifth: 19%, 19%, 22%, 31%, 39%.
  - So real fixtures differ about twice as much as the model says.
- **Injury news in FPL's data** comes in these forms (status / `news`):
  `u` "Has joined York City on loan…" (McNally), `i` "Hamstring injury -
  Expected back 24 Oct" (Tzolis), `i` "Thigh injury - Unknown return date"
  (Isak), `d` "Ankle injury - 75% chance of playing" (Robinson), plus
  "Suspended until …" and permanent departures. `predict.py` currently drops
  status `u` players from its table, so a planner must still list them as
  sellable squad members.

## How it fits together

```
FPL public API (team 5234632) ──► myteam: squad, purchase & selling prices,
                                  bank, free transfers, chips left
part 2 model ──► xP per player per gameweek
  + team ratings (clean-sheet fix)
  + availability per week (injury news)
                    │
6-week planner (PuLP solver) ──► transfers/save, hits, captain, XI, chips in the window
chip scan to end of chip set ──► value of holding each chip, best weeks
                    │
recommend.py ──► plain-English advice + data/processed/plan.csv
simulate.py  ──► 2025/26 season simulation report
```

## Files

| File | Purpose |
|---|---|
| `fpl/settings.py` | All tunable settings in one place (below) |
| `fpl/api.py` (extended) | Entry endpoints; dated fixture copies (vaastav commits, Internet Archive); archived 2025/26 `bootstrap-static`. Still the only file that uses the internet |
| `fpl/storage.py` (extended) | Caches benchmark files and fixture copies under `data/raw/benchmarks/` and `data/raw/fixture_snapshots/2025-26/` |
| `fpl/myteam.py` | Team state from API responses: squad, purchase/selling prices, bank, free transfers, chips left |
| `fpl/rules.py` | FPL game rules shared by planner and simulation: selling price, free-transfer update, chip windows, auto-subs, gameweek scoring |
| `fpl/availability.py` | News → chance of being available per future gameweek; recovery curve from history |
| `fpl/team_ratings.py` | Time-weighted attack/defence ratings per team |
| `fpl/optimizer.py` | Multi-week squad/transfer/captain/chip solver |
| `fpl/chips.py` | Chip scan: value of each chip per week to the end of its window |
| `fpl/simulate.py` | Season simulation loop and report |
| `fpl/benchmarks.py` | Average manager, points-to-rank curve |
| `fpl/forecast.py` | Live predictions with availability, shared by `predict.py` and `recommend.py` |
| `fpl/advice.py` | Plain-English rendering for `recommend.py` |
| `recommend.py` | The command the user runs |
| `simulate.py` | Runs the 2025/26 simulation (and the settings test runs) |
| `pull_benchmarks.py` | One-off download: archived 2025/26 data, manager sample, fixture copies |

New dependencies: `pulp` (includes the CBC solver; works on Windows),
`highspy` (the HiGHS solver: faster here; CBC is the automatic fallback when it
isn't installed) and `tzdata` (time zone data for `zoneinfo` on Windows).

### Settings (`fpl/settings.py`)

| Setting | Default | Meaning |
|---|---|---|
| `TEAM_ID` | 5234632 | Overridden by `--team` |
| `TIMEZONE` | `Africa/Nairobi` | All printed times |
| `HORIZON` | 6 | Weeks planned (4–6 via `--horizon`) |
| `DISCOUNT` | 0.85 | Weight of week *k* is 0.85^(k−1) (was 0.9; changed after the 2025/26 simulation, see §7) |
| `HIT_MARGIN` | 1.0 | A hit must gain more than 4 + this (discounted) |
| `FT_END_VALUE` | 1.5 | Points credited per free transfer saved at the plan's end |
| `BENCH_WEIGHT` | 0.1 | Share of bench players' xP counted (cover value) |
| `FIRST_WEEK_BACK` | 0.8 | Availability in a player's first week back from a dated injury |
| `SOLVER_TIME_LIMIT` | 180 s | Per solve in recommend.py (run weekly, so a proven plan is worth the wait) |
| `SOLVER_GAP` | 0.001 | A plan proven within 0.1% of the best possible counts as optimal |
| `SIM_TIME_LIMIT` | 60 s | Per solve in the season simulation (at 30 s chip-heavy solves were left far from proven) |
| `DATA_MAX_AGE_HOURS` | 6 | `recommend.py` refreshes FPL data if older |

## 1. Loading the user's team (`fpl/myteam.py`)

- **Squad:** picks for the latest gameweek whose deadline has passed. If a
  Free Hit was played that week, use the previous week's picks, because a
  Free Hit squad reverts.
- **Purchase price** of each current player:
  - bought by transfer: `element_in_cost` from his latest transfer in;
  - in the team's starting squad (picks for the first gameweek in
    `history.current`, so managers who joined after GW1 work too) and never
    re-bought: his price at that gameweek, from part 1's `player_history`
    (`value` at that round, or at his next match if his team had none),
    else his current price (flagged as unknown).
  - Transfers made in a Free Hit week are ignored.
- **Selling price** = purchase + floor((now − purchase) / 2) when the price has
  risen (in £0.1m units); = now when it has fallen.
- **Bank:** `bank` from the latest picks' `entry_history`.
- **Free transfers:**
  - start GW2 with 1;
  - each normal week: next = min(5, max(free − transfers made, 0) + 1);
  - Wildcard / Free Hit week: next = free (kept, no +1).
  - Checked by test: the rebuild must also reproduce every `event_transfers_cost`.
- **Chips left:** FPL's chip windows minus chips in `history.chips`, so a
  chip used in GW1–19 uses up the first set's copy.

## 2. Injury flags in multi-week plans (`fpl/availability.py`)

Each player gets an **availability** a(w) ∈ [0, 1] for every planned gameweek
w. P(plays) and P(60+) are multiplied by a(w) and xP is recomputed. This
replaces `models.apply_availability`, which only capped the next gameweek.

| FPL status / news | a(w) |
|---|---|
| status `a`, no news | 1 |
| status `u` (loan, left the club) | 0 every week |
| "Expected back DATE" or "Suspended until DATE" | 0 for gameweeks whose last match is before DATE (the gameweek ends before he's back); `FIRST_WEEK_BACK` for the first gameweek ending on or after it; 1 after that |
| "N% chance of playing" (25/50/75) | N/100 next gameweek; halfway to 1 the week after; 1 after that |
| "Unknown return date", or status `i`/`s` with 0% and no date | next gameweek: FPL's `chance_of_playing_next_round`/100 when given (Isak: 0%); from the following week, the **recovery curve** (below) |
| status `n` (e.g. ineligible against parent club) | `chance_of_playing_next_round`/100 for next gameweek only |
| anything else | cap next gameweek at `chance_of_playing_next_round`/100 (old behaviour); the raw news is printed for the user |

**Recovery curve:** estimated once from 2022/23–2025/26 match history.
- Look at regular starters (60+ minutes in at least 6 of their last 10 team
  fixtures) who then had 0 minutes for *m* straight team fixtures.
- Measure the chance they play in the next fixture, for m = 1, 2, … (pooled
  above 8).
- For a live player with an unknown return, *m* = his team's fixtures since his
  last appearance, and the curve gives a(w) week by week.
- The curve also captures non-injury absences, so it is a rough guide. It is
  printed in the plan so the user can judge it.

**Form on return:** while a player is out, his recent-form features fill up
with zero-minute matches, so the model would under-rate him once he is back.
For flagged players in live prediction, features are built from his history
**up to his last appearance**, so his matches missed while flagged are
excluded. This applies to live prediction only, because historical news isn't stored.

## 3. Clean-sheet fix (`fpl/team_ratings.py`)

- For every team, fit an **attack and defence rating** plus a home advantage
  with a weighted Poisson regression. The target is each team's xG in each
  match; using xG instead of goals is less noisy.
- Fitting uses all matches before the cutoff (2022/23 onward), with weights
  halving every 180 days.
- Promoted teams start from the part 2 promoted-team prior and blend into
  their own rating as matches accumulate (same formula, k = 6).
- **New team features for each fixture:** the ratings' expected goals for and
  against (`rating_xg_for`, `rating_xg_against`) and `rating_cs = exp(−rating_xg_against)`.
- Added to `TEAM_FEATURES` for the clean-sheet and goals-conceded models only.
  Player models keep their part 2 features.
- No future data: fitted only on matches before each cutoff, with the same
  leakage test as part 2.
- **Accepted if**, on the 2025/26 backtest (1 week ahead):
  - the 5th–95th percentile range of P(CS) is wider than 0.17–0.33;
  - log loss is no worse than the part 2 model (measured when the build starts);
  - the bucket calibration stays close to the diagonal.
  - If the wider spread costs accuracy, stop and report before going further.

## 4. The planner (`fpl/optimizer.py`)

**Players considered:**
- the user's 15;
- the top 40 per position by discounted horizon xP (and the top 15 per
  position by xP per £m);
- every player in at least one of those lists, which keeps the solve fast.

**Choices per gameweek *w* in the horizon:**
- the 15-man squad, starting 11 and captain;
- transfers in and out;
- one of {no chip, Wildcard, Free Hit, Bench Boost, Triple Captain};
- a separate Free Hit squad, used only if a Free Hit is played that week.

**Rules enforced:**
- **Squad:** 2 GK, 5 DEF, 5 MID, 3 FWD; at most 3 per club.
- **Starting 11:** 1 GK, 3–5 DEF, 2–5 MID, 1–3 FWD.
- **Money:**
  - selling gives the selling price (fixed per owned player);
  - buying costs the current price;
  - bank after every week ≥ 0;
  - prices are assumed constant over the horizon.
- **Free transfers:** follow §1's update rule (cap 5, chip weeks keep the count
  with no +1). Paid transfers = transfers − free, at 4 points each.
- **Wildcard:** transfers that week cost nothing.
- **Free Hit:** that week uses the Free Hit squad (budget = selling value + bank);
  the normal squad is unchanged.
- **Chips:** at most one per gameweek, only chips left, only inside their
  window from FPL's data.

**Objective** (maximise):

    Σ_w DISCOUNT^(w−1) · [ xP(starting 11) + xP(captain) · (1, or 2 with Triple Captain)
                           + BENCH_WEIGHT · xP(bench), or full bench xP with Bench Boost
                           − 4 · paid transfers − HIT_MARGIN · paid transfers ]
    + FT_END_VALUE · free transfers left after the last week
    + Σ_chips HOLD_VALUE(chip) · (chip not played in the horizon)

Triple Captain and Bench Boost extras (the second captain multiple; the bench's
xP beyond `BENCH_WEIGHT`) are **not discounted**: they count in full in every
plan week, so discounting never pulls a chip towards the soonest week. `Plan.value`
includes them the same way. Everything else keeps its discount.

The `HIT_MARGIN` term means a hit is only taken when it gains more than
4 + margin points after discounting. Points shown to the user exclude the
margin.

A tie-break of 0.01 points per transfer keeps the planner from making moves that gain nothing.

**Solving** (measured on the real GW7–12 case: one MIP with every chip free was
still 2–7% from proven after 180 s and up to 10 minutes; the search below proves
it within 0.1% in about 50 s):
- The LP relaxation is kept tight, because it is what proves a plan best:
  - Triple Captain: at most one tripled captain per week, gated by the chip.
  - Bench Boost: at most 4 boosted bench players per week, gated by the chip.
  - Free Hit: the playing squad is "kept squad + Free Hit squad", the kept part
    being the normal squad scaled by (1 − Free Hit), the Free Hit squad's shape,
    club limits and budget scaled by the Free Hit flag; no transfers that week.
  - Wildcard: transfers made under it are marked per player (each at most the
    Wildcard flag, at most 15 in all), and only the others count against free
    transfers, so a fraction of a Wildcard frees only that fraction of each move.
  - Variables that are products of 0/1 choices are continuous (they come out
    whole anyway).
- **Chip search:** a fractional Wildcard, Free Hit or Bench Boost still makes the
  LP look much better than any real plan, so these chips are fixed one at a
  time by branch and bound: each chip goes to a week or to "not in this plan",
  the LP with those fixed bounds the branch (best bound first), branches that
  can't beat the best plan by more than `SOLVER_GAP` are dropped, and a MIP is
  solved once all three are fixed (Triple Captain stays in the MIP; its LP is
  already tight).
- Solver: HiGHS (one model kept between solves, so a chip fix is a bound
  change); CBC if `highspy` isn't installed.
- **Status:** "Optimal" means the search finished and nothing it left out can
  beat the plan by more than `SOLVER_GAP`. If time runs out, the best plan so far
  is used and the status is "time limit (gap X%)", X being how much better the
  best possible plan could still be; the advice says so.

**After solving:**
- Vice-captain = the starter with the second-highest xP.
- Bench order: the backup GK goes in the GK slot; outfield subs are ordered by
  that week's xP.
- Only week 1's moves are advice to act on; later weeks are a suggested path,
  re-planned every week.
- If the solver hits the time limit, it uses the best plan found and says so.

**Explaining hits:** the planner solves a second time with week 1's hits fixed the other way: no hit if the plan takes one, or at least one genuine extra transfer (more transfers than free ones, no Wildcard/Free Hit) if it doesn't. Hits always equal exactly the transfers beyond the free ones. Both solves stop within `SOLVER_GAP`, so the second can come out ahead; then it becomes the advice (and the advice says the check found it), so the verdict never contradicts the plan. The gain is reported in expected points after paying the 4-point hits but before the safety margin: the difference in the planner's objective (chips kept and free transfers saved included) plus the discounted `HIT_MARGIN` terms, e.g. "Taking a -8 hit: over 6 weeks this plan is expected to score about 2.4 pts more than the best plan without a hit, after paying the 8." A gain under 2 points is called a close call and the best no-hit plan's week-1 moves are printed; if either solve stopped at its time limit the comparison is called approximate. Without a hit, the advice says how many points more (too few to be worth the risk) or fewer the best plan with a hit this week would score.

## 5. Chip timing (`fpl/chips.py`)

The **chip scan** runs over every gameweek from the week after the horizon to
the end of each unused chip's window. The weeks it can see are limited by the
predictions available; it predicts up to 13 weeks ahead, which covers GW19 from now.
When a window ends more than 13 weeks away (early in the simulation, or the
second set), only the first 13 weeks are scanned. It uses the squad at the end
of the plan: (The hold values fed to the planner use the current squad, because the plan isn't known yet. The printed chip advice re-scans with the plan's final squad.)

- **Bench Boost:** the sum of the four bench players' xP that week.
- **Triple Captain:** the top starter's xP that week (the extra captain multiple).
- **Free Hit:** the best squad's xP for that week (budget = squad value) minus
  the planned squad's starting-11 xP.
- **Wildcard:** value = the best fixed squad's xP over the (up to 6-week)
  window from that week, minus the best squad reachable from the current squad
  with F = window length // 2 free transfers, no hits and no transfers after
  week 1 (both solved with the same flat settings). It is what a Wildcard adds
  beyond what ordinary transfers could do. The scan pool always includes the
  current squad.

`HOLD_VALUE(chip)` = the best scan value in a week after the horizon,
undiscounted. It is 0 when no weeks are left in the window, so a
chip about to expire gets played. Predictions far ahead are flatter than near
ones, which tends to make hold values a little low; the simulation shows
whether chips get played too early.

**Printed:** for each unused first-set chip, either "play it this week (worth
+X)" or "hold — best week GWn (+X), because …". The reason names double or
blank gameweeks from the fixture list when they are involved. Second-set chips
are mentioned once GW19 has passed.

## 6. `recommend.py`

```
python recommend.py               # team 5234632, 6-week plan
python recommend.py --team 123 --horizon 4
```

1. If the processed data is older than `DATA_MAX_AGE_HOURS`, re-run the part 1
   pull (`pull_data.py` logic).
2. Load the team (§1); train and predict 13 weeks ahead (6 for the plan, more
   for the chip scan) with the clean-sheet fix and availability (§2–3).
3. Plan (§4), scan chips (§5), print, and save the full plan to
   `data/processed/plan.csv`.

The printout has these sections, in plain English:

- **Header:** team name, gameweek, deadline in Nairobi time, bank, free
  transfers, chips left and the first set's last week.
- **This week:**
  - transfers out → in, with price, reason and xP over the horizon;
  - or "save your free transfer", with how many you will have next week;
  - the hit verdict;
  - captain and vice-captain;
  - the starting XI with each player's xP;
  - bench order.
- **Injury watch:** every flagged squad player, with FPL's news and his
  availability for each planned week.
- **Next weeks:** the suggested moves for weeks 2–6, labelled as a plan
  that will be re-run.
- **Chips:** the advice from §5.

## 7. 2025/26 season simulation (`fpl/simulate.py`, `simulate.py`)

**Loop:**
- Before GW1, pick the best £100m squad for the first 6 weeks.
- Then, for each gameweek: retrain on data before its deadline, predict, plan,
  play week 1 of the plan, score it with real points, update the state
  (squad, purchase prices, bank, free transfers, chips).

**What the simulated manager knows at each deadline:**
- matches already played (part 2 leakage rules);
- prices as they actually were that gameweek (vaastav `value`);
- the **newest fixture copy saved before that deadline** (§ verified facts).
  Fixtures without a gameweek count as blanks until a later copy places them.
  With gaps of up to 2–3 months between copies, the simulation may learn of a
  double gameweek later than real managers did, never earlier.
- no injury news (none was stored).

**2025/26 rules:**
- chip windows from the archived 2025/26 `bootstrap-static`;
- free transfers as in §1;
- the AFCON top-up: free transfers set to 5 in GW16 (premierleague.com/en/news/4362102); chip windows in the archived 2025/26 data match this season's (set 1: GW1/2–19, set 2: GW20–38).
- DefCon scoring and 2025/26 scoring rules (part 2's `rules_for`).

**Scoring a week** (`fpl/rules.py`):
- real points of the starting 11;
- **auto-subs:** a starter with 0 minutes is replaced by the first bench
  player (in bench order) who played and keeps the formation legal; GK only
  by GK;
- the captain is doubled, or the vice-captain if the captain didn't play
  (tripled with Triple Captain);
- Bench Boost counts all 15;
- hits are subtracted.

**Speed:**
- Predictions (one walk-forward run, up to 13 weeks ahead) are computed once,
  about 15–20 minutes, and cached.
- A second set of predictions uses the final fixture list.
- Each settings run only re-runs the planner (38 solves).

**Settings test** (on the dated-fixture predictions), one setting at a time
around the defaults, 7 runs:
- discount 0.85 / 0.9 / 0.95;
- hit margin 0 / 1 / 2;
- saved-transfer value 1 / 1.5 / 2.

Outcome (2026-10-11): with a 60 s solve limit, GW1-19 plans were often unproven (gaps up to
57%), so the runs were noisy. Re-run like for like at 180 s, every plan proven best: discount
0.8 = 2,256, 0.85 = 2,208, 0.9 = 2,173. 0.85 beats 0.9 by 35 and its neighbour 0.8 also does
well, so the default became 0.85 (0.8 not chosen: edge of the tested range). The simulation's
limit became 180 s, and the settings table is re-run around 0.85 (discount 0.8 / 0.85 / 0.9).
After the pre-Nov-2022 xG fix (2026-10-12), full table at 180 s: 0.85 = 2,249, 0.8 = 2,149,
0.9 = 2,157; hit margin 0 = 2,205, 2 = 2,222; FT value 1 = 2,212, 2 = 2,147 (97-100% of plans
proven best). 0.85 beat 0.9 in both 180 s runs (+35, +92), so it stays; the other defaults stay.
The same setting moved up to ~100 pts between runs, so treat gaps under ~100 as noise.

Keep the defaults unless another value wins by more than about 30 points
*and* its neighbouring values also do well. The full table is reported.

**Report** (printed and saved to `data/processed/simulation_report.csv` and
`simulation_weeks.csv`):
- Season total vs the **average manager (1,895)** and **the user's 2,061**.
- **Estimated overall rank**, from a points-to-rank curve of about 3,000
  sampled managers' 2025/26 totals and ranks (interpolated on log rank). It
  is labelled as roughly ±10–20%, and less precise at very high ranks.
- Week by week: points, transfers, hits, captain (and whether he was the
  best choice in the XI), chip.
- **Chip dependence:** points each chip added over a normal week, and the
  season total without them.
- **Fixture knowledge:** the season total with the final fixture list, next to
  the total with dated fixture copies.
- **Pessimistic factors:** no injury news; no press conferences; fixture copies
  may lag what managers knew.
- **Optimistic factors:**
  - fixture copies may miss changes made between copies;
  - player positions and the player list are end-of-season versions (players
    who joined mid-season are only buyable from their first match);
  - the model's settings were chosen in part 2 partly on 2025/26 data;
  - the planner settings are tested on the same season they are reported for
    (limited to 7 runs to keep this small).

## Testing

All tests run offline, using small saved files built from real responses.

| Test file | Checks |
|---|---|
| `tests/test_myteam.py` | Selling price on hand-worked cases (rise with odd profit, fall, unchanged). The saved responses for team 5234632 rebuild 1 free transfer, £0.6m bank, no chips. Free Hit week picks/transfers ignored |
| `tests/test_rules.py` | Free-transfer update on saved real managers' histories (with Wildcard and Free Hit) reproduces every hit FPL charged; cap at 5; auto-subs, vice-captain, Triple Captain, Bench Boost and hits on hand-worked weeks |
| `tests/test_availability.py` | Each news form → correct a(w) using real deadlines (the user's four players plus a suspension); recovery curve is decreasing and between 0 and 1 on the test league; form-on-return excludes missed matches |
| `tests/test_team_ratings.py` | Changing a future match doesn't change earlier ratings; a strong defence gets a higher `rating_cs`; promoted prior applies |
| `tests/test_optimizer.py` | On small made-up pools: squad, formation, club and budget limits; selling prices used; hit only when gain > 4 + margin; saves a transfer when that wins next week; cap 5; Wildcard/Free Hit free-transfer rule; Free Hit squad reverts; chip windows and one chip per week; captain is the top starter; a status-`u` player is sold or benched |
| `tests/test_chips.py` | Bench Boost prefers a double gameweek; hold value 0 at window end |
| `tests/test_simulate.py` | Each week uses only data and fixture copies from before its deadline; a short run on the made-up league completes and the season total equals the sum of scored weeks |
| `tests/test_benchmarks.py` | Points-to-rank curve is monotonic and reproduces sampled points |
| `tests/test_recommend.py` | End to end on a saved copy of team 5234632: every section prints; the deadline shows as Sat 17 Oct 13:00 |

## Success criteria

- All tests pass, including the part 2 tests.
- Clean-sheet fix meets the §3 acceptance check on 2025/26.
- `simulate.py` runs 2025/26 end to end and prints the report with both
  fixture versions, chip dependence, rank estimate and both lists of factors.
- `python recommend.py` prints advice for team 5234632 that passes the
  user's own sanity check before anything is merged.
