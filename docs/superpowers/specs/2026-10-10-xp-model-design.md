# Part 2: Expected Points (xP) Model — Design

Date: 2026-10-10 · Status: approved

## Goal

Predict each Premier League player's expected FPL points (xP) for the next
gameweek, then for the next 3–6 gameweeks. Prove it works with a walk-forward
backtest on 2025/26 that beats a simple baseline, with no future information
leaking into training. Print a ranked table of top players for a sanity check.

Out of scope for this part: press-conference / LLM news scoring (later), the
squad optimizer and transfer planner (later parts).

## Verified facts this design relies on

- **vaastav/Fantasy-Premier-League** has full GW1–38 player-by-match files
  (`data/<season>/gws/merged_gw.csv`) for every season through 2025/26.
  2025/26 has 29,757 rows and includes DefCon columns
  (`clearances_blocks_interceptions`, `tackles`, `recoveries`,
  `defensive_contribution`). Its 2026/27 folder only has GW1.
- FPL's own `expected_goals` / `expected_assists` exist from 2022/23 onward,
  with no missing values. Earlier seasons lack them, so training uses
  **2022/23 onward**; Understat/FBref are not needed yet.
- Our part-1 `data/processed/player_history.csv` (2026/27, GW1–6, 3,883 rows)
  uses the same column names as vaastav.
- The scoring rules below, coded up, reproduce `total_points` **exactly** for
  100% of rows in both 2025/26 (vaastav) and 2026/27 (our pull).

## FPL scoring rules (2026/27)

| Event | GK | DEF | MID | FWD |
|---|---|---|---|---|
| 1–59 minutes | 1 | 1 | 1 | 1 |
| 60+ minutes | 2 | 2 | 2 | 2 |
| Goal | 10 | 6 | 5 | 4 |
| Assist | 3 | 3 | 3 | 3 |
| Clean sheet (needs 60+ min) | 4 | 4 | 1 | 0 |
| Every 2 goals conceded | −1 | −1 | 0 | 0 |
| Every 3 saves | +1 | — | — | — |
| Yellow / red card | −1 / −3 | same | same | same |
| DefCon (max 2 per match) | — | 2 pts at 10+ CBIT | 2 pts at 12+ CBIRT | 2 pts at 12+ CBIRT |
| Bonus | 0–3 | 0–3 | 0–3 | 0–3 |

CBIT = clearances, blocks, interceptions + tackles. CBIRT = CBIT + recoveries.
Penalty saves (+5), penalty misses (−2) and own goals (−2) are scored when
rebuilding actual points but **not predicted** (each is under 0.05 pts/match on
average).

Season differences: DefCon exists only from 2025/26. Goalkeeper goals were 6
points before 2026/27.

**Rules are season-aware.** `scoring.py` holds one rules table per season
(`rules_for(season)`), and every conversion from component predictions to xP,
and every rebuild of actual points, uses the rules of the season being scored:
GK goal = 6 before 2026/27 and 10 from 2026/27; DefCon points = 0 before
2025/26. So the 2025/26 backtest scores GK goals at 6, the 2026/27 backtest
and live predictions at 10. (The earlier 100% check passed with 10 for
2025/26 only because no goalkeeper scored in any of 2022/23–2025/26 —
checked; the rebuild test will use the season-aware rules.) The 2026/27 BPS rework changed how bonus is earned
(no −1 for being tackled, CBI 1 BPS per 3 instead of per 2, new GK save
values) but not the 0–3 bonus scale.

## Approach

**A (main): one small model per scoring event, then apply the rules.**
**B (yardstick): one LightGBM predicting total points directly.**
**Baseline: a player's points per game over his last 5 matches.**

A and B are both LightGBM (gradient-boosted trees). B and the baseline exist
only to measure whether A's structure is worth it.

```
vaastav CSVs (2022/23–2025/26)  ┐
our player_history.csv (2026/27) ┴─► combined match table ─► features (only from earlier GWs)
                                                                    │
         ┌───────────┬──────────┬───────────┬──────────┬───────────┼──────────┐
      minutes     goals &     team CS &    GK saves    cards     DefCon     bonus
                  assists     goals conc.
         └───────────┴──────────┴───────────┴──────────┴───────────┴──────────┘
                              FPL scoring rules per position
                                            │
                       xP per player ─► ranked table / backtest report
```

## Component models

Predictions are per player **per fixture**; a double gameweek sums its
fixtures, a blank gameweek is 0.

| Part | Target | Model | Trained on | Contribution to xP |
|---|---|---|---|---|
| Minutes | P(minutes > 0), P(minutes ≥ 60) | two LightGBM classifiers | all rows | 1·(P(plays) − P(60+)) + 2·P(60+) |
| Goals | goals if he plays | LightGBM, Poisson objective | rows with minutes > 0 | × P(plays) × goal pts for position |
| Assists | assists if he plays | LightGBM, Poisson objective | rows with minutes > 0 | × P(plays) × 3 |
| Team clean sheet | P(team concedes 0) | LightGBM classifier, one row per team-fixture | all team-fixtures | × **P(60+)** × CS pts (P(CS and 60+), assuming independence) |
| Goals conceded (GK/DEF) | team goals conceded λ | LightGBM, Poisson objective, team-fixture rows | all team-fixtures | −E[floor(GC/2)] from the Poisson distribution × P(60+) |
| Saves (GK) | saves if he plays | LightGBM, Poisson objective | GK rows with minutes > 0 | +E[floor(saves/3)] from the Poisson distribution × P(plays) |
| Cards | P(yellow), P(red) if he plays | no model: his cards per appearance over the last 38 fixtures, blended with the newcomer prior like every other feature | all rows | −1·P(yellow)·P(plays) − 3·P(red)·P(plays) |
| DefCon | P(reaches position threshold) if he plays | LightGBM classifier, DEF/MID/FWD only | **2025/26 and 2026/27 rows only**, minutes > 0 | × 2 × P(plays) |
| Bonus | bonus if he plays | LightGBM, Poisson objective | rows with minutes > 0 | × P(plays) |

The goals-conceded term uses P(60+) as an approximation: players subbed on
for a few minutes rarely concede two goals while on the pitch.

### Bonus rework handling

Every row carries a `rules_era` feature: `pre_2025`, `2025_26`, `2026_27`.
The bonus model sees it, and 2026/27 rows get sample weight 3, so the model
leans towards the new system as this season's data grows. The only place the
new bonus rules can be tested is the small 2026/27 backtest (below).

### Model B

Trained on `total_points` as actually scored at the time (so older seasons
have no DefCon and old GK goal values), with `rules_era` as a feature. Same
input features as A. Used only in the backtest.

## Data

### Sources

- vaastav, seasons 2022-23, 2023-24, 2024-25, 2025-26: `gws/merged_gw.csv`,
  `players_raw.csv`, `teams.csv`, `fixtures.csv`. Downloaded once, cached to
  `data/raw/vaastav/<season>/`, re-download only with `--force`.
- 2026/27: our `data/processed/player_history.csv`, `players.csv`,
  `teams.csv`, `fixtures.csv` from part 1.

### Combined match table (one row per player per fixture)

Columns: `season`, `gw`, `kickoff_time`, `fixture_id`, `player_code`,
`player_name`, `position` (GK/DEF/MID/FWD), `team`, `opponent`, `was_home`,
`price`, `minutes`, `starts`, `goals_scored`, `assists`, `clean_sheets`,
`goals_conceded`, `saves`, `yellow_cards`, `red_cards`, `penalties_saved`,
`penalties_missed`, `own_goals`, `bonus`, `bps`, `expected_goals`,
`expected_assists`, `expected_goals_conceded`, `threat`,
`clearances_blocks_interceptions`, `tackles`, `recoveries` (DefCon columns
null before 2025/26), `total_points`, `rules_era`.

- **Player linking:** element IDs reset each season. Each season's
  `players_raw.csv` (and our bootstrap) has a permanent `code`, used as
  `player_code`.
- **Team linking:** team IDs reset each season; teams are linked by full name
  (with a small alias map if any name differs between sources).
- **Position:** taken per season (a player reclassified MID→DEF is scored by
  his position that season).
- vaastav files are read with `encoding="latin-1"` (accented names).
- 2024/25 Assistant Manager rows (position `AM`) are dropped.

## Features

All features for gameweek N use **only matches with kickoff before gameweek
N's first kickoff**. Rolling values are computed and then shifted one match.

Player features (follow the player via `player_code`, across clubs and seasons):
- For each stat, a rate over his last 3, 5, 10 and 38 fixtures (38 ≈ one
  season, and unlike "season to date" it doesn't go empty at the start of a
  season):
  - per-fixture averages: minutes, starts, played (minutes > 0), played 60+;
  - per-90 rates: xG, xA, goals, assists, threat, BPS, bonus, saves, CBIT,
    CBIRT (CBIT/CBIRT only over minutes that have DefCon data);
  - per-appearance rates: DefCon-threshold hits, yellow cards, red cards.
- `matches_played` (career, in our data), `matches_this_season`,
  `matches_for_current_team`.
- Position, price before the deadline, home/away, `rules_era`.

Team features (use the team **in the current fixture row**, so a player who
moved clubs gets his new team's strength):
- Own team rolling xG for, xG against, goals for/against (last 6 matches).
- Opponent rolling xG for, xG against (last 6 matches).

Deliberately excluded:
- FPL's team `strength_*` ratings — vaastav stores end-of-season values,
  which would leak the future.
- `transfers_in/out/balance`, `selected` for gameweek N — unclear when they
  were recorded, so they may leak.
- Injury news — not stored historically. Training never sees it; in live
  prediction only, P(plays) and P(60+) are capped at
  `chance_of_playing_next_round / 100` when FPL has flagged the player.
  The backtest cannot measure this.

### Players and teams with little or no history

Every rolling or per-90 player feature is blended with a prior:

    feature = (n · player_value + k · prior) / (n + k)

where n = the player's **total** number of PL matches before the cutoff
(not capped at the window size), counted over the kind of match the feature
measures: all his fixtures for per-fixture features, his appearances for
per-90 and per-appearance features, his appearances with DefCon data for
CBIT/CBIRT/DefCon features. k = 5. With no matches
the feature equals the prior; after 20 matches the prior's weight is 5/25 =
20%, after 95 it is 5%. So the prior fades out as his history grows, and an
established player's last-3 form is not pulled towards the newcomer average.
If a window has fewer matches than its size (e.g. only 2 matches for a last-3
average), the player value is the mean of the matches available.

- **Player prior:** average of that feature over *newcomer* matches (a
  player's first 5 PL matches in our data) in the same position and price
  band (bands: ≤£5.0m, £5.5–7.0m, £7.5–9.0m, ≥£9.5m), computed only from
  training data before the cutoff. Covers new signings from other leagues,
  youngsters and players returning after long absences. Price is the main
  quality signal for new signings.
- **Promoted team prior:** a team counts as promoted if it was not in the
  previous season's PL. Its team features blend with the average of
  promoted teams' stats over their first 6 matches in earlier seasons
  (k = 6, n = the team's total PL matches since promotion, same formula).
  Non-promoted teams at season start carry over their last 6 PL
  matches from the previous season.
- **2022/23 start:** no earlier season with xG is loaded, so all teams start
  2022/23 with the league-average prior. Those rows are still used for
  training; this only affects the first few weeks of the oldest season.
- **Transferred players:** player features follow him; team features use the
  new team; `matches_for_current_team` lets the minutes model learn that a
  new arrival's minutes are less certain.
- **Live prediction for a player with no rows at all** (e.g. a just-signed
  player): built from `players.csv` (position, price, team), so all player
  features equal the prior.

## Leakage protection

1. Features for GW N only see matches with kickoff before GW N's first
   kickoff. Priors are computed only from training data.
2. Test: change a future match's stats and assert features for earlier
   gameweeks are identical.
3. Each backtest step asserts the latest training kickoff is before that
   gameweek's first kickoff, and the train-and-predict function refuses any
   history at or after the prediction time.
4. Only **finished** fixtures enter the match table, so a gameweek in
   progress (like GW6 on 2026-10-10) never leaks half-played matches.
5. Fixture IDs restart every season, so every join keys on season too.
6. For backtest predictions 2+ weeks ahead, each player's price is the one
   known at the prediction gameweek, not the later one.

## Backtest

- **Main:** 2025/26, GW 1–38. For each GW, retrain all models from scratch
  on data before it, predict it, compare to actual points.
- **Sanity check:** 2026/27, GW2 to the last fully finished gameweek (GW5 as
  of 2026-10-10, since GW6 is in progress). The only test of the new bonus
  rules.
- **Compared:** A, B, baseline = his average points over his last 5
  fixtures in the data, counting 0 for fixtures he didn't play (like FPL's
  "form"); fewer than 5 fixtures uses what there is, none gives 0.
- **Players scored:** everyone whose team had a fixture that gameweek and
  who was in the data (including those who played 0 minutes).
- **Measures:**
  - MAE (average absolute error per player).
  - Spearman rank correlation between predicted and actual points.
  - Top-20 test: each week, the actual points scored by the 20 players each
    method ranked highest (closest to real use).
  - Calibration of P(plays), P(60+), P(clean sheet), P(DefCon): predicted
    vs actual rates in 10 buckets.
- **Horizons:** at each GW also predict GW+1…GW+5 using features frozen at
  GW, and report MAE and top-20 by horizon.
- Runtime estimate: 38 retrains × ~10 small models, about 5–10 minutes.

## Multi-gameweek prediction

From the next gameweek, player and team form features are frozen at the
latest data; only fixture features (opponent, home/away) change per week.
Output: one xP column per gameweek plus a total. No discount is applied
here — that belongs to the planner in a later part.

## Output

`predict.py` trains on all data, predicts FPL's next gameweek (`is_next` in
`gameweeks.csv`; GW7 on 2026-10-10) and:
- prints the top 30 overall and top 10 per position for the next gameweek;
- columns: player, team, position, price, opponent(s) with H/A, P(plays),
  P(60+), xG, xA, P(CS), P(DefCon), E[bonus], xP, baseline;
- saves `data/processed/xp_next.csv` (all players); with `--horizon N`
  (max 6) adds per-GW columns and a total.

`backtest.py` prints the summary table and saves
`data/processed/backtest_summary.csv` and `backtest_predictions.csv`.

## Files

| File | Purpose |
|---|---|
| `fpl/seasons.py` | Which seasons we use; season-label helpers |
| `fpl/api.py` (extended) | Adds `fetch_vaastav_csv(season, path)`; stays the only file that uses the internet |
| `fpl/storage.py` (extended) | Saves/loads vaastav raw files under `data/raw/vaastav/<season>/` |
| `pull_history.py` | One-off script to download past seasons |
| `fpl/matches.py` | Builds the combined match table (seasons merged, IDs linked, AM rows dropped) |
| `fpl/features.py` | Leakage-safe rolling features, team strength, priors for newcomers and promoted teams |
| `fpl/scoring.py` | FPL rules: points table, DefCon thresholds, actual-points rebuild, component predictions → xP |
| `fpl/models.py` | Component LightGBM models, card rates, model B, baseline |
| `fpl/backtest.py` | Walk-forward loop and measures |
| `backtest.py` | Runs the backtest, prints and saves the report |
| `predict.py` | Ranked xP table for the next gameweek(s) |
| `tests/test_scoring.py` | Rules reproduce actual points exactly on real sample rows; xP formula on hand-worked examples |
| `tests/test_features.py` | Leakage test; newcomer and promoted-team priors; transferred player uses new team's strength |
| `tests/test_matches.py` | Season merging, player-code linking, AM rows dropped |
| `tests/test_backtest.py` | Training cutoff check; metrics on hand-made numbers; double gameweeks summed |
| `tests/test_models.py` | Part predictions are valid; brand-new player; refuses future history; availability cap |
| `tests/test_predict.py` | Ranked table sums double gameweeks and handles blank gameweeks |
| `tests/conftest.py` | A small made-up league (promoted team, transferred player, newcomers) shaped like the real match table |

Test fixtures are small CSVs in `tests/` built from real rows, so tests never
need the internet.

New dependency: `lightgbm` (added to `requirements.txt`).

## Success criteria

- All tests pass, including the leakage tests.
- Backtest runs end to end on 2025/26 and reports A, B and baseline.
- A beats the baseline on MAE and on the top-20 test over 2025/26. If it
  doesn't, we stop and investigate before moving on to part 3.
- `predict.py` prints a ranked table that passes a human sanity check.
