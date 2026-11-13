"""Turn the match table into model inputs ("features") without peeking at the future.

Every feature for a row uses only matches that kicked off before the row's
`asof` time (its gameweek's first kickoff). FeatureBuilder only ever sees the
history it is given, so the backtest hands it data from before each cutoff.

Players and teams with little history are blended towards a sensible default
(a "prior"): feature = (n * own_value + K * prior) / (n + K), where n is the
player's (or team's) total number of past matches. The prior fades as n grows.
"""
import numpy as np
import pandas as pd

from fpl import scoring, team_ratings
from fpl.seasons import previous_season

TEAM_WINDOW = 6
TEAM_K = 6
TEAM_STATS = ["xg_for", "xg_against", "goals_for", "goals_against"]
TEAM_FEATURES = ([f"team_{s}" for s in TEAM_STATS] + [f"opp_{s}" for s in TEAM_STATS]
                 + ["was_home", "rules_era_code"])
WINDOWS = (3, 5, 10, 38)          # last N fixtures; 38 is about a season
BASELINE_WINDOW = 5               # must be one of WINDOWS
PLAYER_K = 5
NEWCOMER_MATCHES = 5
PRICE_BINS = [-np.inf, 5.0, 7.0, 9.0, np.inf]
# Smallest denominator total a prior group needs before we trust it.
MIN_PRIOR_DENOMINATOR = {"fixture": 10, "played": 10, "played_dc": 10,
                         "minutes": 900, "minutes_dc": 900, "minutes_xg": 900}
POSITION_CODE = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}

# (feature name, numerator column, denominator column, scale):
# value over a window = sum(numerator) / sum(denominator) * scale
RATE_STATS = [
    ("minutes", "minutes", "fixture", 1),
    ("starts", "starts", "fixture", 1),
    ("played", "played", "fixture", 1),
    ("played60", "played60", "fixture", 1),
    ("xg90", "xg_known", "minutes_xg", 90),
    ("xa90", "xa_known", "minutes_xg", 90),
    ("goals90", "goals_scored", "minutes", 90),
    ("assists90", "assists", "minutes", 90),
    ("threat90", "threat", "minutes", 90),
    ("bps90", "bps", "minutes", 90),
    ("bonus90", "bonus", "minutes", 90),
    ("saves90", "saves", "minutes", 90),
    ("cbit90", "cbit", "minutes_dc", 90),
    ("cbirt90", "cbirt", "minutes_dc", 90),
    ("defcon_rate", "defcon_hits", "played_dc", 1),
    ("yellow_rate", "yellow_cards", "played", 1),
    ("red_rate", "red_cards", "played", 1),
]
DENOMINATORS = sorted({den for _, _, den, _ in RATE_STATS})
RATE_FEATURES = [f"{name}_{w}" for name, *_ in RATE_STATS for w in WINDOWS]
PLAYER_HISTORY_FEATURES = ["matches_played", "apps", "matches_this_season",
                           "matches_for_current_team"]
PLAYER_FEATURES = (RATE_FEATURES + PLAYER_HISTORY_FEATURES + ["position_code", "price"]
                   + TEAM_FEATURES)


def blend(value: pd.Series, n: pd.Series, prior: pd.Series, k: float) -> pd.Series:
    """Mix own value with a prior: (n * value + k * prior) / (n + k).

    No history (n = 0) gives the prior. History but nothing in this window
    (e.g. no minutes lately) gives NaN, which LightGBM treats as "missing".
    """
    mixed = (n * value + k * prior) / (n + k)
    return mixed.where(value.notna(), prior.where(n == 0))


def team_matches(history: pd.DataFrame) -> pd.DataFrame:
    """One row per team per fixture: goals and xG for and against."""
    keys = ["season", "gw", "fixture_id", "kickoff_time", "team", "opponent", "was_home"]
    t = history.groupby(keys, as_index=False).agg(
        goals_for=("team_goals", "first"), goals_against=("opp_goals", "first"),
        xg_for=("expected_goals", lambda x: x.sum(min_count=1)))  # unknown xG stays NaN
    against = t[["season", "fixture_id", "team", "xg_for"]].rename(
        columns={"team": "opponent", "xg_for": "xg_against"})
    t = t.merge(against, on=["season", "fixture_id", "opponent"], how="left")
    t["rules_era"] = t["season"].map(scoring.rules_era)
    return t.sort_values(["kickoff_time", "fixture_id", "team"], ignore_index=True)


def _check_asof(rows: pd.DataFrame) -> None:
    if "asof" not in rows or rows["asof"].isna().any():
        raise ValueError("every row needs an `asof` time (see matches.with_asof)")


def _team_state(tm: pd.DataFrame, spell_start: list) -> pd.DataFrame:
    """Rolling sums of each team's last TEAM_WINDOW matches within its current PL spell."""
    spell = pd.Series(spell_start, index=tm.index)
    keys = [tm["team"], spell]
    cols = {"team": tm["team"], "kickoff_time": tm["kickoff_time"], "spell_start": spell}
    for stat in TEAM_STATS:
        # Average over the matches in the window where the stat is known (xG can be missing).
        known = tm[stat].notna().astype(int)
        running = tm[stat].fillna(0).groupby(keys).cumsum()
        n_known = known.groupby(keys).cumsum()
        total = running - running.groupby(keys).shift(TEAM_WINDOW).fillna(0)
        n_window = n_known - n_known.groupby(keys).shift(TEAM_WINDOW).fillna(0)
        cols[f"{stat}__sum"] = total.where(n_window > 0)
        cols[f"{stat}__n"] = n_window
    count = tm.groupby(keys).cumcount() + 1
    cols["count"] = count
    return pd.DataFrame(cols)


def price_band(price):
    """0: up to £5.0m, 1: £5.5–7.0m, 2: £7.5–9.0m, 3: £9.5m+ (-1 if unknown)."""
    values = np.atleast_1d(np.asarray(price, dtype=float))
    bands = np.nan_to_num(pd.cut(values, PRICE_BINS, labels=False), nan=-1).astype(int)
    return bands if np.ndim(price) else int(bands[0])


def _prepare_players(h: pd.DataFrame) -> pd.DataFrame:
    """Add the helper columns RATE_STATS and the priors need."""
    h = h.copy()
    h["fixture"] = 1.0
    h["played"] = (h["minutes"] > 0).astype(float)
    h["played60"] = (h["minutes"] >= 60).astype(float)
    known = h["clearances_blocks_interceptions"].notna()
    h["cbit"] = (h["clearances_blocks_interceptions"] + h["tackles"]).fillna(0)
    h["cbirt"] = (h["clearances_blocks_interceptions"] + h["tackles"] + h["recoveries"]).fillna(0)
    h["minutes_dc"] = h["minutes"].where(known, 0)
    # xG/xA are missing before FPL published them (matches.blank_unpublished_expected):
    # count only the minutes where they are known, like DefCon above.
    xg_known = h["expected_goals"].notna() & h["expected_assists"].notna()
    h["minutes_xg"] = h["minutes"].where(xg_known, 0)
    h["xg_known"] = h["expected_goals"].where(xg_known, 0)
    h["xa_known"] = h["expected_assists"].where(xg_known, 0)
    hit = scoring.defcon_hit(h)
    h["played_dc"] = h["played"] * hit.notna()   # outfield appearances with DefCon data
    h["defcon_hits"] = hit.fillna(0) * h["played_dc"]
    h["career_index"] = h.groupby("player_code").cumcount()
    h["debut_season"] = h.groupby("player_code")["season"].transform("first")
    h["price_band"] = price_band(h["price"].to_numpy())
    return h


def _player_state(h: pd.DataFrame) -> pd.DataFrame:
    """Running window sums after each of a player's matches, for as-of lookups."""
    player = h["player_code"]
    cols = {"player_code": player, "kickoff_time": h["kickoff_time"],
            "season": h["season"], "team": h["team"]}
    sum_cols = sorted({num for _, num, _, _ in RATE_STATS} | set(DENOMINATORS) | {"total_points"})
    for col in sum_cols:
        running = h[col].groupby(player).cumsum()
        for w in WINDOWS:
            cols[f"{col}__{w}"] = running - running.groupby(player).shift(w).fillna(0)
    for den in DENOMINATORS:
        cols[f"n_{den}"] = (h[den] > 0).astype(int).groupby(player).cumsum()
    cols["season_count"] = h.groupby(["player_code", "season"]).cumcount() + 1
    new_team = h["team"] != h.groupby("player_code")["team"].shift()
    run = new_team.astype(int).groupby(player).cumsum()
    cols["team_streak"] = h.groupby([player, run]).cumcount() + 1
    return pd.DataFrame(cols)


def _pooled(frame: pd.DataFrame, num: str, den: str, scale: float) -> float:
    total = frame[den].sum()
    return frame[num].sum() / total * scale if total >= MIN_PRIOR_DENOMINATOR[den] else np.nan


def _player_priors(h: pd.DataFrame, first_season: str) -> pd.DataFrame:
    """Default for each rate stat for a player with no history, by (position, price band).

    Learned from newcomers: each player's first NEWCOMER_MATCHES fixtures, for
    players who debut after the first season in the data (in the first season
    everyone looks new). Thin groups fall back to the position, then to all
    newcomers, then to everyone.
    """
    newcomers = h[(h["career_index"] < NEWCOMER_MATCHES) & (h["debut_season"] != first_season)]
    index = pd.MultiIndex.from_product([scoring.POSITIONS, range(len(PRICE_BINS) - 1)],
                                       names=["position", "price_band"])
    table = pd.DataFrame(index=index, columns=[name for name, *_ in RATE_STATS], dtype=float)
    for position, band in index:
        same_position = newcomers[newcomers["position"] == position]
        groups = [same_position[same_position["price_band"] == band], same_position, newcomers, h]
        for name, num, den, scale in RATE_STATS:
            values = (_pooled(g, num, den, scale) for g in groups)
            table.loc[(position, band), name] = next((v for v in values if not np.isnan(v)), np.nan)
    return table


class FeatureBuilder:
    """Computes features from `history` only. Give it nothing you couldn't have known."""

    def __init__(self, history: pd.DataFrame):
        self.history = history.sort_values(["kickoff_time", "fixture_id", "player_code"],
                                           ignore_index=True)
        self.first_season = self.history["season"].min()
        self.history = _prepare_players(self.history)
        self.player_state = _player_state(self.history)
        self.player_priors = _player_priors(self.history, self.first_season)

        self.team_matches = team_matches(self.history)
        tm = self.team_matches
        self._team_seasons = set(zip(tm["team"], tm["season"]))
        spell = [self._spell_start(t, s) for t, s in zip(tm["team"], tm["season"])]
        self.team_state = _team_state(tm, spell)
        league = tm[TEAM_STATS].mean()
        promoted_start = (np.array(spell, dtype=object) > self.first_season) & \
            (self.team_state["count"] <= TEAM_WINDOW).to_numpy()
        first_matches = tm[promoted_start]
        self.team_priors = {
            "league": league,
            "promoted": first_matches[TEAM_STATS].mean() if len(first_matches) >= TEAM_WINDOW else league,
        }

    def _spell_start(self, team: str, season: str) -> str:
        """First season of the team's current unbroken PL run (within our data)."""
        while (team, previous_season(season)) in self._team_seasons:
            season = previous_season(season)
        return season

    def _team_values(self, team: pd.Series, season: pd.Series, asof: pd.Series) -> dict:
        """Blended rolling team stats as of each row's asof, as numpy arrays in row order."""
        left = pd.DataFrame({"team": team.reset_index(drop=True),
                             "season": season.reset_index(drop=True),
                             "asof": asof.reset_index(drop=True)})
        left["_row"] = np.arange(len(left))
        spells = {key: self._spell_start(*key) for key in set(zip(left["team"], left["season"]))}
        left["spell_start"] = [spells[key] for key in zip(left["team"], left["season"])]
        state = self.team_state.add_prefix("s_").rename(columns={"s_team": "team"})
        m = pd.merge_asof(left.sort_values("asof", kind="stable"), state,
                          left_on="asof", right_on="s_kickoff_time", by="team",
                          allow_exact_matches=False, direction="backward")
        m = m.sort_values("_row").reset_index(drop=True)
        # A relegated-then-promoted team starts fresh: old-spell matches don't count.
        same_spell = m["s_spell_start"] == m["spell_start"]
        n = m["s_count"].where(same_spell, 0).fillna(0)
        promoted = m["spell_start"] > self.first_season
        out = {}
        for stat in TEAM_STATS:
            value = (m[f"s_{stat}__sum"] / m[f"s_{stat}__n"]).where(same_spell)
            prior = pd.Series(np.where(promoted, self.team_priors["promoted"][stat],
                                       self.team_priors["league"][stat]), index=m.index)
            out[stat] = blend(value, n, prior, TEAM_K).to_numpy()
            out[f"prior_{stat}"] = prior.to_numpy()
        out["n"] = n.to_numpy()
        return out

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
                # An unrated team or opponent (e.g. newly promoted) falls back to the prior too.
                rows[col] = blend(raw[col].fillna(prior), n, prior, TEAM_K).to_numpy()
            rows["rating_cs"] = np.exp(-rows["rating_xg_against"])
        rows["was_home"] = rows["was_home"].astype(int)
        rows["rules_era_code"] = rows["season"].map(
            lambda s: scoring.RULES_ERA_CODE[scoring.rules_era(s)])
        return rows

    def team_features(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Add TEAM_FEATURES and the rating features to team rows (TEAM_ROW_COLUMNS + asof)."""
        _check_asof(rows)
        return self._add_team_features(rows.copy(), with_ratings=True)

    def player_features(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Add PLAYER_FEATURES and pts_last5 (the baseline) to player rows.

        rows need PLAYER_ROW_COLUMNS plus asof. Returns a copy, same index and order.
        """
        _check_asof(rows)
        left = rows.reset_index(drop=True)
        left = left.assign(_row=np.arange(len(left))).sort_values("asof", kind="stable")
        state = self.player_state.add_prefix("s_").rename(columns={"s_player_code": "player_code"})
        m = pd.merge_asof(left, state, left_on="asof", right_on="s_kickoff_time", by="player_code",
                          allow_exact_matches=False, direction="backward")
        m = m.sort_values("_row").reset_index(drop=True)
        keys = pd.MultiIndex.from_arrays([m["position"], price_band(m["price"].to_numpy())])
        priors = self.player_priors.reindex(keys).reset_index(drop=True)

        feats = {}
        for name, num, den, scale in RATE_STATS:
            n = m[f"s_n_{den}"].fillna(0)  # his total past matches of this kind
            for w in WINDOWS:
                total = m[f"s_{den}__{w}"]
                value = m[f"s_{num}__{w}"] / total.where(total > 0) * scale
                feats[f"{name}_{w}"] = blend(value, n, priors[name], PLAYER_K)
        feats["matches_played"] = m["s_n_fixture"].fillna(0)
        feats["apps"] = m["s_n_played"].fillna(0)
        feats["matches_this_season"] = m["s_season_count"].where(m["s_season"] == m["season"], 0).fillna(0)
        feats["matches_for_current_team"] = m["s_team_streak"].where(m["s_team"] == m["team"], 0).fillna(0)
        feats["position_code"] = m["position"].map(POSITION_CODE)
        feats["pts_last5"] = (m[f"s_total_points__{BASELINE_WINDOW}"]
                              / m[f"s_fixture__{BASELINE_WINDOW}"]).fillna(0)

        out = rows.copy()
        for col, values in feats.items():
            out[col] = values.to_numpy()
        return self._add_team_features(out)
