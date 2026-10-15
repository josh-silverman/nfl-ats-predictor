"""Team-level features built from play-by-play. Leakage rules live here, so read CLAUDE.md first."""

import pandas as pd

from nfl_ats.data import TEAM_MAP, load_pbp

PBP_COLS = ["game_id", "season", "week", "season_type", "posteam", "defteam",
            "play_type", "two_point_attempt", "epa", "wp"]

# Garbage time: the offense's win probability is outside [10%, 90%], so the game is
# effectively decided. A common analytics convention, fixed up front (not tuned).
GARBAGE_WP = (0.10, 0.90)


def _scrimmage_plays(pbp: pd.DataFrame, drop_garbage_time: bool = False) -> pd.DataFrame:
    """Regular-season pass and run plays with EPA. Drops kneels, spikes, penalties-only, and 2-pt tries.

    drop_garbage_time also drops plays outside GARBAGE_WP (experiment #5).
    """
    keep = (
        (pbp["season_type"] == "REG")
        & pbp["play_type"].isin(["pass", "run"])
        & (pbp["two_point_attempt"] != 1)
        & pbp["epa"].notna()
    )
    if drop_garbage_time:
        keep &= pbp["wp"].between(*GARBAGE_WP)
    plays = pbp.loc[keep, PBP_COLS].copy()
    for col in ["posteam", "defteam"]:
        plays[col] = plays[col].replace(TEAM_MAP)
    return plays


def team_game_stats(pbp: pd.DataFrame, drop_garbage_time: bool = False) -> pd.DataFrame:
    """One row per team per game: offensive EPA/play, defensive EPA/play allowed, and plays run.

    A game's own stats are in its own row here. They must be lagged (Phase 4, step 2)
    before they can be used as features for anything.
    """
    plays = _scrimmage_plays(pbp, drop_garbage_time)

    offense = (plays.groupby(["season", "week", "game_id", "posteam", "defteam"], as_index=False)
                    .agg(off_epa=("epa", "mean"), off_plays=("epa", "size"))
                    .rename(columns={"posteam": "team", "defteam": "opponent"}))

    # A team's defense allowed whatever its opponent's offense did in the same game.
    defense = offense[["game_id", "team", "off_epa", "off_plays"]].rename(
        columns={"team": "opponent", "off_epa": "def_epa", "off_plays": "def_plays"})

    return offense.merge(defense, on=["game_id", "opponent"], how="left", validate="one_to_one")


def load_team_games(seasons: list[int], drop_garbage_time: bool = False) -> pd.DataFrame:
    """team_game_stats for several seasons, loading play-by-play one season at a time to save memory."""
    frames = [team_game_stats(load_pbp([season]), drop_garbage_time) for season in seasons]
    return pd.concat(frames, ignore_index=True)


def season_to_date(team_games: pd.DataFrame) -> pd.DataFrame:
    """Each team's season-to-date stats as of kickoff, using only games it has already played.

    EPA is play-weighted (total EPA / total plays), so a 75-play game counts more than a
    45-play game. Cumulative sums are shifted one game within team-season, so a game's own
    result never appears in its own row. Week 1 (no prior games) is NaN; step 3 fills it.
    """
    tg = team_games.sort_values(["season", "team", "week"]).copy()
    tg["off_epa_total"] = tg["off_epa"] * tg["off_plays"]
    tg["def_epa_total"] = tg["def_epa"] * tg["def_plays"]

    sums = ["off_epa_total", "off_plays", "def_epa_total", "def_plays"]
    by_team = tg.groupby(["season", "team"])
    prior = by_team[sums].cumsum().groupby([tg["season"], tg["team"]]).shift(1)  # the lag

    out = tg[["season", "week", "game_id", "team"]].copy()
    out["games_played"] = by_team.cumcount()  # games before this one
    out["off_epa_to_date"] = prior["off_epa_total"] / prior["off_plays"]
    out["def_epa_to_date"] = prior["def_epa_total"] / prior["def_plays"]
    out["pace_to_date"] = prior["off_plays"] / out["games_played"]
    return out.reset_index(drop=True)


# Each feature: (to-date column, prior-season column, the per-game actual it is trying to predict).
SHRINK_STATS = {
    "off_epa": ("off_epa_to_date", "off_epa_prev", "off_epa"),
    "def_epa": ("def_epa_to_date", "def_epa_prev", "def_epa"),
    "pace": ("pace_to_date", "pace_prev", "off_plays"),
}


def prior_season(team_games: pd.DataFrame) -> pd.DataFrame:
    """Each team's full-season values from LAST season, plus last season's league mean.

    Keyed by the season they are used in (season + 1), so they are known before Week 1.
    """
    tg = team_games.assign(off_epa_total=team_games["off_epa"] * team_games["off_plays"],
                           def_epa_total=team_games["def_epa"] * team_games["def_plays"])
    totals = tg.groupby(["season", "team"]).agg(
        off_epa_total=("off_epa_total", "sum"), off_plays=("off_plays", "sum"),
        def_epa_total=("def_epa_total", "sum"), def_plays=("def_plays", "sum"),
        games=("game_id", "size"))
    prev = pd.DataFrame({
        "off_epa_prev": totals["off_epa_total"] / totals["off_plays"],
        "def_epa_prev": totals["def_epa_total"] / totals["def_plays"],
        "pace_prev": totals["off_plays"] / totals["games"],
    }).reset_index()

    league = prev.groupby("season")[["off_epa_prev", "def_epa_prev", "pace_prev"]].transform("mean")
    prev[["off_epa_league", "def_epa_league", "pace_league"]] = league.to_numpy()
    prev["season"] += 1  # last season's numbers become this season's prior
    return prev


def add_shrunk_features(to_date: pd.DataFrame, prior: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Blend each prior toward this season's to-date value as games accumulate.

    prior     = league + r * (last_season - league)   # regress last season toward the mean
    w         = k0 / (k0 + games_played)              # Week 1: w = 1, pure prior
    feature   = w * prior + (1 - w) * to_date

    params maps stat -> (r, k0). Fit them with fit_shrinkage on training seasons only.
    """
    out = to_date.merge(prior, on=["season", "team"], how="left", validate="many_to_one")
    for stat, (to_date_col, prev_col, _) in SHRINK_STATS.items():
        r, k0 = params[stat]
        league = out[f"{stat}_league"]
        regressed = league + r * (out[prev_col] - league)
        w = k0 / (k0 + out["games_played"])
        out[stat] = w * regressed + (1 - w) * out[to_date_col].fillna(0)  # fillna only matters when w = 1
    return out


R_GRID = [round(0.1 * i, 1) for i in range(11)]
K0_GRID = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48]


def fit_shrinkage(team_games: pd.DataFrame, train_seasons: list[int]) -> dict:
    """Grid-search (r, k0) per stat to minimize MSE against each game's actual value.

    Uses only rows from train_seasons (their priors come from the season before, which is
    earlier still). Call this inside each walk-forward fold, never on the full dataset.
    """
    to_date = season_to_date(team_games)
    prior = prior_season(team_games)
    actual = team_games[["season", "game_id", "team", "off_epa", "def_epa", "off_plays"]]
    train = (to_date[to_date["season"].isin(train_seasons)]
             .merge(prior, on=["season", "team"], how="inner", validate="many_to_one")
             .merge(actual, on=["season", "game_id", "team"], validate="one_to_one"))

    params = {}
    for stat, (to_date_col, prev_col, actual_col) in SHRINK_STATS.items():
        league = train[f"{stat}_league"]
        best = None
        for r in R_GRID:
            regressed = league + r * (train[prev_col] - league)
            for k0 in K0_GRID:
                w = k0 / (k0 + train["games_played"])
                pred = w * regressed + (1 - w) * train[to_date_col].fillna(0)
                mse = ((pred - train[actual_col]) ** 2).mean()
                if best is None or mse < best[0]:
                    best = (mse, r, k0)
        params[stat] = (best[1], best[2])
    return params


def adjust_for_opponent(team_games: pd.DataFrame, team_features: pd.DataFrame) -> pd.DataFrame:
    """Per-game EPA adjusted for the opponent's strength AS OF KICKOFF (experiment #6).

    off_epa_adj = off_epa - (opponent's def_epa allowed - league): gaining yards on a weak
    defense counts for less. def_epa_adj = def_epa - (opponent's off_epa - league): allowing a
    lot to a strong offense counts for less. Opponent strength comes from team_features (shrunk,
    lagged), so it uses only the opponent's games before this one. Where it's unknown (the
    first loaded season has no prior), no adjustment is made.
    """
    opp = team_features[["season", "game_id", "team", "off_epa", "def_epa", "off_epa_league", "def_epa_league"]]
    opp = opp.rename(columns={"team": "opponent", "off_epa": "opp_off", "def_epa": "opp_def",
                              "off_epa_league": "opp_off_league", "def_epa_league": "opp_def_league"})
    out = team_games.merge(opp, on=["season", "game_id", "opponent"], how="left", validate="one_to_one")
    opp_def_weakness = (out["opp_def"] - out["opp_def_league"]).fillna(0)
    opp_off_strength = (out["opp_off"] - out["opp_off_league"]).fillna(0)
    return team_games.assign(off_epa=(out["off_epa"] - opp_def_weakness).to_numpy(),
                             def_epa=(out["def_epa"] - opp_off_strength).to_numpy())


FEATURES = ["off_epa_diff", "def_epa_diff", "pace_diff", "rest_diff"]


def build_game_features(games: pd.DataFrame, team_features: pd.DataFrame) -> pd.DataFrame:
    """One row per game: the target columns plus home − away feature differences.

    `games` comes from target.add_home_cover; `team_features` from add_shrunk_features.
    Sign guide: positive off_epa_diff favors home; def_epa is EPA *allowed*, so a
    NEGATIVE def_epa_diff favors home (home defense allows less).
    """
    side_cols = ["season", "game_id", "team", "off_epa", "def_epa", "pace"]
    team = team_features[side_cols]

    out = games[["season", "week", "game_id", "gameday", "home_team", "away_team", "location",
                 "spread_line", "home_rest", "away_rest", "margin", "is_push", "home_cover"]]
    for side in ["home", "away"]:
        renamed = team.rename(columns={c: f"{side}_{c}" for c in ["team", "off_epa", "def_epa", "pace"]})
        out = out.merge(renamed, on=["season", "game_id", f"{side}_team"], how="left", validate="one_to_one")

    for stat in ["off_epa", "def_epa", "pace"]:
        out[f"{stat}_diff"] = out[f"home_{stat}"] - out[f"away_{stat}"]
    out["rest_diff"] = out["home_rest"] - out["away_rest"]  # known before kickoff

    # The left merges above leave NaN for any game without team features (e.g. unplayed games).
    # Fail loudly. Phase 8 adds an explicit as-of path for unplayed games; don't patch around this.
    missing = out.loc[out[FEATURES].isna().any(axis=1), "game_id"]
    if not missing.empty:
        raise ValueError(f"{len(missing)} games missing features, e.g. {missing.head(5).tolist()}")
    return out
