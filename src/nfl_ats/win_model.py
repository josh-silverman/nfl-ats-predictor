"""v2: P(home team wins) from Elo. See CLAUDE.md (success criteria) and design doc section 9.

Elo (src/nfl_ats/elo.py) has fixed published parameters and updates only after each game, so its
pregame numbers never use the game's own result and nothing about it is fit on test data. The only
thing fit is the logistic regression, and it is fit inside each walk-forward fold.
"""

import pandas as pd

from nfl_ats.data import regular_season
from nfl_ats.diagnostics import moneyline_prob
from nfl_ats.elo import run_elo
from nfl_ats.model import check_features, make_model

WIN_FEATURES = ["elo_diff"]
TARGET = "home_win"


def build_win_table(schedules: pd.DataFrame) -> pd.DataFrame:
    """One row per regular-season game: pregame Elo, the target, ties flagged, and the market probability.

    `schedules` should start a season before the first training season (2015) so Elo can warm up.
    home_win is 1/0 for decided games and NaN for ties and unplayed games, which are never trained on.
    """
    reg = regular_season(schedules)
    elo = run_elo(reg)
    cols = ["season", "week", "gameday", "game_id", "home_team", "away_team", "location",
            "home_score", "away_score", "spread_line", "home_moneyline", "away_moneyline"]
    table = reg[cols].merge(elo, on="game_id", how="left", validate="one_to_one")

    table["margin"] = table["home_score"] - table["away_score"]
    played = table["margin"].notna()
    table["is_tie"] = played & (table["margin"] == 0)
    table[TARGET] = (table["margin"] > 0).astype(float).where(played & ~table["is_tie"])
    table["p_market"] = moneyline_prob(table["home_moneyline"], table["away_moneyline"])
    return table


def fit_fold(table: pd.DataFrame, features: list[str], test_season: int, first_train: int = 2016, C: float = 1.0):
    """The fitted Pipeline for one fold: trained on decided games from first_train through test_season − 1."""
    graded = table[table[TARGET].notna()]
    train = graded[graded["season"].between(first_train, test_season - 1)]
    return make_model(C).fit(train[features], train[TARGET])


def win_walk_forward(table: pd.DataFrame, features: list[str], test_seasons: list[int],
                     first_train: int = 2016, C: float = 1.0) -> pd.DataFrame:
    """Out-of-sample P(home win): each test season predicted by a model fit on all earlier seasons.

    Returns one row per decided test game: season, game_id, home_win, p.
    """
    check_features(features)
    graded = table[table[TARGET].notna()]
    preds = []
    for test_season in test_seasons:
        test = graded[graded["season"] == test_season]
        model = fit_fold(table, features, test_season, first_train, C)
        preds.append(test[["season", "game_id", TARGET]].assign(p=model.predict_proba(test[features])[:, 1]))
    return pd.concat(preds, ignore_index=True)


def benchmark_predictions(table: pd.DataFrame, test_seasons: list[int], first_train: int = 2016) -> dict:
    """The three benchmarks on the same decided test games, in win_walk_forward's format.

    base_rate: each fold's training home-win rate. elo: raw Elo p_home_win (no fitting).
    market: devigged schedule moneyline (not verified as closing; source changes around 2023).
    """
    graded = table[table[TARGET].notna()]
    test = graded[graded["season"].isin(test_seasons)]
    base_rate = test["season"].map(
        {s: graded.loc[graded["season"].between(first_train, s - 1), TARGET].mean() for s in test_seasons})
    keep = ["season", "game_id", TARGET]
    return {
        "base_rate": test[keep].assign(p=base_rate.to_numpy()).reset_index(drop=True),
        "elo": test[keep].assign(p=test["p_home_win"].to_numpy()).reset_index(drop=True),
        "market": test[keep].assign(p=test["p_market"].to_numpy()).reset_index(drop=True),
    }
