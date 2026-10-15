"""The v1 model: a scikit-learn Pipeline of StandardScaler + LogisticRegression."""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nfl_ats.features import (FEATURES, add_shrunk_features, adjust_for_opponent, build_game_features,
                              fit_shrinkage, prior_season, season_to_date)


# Columns that are only known after the game (or ARE the answer). Never valid model inputs.
# The QB columns record who ACTUALLY started (post-hoc), so they'd leak late scratches and benchings.
OUTCOME_COLUMNS = {"margin", "home_cover", "is_push", "home_score", "away_score", "result", "total",
                   "overtime", "home_qb_id", "away_qb_id", "home_qb_name", "away_qb_name",
                   "home_win", "is_tie"}


def check_features(features: list[str]) -> None:
    """Refuse any feature list that includes a post-game column."""
    leaked = OUTCOME_COLUMNS.intersection(features)
    if leaked:
        raise ValueError(f"Outcome columns can't be features: {sorted(leaked)}")


def make_model(C: float = 1.0) -> Pipeline:
    """Scale each feature to mean 0 / SD 1, then fit a logistic regression.

    Both steps live in one Pipeline, so .fit() learns the scaling from training rows only.
    C is the inverse regularization strength: smaller C pulls coefficients harder toward 0.
    """
    return Pipeline([
        ("scale", StandardScaler()),
        ("logreg", LogisticRegression(C=C)),
    ])


def _team_features(team_games: pd.DataFrame, train_seasons: list[int]) -> pd.DataFrame:
    params = fit_shrinkage(team_games, train_seasons)
    return add_shrunk_features(season_to_date(team_games), prior_season(team_games), params)


def fold_data(games: pd.DataFrame, team_games: pd.DataFrame, train_seasons: list[int], test_season: int,
              opponent_adjust: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Game features for one walk-forward fold, with shrinkage fit on train_seasons only.

    opponent_adjust (experiment #6): build unadjusted strengths first, use them to adjust each
    game's EPA for its opponent, then rebuild everything from the adjusted games. Both passes
    fit shrinkage on train_seasons only.
    Returns (train, test), both with pushes dropped. Select model inputs with FEATURES only.
    """
    team_features = _team_features(team_games, train_seasons)
    if opponent_adjust:
        adjusted = adjust_for_opponent(team_games, team_features)
        team_features = _team_features(adjusted, train_seasons)
    seasons = train_seasons + [test_season]
    table = build_game_features(games[games["season"].isin(seasons)], team_features)
    table = table.dropna(subset=["home_cover"])  # pushes are counted elsewhere, never trained on
    return table[table["season"].isin(train_seasons)], table[table["season"] == test_season]


def coefficients(model: Pipeline) -> pd.Series:
    """Logistic-regression coefficients per feature, in standardized units (per 1 SD of the feature)."""
    return pd.Series(model.named_steps["logreg"].coef_[0], index=model.feature_names_in_).sort_values()


def walk_forward(games: pd.DataFrame, team_games: pd.DataFrame, features: list[str],
                 test_seasons: list[int], first_train: int = 2016, C: float = 1.0,
                 opponent_adjust: bool = False) -> pd.DataFrame:
    """Out-of-sample predictions: each test season is predicted by a model trained on all earlier seasons.

    Shrinkage, scaling, and the logistic regression are all refit inside every fold.
    Returns one row per graded test game: season, game_id, home_cover, p.
    """
    check_features(features)
    preds = []
    for test_season in test_seasons:
        train_seasons = list(range(first_train, test_season))
        train, test = fold_data(games, team_games, train_seasons, test_season, opponent_adjust)
        model = make_model(C).fit(train[features], train["home_cover"])
        preds.append(test[["season", "game_id", "home_cover"]].assign(p=model.predict_proba(test[features])[:, 1]))
    return pd.concat(preds, ignore_index=True)
