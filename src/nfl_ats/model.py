"""The v1 model: a scikit-learn Pipeline of StandardScaler + LogisticRegression."""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nfl_ats.features import (FEATURES, add_shrunk_features, build_game_features, fit_shrinkage,
                              prior_season, season_to_date)


def make_model(C: float = 1.0) -> Pipeline:
    """Scale each feature to mean 0 / SD 1, then fit a logistic regression.

    Both steps live in one Pipeline, so .fit() learns the scaling from training rows only.
    C is the inverse regularization strength: smaller C pulls coefficients harder toward 0.
    """
    return Pipeline([
        ("scale", StandardScaler()),
        ("logreg", LogisticRegression(C=C)),
    ])


def fold_data(games: pd.DataFrame, team_games: pd.DataFrame,
              train_seasons: list[int], test_season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Game features for one walk-forward fold, with shrinkage fit on train_seasons only.

    Returns (train, test), both with pushes dropped. Select model inputs with FEATURES only.
    """
    params = fit_shrinkage(team_games, train_seasons)
    team_features = add_shrunk_features(season_to_date(team_games), prior_season(team_games), params)
    seasons = train_seasons + [test_season]
    table = build_game_features(games[games["season"].isin(seasons)], team_features)
    table = table.dropna(subset=["home_cover"])  # pushes are counted elsewhere, never trained on
    return table[table["season"].isin(train_seasons)], table[table["season"] == test_season]


def coefficients(model: Pipeline) -> pd.Series:
    """Logistic-regression coefficients per feature, in standardized units (per 1 SD of the feature)."""
    return pd.Series(model.named_steps["logreg"].coef_[0], index=model.feature_names_in_).sort_values()


def walk_forward(games: pd.DataFrame, team_games: pd.DataFrame, features: list[str],
                 test_seasons: list[int], first_train: int = 2016, C: float = 1.0) -> pd.DataFrame:
    """Out-of-sample predictions: each test season is predicted by a model trained on all earlier seasons.

    Shrinkage, scaling, and the logistic regression are all refit inside every fold.
    Returns one row per graded test game: season, game_id, home_cover, p.
    """
    preds = []
    for test_season in test_seasons:
        train_seasons = list(range(first_train, test_season))
        train, test = fold_data(games, team_games, train_seasons, test_season)
        model = make_model(C).fit(train[features], train["home_cover"])
        preds.append(test[["season", "game_id", "home_cover"]].assign(p=model.predict_proba(test[features])[:, 1]))
    return pd.concat(preds, ignore_index=True)
