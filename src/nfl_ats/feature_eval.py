"""A reusable process for deciding whether a feature (or any model change) earns its place.

Three checks, all on walk-forward test folds:
1. Paired bootstrap on the same games: log loss (primary), Brier, accuracy.
   Dispersion guard: a change that only makes the model less confident can "win" on log loss without
   adding information. Shrinking predictions toward 50% keeps their rank order, so AUC (a pure ranking
   measure) is unchanged; new information improves the ranking. (Brier resolution with equal-count bins
   is also rank-based, so it CAN'T catch this: mistake #8.)
2. Stability: the per-season log-loss change should point the same way in most seasons.
3. Permutation importance: shuffle one feature in each test fold and measure how much log loss worsens.

KEEP RULE (fixed before running, never tuned after):
  keep a change only if the paired log-loss CI is entirely below 0, it improves log loss in at least
  5 of 7 seasons, and its AUC is higher (it ranks games better, not just more cautiously).
"""

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

from nfl_ats.evaluate import TEST_SEASONS, brier_decomposition, log_loss, paired_diff
from nfl_ats.model import fold_data, make_model, walk_forward

MIN_SEASONS_IMPROVED = 5


def compare_predictions(base: pd.DataFrame, new: pd.DataFrame) -> dict:
    """Score `new` against `base` (both from walk_forward, same games). Returns a summary and per-season table."""
    if not (base["game_id"].to_numpy() == new["game_id"].to_numpy()).all():
        raise ValueError("base and new must be predictions for the same games in the same order")
    y = base["home_cover"].to_numpy()
    diff = paired_diff(y, base["p"], new["p"])

    per_season = pd.DataFrame({
        season: {"log_loss_diff": log_loss(g["home_cover"].to_numpy(), new.loc[g.index, "p"].to_numpy())
                                  - log_loss(g["home_cover"].to_numpy(), g["p"].to_numpy())}
        for season, g in base.groupby("season")
    }).T
    improved = int((per_season["log_loss_diff"] < 0).sum())

    res_base = brier_decomposition(y, base["p"])["resolution"]
    res_new = brier_decomposition(y, new["p"])["resolution"]
    auc_base, auc_new = roc_auc_score(y, base["p"]), roc_auc_score(y, new["p"])
    summary = {
        "log_loss_diff": diff.loc["log_loss", "diff"],
        "log_loss_ci": (diff.loc["log_loss", "ci_low"], diff.loc["log_loss", "ci_high"]),
        "brier_diff": diff.loc["brier", "diff"],
        "accuracy_diff": diff.loc["accuracy", "diff"],
        "accuracy_ci": (diff.loc["accuracy", "ci_low"], diff.loc["accuracy", "ci_high"]),
        "seasons_improved": f"{improved}/{len(per_season)}",
        "pred_sd": (float(base["p"].std()), float(new["p"].std())),
        "resolution": (float(res_base), float(res_new)),
        "auc": (float(auc_base), float(auc_new)),
    }
    summary["keep"] = bool(diff.loc["log_loss", "ci_high"] < 0
                           and improved >= MIN_SEASONS_IMPROVED
                           and auc_new > auc_base)
    return {"summary": summary, "per_season": per_season}


def evaluate_feature(games: pd.DataFrame, team_games: pd.DataFrame, base_features: list[str], feature: str,
                     mode: str = "drop", test_seasons: list[int] = TEST_SEASONS, **walk_kwargs) -> dict:
    """Single-change test of one feature against the current kept feature set.

    mode="drop": does REMOVING `feature` from base_features hurt? (Does an existing feature earn its place?)
    mode="add":  does ADDING `feature` to base_features help? (Should a candidate be kept?)
    For "drop", `keep` in the result means the feature should be KEPT (removing it would hurt).
    """
    if mode == "drop":
        with_f, without_f = base_features, [f for f in base_features if f != feature]
    elif mode == "add":
        with_f, without_f = base_features + [feature], base_features
    else:
        raise ValueError("mode must be 'drop' or 'add'")
    p_with = walk_forward(games, team_games, with_f, test_seasons, **walk_kwargs)
    p_without = walk_forward(games, team_games, without_f, test_seasons, **walk_kwargs)
    return compare_predictions(p_without, p_with)  # base = without, new = with the feature


def base_rate_walk_forward(games: pd.DataFrame, team_games: pd.DataFrame, test_seasons: list[int] = TEST_SEASONS,
                           first_train: int = 2016) -> pd.DataFrame:
    """The 'no features' model: predict each fold's training home-cover rate for every test game."""
    preds = []
    for test_season in test_seasons:
        train, test = fold_data(games, team_games, list(range(first_train, test_season)), test_season)
        preds.append(test[["season", "game_id", "home_cover"]].assign(p=train["home_cover"].mean()))
    return pd.concat(preds, ignore_index=True)


def permutation_importance_by_fold(games: pd.DataFrame, team_games: pd.DataFrame, features: list[str],
                                   test_seasons: list[int] = TEST_SEASONS, first_train: int = 2016,
                                   n_repeats: int = 30, seed: int = 0) -> pd.DataFrame:
    """Increase in TEST log loss when each feature is shuffled, per fold. Positive = the feature helps.

    Computed on the held-out season with the model fit on earlier seasons, so it measures out-of-sample
    usefulness, not in-sample fit.
    """
    rows = []
    for test_season in test_seasons:
        train, test = fold_data(games, team_games, list(range(first_train, test_season)), test_season)
        model = make_model().fit(train[features], train["home_cover"])
        result = permutation_importance(model, test[features], test["home_cover"], scoring="neg_log_loss",
                                        n_repeats=n_repeats, random_state=seed)
        for i, f in enumerate(features):
            rows.append({"test_season": test_season, "feature": f, "importance": result.importances_mean[i]})
    return pd.DataFrame(rows).pivot(index="test_season", columns="feature", values="importance")[features]
