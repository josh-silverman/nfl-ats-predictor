"""Diagnostics for 'why isn't the model improving?'. Nothing here is used by the production pipeline."""

import numpy as np
import pandas as pd

from nfl_ats.data import load_schedules, regular_season
from nfl_ats.features import (FEATURES, add_shrunk_features, build_game_features, fit_shrinkage,
                              prior_season)
from nfl_ats.model import fold_data, make_model
from nfl_ats.target import add_home_cover


def lookup_games(games: pd.DataFrame, game_ids: list[str]) -> pd.DataFrame:
    """The columns needed to check the spread sign and target by hand against a real box score."""
    cols = ["game_id", "gameday", "away_team", "away_score", "home_team", "home_score",
            "spread_line", "margin", "is_push", "home_cover"]
    out = games.set_index("game_id").loc[game_ids].reset_index()[cols]
    out["favorite"] = np.where(out["spread_line"] > 0, out["home_team"],
                               np.where(out["spread_line"] < 0, out["away_team"], "pick'em"))
    return out


def leaky_season_to_date(team_games: pd.DataFrame) -> pd.DataFrame:
    """DELIBERATELY WRONG: season-to-date stats that INCLUDE the game itself (no shift(1)).

    Same output shape as features.season_to_date, so it can drop into the same pipeline and
    show what leakage does to the backtest. Never use outside diagnostics.
    """
    tg = team_games.sort_values(["season", "team", "week"]).copy()
    tg["off_epa_total"] = tg["off_epa"] * tg["off_plays"]
    tg["def_epa_total"] = tg["def_epa"] * tg["def_plays"]
    sums = ["off_epa_total", "off_plays", "def_epa_total", "def_plays"]
    by_team = tg.groupby(["season", "team"])
    running = by_team[sums].cumsum()  # <- the bug: no .shift(1), so this game's result is included

    out = tg[["season", "week", "game_id", "team"]].copy()
    out["games_played"] = by_team.cumcount() + 1
    out["off_epa_to_date"] = running["off_epa_total"] / running["off_plays"]
    out["def_epa_to_date"] = running["def_epa_total"] / running["def_plays"]
    out["pace_to_date"] = running["off_plays"] / out["games_played"]
    return out.reset_index(drop=True)


def leaky_walk_forward(games: pd.DataFrame, team_games: pd.DataFrame, test_seasons: list[int],
                       first_train: int = 2016) -> pd.DataFrame:
    """walk_forward, but with leaky_season_to_date. Shrinkage params are reused from the honest fit."""
    preds = []
    for test_season in test_seasons:
        train_seasons = list(range(first_train, test_season))
        params = fit_shrinkage(team_games, train_seasons)
        tf = add_shrunk_features(leaky_season_to_date(team_games), prior_season(team_games), params)
        table = build_game_features(games[games["season"].isin(train_seasons + [test_season])], tf)
        table = table.dropna(subset=["home_cover"])
        train, test = table[table["season"].isin(train_seasons)], table[table["season"] == test_season]
        model = make_model().fit(train[FEATURES], train["home_cover"])
        preds.append(test[["season", "game_id", "home_cover"]].assign(p=model.predict_proba(test[FEATURES])[:, 1]))
    return pd.concat(preds, ignore_index=True)


def row_counts(team_games: pd.DataFrame, train_seasons: list[int], test_season: int) -> pd.Series:
    """Rows surviving each pipeline step, for one fold. Every drop should be explainable."""
    raw = load_schedules()
    reg = regular_season(raw)
    games = add_home_cover(reg)
    in_fold = games["season"].isin(train_seasons + [test_season])
    train, test = fold_data(games, team_games, train_seasons, test_season)
    return pd.Series({
        "schedule rows (incl. playoffs)": len(raw),
        "regular season": len(reg),
        "with target (incl. pushes)": len(games),
        "team-games (2 per game, incl. 2015)": len(team_games),
        "fold games (train + test seasons)": int(in_fold.sum()),
        "fold pushes dropped": int(games.loc[in_fold, "is_push"].sum()),
        "fold train rows": len(train),
        "fold test rows": len(test),
    })


def feature_health(table: pd.DataFrame) -> pd.DataFrame:
    """Per feature: NaN share, distinct values, SD, and the Week 1 mean and SD (Week 1 = pure prior)."""
    week1 = table[table["week"] == 1]
    return pd.DataFrame({
        "nan_share": table[FEATURES].isna().mean(),
        "n_unique": table[FEATURES].nunique(),
        "sd": table[FEATURES].std(),
        "week1_nan": week1[FEATURES].isna().sum(),
        "week1_sd": week1[FEATURES].std(),
    })


def scaler_fit_check(games: pd.DataFrame, team_games: pd.DataFrame, test_seasons: list[int],
                     first_train: int = 2016) -> pd.DataFrame:
    """Per fold: max gap between the fitted scaler's means and the TRAIN rows' means vs. the TEST rows' means.

    A scaler fit on training rows only matches the train means exactly (gap 0) and the test means only roughly.
    """
    rows = []
    for test_season in test_seasons:
        train_seasons = list(range(first_train, test_season))
        train, test = fold_data(games, team_games, train_seasons, test_season)
        scaler = make_model().fit(train[FEATURES], train["home_cover"]).named_steps["scale"]
        rows.append({
            "test_season": test_season,
            "gap_vs_train_mean": float(np.abs(scaler.mean_ - train[FEATURES].mean().to_numpy()).max()),
            "gap_vs_test_mean": float(np.abs(scaler.mean_ - test[FEATURES].mean().to_numpy()).max()),
        })
    return pd.DataFrame(rows).set_index("test_season")


Z_80_POWER = 1.96 + 0.84  # two-sided 5% test with 80% power


def paired_power(y, p_base, p_new) -> pd.DataFrame:
    """How big a true difference this paired comparison could reliably detect (80% power, 5% test).

    Accuracy: games where both models pick the same side add nothing, so the standard error depends
    on the disagreement rate d: SE ≈ sqrt(d / n). Log loss: SE of the per-game log-loss differences.
    MDE (minimum detectable effect) = 2.8 × SE.
    """
    y, p_base, p_new = (np.asarray(a, dtype=float) for a in (y, p_base, p_new))
    n = len(y)
    pick_base, pick_new = p_base > 0.5, p_new > 0.5
    acc_diff = (pick_new == y).astype(float) - (pick_base == y).astype(float)
    eps = 1e-15
    ll = lambda p: -(y * np.log(np.clip(p, eps, 1)) + (1 - y) * np.log(np.clip(1 - p, eps, 1)))
    ll_diff = ll(p_new) - ll(p_base)
    rows = {
        "accuracy": {"observed_diff": acc_diff.mean(), "se": acc_diff.std(ddof=1) / np.sqrt(n),
                     "disagreement_rate": (pick_base != pick_new).mean()},
        "log_loss": {"observed_diff": ll_diff.mean(), "se": ll_diff.std(ddof=1) / np.sqrt(n),
                     "disagreement_rate": np.nan},
    }
    out = pd.DataFrame(rows).T
    out["mde_80"] = Z_80_POWER * out["se"]
    return out


def accuracy_mde_grid(n: int, disagreement_rates=(0.05, 0.1, 0.2, 0.3, 0.5)) -> pd.Series:
    """Accuracy MDE for a paired comparison of n games at several disagreement rates: 2.8·sqrt(d/n)."""
    return pd.Series({d: Z_80_POWER * np.sqrt(d / n) for d in disagreement_rates}, name="accuracy_mde_80")


def _fold_tables(games: pd.DataFrame, team_games: pd.DataFrame, test_seasons: list[int], first_train: int = 2016):
    """Yield (train, test) game tables per fold, KEEPING pushes (margin targets don't exclude them)."""
    from nfl_ats.model import _team_features  # shrinkage refit on the fold's training seasons
    for test_season in test_seasons:
        train_seasons = list(range(first_train, test_season))
        tf = _team_features(team_games, train_seasons)
        table = build_game_features(games[games["season"].isin(train_seasons + [test_season])], tf)
        table = table[table["margin"].notna()]
        yield table[table["season"].isin(train_seasons)], table[table["season"] == test_season]


def margin_walk_forward(games: pd.DataFrame, team_games: pd.DataFrame, test_seasons: list[int]) -> pd.DataFrame:
    """Out-of-sample margin predictions: spread taken as-is, a fitted spread-only model, and spread + FEATURES.

    `spread_as_is` uses margin = spread_line with nothing fit. The two LinearRegressions are refit per fold.
    """
    from sklearn.linear_model import LinearRegression
    out = []
    for train, test in _fold_tables(games, team_games, test_seasons):
        spread_only = LinearRegression().fit(train[["spread_line"]], train["margin"])
        with_feats = LinearRegression().fit(train[["spread_line"] + FEATURES], train["margin"])
        out.append(test[["season", "week", "game_id", "margin", "spread_line"] + FEATURES].assign(
            spread_as_is=test["spread_line"],
            spread_only=spread_only.predict(test[["spread_line"]]),
            spread_plus_features=with_feats.predict(test[["spread_line"] + FEATURES]),
        ))
    return pd.concat(out, ignore_index=True)


def paired_mse_diff(actual, pred_base, pred_new, n_boot: int = 2000, seed: int = 0) -> pd.Series:
    """MSE of each, the percent change, and a paired-bootstrap CI on (new − base) MSE."""
    actual, pred_base, pred_new = (np.asarray(a, dtype=float) for a in (actual, pred_base, pred_new))
    d = (pred_new - actual) ** 2 - (pred_base - actual) ** 2
    idx = np.random.default_rng(seed).integers(0, len(d), size=(n_boot, len(d)))
    boot = d[idx].mean(axis=1)
    base_mse = float(((pred_base - actual) ** 2).mean())
    return pd.Series({"mse_base": base_mse, "mse_new": float(((pred_new - actual) ** 2).mean()),
                      "diff": float(d.mean()), "ci_low": float(np.quantile(boot, 0.025)),
                      "ci_high": float(np.quantile(boot, 0.975)), "pct": float(d.mean() / base_mse)})


def residual_correlations(table: pd.DataFrame, by: str = "season") -> pd.DataFrame:
    """Correlation of each feature with the spread residual (margin − spread_line), per group, with Fisher 95% CIs.

    A feature with information the line misses should correlate with the residual in most groups, not one.
    """
    rows = []
    groups = list(table.groupby(by)) + [("all", table)]
    for key, g in groups:
        resid = g["margin"] - g["spread_line"]
        for f in FEATURES:
            r, n = g[f].corr(resid), len(g)
            half = 1.96 / np.sqrt(n - 3)
            rows.append({by: key, "feature": f, "n": n, "r": r,
                         "ci_low": np.tanh(np.arctanh(r) - half), "ci_high": np.tanh(np.arctanh(r) + half)})
    return pd.DataFrame(rows)
