"""Metrics with bootstrap CIs, and the baselines every model is compared against."""

import numpy as np
import pandas as pd

BREAKEVEN = 110 / 210  # 52.4% at -110
TEST_SEASONS = list(range(2019, 2026))


# Metrics take arrays whose LAST axis is games, so they work on one sample (n,) or on a
# stack of bootstrap resamples (n_boot, n) without a Python loop.
def accuracy(y, pick):
    return (y == pick).mean(axis=-1)


def log_loss(y, p, eps=1e-15):
    p = np.clip(p, eps, 1 - eps)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean(axis=-1)


def brier(y, p):
    return ((p - y) ** 2).mean(axis=-1)


def bootstrap_ci(metric, y, pred, n_boot=2000, seed=0, level=0.95):
    """Point estimate and percentile CI, resampling games with replacement."""
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(y), size=(n_boot, len(y)))
    stats = metric(y[idx], pred[idx])
    tail = (1 - level) / 2
    return float(metric(y, pred)), float(np.quantile(stats, tail)), float(np.quantile(stats, 1 - tail))


def baseline_picks(games: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Each baseline's pick (1 = home covers) for graded games. Pick'em games count as home for 'favorite'."""
    return pd.DataFrame({
        "always_home": np.ones(len(games)),
        "always_favorite": (games["spread_line"] >= 0).astype(float).to_numpy(),
        "coin_flip": np.random.default_rng(seed).integers(0, 2, len(games)).astype(float),
    }, index=games.index)


def baseline_table(games: pd.DataFrame, seasons: list[int] = TEST_SEASONS) -> pd.DataFrame:
    """ATS accuracy with 95% CIs for each baseline over `seasons`, pushes excluded."""
    graded = games[games["season"].isin(seasons) & games["home_cover"].notna()]
    picks = baseline_picks(graded)
    rows = []
    for name in picks.columns:
        acc, lo, hi = bootstrap_ci(accuracy, graded["home_cover"], picks[name])
        rows.append({"baseline": name, "games": len(graded), "accuracy": acc, "ci_low": lo, "ci_high": hi,
                     "edge": lo > BREAKEVEN})
    return pd.DataFrame(rows).set_index("baseline")
