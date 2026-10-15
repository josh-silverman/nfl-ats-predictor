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


def score(y, p, n_boot=2000, seed=0) -> pd.DataFrame:
    """ATS accuracy (pick home when p > 0.5), log loss, and Brier, each with a bootstrap 95% CI."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    pick = (p > 0.5).astype(float)
    rows = {
        "accuracy": bootstrap_ci(accuracy, y, pick, n_boot, seed),
        "log_loss": bootstrap_ci(log_loss, y, p, n_boot, seed),
        "brier": bootstrap_ci(brier, y, p, n_boot, seed),
    }
    return pd.DataFrame(rows, index=["estimate", "ci_low", "ci_high"]).T


def paired_diff(y, p_base, p_new, n_boot=2000, seed=0) -> pd.DataFrame:
    """New minus base for each metric, with a CI from resampling the SAME games for both models.

    Pairing removes the shared game-to-game noise, so the CI on the difference is much tighter
    than comparing two separate CIs. Accuracy: higher is better. Log loss, Brier: lower is better.
    """
    y = np.asarray(y, dtype=float)
    p_base, p_new = np.asarray(p_base, dtype=float), np.asarray(p_new, dtype=float)
    pick_base, pick_new = (p_base > 0.5).astype(float), (p_new > 0.5).astype(float)
    idx = np.random.default_rng(seed).integers(0, len(y), size=(n_boot, len(y)))
    diffs = {
        "accuracy": lambda i: accuracy(y[i], pick_new[i]) - accuracy(y[i], pick_base[i]),
        "log_loss": lambda i: log_loss(y[i], p_new[i]) - log_loss(y[i], p_base[i]),
        "brier": lambda i: brier(y[i], p_new[i]) - brier(y[i], p_base[i]),
    }
    everything = np.arange(len(y))
    rows = {name: (float(f(everything)), *np.quantile(f(idx), [0.025, 0.975])) for name, f in diffs.items()}
    return pd.DataFrame(rows, index=["diff", "ci_low", "ci_high"]).T


def season_table(preds: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Model metrics with CIs per test season and pooled, next to the baselines on the same games.

    `preds` comes from model.walk_forward (season, game_id, home_cover, p).
    """
    graded = preds.merge(games[["season", "game_id", "spread_line"]], on=["season", "game_id"],
                         validate="one_to_one")
    rows = []
    for season, g in list(graded.groupby("season")) + [("pooled", graded)]:
        acc, lo, hi = bootstrap_ci(accuracy, g["home_cover"], (g["p"] > 0.5).astype(float))
        picks = baseline_picks(g)
        rows.append({
            "season": season, "games": len(g),
            "accuracy": acc, "ci_low": lo, "ci_high": hi,
            "log_loss": float(log_loss(g["home_cover"].to_numpy(), g["p"].to_numpy())),
            "brier": float(brier(g["home_cover"].to_numpy(), g["p"].to_numpy())),
            **{name: float(accuracy(g["home_cover"].to_numpy(), picks[name].to_numpy())) for name in picks},
            "edge": lo > BREAKEVEN,
        })
    return pd.DataFrame(rows).set_index("season")


def reliability_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    """Bin games by predicted probability (equal-count bins) and compare mean prediction with outcome rate."""
    df = pd.DataFrame({"y": np.asarray(y, dtype=float), "p": np.asarray(p, dtype=float)})
    df["bin"] = pd.qcut(df["p"], n_bins, duplicates="drop")
    return df.groupby("bin", observed=True).agg(games=("y", "size"), mean_p=("p", "mean"), cover_rate=("y", "mean"))


def brier_decomposition(y, p, n_bins: int = 10) -> pd.Series:
    """Murphy decomposition: Brier ≈ reliability − resolution + uncertainty.

    reliability: how far predictions sit from observed rates (lower is better; 0 = perfectly calibrated)
    resolution:  how much predictions separate games from the overall rate (higher is better; 0 = no skill)
    uncertainty: base_rate * (1 - base_rate), fixed by the data and about 0.25 for a coin-flip outcome
    The identity is exact only when p is constant within each bin, so a small residual remains.
    """
    y = np.asarray(y, dtype=float)
    bins = reliability_table(y, p, n_bins)
    w = bins["games"] / bins["games"].sum()
    base = y.mean()
    return pd.Series({
        "brier": float(brier(y, np.asarray(p, dtype=float))),
        "reliability": float((w * (bins["mean_p"] - bins["cover_rate"]) ** 2).sum()),
        "resolution": float((w * (bins["cover_rate"] - base) ** 2).sum()),
        "uncertainty": float(base * (1 - base)),
    })


def calibration_slope(y, p, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Recalibration slope: fit outcome ~ a + b·logit(p) on held-out predictions. Returns (b, ci_low, ci_high).

    b ≈ 1: probabilities are well scaled. b < 1: overconfident (too extreme). b > 1: too timid.
    """
    from sklearn.linear_model import LogisticRegression
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    x = np.log(p / (1 - p)).reshape(-1, 1)

    def fit(idx):
        return LogisticRegression(C=np.inf).fit(x[idx], y[idx]).coef_[0, 0]  # unpenalized (penalty=None is deprecated)

    rng = np.random.default_rng(seed)
    boots = [fit(rng.integers(0, len(y), len(y))) for _ in range(n_boot)]
    return float(fit(np.arange(len(y)))), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))
