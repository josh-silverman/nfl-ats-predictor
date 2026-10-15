"""The leak demo must really leak, or Step 1's comparison means nothing."""

import unittest

import numpy as np

import pandas as pd

from nfl_ats.diagnostics import (accuracy_mde_grid, leaky_season_to_date, paired_mse_diff, paired_power,
                                 residual_correlations)
from nfl_ats.features import FEATURES
from nfl_ats.features import season_to_date
from tests.test_features import _toy_games


class TestLeakyDemo(unittest.TestCase):

    def test_leaky_version_includes_the_game_itself(self):
        leaky = leaky_season_to_date(_toy_games()).set_index(["team", "week"])
        honest = season_to_date(_toy_games()).set_index(["team", "week"])
        self.assertAlmostEqual(leaky.loc[("A", 1), "off_epa_to_date"], 0.10)  # week 1 sees its own game
        self.assertTrue(honest["off_epa_to_date"].isna().loc[("A", 1)])        # honest version sees nothing


class TestPairedPower(unittest.TestCase):

    def test_identical_models_have_zero_se(self):
        rng = np.random.default_rng(8)
        y, p = rng.integers(0, 2, 500).astype(float), rng.uniform(0.4, 0.6, 500)
        out = paired_power(y, p, p)
        self.assertEqual(out.loc["accuracy", "se"], 0)
        self.assertEqual(out.loc["accuracy", "disagreement_rate"], 0)

    def test_accuracy_se_matches_disagreement_formula(self):
        """With random outcomes, the per-game accuracy difference is ±1 on disagreements: SE ≈ sqrt(d/n)."""
        rng = np.random.default_rng(9)
        n = 20000
        y = rng.integers(0, 2, n).astype(float)
        base = np.full(n, 0.6)
        new = np.where(rng.uniform(size=n) < 0.2, 0.4, 0.6)  # disagree on about 20% of games
        out = paired_power(y, base, new)
        self.assertAlmostEqual(out.loc["accuracy", "se"], np.sqrt(0.2 / n), delta=0.0003)
        self.assertAlmostEqual(accuracy_mde_grid(n)[0.2], 2.8 * np.sqrt(0.2 / n))


class TestSignalTools(unittest.TestCase):

    def test_paired_mse_diff_sign_and_identity(self):
        rng = np.random.default_rng(10)
        actual = rng.normal(0, 10, 2000)
        good, noisy = actual + rng.normal(0, 1, 2000), actual + rng.normal(0, 5, 2000)
        out = paired_mse_diff(actual, noisy, good)
        self.assertLess(out["ci_high"], 0)  # the better predictor lowers MSE
        self.assertEqual(paired_mse_diff(actual, good, good)["diff"], 0)

    def test_residual_correlation_finds_planted_signal(self):
        rng = np.random.default_rng(11)
        n = 3000
        t = pd.DataFrame({f: rng.normal(size=n) for f in FEATURES})
        t["season"] = rng.choice([2020, 2021], n)
        t["spread_line"] = rng.normal(0, 6, n)
        t["margin"] = t["spread_line"] + 3 * t["off_epa_diff"] + rng.normal(0, 12, n)  # only off_epa matters
        rc = residual_correlations(t).query("season == 'all'").set_index("feature")
        self.assertGreater(rc.loc["off_epa_diff", "ci_low"], 0)
        self.assertLess(rc.loc["rest_diff", "ci_low"], 0)
        self.assertGreater(rc.loc["rest_diff", "ci_high"], 0)


if __name__ == "__main__":
    unittest.main()
