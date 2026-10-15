"""Tests for metrics and the bootstrap. Run: python -m unittest discover tests"""

import unittest

import numpy as np
from sklearn.metrics import brier_score_loss
from sklearn.metrics import log_loss as sk_log_loss

from nfl_ats.evaluate import accuracy, bootstrap_ci, brier, brier_decomposition, log_loss, paired_diff, score


class TestMetrics(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(1)
        self.y = rng.integers(0, 2, 500).astype(float)
        self.p = rng.uniform(0.05, 0.95, 500)

    def test_match_sklearn(self):
        self.assertAlmostEqual(log_loss(self.y, self.p), sk_log_loss(self.y, self.p))
        self.assertAlmostEqual(brier(self.y, self.p), brier_score_loss(self.y, self.p))

    def test_vectorized_over_resamples(self):
        stacked = np.vstack([self.y, self.y])
        np.testing.assert_allclose(accuracy(stacked, stacked), [1.0, 1.0])


class TestBootstrap(unittest.TestCase):

    def test_ci_brackets_estimate_and_is_reproducible(self):
        y = np.random.default_rng(2).integers(0, 2, 1900).astype(float)
        pick = np.ones(1900)
        est, lo, hi = bootstrap_ci(accuracy, y, pick, seed=7)
        self.assertLess(lo, est)
        self.assertLess(est, hi)
        self.assertEqual((est, lo, hi), bootstrap_ci(accuracy, y, pick, seed=7))
        # n = 1,900 at about 50%: the half-width should be about 2.25 points.
        self.assertAlmostEqual((hi - lo) / 2, 1.96 * np.sqrt(0.25 / 1900), delta=0.003)


class TestScore(unittest.TestCase):

    def test_know_nothing_model_scores_the_reference_values(self):
        y = np.random.default_rng(3).integers(0, 2, 400).astype(float)
        out = score(y, np.full(400, 0.5))
        self.assertAlmostEqual(out.loc["log_loss", "estimate"], np.log(2))
        self.assertAlmostEqual(out.loc["brier", "estimate"], 0.25)
        self.assertAlmostEqual(out.loc["accuracy", "estimate"], 1 - y.mean())  # p = 0.5 picks away


class TestPairedDiff(unittest.TestCase):

    def test_identical_models_differ_by_exactly_zero(self):
        rng = np.random.default_rng(4)
        y, p = rng.integers(0, 2, 300).astype(float), rng.uniform(0.3, 0.7, 300)
        out = paired_diff(y, p, p)
        self.assertTrue((out.abs() < 1e-12).all().all())

    def test_better_model_shows_lower_log_loss(self):
        rng = np.random.default_rng(5)
        y = rng.integers(0, 2, 1000).astype(float)
        informed = np.where(y == 1, 0.6, 0.4)
        out = paired_diff(y, np.full(1000, 0.5), informed)
        self.assertLess(out.loc["log_loss", "ci_high"], 0)


class TestBrierDecomposition(unittest.TestCase):

    def test_parts_add_up_and_extremes_behave(self):
        rng = np.random.default_rng(6)
        p = rng.uniform(0.2, 0.8, 5000)
        y = (rng.uniform(size=5000) < p).astype(float)  # perfectly calibrated by construction
        d = brier_decomposition(y, p, n_bins=20)
        self.assertAlmostEqual(d["brier"], d["reliability"] - d["resolution"] + d["uncertainty"], delta=0.003)
        self.assertLess(d["reliability"], 0.002)
        self.assertGreater(d["resolution"], 0.02)

    def test_uninformative_forecast_resolution_sits_at_the_noise_floor(self):
        """Binning random outcomes creates 'resolution' of about var / games_per_bin even with zero skill."""
        rng = np.random.default_rng(7)
        floors = []
        for seed in range(30):
            y = np.random.default_rng(seed).integers(0, 2, 1000).astype(float)
            p = 0.5 + rng.normal(0, 1e-6, 1000)  # no information, tiny jitter so bins can form
            floors.append(brier_decomposition(y, p, n_bins=10)["resolution"])
        self.assertAlmostEqual(np.mean(floors), 0.25 / 100, delta=0.0008)

if __name__ == "__main__":
    unittest.main()
