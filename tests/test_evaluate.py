"""Tests for metrics and the bootstrap. Run: python -m unittest discover tests"""

import unittest

import numpy as np
from sklearn.metrics import brier_score_loss
from sklearn.metrics import log_loss as sk_log_loss

from nfl_ats.evaluate import accuracy, bootstrap_ci, brier, log_loss


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


if __name__ == "__main__":
    unittest.main()
