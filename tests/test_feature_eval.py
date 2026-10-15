"""The feature-evaluation keep rule must reject no-ops and accept a real improvement."""

import unittest

import numpy as np
import pandas as pd

from nfl_ats.feature_eval import compare_predictions


def _preds(y, p, seasons):
    return pd.DataFrame({"season": seasons, "game_id": [f"g{i}" for i in range(len(y))], "home_cover": y, "p": p})


class TestKeepRule(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(12)
        self.n = 2100
        self.y = rng.integers(0, 2, self.n).astype(float)
        self.seasons = np.repeat(np.arange(2019, 2026), self.n // 7)
        self.noise = 0.5 + rng.normal(0, 0.01, self.n)  # an uninformative model

    def test_identical_models_are_not_kept(self):
        base = _preds(self.y, self.noise, self.seasons)
        out = compare_predictions(base, base.copy())
        self.assertFalse(out["summary"]["keep"])
        self.assertEqual(out["summary"]["log_loss_diff"], 0)

    def test_informative_model_is_kept(self):
        informed = np.where(self.y == 1, 0.56, 0.44) + np.random.default_rng(13).normal(0, 0.01, self.n)
        out = compare_predictions(_preds(self.y, self.noise, self.seasons), _preds(self.y, informed, self.seasons))
        self.assertTrue(out["summary"]["keep"])

    def test_shrinking_toward_half_is_not_kept(self):
        """The dispersion guard: just being less confident must not count as an improvement."""
        rng = np.random.default_rng(14)
        overconfident = 0.5 + rng.normal(0, 0.06, self.n)  # noise, but spread out
        shrunk = 0.5 + (overconfident - 0.5) * 0.2          # same noise, pulled toward 50%
        out = compare_predictions(_preds(self.y, overconfident, self.seasons), _preds(self.y, shrunk, self.seasons))
        self.assertLess(out["summary"]["log_loss_diff"], 0)  # log loss "improves"...
        self.assertFalse(out["summary"]["keep"])             # ...but the rule rejects it

    def test_mismatched_games_raise(self):
        base = _preds(self.y, self.noise, self.seasons)
        with self.assertRaises(ValueError):
            compare_predictions(base, base.iloc[::-1].reset_index(drop=True))


if __name__ == "__main__":
    unittest.main()
