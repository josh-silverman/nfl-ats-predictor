"""v2 win model: ties counted, Elo as-of, and the walk-forward never learns from the test season."""

import unittest

import numpy as np
import pandas as pd

from nfl_ats.feature_eval import compare_predictions
from nfl_ats.model import check_features
from nfl_ats.win_model import (TARGET, WIN_FEATURES, benchmark_predictions, build_win_table, fit_fold, fit_qb_points,
                                qb_adjusted_p, qb_walk_forward, win_walk_forward)

TEAMS = ["AAA", "BBB", "CCC", "DDD"]


def _schedules(seasons=(2015, 2016, 2017, 2018), seed=0) -> pd.DataFrame:
    """A tiny round-robin league with random scores, in nflverse schedule format (regular season)."""
    rng = np.random.default_rng(seed)
    rows = []
    for season in seasons:
        week = 0
        for i, home in enumerate(TEAMS):
            for away in TEAMS[i + 1:] + TEAMS[:i]:
                week += 1
                hs, as_ = int(rng.integers(0, 40)), int(rng.integers(0, 40))
                rows.append({"season": season, "game_type": "REG", "week": week,
                             "gameday": f"{season}-09-{week:02d}", "game_id": f"{season}_{week:02d}_{away}_{home}",
                             "home_team": home, "away_team": away, "location": "Home",
                             "home_score": hs, "away_score": as_, "spread_line": 0.0,
                             "home_moneyline": -110.0, "away_moneyline": -110.0})
    df = pd.DataFrame(rows)
    df.loc[3, ["home_score", "away_score"]] = [17, 17]  # one tie in 2015
    df.loc[20, ["home_score", "away_score"]] = [20, 20]  # one tie in 2016
    return df


class TestBuildWinTable(unittest.TestCase):

    def test_ties_are_flagged_counted_and_untargeted(self):
        t = build_win_table(_schedules())
        self.assertEqual(int(t["is_tie"].sum()), 2)
        self.assertTrue(t.loc[t["is_tie"], TARGET].isna().all())
        decided = t[~t["is_tie"]]
        self.assertTrue(((decided[TARGET] == 1) == (decided["margin"] > 0)).all())

    def test_market_probability_is_half_at_even_odds(self):
        t = build_win_table(_schedules())
        self.assertTrue(np.allclose(t["p_market"], 0.5))

    def test_target_columns_are_blocked_as_features(self):
        with self.assertRaises(ValueError):
            check_features(WIN_FEATURES + [TARGET])


class TestWinWalkForward(unittest.TestCase):

    def test_wrecking_the_test_season_changes_nothing_before_it(self):
        """Flip every 2018 result: 2018's pregame Elo, and so its predictions, must not move."""
        sched = _schedules()
        wrecked = sched.copy()
        hit = wrecked["season"] == 2018
        wrecked.loc[hit, ["home_score", "away_score"]] = wrecked.loc[hit, ["away_score", "home_score"]].to_numpy()

        base = win_walk_forward(build_win_table(sched), WIN_FEATURES, [2018])
        after = win_walk_forward(build_win_table(wrecked), WIN_FEATURES, [2018])
        # Same games are graded only where both versions are decided (ties stay ties when flipped)
        merged = base.merge(after, on="game_id", suffixes=("", "_w"))
        # The first 2018 game's prediction depends only on pre-2018 games, so it must be identical
        first = merged.sort_values("game_id").iloc[0]
        self.assertAlmostEqual(first["p"], first["p_w"])

    def test_fold_weights_ignore_the_test_season(self):
        """Direct check: the 2018 fold's fitted scaler and coefficients are identical when 2018 is wrecked."""
        sched = _schedules()
        wrecked = sched.copy()
        hit = wrecked["season"] == 2018
        wrecked.loc[hit, ["home_score", "away_score"]] = [99, 0]
        m1 = fit_fold(build_win_table(sched), WIN_FEATURES, 2018)
        m2 = fit_fold(build_win_table(wrecked), WIN_FEATURES, 2018)
        np.testing.assert_array_equal(m1.named_steps["logreg"].coef_, m2.named_steps["logreg"].coef_)
        np.testing.assert_array_equal(m1.named_steps["scale"].mean_, m2.named_steps["scale"].mean_)

    def test_predictions_cover_only_decided_test_games(self):
        t = build_win_table(_schedules())
        preds = win_walk_forward(t, WIN_FEATURES, [2017, 2018])
        self.assertEqual(set(preds["season"]), {2017, 2018})
        self.assertTrue(preds[TARGET].notna().all())
        self.assertEqual(len(preds), int(t[t["season"].isin([2017, 2018])][TARGET].notna().sum()))

    def test_benchmarks_line_up_with_model_predictions(self):
        t = build_win_table(_schedules())
        preds = win_walk_forward(t, WIN_FEATURES, [2017, 2018])
        bench = benchmark_predictions(t, [2017, 2018])
        for name, b in bench.items():
            self.assertTrue((b["game_id"].to_numpy() == preds["game_id"].to_numpy()).all(), name)
        out = compare_predictions(bench["elo"], preds, target=TARGET)
        self.assertIn("keep", out["summary"])



class TestQbAdjustedElo(unittest.TestCase):
    """H4: QB_PTS is fit on training seasons only, and 0 points reproduces raw Elo exactly."""

    def _table(self):
        t = build_win_table(_schedules(seasons=(2015, 2016, 2017, 2018)))
        rng = np.random.default_rng(1)
        t["home_qb_out"] = (rng.random(len(t)) < 0.1).astype(int)
        t["away_qb_out"] = (rng.random(len(t)) < 0.1).astype(int)
        return t

    def test_zero_points_is_raw_elo(self):
        t = self._table()
        np.testing.assert_allclose(qb_adjusted_p(t, 0), t["p_home_win"])

    def test_fit_ignores_the_test_season(self):
        t = self._table()
        wrecked = t.copy()
        test = wrecked["season"] == 2018
        wrecked.loc[test, TARGET] = 1 - wrecked.loc[test, TARGET]
        wrecked.loc[test, "home_qb_out"] = 1
        self.assertEqual(fit_qb_points(t, 2018), fit_qb_points(wrecked, 2018))

    def test_walk_forward_returns_one_fit_per_season(self):
        preds, pts = qb_walk_forward(self._table(), [2017, 2018])
        self.assertEqual(sorted(pts), [2017, 2018])
        self.assertTrue(preds["p"].between(0, 1).all())


if __name__ == "__main__":
    unittest.main()
