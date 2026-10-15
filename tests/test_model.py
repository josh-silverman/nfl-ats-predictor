"""Walk-forward leakage tests (Phase 4 skeptic requirements). Run: python -m unittest discover tests"""

import unittest

import pandas as pd

from nfl_ats.features import (FEATURES, add_shrunk_features, adjust_for_opponent, fit_shrinkage,
                              prior_season, season_to_date)
from nfl_ats.model import check_features, fold_data, walk_forward
from tests.test_features import _toy_games

SEASONS = [2019, 2020, 2021, 2022]


def _team_games() -> pd.DataFrame:
    """The toy schedule repeated over four seasons, with results that differ each season."""
    frames = [_toy_games().assign(season=s, off_epa=lambda d, k=k: d["off_epa"] * (1 + 0.3 * k),
                                  game_id=lambda d, s=s: f"{s}_" + d["game_id"])
              for k, s in enumerate(SEASONS)]
    return pd.concat(frames, ignore_index=True)


def _games(team_games: pd.DataFrame) -> pd.DataFrame:
    """One schedule row per toy game: A at home, alternating cover results."""
    g = team_games[team_games["team"] == "A"][["season", "week", "game_id"]].copy()
    g = g.assign(gameday="2020-01-01", home_team="A", away_team="B", location="Home",
                 spread_line=1.0, home_rest=7, away_rest=7, is_push=False)
    g["margin"] = [5.0, -5.0] * (len(g) // 2) + [5.0] * (len(g) % 2)
    g["home_cover"] = (g["margin"] > g["spread_line"]).astype(float)
    return g.reset_index(drop=True)


class TestFoldIsolation(unittest.TestCase):

    def setUp(self):
        self.tg = _team_games()
        self.games = _games(self.tg)
        self.train_seasons, self.test_season = [2020, 2021], 2022
        wrecked = self.tg.copy()
        hit = wrecked["season"] == self.test_season
        wrecked.loc[hit, ["off_epa", "def_epa", "off_plays", "def_plays"]] = [99.0, -99.0, 999, 999]
        self.wrecked_tg = wrecked
        wrecked_games = self.games.copy()
        wrecked_games.loc[wrecked_games["season"] == self.test_season, "home_cover"] = 1 - wrecked_games["home_cover"]
        self.wrecked_games = wrecked_games

    def test_wrecking_test_season_leaves_shrinkage_params_unchanged(self):
        self.assertEqual(fit_shrinkage(self.tg, self.train_seasons),
                         fit_shrinkage(self.wrecked_tg, self.train_seasons))

    def test_wrecking_test_season_leaves_training_rows_unchanged(self):
        train, _ = fold_data(self.games, self.tg, self.train_seasons, self.test_season)
        train_w, _ = fold_data(self.wrecked_games, self.wrecked_tg, self.train_seasons, self.test_season)
        pd.testing.assert_frame_equal(train, train_w)

    def test_opponent_adjustment_keeps_fold_isolation(self):
        train, _ = fold_data(self.games, self.tg, self.train_seasons, self.test_season, opponent_adjust=True)
        train_w, _ = fold_data(self.wrecked_games, self.wrecked_tg, self.train_seasons, self.test_season,
                               opponent_adjust=True)
        pd.testing.assert_frame_equal(train, train_w)

    def test_opponent_adjustment_never_uses_the_game_itself(self):
        """Wreck one 2021 game with shrinkage params held fixed: only LATER rows may change.

        (Params must be fixed: refitting them on a wrecked training season legitimately changes
        every training row, just as refitting a scaler would. That's fitting, not leakage.)
        """
        params = {"off_epa": (0.5, 4), "def_epa": (0.5, 4), "pace": (0.5, 4)}

        def adjusted_features(tg):
            tf = add_shrunk_features(season_to_date(tg), prior_season(tg), params)
            adj = adjust_for_opponent(tg, tf)
            out = add_shrunk_features(season_to_date(adj), prior_season(adj), params)
            return out.set_index(["season", "team", "week"])[["off_epa", "def_epa"]].sort_index()

        wrecked = self.tg.copy()
        hit = (wrecked["season"] == 2021) & (wrecked["week"] == 3)
        wrecked.loc[hit, ["off_epa", "def_epa"]] = [99.0, -99.0]
        before, after = adjusted_features(self.tg), adjusted_features(wrecked)
        changed = (before - after).abs().sum(axis=1) > 1e-12
        changed_keys = changed[changed].index
        self.assertTrue(len(changed_keys) > 0)  # the wreck does reach later games
        for season, _, week in changed_keys:
            self.assertTrue(season > 2021 or week > 3, f"row {season} wk {week} saw its own or a later game")

class TestFeatureGuard(unittest.TestCase):

    def test_outcome_columns_are_refused(self):
        with self.assertRaisesRegex(ValueError, "margin"):
            check_features(FEATURES + ["margin"])
        check_features(FEATURES)  # the real feature list passes

    def test_walk_forward_refuses_leaky_features(self):
        tg = _team_games()
        with self.assertRaises(ValueError):
            walk_forward(_games(tg), tg, FEATURES + ["home_cover"], [2022], first_train=2020)

    def test_model_sees_exactly_the_requested_columns(self):
        tg = _team_games()
        train, _ = fold_data(_games(tg), tg, [2020, 2021], 2022)
        self.assertEqual(list(train[FEATURES].columns), FEATURES)


if __name__ == "__main__":
    unittest.main()
