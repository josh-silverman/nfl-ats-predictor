"""Leakage tests for season-to-date features. Run: python -m unittest discover tests"""

import unittest

import numpy as np
import pandas as pd

from nfl_ats.features import (FEATURES, add_shrunk_features, build_game_features, fit_shrinkage,
                              prior_season, season_to_date)


def _toy_games() -> pd.DataFrame:
    """Two teams, three games each, with a bye in week 2 and rows deliberately out of order."""
    return pd.DataFrame({
        "season":    [2020] * 6,
        "week":      [3, 1, 4, 1, 3, 4],
        "game_id":   ["g3", "g1", "g4", "g1", "g3", "g4"],
        "team":      ["A", "A", "A", "B", "B", "B"],
        "opponent":  ["B", "B", "B", "A", "A", "A"],
        "off_epa":   [0.30, 0.10, 0.50, -0.10, -0.30, -0.50],
        "off_plays": [50, 70, 60, 60, 50, 60],
        "def_epa":   [-0.30, -0.10, -0.50, 0.10, 0.30, 0.50],
        "def_plays": [50, 60, 60, 70, 50, 60],
    })


class TestSeasonToDate(unittest.TestCase):

    def test_week1_has_no_history(self):
        std = season_to_date(_toy_games())
        week1 = std[std["week"] == 1]
        self.assertTrue(week1["off_epa_to_date"].isna().all())
        self.assertTrue((week1["games_played"] == 0).all())

    def test_values_are_play_weighted_prior_games_only(self):
        std = season_to_date(_toy_games()).set_index(["team", "week"])
        # Team A, week 3 sees only week 1.
        self.assertAlmostEqual(std.loc[("A", 3), "off_epa_to_date"], 0.10)
        # Team A, week 4 sees weeks 1 and 3: (0.10*70 + 0.30*50) / 120, not the plain mean 0.20.
        self.assertAlmostEqual(std.loc[("A", 4), "off_epa_to_date"], (0.10 * 70 + 0.30 * 50) / 120)
        self.assertAlmostEqual(std.loc[("A", 4), "pace_to_date"], 60.0)
        self.assertEqual(std.loc[("A", 4), "games_played"], 2)

    def test_changing_week_k_never_changes_weeks_up_to_k(self):
        """The core leakage test: wreck one game's stats, and only LATER rows may change."""
        games = _toy_games()
        before = season_to_date(games).set_index(["team", "week"])

        wrecked = games.copy()
        hit = (wrecked["team"] == "A") & (wrecked["week"] == 3)
        wrecked.loc[hit, ["off_epa", "def_epa", "off_plays"]] = [99.0, 99.0, 999]
        after = season_to_date(wrecked).set_index(["team", "week"])

        cols = ["off_epa_to_date", "def_epa_to_date", "pace_to_date"]
        for week in [1, 3]:  # up to and including the wrecked week: must be identical
            pd.testing.assert_series_equal(before.loc[("A", week), cols], after.loc[("A", week), cols])
        self.assertFalse(np.isclose(before.loc[("A", 4), "off_epa_to_date"], after.loc[("A", 4), "off_epa_to_date"]))
        pd.testing.assert_frame_equal(before.loc["B"], after.loc["B"])  # other teams untouched

    def test_new_season_starts_fresh(self):
        games = _toy_games()
        next_season = games.assign(season=2021)
        std = season_to_date(pd.concat([games, next_season], ignore_index=True))
        first_2021 = std[(std["season"] == 2021) & (std["week"] == 1)]
        self.assertTrue(first_2021["off_epa_to_date"].isna().all())


def _two_seasons() -> pd.DataFrame:
    """The toy games in 2020, then the same schedule in 2021 with different results."""
    later = _toy_games().assign(season=2021, off_epa=lambda d: d["off_epa"] * 2)
    return pd.concat([_toy_games(), later], ignore_index=True)


class TestShrinkage(unittest.TestCase):

    def test_prior_comes_from_last_season_only(self):
        prior = prior_season(_two_seasons()).set_index(["season", "team"])
        # 2021's prior for team A is A's full 2020 play-weighted EPA.
        expected = (0.30 * 50 + 0.10 * 70 + 0.50 * 60) / 180
        self.assertAlmostEqual(prior.loc[(2021, "A"), "off_epa_prev"], expected)
        # The first season has no prior: there is no 2019 data to carry over.
        self.assertNotIn(2020, prior.index.get_level_values("season"))

    def test_week1_is_pure_regressed_prior(self):
        games = _two_seasons()
        feat = add_shrunk_features(season_to_date(games), prior_season(games),
                                   {"off_epa": (0.5, 4), "def_epa": (0.5, 4), "pace": (0.5, 4)})
        row = feat[(feat["season"] == 2021) & (feat["team"] == "A") & (feat["week"] == 1)].iloc[0]
        expected = row["off_epa_league"] + 0.5 * (row["off_epa_prev"] - row["off_epa_league"])
        self.assertAlmostEqual(row["off_epa"], expected)

    def test_fit_ignores_seasons_outside_training(self):
        """Wrecking a non-training season must not change the fitted parameters."""
        games = _two_seasons()
        third = _toy_games().assign(season=2022)
        base = fit_shrinkage(pd.concat([games, third], ignore_index=True), train_seasons=[2021])
        wrecked = third.assign(off_epa=99.0, def_epa=-99.0, off_plays=999)
        after = fit_shrinkage(pd.concat([games, wrecked], ignore_index=True), train_seasons=[2021])
        self.assertEqual(base, after)


class TestGameFeatures(unittest.TestCase):

    def test_game_result_never_reaches_its_own_features(self):
        """End to end: wreck one game's play-by-play stats; that game's feature row must not move."""
        team_games = _two_seasons()
        games = pd.DataFrame({
            "season": [2021], "week": [3], "game_id": ["g3"], "gameday": ["2021-09-26"],
            "home_team": ["A"], "away_team": ["B"], "location": ["Home"], "spread_line": [3.0],
            "home_rest": [14], "away_rest": [7], "margin": [10.0], "is_push": [False], "home_cover": [1.0],
        })
        params = {"off_epa": (0.5, 4), "def_epa": (0.5, 4), "pace": (0.5, 4)}

        def features(tg):
            tf = add_shrunk_features(season_to_date(tg), prior_season(tg), params)
            return build_game_features(games, tf)[FEATURES]

        wrecked = team_games.copy()
        hit = (wrecked["season"] == 2021) & (wrecked["game_id"] == "g3")
        wrecked.loc[hit, ["off_epa", "def_epa", "off_plays", "def_plays"]] = [99.0, -99.0, 999, 999]
        pd.testing.assert_frame_equal(features(team_games), features(wrecked))
        self.assertEqual(features(team_games)["rest_diff"].iloc[0], 7)

    def test_game_without_team_features_raises(self):
        """A game with no team-features row must raise, not silently produce NaN features."""
        team_games = _two_seasons()
        tf = add_shrunk_features(season_to_date(team_games), prior_season(team_games),
                                 {"off_epa": (0.5, 4), "def_epa": (0.5, 4), "pace": (0.5, 4)})
        unplayed = pd.DataFrame({
            "season": [2021], "week": [5], "game_id": ["g5"], "gameday": ["2021-10-10"],
            "home_team": ["A"], "away_team": ["B"], "location": ["Home"], "spread_line": [3.0],
            "home_rest": [7], "away_rest": [7], "margin": [np.nan], "is_push": [False],
            "home_cover": [np.nan],
        })
        with self.assertRaisesRegex(ValueError, "missing features"):
            build_game_features(unplayed, tf)


if __name__ == "__main__":
    unittest.main()
