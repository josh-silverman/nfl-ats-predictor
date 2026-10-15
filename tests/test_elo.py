"""Elo must be as-of: a game's own result never changes its own pregame probability."""

import unittest

import pandas as pd

from nfl_ats.elo import ELO_PER_POINT, HOME_FIELD, run_elo, trailing_home_field, win_prob


def _games():
    return pd.DataFrame({
        "season": [2020] * 4 + [2021],
        "week": [1, 1, 2, 2, 1],
        "gameday": ["2020-09-10", "2020-09-13", "2020-09-20", "2020-09-20", "2021-09-12"],
        "game_id": ["a", "b", "c", "d", "e"],
        "home_team": ["A", "C", "A", "B", "A"],
        "away_team": ["B", "D", "C", "D", "B"],
        "home_score": [30, 10, 21, 17, None],
        "away_score": [10, 20, 24, 17, None],
        "location": ["Home", "Home", "Home", "Neutral", "Home"],
    })


class TestElo(unittest.TestCase):

    def test_first_game_is_home_field_only(self):
        out = run_elo(_games()).set_index("game_id")
        self.assertAlmostEqual(out.loc["a", "p_home_win"], win_prob(48))

    def test_neutral_site_has_no_home_field(self):
        out = run_elo(_games()).set_index("game_id")
        self.assertAlmostEqual(out.loc["d", "elo_diff"], out.loc["d", "home_elo"] - out.loc["d", "away_elo"])

    def test_own_result_never_changes_own_prediction(self):
        """Flip game c's result: c's pregame numbers must not move; later games may."""
        base = run_elo(_games()).set_index("game_id")
        flipped = _games()
        flipped.loc[flipped["game_id"] == "c", ["home_score", "away_score"]] = [45, 0]
        after = run_elo(flipped).set_index("game_id")
        for g in ["a", "b", "c", "d"]:
            self.assertAlmostEqual(base.loc[g, "p_home_win"], after.loc[g, "p_home_win"])
        self.assertNotAlmostEqual(base.loc["e", "p_home_win"], after.loc["e", "p_home_win"])

    def test_unplayed_game_gets_prediction_and_season_regression(self):
        out = run_elo(_games()).set_index("game_id")
        self.assertTrue(0 < out.loc["e", "p_home_win"] < 1)
        # A won game a by 20 and lost c by 3, so after 1/3 regression A is still above the mean
        self.assertGreater(out.loc["e", "home_elo"], 1500)


class TestTrailingHomeField(unittest.TestCase):

    def _league(self):
        rows = []
        for season, margin in [(2015, 4), (2016, 2), (2017, 0), (2018, 6)]:
            rows += [{"season": season, "home_score": 20 + margin, "away_score": 20, "location": "Home"},
                     {"season": season, "home_score": 99, "away_score": 0, "location": "Neutral"}]  # ignored
        return pd.DataFrame(rows)

    def test_uses_only_earlier_seasons_and_ignores_neutral_sites(self):
        hfa = trailing_home_field(self._league(), window=3)
        self.assertEqual(hfa[2015], HOME_FIELD)                       # nothing earlier
        self.assertAlmostEqual(hfa[2016], 4 * ELO_PER_POINT)          # 2015 only
        self.assertAlmostEqual(hfa[2018], (4 + 2 + 0) / 3 * ELO_PER_POINT)  # 2015-2017, not 2018 itself

    def test_season_dict_is_applied(self):
        g = _games()
        out = run_elo(g, home_field={2020: 0.0, 2021: 0.0}).set_index("game_id")
        self.assertAlmostEqual(out.loc["a", "p_home_win"], 0.5)


if __name__ == "__main__":
    unittest.main()
