"""H2 QB feature: the expected starter comes from the PREVIOUS game, and only Out/Doubtful counts."""

import unittest

import pandas as pd

from nfl_ats.qb import add_qb_feature, expected_starters, game_starters, qb_out_flags


def _pbp():
    rows = []
    # Team A: QB "a1" throws most in weeks 1-2, backup "a2" throws most in week 3 (a1 hurt mid-week-2 plan)
    for week, passers in [(1, ["a1"] * 30 + ["a2"] * 2), (2, ["a1"] * 25 + ["a2"] * 10), (3, ["a2"] * 28)]:
        rows += [{"season": 2020, "week": week, "game_id": f"g{week}", "posteam": "A", "season_type": "REG",
                  "pass_attempt": 1, "passer_player_id": p} for p in passers]
    rows.append({"season": 2020, "week": 3, "game_id": "g3", "posteam": "A", "season_type": "REG",
                 "pass_attempt": 0, "passer_player_id": None})
    return pd.DataFrame(rows)


class TestQbFeature(unittest.TestCase):

    def test_starter_is_most_attempts(self):
        s = game_starters(_pbp()).set_index("week")
        self.assertEqual(s.loc[1, "starter_id"], "a1")
        self.assertEqual(s.loc[3, "starter_id"], "a2")

    def test_expected_starter_is_previous_games_starter(self):
        e = expected_starters(game_starters(_pbp())).set_index("week")
        self.assertTrue(pd.isna(e.loc[1, "expected_id"]))   # week 1: no prior game
        self.assertEqual(e.loc[3, "expected_id"], "a1")     # NOT a2, who actually started week 3

    def test_only_out_or_doubtful_flags(self):
        e = expected_starters(game_starters(_pbp()))
        inj = pd.DataFrame({"season": [2020.0, 2020.0], "week": [2.0, 3.0], "team": ["A", "A"],
                            "gsis_id": ["a1", "a1"], "game_type": ["REG", "REG"],
                            "report_status": ["Questionable", "Out"]})
        f = qb_out_flags(e, inj).set_index("week")
        self.assertEqual(f.loc[2, "qb_out"], 0)  # Questionable = available
        self.assertEqual(f.loc[3, "qb_out"], 1)

    def test_diff_is_home_minus_away(self):
        table = pd.DataFrame({"game_id": ["g3"], "home_team": ["A"], "away_team": ["B"]})
        flags = pd.DataFrame({"game_id": ["g3", "g3"], "team": ["A", "B"], "qb_out": [1, 0]})
        out = add_qb_feature(table, flags)
        self.assertEqual(out["qb_out_diff"].iloc[0], 1)


if __name__ == "__main__":
    unittest.main()
