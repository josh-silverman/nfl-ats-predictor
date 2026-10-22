"""H2 QB feature: the expected starter comes from the PREVIOUS game, and only Out/Doubtful counts."""

import unittest

import pandas as pd

from nfl_ats.qb import add_qb_feature, expected_starters, game_starters, qb_out_flags, qb_unavailable, usual_starters


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



class TestUsualStarter(unittest.TestCase):
    """H4: the No. 1 QB comes from EARLIER games only; Out/Doubtful or the reserve list flags him."""

    def _starters(self):
        # a1 starts weeks 1-5 of 2020, gets hurt, backup a2 starts weeks 6-7
        return pd.DataFrame({"season": 2020, "week": range(1, 8), "game_id": [f"g{w}" for w in range(1, 8)],
                             "team": "A", "starter_id": ["a1"] * 5 + ["a2"] * 2})

    def _games(self, weeks):
        return pd.DataFrame({"season": 2020, "week": weeks, "game_id": [f"g{w}" for w in weeks], "team": "A"})

    def test_usual_starter_survives_a_few_missed_games(self):
        u = usual_starters(self._starters(), self._games([1, 6, 8])).set_index("week")
        self.assertTrue(pd.isna(u.loc[1, "usual_id"]))   # no earlier game
        self.assertEqual(u.loc[6, "usual_id"], "a1")
        self.assertEqual(u.loc[8, "usual_id"], "a1")     # unplayed game: 5 of the last 7 starts are a1's

    def test_current_game_never_counts(self):
        wrecked = self._starters()
        wrecked.loc[wrecked["week"] == 6, "starter_id"] = "zzz"
        a = usual_starters(self._starters(), self._games([6]))
        b = usual_starters(wrecked, self._games([6]))
        self.assertEqual(a["usual_id"].iloc[0], b["usual_id"].iloc[0])

    def test_window_decays_to_the_backup(self):
        s = pd.DataFrame({"season": 2020, "week": range(1, 13), "game_id": [f"g{w}" for w in range(1, 13)],
                          "team": "A", "starter_id": ["a1"] * 3 + ["a2"] * 9})
        self.assertEqual(usual_starters(s, self._games([13]))["usual_id"].iloc[0], "a2")

    def test_reserve_and_out_flag_but_questionable_does_not(self):
        usual = pd.DataFrame({"season": 2020, "week": [6, 7, 8], "game_id": ["g6", "g7", "g8"], "team": "A",
                              "usual_id": "a1"})
        inj = pd.DataFrame({"season": [2020.0, 2020.0], "week": [6.0, 8.0], "team": ["A", "A"], "gsis_id": ["a1", "a1"],
                            "game_type": ["REG", "REG"], "report_status": ["Questionable", "Doubtful"]})
        rost = pd.DataFrame({"season": 2020, "week": [6, 7], "team": "A", "gsis_id": "a1", "full_name": "Al One",
                             "status": ["ACT", "RES"], "status_description_abbr": ["A01", "R01"]})
        f = qb_unavailable(usual, inj, rost).set_index("week")
        self.assertEqual(f.loc[6, "qb_out"], 0)
        self.assertEqual((f.loc[7, "qb_out"], f.loc[7, "reason"]), (1, "injured reserve"))
        self.assertEqual((f.loc[8, "qb_out"], f.loc[8, "reason"]), (1, "doubtful"))
        self.assertEqual(f.loc[7, "player"], "Al One")


if __name__ == "__main__":
    unittest.main()
