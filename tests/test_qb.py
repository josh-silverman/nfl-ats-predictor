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
        rost = pd.DataFrame({"season": 2020, "week": [6, 7, 8], "team": "A", "gsis_id": "a1", "full_name": "Al One",
                             "status": ["ACT", "RES", "ACT"], "status_description_abbr": ["A01", "R01", "A01"]})
        f = qb_unavailable(usual, inj, rost).set_index("week")
        self.assertEqual(f.loc[6, "qb_out"], 0)
        self.assertEqual((f.loc[7, "qb_out"], f.loc[7, "reason"]), (1, "injured reserve"))
        self.assertEqual((f.loc[8, "qb_out"], f.loc[8, "reason"]), (1, "doubtful"))
        self.assertEqual(f.loc[7, "player"], "Al One")
        self.assertEqual(f["roster_week"].tolist(), [6, 7, 8])         # each game used its own week's roster

    def test_unpublished_roster_falls_back_to_the_latest_earlier_week(self):
        """Week 9's roster isn't out yet (a Wednesday lock): the week-8 reserve listing still flags the QB."""
        usual = pd.DataFrame({"season": 2026, "week": [8, 9], "game_id": ["g8", "g9"], "team": "PHI", "usual_id": "h1"})
        inj = pd.DataFrame(columns=["season", "week", "team", "gsis_id", "game_type", "report_status"])
        rost = pd.DataFrame({"season": [2025, 2026], "week": [17, 8], "team": "PHI", "gsis_id": "h1",
                             "full_name": "Jay Hurt", "status": ["ACT", "RES"], "status_description_abbr": ["A01", "R01"]})
        f = qb_unavailable(usual, inj, rost).set_index("week")
        self.assertEqual((f.loc[9, "qb_out"], f.loc[9, "reason"], f.loc[9, "roster_week"]), (1, "injured reserve", 8))
        week1 = qb_unavailable(usual.assign(week=1), inj, rost)            # no 2026 roster yet: never borrows 2025's
        self.assertTrue(week1["roster_week"].isna().all())
        self.assertEqual(week1["qb_out"].sum(), 0)


class TestStickyUsualStarter(unittest.TestCase):
    """H8: a fill-in doesn't inherit 'usual' while the starter is flagged out; sticky=False is exactly H4."""

    @staticmethod
    def _season(starts, out_weeks, ir_week=None):
        """One team, weeks 1..n. starts: starter per week (None = not played yet). out_weeks: weeks 'a1' is Out.
        ir_week: a week where the fill-in 'b2' is on injured reserve."""
        from nfl_ats.qb import sequential_usual_flags  # noqa: F401
        n = len(starts)
        games = pd.DataFrame({"season": 2026, "week": range(1, n + 1), "game_id": [f"g{w}" for w in range(1, n + 1)], "team": "BAL"})
        starters = pd.DataFrame([{"season": 2026, "week": w, "game_id": f"g{w}", "team": "BAL", "starter_id": s}
                                 for w, s in enumerate(starts, start=1) if s])
        inj = pd.DataFrame({"season": 2026.0, "week": [float(w) for w in out_weeks], "team": "BAL", "gsis_id": "a1",
                            "game_type": "REG", "report_status": "Out"})
        rost = pd.DataFrame([{"season": 2026, "week": w, "team": "BAL", "gsis_id": q, "full_name": q,
                              "status": "RES" if (q == "b2" and w == ir_week) else "ACT", "status_description_abbr": "R01" if q == "b2" else "A01"}
                             for w in range(1, n + 1) for q in ("a1", "b2")])
        return starters, games, inj, rost  # sequential_usual_flags' argument order

    def test_returning_starter_is_not_replaced_by_his_fill_in(self):
        """The 2026 BAL case: a1 out weeks 5-7 and 9, b2 fills in, then b2 goes on IR in week 10 while a1 is back."""
        from nfl_ats.qb import sequential_usual_flags
        starts = ["a1"] * 4 + ["b2"] * 3 + ["a1"] + ["b2"] + [None]
        args = self._season(starts, out_weeks=[5, 6, 7, 9], ir_week=10)
        h8 = sequential_usual_flags(*args).set_index("week")
        h4 = sequential_usual_flags(*args, sticky=False).set_index("week")
        self.assertEqual((h8.loc[10, "usual_id"], h8.loc[10, "qb_out"]), ("a1", 0))              # a1 is back: no penalty
        self.assertEqual((h4.loc[10, "usual_id"], h4.loc[10, "qb_out"]), ("b2", 1))              # H4's miss
        self.assertEqual(h8.loc[5:7, "qb_out"].tolist(), [1, 1, 1])                               # a1's absence is flagged

    def test_long_absence_stays_flagged(self):
        from nfl_ats.qb import sequential_usual_flags
        starts = ["a1"] * 3 + ["b2"] * 10
        args = self._season(starts, out_weeks=range(4, 14))
        h8 = sequential_usual_flags(*args).set_index("week")
        h4 = sequential_usual_flags(*args, sticky=False).set_index("week")
        self.assertTrue((h8.loc[4:13, "qb_out"] == 1).all())                                      # stated side effect
        self.assertEqual(h4.loc[13, "usual_id"], "b2")                                            # H4 fades to the fill-in

    def test_non_sticky_matches_h4_exactly(self):
        from nfl_ats.qb import sequential_usual_flags
        starts = ["a1"] * 4 + ["b2"] * 3 + ["a1"] + ["b2"] + [None]
        starters, games, inj, rost = self._season(starts, out_weeks=[5, 6, 7, 9], ir_week=10)
        h4 = qb_unavailable(usual_starters(starters, games), inj, rost)
        seq = sequential_usual_flags(starters, games, inj, rost, sticky=False)
        cols = ["week", "usual_id", "qb_out", "reason"]
        pd.testing.assert_frame_equal(h4[cols].reset_index(drop=True), seq[cols].reset_index(drop=True), check_dtype=False)


class TestQbValue(unittest.TestCase):
    """H7: a QB's value uses only his EARLIER games, shrunk, over a fixed window."""

    @staticmethod
    def _db(n_games=25, epa=10.0, dropbacks=40):
        return pd.DataFrame({"season": 2020, "week": range(1, n_games + 1), "game_id": [f"g{w}" for w in range(1, n_games + 1)],
                             "qb_id": "q1", "epa": epa, "dropbacks": dropbacks})

    def test_value_uses_only_earlier_games_and_shrinks(self):
        from nfl_ats.qb import qb_values
        db = self._db(n_games=2)
        t = pd.DataFrame({"season": 2020, "week": [1, 2, 3], "qb_id": "q1"})
        v = qb_values(db, t)
        self.assertEqual(v.iloc[0], 0.0)                             # week 1: no earlier game
        self.assertAlmostEqual(v.iloc[1], 10 / (40 + 200))           # week 2: only week 1 counts
        self.assertAlmostEqual(v.iloc[2], 20 / (80 + 200))           # week 3: weeks 1-2
        wrecked = db.copy()
        wrecked.loc[wrecked["week"] == 2, "epa"] = 999.0              # the week-2 game's own plays
        self.assertAlmostEqual(qb_values(wrecked, t).iloc[1], v.iloc[1])

    def test_window_keeps_the_last_20_games(self):
        from nfl_ats.qb import qb_values
        db = self._db(n_games=25)
        db.loc[db["week"] <= 5, "epa"] = -100.0                     # old bad games fall out of the window
        v = qb_values(db, pd.DataFrame({"season": [2020], "week": [26], "qb_id": ["q1"]}))
        self.assertAlmostEqual(v.iloc[0], 20 * 10 / (20 * 40 + 200))

    def test_unknown_or_missing_qb_is_zero(self):
        from nfl_ats.qb import qb_values
        t = pd.DataFrame({"season": 2020, "week": [5, 5], "qb_id": ["nobody", None]})
        self.assertEqual(qb_values(self._db(), t).tolist(), [0.0, 0.0])

    def test_scrambles_credit_the_rusher(self):
        from nfl_ats.qb import dropback_games
        pbp = pd.DataFrame({"season": 2020, "week": 1, "game_id": "g1", "season_type": "REG", "qb_dropback": [1, 1, 0],
                            "passer_player_id": ["q1", None, None], "rusher_player_id": [None, "q1", "rb"],
                            "qb_epa": [0.5, 1.0, 3.0]})
        d = dropback_games(pbp)
        self.assertEqual((d.loc[0, "qb_id"], d.loc[0, "dropbacks"], d.loc[0, "epa"]), ("q1", 2, 1.5))


if __name__ == "__main__":
    unittest.main()
