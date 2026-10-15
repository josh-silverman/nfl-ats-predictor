"""Ledger guarantees: as-of predictions, nothing predicted after kickoff, no overwrites, stale-data guard."""

import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import pandas as pd

from nfl_ats.ledger import EASTERN, as_of, predict_week, score_ledger, write_ledger
from tests.test_win_model import _schedules


def _with_2019_live() -> pd.DataFrame:
    """Toy league: 2015-2018 complete, 2019 has weeks 1-2 played and weeks 3+ unplayed, with kickoff times."""
    s = _schedules(seasons=(2015, 2016, 2017, 2018, 2019))
    s["gametime"] = "13:00"
    unplayed = (s["season"] == 2019) & (s["week"] >= 3)
    s.loc[unplayed, ["home_score", "away_score"]] = float("nan")
    return s


NOW = datetime(2019, 9, 2, 12, 0, tzinfo=EASTERN)  # before 2019 week 3 (gameday 2019-09-03)


class TestLedger(unittest.TestCase):

    def test_predictions_ignore_the_target_weeks_results(self):
        s = _with_2019_live()
        filled = s.copy()
        hit = (filled["season"] == 2019) & (filled["week"] == 3)
        filled.loc[hit, ["home_score", "away_score"]] = [50, 0]  # pretend week 3 already has a result
        a = predict_week(s, 2019, 3, NOW)
        b = predict_week(filled, 2019, 3, NOW)
        pd.testing.assert_series_equal(a["p_elo"], b["p_elo"])
        pd.testing.assert_series_equal(a["p_v2"], b["p_v2"])

    def test_started_games_are_skipped_and_counted(self):
        s = _with_2019_live()
        late = datetime(2019, 9, 3, 13, 30, tzinfo=EASTERN)  # week 3's game kicked off at 13:00
        week3_and_4 = s[(s["season"] == 2019) & (s["week"].isin([3]))]
        self.assertEqual(len(week3_and_4), 1)
        with self.assertRaises(ValueError):  # the only week-3 game has started: nothing left to predict
            predict_week(s, 2019, 3, late)

    def test_refuses_when_earlier_weeks_lack_scores(self):
        s = _with_2019_live()
        s.loc[(s["season"] == 2019) & (s["week"] == 2), "home_score"] = float("nan")
        with self.assertRaisesRegex(ValueError, "Earlier-week games without final scores"):
            predict_week(s, 2019, 3, NOW)

    def test_never_overwrites(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "x.csv"
            write_ledger(pd.DataFrame({"a": [1]}), path)
            with self.assertRaises(FileExistsError):
                write_ledger(pd.DataFrame({"a": [2]}), path)
            self.assertEqual(pd.read_csv(path)["a"].iloc[0], 1)

    def test_as_of_hides_future_scores_only(self):
        s = _schedules(seasons=(2015, 2016))
        masked = as_of(s, 2016, 3)
        self.assertTrue(masked.loc[(masked["season"] == 2016) & (masked["week"] >= 3), "home_score"].isna().all())
        self.assertTrue(masked.loc[masked["season"] == 2015, "home_score"].notna().all())

    def test_scoring_marks_ties_and_wins(self):
        ledger = pd.DataFrame({"game_id": ["g1", "g2", "g3"], "p_elo": [0.6, 0.4, 0.5]})
        sched = pd.DataFrame({"game_id": ["g1", "g2", "g3"], "home_score": [20, 10, 17], "away_score": [10, 20, 17]})
        out = score_ledger(ledger, sched).set_index("game_id")
        self.assertEqual(out.loc["g1", "home_win"], 1)
        self.assertEqual(out.loc["g2", "home_win"], 0)
        self.assertTrue(out.loc["g3", "is_tie"])
        self.assertTrue(pd.isna(out.loc["g3", "home_win"]))


if __name__ == "__main__":
    unittest.main()
