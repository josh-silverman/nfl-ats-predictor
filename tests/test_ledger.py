"""Ledger guarantees: as-of predictions, nothing predicted after kickoff, no overwrites, stale-data guard."""

import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import pandas as pd

import numpy as np

from nfl_ats.ledger import (EASTERN, MODEL_COLUMNS, as_of, attach_qb, check_final_injury_report, live_report, load_qb_ledger,
                            market_move_summary, market_moves,
                            predict_week, score_ledger, write_ledger)
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


class TestLiveReport(unittest.TestCase):

    def _scored(self, n=400, bias=0.0, seed=0):
        rng = np.random.default_rng(seed)
        p = rng.uniform(0.2, 0.8, n)
        won = (rng.uniform(size=n) < p).astype(float)
        return pd.DataFrame({"week": np.repeat(np.arange(1, 9), n // 8), "is_tie": False, "home_win": won,
                             "p_elo": np.clip(p + bias, 0.01, 0.99), "p_v2": p, "p_market_at_pull": p})

    def test_calibrated_model_passes_and_biased_one_fails(self):
        rep = live_report(self._scored(bias=0.10), n_sims=200)
        self.assertTrue(rep["models"]["v2 (Elo + logistic)"]["consistent_with_calibration"])
        self.assertFalse(rep["models"][MODEL_COLUMNS["p_elo"]]["consistent_with_calibration"])

    def test_too_few_games_returns_note(self):
        rep = live_report(self._scored(n=8))
        self.assertIn("note", rep)



class TestQbLedger(unittest.TestCase):
    """H4 live columns: the adjustment moves only flagged games, the latest pre-kickoff QB lock counts."""

    def test_qb_columns_move_only_flagged_games(self):
        s = _with_2019_live()
        base = predict_week(s, 2019, 3, NOW)
        g = base.iloc[0]
        flags = pd.DataFrame({"game_id": [g.game_id, g.game_id], "team": [g.home_team, g.away_team],
                              "player": ["Home QB", "Away QB"], "qb_out": [1, 0]})
        out = predict_week(s, 2019, 3, NOW, qb=(flags, 80))
        self.assertLess(out.loc[0, "p_elo_qb"], out.loc[0, "p_elo"])          # home QB out → home chance drops
        self.assertAlmostEqual(out.loc[0, "elo_spread_qb"], round(out.loc[0, "elo_spread"] - 80 / 25, 1), places=1)
        np.testing.assert_allclose(out.loc[1:, "p_elo_qb"], out.loc[1:, "p_elo"], atol=1e-4)  # unflagged: same as Elo
        self.assertEqual(out.loc[0, "home_qb"], "Home QB")

    def test_latest_qb_lock_wins_and_raw_elo_is_untouched(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "qb").mkdir()
            row = {"game_id": "g1", "home_qb": "A", "away_qb": "B", "home_qb_out": 0, "away_qb_out": 0, "qb_pts": 80,
                   "elo_spread_qb": 2.0, "p_elo": 0.6, "p_elo_qb": 0.6, "pulled_at_et": "2026-10-29 15:00:00"}
            pd.DataFrame([row]).to_csv(d / "2026_wk08_20261029T150000.csv", index=False)
            pd.DataFrame([{**row, "home_qb_out": 1, "p_elo_qb": 0.49, "pulled_at_et": "2026-10-31 11:00:00"}]).to_csv(
                d / "qb" / "2026_wk08_qb_20261031T110000.csv", index=False)
            q = load_qb_ledger(2026, d)
            self.assertEqual(q.loc[0, "p_elo_qb"], 0.49)
            live = pd.DataFrame([{"game_id": "g1", "p_elo": 0.6, "p_elo_qb": 0.6}])
            merged = attach_qb(live, 2026, d)
            self.assertEqual((merged.loc[0, "p_elo"], merged.loc[0, "p_elo_qb"]), (0.6, 0.49))

    def test_saturday_lock_refuses_without_final_statuses(self):
        inj = pd.DataFrame({"season": 2026, "week": 8, "team": ["A", "B", "C"], "report_status": [None, None, "Out"]})
        with self.assertRaises(ValueError):
            check_final_injury_report(inj, 2026, 8, ["A", "B", "C", "D"])
        check_final_injury_report(inj.assign(report_status="Out"), 2026, 8, ["A", "B", "C", "D"])



class TestMarketMoves(unittest.TestCase):
    """Pre-registered market-move check: signs, the 1-point gap rule, no-move, the QB lock's own line."""

    def _games(self):
        # ours / lock / close are HOME margins. g1: we like home more (+2) and the line moved +1.5 → toward.
        # g2: we like away more (-3) and the line moved +1 (toward home) → away. g3: gap 0.5 → excluded.
        # g4: gap +2, no move. g5: not played yet → excluded. g6: QB lock uses its own (Saturday) line.
        scored = pd.DataFrame({
            "game_id": [f"g{i}" for i in range(1, 7)], "week": 8,
            "home_score": [20, 20, 20, 20, np.nan, 20], "away_score": [10] * 6,
            "elo_spread": [5.0, -1.0, 3.5, 4.0, 6.0, 2.0], "spread_line_at_pull": [3.0, 2.0, 3.0, 2.0, 1.0, 2.0],
            "p_elo": [0.65, 0.47, 0.6, 0.62, 0.7, 0.58], "p_market_at_pull": [0.60, 0.55, 0.6, 0.57, 0.5, 0.57],
            "p_elo_qb": [np.nan] * 5 + [0.45], "elo_spread_qb": [np.nan] * 5 + [-1.6],
            "qb_spread_line_at_pull": [np.nan] * 5 + [-3.0], "qb_p_market_at_pull": [np.nan] * 5 + [0.40]})
        sched = pd.DataFrame({"game_id": [f"g{i}" for i in range(1, 7)],
                              "spread_line": [4.5, 3.0, 3.5, 2.0, 1.0, -2.0],
                              "home_moneyline": [-200, -150, -160, -130, -110, 140],
                              "away_moneyline": [170, 130, 140, 110, -110, -160]})
        return scored, sched

    def test_signs_gap_rule_and_no_move(self):
        m = market_moves(*self._games()).set_index("game_id")
        self.assertEqual(sorted(m.index), ["g1", "g2", "g4", "g6"])        # g3 gap < 1, g5 unplayed
        self.assertEqual(m.loc["g1", "toward"], 1.0)
        self.assertEqual(m.loc["g2", "toward"], 0.0)
        self.assertTrue(np.isnan(m.loc["g4", "toward"]))                   # no move
        self.assertAlmostEqual(m.loc["g2", "dir_move"], -1.0)

    def test_qb_lock_is_judged_against_its_own_line(self):
        m = market_moves(*self._games()).set_index("game_id")
        # QB pick: home by -1.6 vs Saturday line -3.0 → gap +1.4; close -2.0 → moved +1.0 toward us.
        self.assertTrue(m.loc["g6", "qb_lock"])
        self.assertAlmostEqual(m.loc["g6", "gap"], 1.4)
        self.assertEqual(m.loc["g6", "toward"], 1.0)
        base = market_moves(*self._games(), use_qb=False).set_index("game_id")
        self.assertNotIn("g6", base.index)                                  # Thursday: gap 0.0 → excluded

    def test_summary_counts(self):
        s = market_move_summary(market_moves(*self._games()))
        self.assertEqual((s["games"], s["toward"], s["away"], s["no_move"]), (4, 2, 1, 1))
        self.assertAlmostEqual(s["share_toward"][0], 2 / 3)


if __name__ == "__main__":
    unittest.main()
