"""Site data: rating history matches Elo, record counts live games only, lock badge honest, JSON-safe cards."""

import json
import unittest
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from nfl_ats.elo import rating_history, run_elo
from nfl_ats.site import (MINUS, TEAMS, confidence, day_group, format_line, game_cards, lock_status,
                          record_summary, team_context)
from tests.test_win_model import _schedules


class TestSiteData(unittest.TestCase):

    def test_all_32_teams_have_a_color(self):
        self.assertEqual(len(TEAMS), 32)
        self.assertTrue(all(c.startswith("#") and len(c) == 7 for _, c in TEAMS.values()))

    def test_rating_history_matches_pregame_elo(self):
        """A team's pregame rating in week w equals its snapshot after its previous played week."""
        s = _schedules(seasons=(2015, 2016))
        s["season_type"] = "REG"
        hist = rating_history(s)
        pre = run_elo(s).merge(s[["game_id", "season", "week", "home_team"]], on="game_id")
        g = pre[(pre["season"] == 2016) & (pre["week"] == 5)].iloc[0]
        snap = hist[(hist["season"] == 2016) & (hist["team"] == g["home_team"]) & (hist["week"] < 5)]
        self.assertAlmostEqual(snap.sort_values("week").iloc[-1]["rating"], g["home_elo"])

    def test_record_counts_decided_games_only(self):
        scored = pd.DataFrame({"week": [6, 6, 6, 7], "home_win": [1.0, 0.0, np.nan, 1.0],
                               "p_elo": [0.7, 0.6, 0.5, 0.4], "p_market_at_pull": [0.6, 0.4, 0.5, np.nan]})
        r = record_summary(scored)
        self.assertEqual(r["games"], 3)                       # the tie / unplayed game is skipped
        self.assertEqual(r["by_week"][-1]["ours"], 1)         # right on game 1 only
        self.assertEqual(r["by_week"][-1]["vegas_games"], 2)  # no market number for the week-7 game
        self.assertEqual(r["by_week"][-1]["vegas"], 2)

    def test_lock_badge_flags_a_late_commit(self):
        kick = pd.Timestamp("2026-10-15 20:15", tz="America/New_York")
        early = lock_status(("abc", datetime(2026, 10, 15, 18, 51, tzinfo=timezone.utc)), kick)
        late = lock_status(("def", datetime(2026, 10, 16, 2, 0, tzinfo=timezone.utc)), kick)
        self.assertTrue(early["before_kickoff"])
        self.assertFalse(late["before_kickoff"])
        self.assertIn("AFTER", late["text"])
        self.assertFalse(lock_status(None, kick)["locked"])

    def _scored(self, home_score=np.nan, away_score=np.nan):
        return pd.DataFrame({
            "game_id": ["g1"], "kickoff_et": ["2026-10-18 13:00"], "home_team": ["PHI"], "away_team": ["CAR"],
            "location": ["Home"], "p_elo": [0.64], "p_market_at_pull": [0.38], "elo_spread": [4.0],
            "spread_line_at_pull": [-3.5], "home_elo": [1521.0], "away_elo": [1469.0],
            "home_score": [home_score], "away_score": [away_score]})

    def test_cards_are_json_safe_and_flag_big_gaps(self):
        cards = game_cards(self._scored())
        json.dumps(cards)
        c = cards[0]
        self.assertTrue(c["big_gap"])
        self.assertEqual(c["our_line"], f"PHI {MINUS}4.0")
        self.assertEqual(c["vegas_line_at_lock"], f"CAR {MINUS}3.5")  # negative home margin = away favored
        self.assertEqual(c["why"], {"home_rating": 1521, "away_rating": 1469, "home_field": 48, "edge": 100})
        self.assertIsNone(c["result"])

    def test_closing_line_hidden_until_kickoff(self):
        before = pd.Timestamp("2026-10-18 12:00", tz="America/New_York")
        after = pd.Timestamp("2026-10-18 13:30", tz="America/New_York")
        self.assertIsNone(game_cards(self._scored(), closing={"g1": -4.0}, now=before)[0]["closing_line"])
        self.assertEqual(game_cards(self._scored(), closing={"g1": -4.0}, now=after)[0]["closing_line"], f"CAR {MINUS}4.0")
        done = game_cards(self._scored(24, 17), closing={"g1": -4.0}, now=before)[0]  # a result also reveals it
        self.assertEqual(done["closing_line"], f"CAR {MINUS}4.0")
        self.assertTrue(done["result"]["we_were_right"])

    def test_line_formatting(self):
        self.assertEqual(format_line(0.0, "A", "B"), "Pick'em")
        self.assertEqual(format_line(2.5, "A", "B"), f"A {MINUS}2.5")
        self.assertIsNone(format_line(np.nan, "A", "B"))

    def test_confidence_and_day_group(self):
        self.assertEqual(confidence(0.549), "Toss-up")
        self.assertEqual(confidence(0.55), "Lean")
        self.assertEqual(confidence(0.30), "Favored")  # away team at 70%
        self.assertEqual(confidence(0.75), "High confidence")
        et = lambda t: pd.Timestamp(t, tz="America/New_York")
        self.assertEqual(day_group(et("2026-10-15 20:15")), "Thursday night")
        self.assertEqual(day_group(et("2026-10-18 09:30")), "International")
        self.assertEqual(day_group(et("2026-10-18 16:25")), "Sunday late")
        self.assertEqual(day_group(et("2026-10-19 20:15")), "Monday night")

    def test_team_context_uses_only_earlier_weeks(self):
        s = _schedules(seasons=(2015, 2016))
        s["season_type"] = "REG"
        hist = rating_history(s)
        base = team_context(s, hist, 2016, 5)
        wrecked = s.copy()
        later = (wrecked["season"] == 2016) & (wrecked["week"] >= 5)
        wrecked.loc[later, ["home_score", "away_score"]] = [99, 0]
        self.assertEqual(base, team_context(wrecked, rating_history(wrecked), 2016, 5))
        played_before = s[(s["season"] == 2016) & (s["week"] < 5)]
        games_for = lambda t: int(((played_before["home_team"] == t) | (played_before["away_team"] == t)).sum())
        rec = base["AAA"]["record"].split("-")
        self.assertEqual(sum(map(int, rec)), games_for("AAA"))

if __name__ == "__main__":
    unittest.main()
