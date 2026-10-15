"""Site data: rating history matches Elo, record counts live games only, lock badge honest, JSON-safe cards."""

import json
import unittest
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from nfl_ats.elo import rating_history, run_elo
from nfl_ats.site import TEAMS, game_cards, lock_status, record_summary
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

    def test_cards_are_json_safe_and_flag_big_gaps(self):
        scored = pd.DataFrame({
            "game_id": ["g1"], "kickoff_et": ["2026-10-18 13:00"], "home_team": ["PHI"], "away_team": ["CAR"],
            "location": ["Home"], "p_elo": [0.64], "p_market_at_pull": [0.38], "elo_spread": [4.0],
            "home_score": [np.nan], "away_score": [np.nan]})
        cards = game_cards(scored)
        json.dumps(cards)
        self.assertTrue(cards[0]["big_gap"])
        self.assertEqual(cards[0]["our_line"], "PHI by 4.0")
        self.assertIsNone(cards[0]["result"])


if __name__ == "__main__":
    unittest.main()
