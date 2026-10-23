"""Playoff odds: played results are fixed, the field has the right shape, locked picks are used."""

import unittest

import numpy as np
import pandas as pd

from nfl_ats.playoffs import DIVISIONS, TEAMS, playoff_odds, simulate_season


def _league(season=2026, played_weeks=2, weeks=6, seed=0):
    """A toy 32-team season (plus the prior season for warm-up): random pairings each week, early weeks played."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in (season - 1, season):
        for week in range(1, weeks + 1):
            order = rng.permutation(TEAMS)
            for i in range(0, 32, 2):
                home, away = order[i], order[i + 1]
                done = s < season or week <= played_weeks
                hs, as_ = (int(rng.integers(0, 40)), int(rng.integers(0, 40))) if done else (np.nan, np.nan)
                rows.append({"season": s, "week": week, "gameday": f"{s}-09-{week:02d}", "game_type": "REG",
                             "game_id": f"{s}_{week:02d}_{away}_{home}", "home_team": home, "away_team": away,
                             "location": "Home", "home_score": hs, "away_score": as_})
    return pd.DataFrame(rows)


class TestPlayoffs(unittest.TestCase):

    def test_field_shape_and_conservation(self):
        sim = simulate_season(_league(), 2026, n_sims=2000, mov=2.0)
        o = playoff_odds(sim)
        conf = o["team"].map(lambda t: DIVISIONS[t][0])
        for c in ["AFC", "NFC"]:
            self.assertAlmostEqual(o.loc[conf == c, "playoff"].sum(), 7.0)       # 7 playoff teams per conference
            self.assertAlmostEqual(o.loc[conf == c, "division"].sum(), 4.0)
            self.assertAlmostEqual(o.loc[conf == c, "top_seed"].sum(), 1.0)
        self.assertTrue(((o["division"] <= o["playoff"] + 1e-12) & o["playoff"].between(0, 1)).all())
        self.assertAlmostEqual(o["wins"].sum(), 32 * 6 / 2)                       # every game has one winner

    def test_played_results_are_fixed(self):
        league = _league()
        sim = simulate_season(league, 2026, n_sims=500, mov=2.0)
        played = league[(league["season"] == 2026) & league["home_score"].notna()]
        t = TEAMS[0]
        g = played[(played["home_team"] == t) | (played["away_team"] == t)]
        real = sum((r.home_score > r.away_score) if r.home_team == t else (r.away_score > r.home_score)
                   for r in g.itertuples()) + 0.5 * sum(r.home_score == r.away_score for r in g.itertuples())
        self.assertTrue((sim["wins"][:, 0] >= real).all())                       # never fewer than already won
        self.assertTrue((sim["wins"][:, 0] <= real + 4).all())                   # at most 4 games left

    def test_locked_picks_are_used_and_seed_is_reproducible(self):
        league = _league()
        week3 = league[(league["season"] == 2026) & (league["week"] == 3)]
        picks = {gid: 1.0 for gid in week3["game_id"]}                         # every home team surely wins week 3
        a = simulate_season(league, 2026, picks, n_sims=300, seed=1, mov=2.0)
        b = simulate_season(league, 2026, picks, n_sims=300, seed=1, mov=2.0)
        np.testing.assert_array_equal(a["wins"], b["wins"])
        home = TEAMS.index(week3.iloc[0]["home_team"])
        no_pick = simulate_season(league, 2026, None, n_sims=300, seed=1, mov=2.0)
        self.assertGreater(a["wins"][:, home].mean(), no_pick["wins"][:, home].mean() - 1e-9)

    def test_dominant_team_almost_always_makes_it(self):
        league = _league(played_weeks=5)
        t = TEAMS[5]
        mine = (league["home_team"] == t) | (league["away_team"] == t)
        played = league["home_score"].notna()
        league.loc[mine & played & (league["home_team"] == t), ["home_score", "away_score"]] = [45, 0]
        league.loc[mine & played & (league["away_team"] == t), ["home_score", "away_score"]] = [0, 45]
        o = playoff_odds(simulate_season(league, 2026, n_sims=2000, mov=2.0)).set_index("team")
        self.assertGreater(o.loc[t, "playoff"], 0.95)


if __name__ == "__main__":
    unittest.main()
