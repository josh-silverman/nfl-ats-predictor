"""Auto-pilot week detection: lock inside the 30-hour window only, skip locked weeks, Thanksgiving locks a day early."""

import unittest

import pandas as pd

from nfl_ats.auto import next_lock_week, qb_due_games, unlocked_soon

ET = "America/New_York"


def _sched(games):
    """games: (week, gameday, gametime) tuples → a minimal nflverse-style schedule."""
    return pd.DataFrame([{"season": 2026, "game_type": "REG", "week": w, "gameday": d, "gametime": t,
                          "game_id": f"2026_{w:02d}_{d[5:]}_{t}", "home_team": "A", "away_team": "B"}
                         for w, d, t in games])


WEEKS = _sched([(8, "2026-10-29", "20:15"), (8, "2026-11-01", "13:00"), (8, "2026-11-02", "20:15"),
                (9, "2026-11-05", "20:15"), (9, "2026-11-08", "13:00")])


class TestAutoWeek(unittest.TestCase):

    def test_locks_thursday_inside_window_only(self):
        tue = pd.Timestamp("2026-10-27 10:00", tz=ET)
        thu = pd.Timestamp("2026-10-29 10:00", tz=ET)
        self.assertIsNone(next_lock_week(WEEKS, 2026, {7}, tue))          # 58 h before kickoff: not yet
        self.assertEqual(next_lock_week(WEEKS, 2026, {7}, thu), 8)        # 10 h before: lock

    def test_skips_locked_week_and_waits_for_the_next(self):
        thu = pd.Timestamp("2026-10-29 10:00", tz=ET)
        self.assertIsNone(next_lock_week(WEEKS, 2026, {7, 8}, thu))      # week 8 done; week 9 is a week away
        nxt = pd.Timestamp("2026-11-05 10:00", tz=ET)
        self.assertEqual(next_lock_week(WEEKS, 2026, {7, 8}, nxt), 9)

    def test_thanksgiving_locks_the_day_before(self):
        tg = _sched([(13, "2026-11-26", "12:30"), (13, "2026-11-29", "13:00")])
        wed = pd.Timestamp("2026-11-25 09:00", tz=ET)                     # 14:00 UTC daily run, EST
        self.assertEqual(next_lock_week(tg, 2026, set(), wed), 13)

    def test_missed_week_is_still_salvaged(self):
        sun = pd.Timestamp("2026-11-01 09:00", tz=ET)                     # Thursday game already played
        self.assertEqual(next_lock_week(WEEKS, 2026, {7}, sun), 8)        # remaining games still get locked

    def test_qb_games_need_a_lock_no_qb_row_and_a_near_kickoff(self):
        sat = pd.Timestamp("2026-10-31 08:00", tz=ET)
        sun1, mon = "2026_08_11-01_13:00", "2026_08_11-02_20:15"
        self.assertEqual(qb_due_games(WEEKS, 2026, {8}, set(), sat), {8: {sun1}})   # Monday's game is 60 h away
        self.assertEqual(qb_due_games(WEEKS, 2026, {8}, {sun1}, sat), {})           # already has a QB row
        self.assertEqual(qb_due_games(WEEKS, 2026, set(), set(), sat), {})          # never locked: nothing to re-lock
        sun = pd.Timestamp("2026-11-01 14:00", tz=ET)
        self.assertEqual(qb_due_games(WEEKS, 2026, {8}, {sun1}, sun), {8: {mon}})   # Monday's turn comes Sunday

    def test_thanksgiving_week_gets_qb_locks_before_each_game(self):
        """2026 week 12: Wed night, three Thursday games, a Friday game, then Sunday. Each is due a day-ish ahead."""
        wk = _sched([(12, "2026-11-25", "20:00"), (12, "2026-11-26", "13:00"), (12, "2026-11-26", "16:30"),
                     (12, "2026-11-26", "20:20"), (12, "2026-11-27", "15:00"), (12, "2026-11-29", "13:00")])
        wed = pd.Timestamp("2026-11-25 08:23", tz=ET)                     # the 13:23 UTC run
        due = qb_due_games(wk, 2026, {12}, set(), wed)[12]
        self.assertEqual(len(due), 4)                                     # Wed + all three Thanksgiving games
        thu = pd.Timestamp("2026-11-26 08:23", tz=ET)
        self.assertIn("2026_12_11-27_15:00", qb_due_games(wk, 2026, {12}, due, thu)[12])  # Friday's game next
        sunday = "2026_12_11-29_13:00"
        self.assertNotIn(sunday, qb_due_games(wk, 2026, {12}, due, thu)[12])          # Sunday waits for Saturday
        self.assertIn(sunday, qb_due_games(wk, 2026, {12}, due, pd.Timestamp("2026-11-28 08:23", tz=ET))[12])

    def test_watchdog_alarms_only_for_unlocked_weeks_close_to_kickoff(self):
        thu_noon = pd.Timestamp("2026-10-29 12:00", tz=ET)                # 8 h 15 m before TNF
        thu_eve = pd.Timestamp("2026-10-29 15:00", tz=ET)                 # 5 h 15 m before TNF
        self.assertEqual(unlocked_soon(WEEKS, 2026, {7}, thu_noon), [])   # inside the lock window, not yet an alarm
        self.assertEqual(unlocked_soon(WEEKS, 2026, {7}, thu_eve), [8])   # unlocked and close: alarm
        self.assertEqual(unlocked_soon(WEEKS, 2026, {7, 8}, thu_eve), [])  # locked: quiet
        sun = pd.Timestamp("2026-11-01 09:00", tz=ET)                     # TNF played, Sunday games 4 h away
        self.assertEqual(unlocked_soon(WEEKS, 2026, {7}, sun), [8])       # a missed lock still alarms


class TestPruneRawCache(unittest.TestCase):

    def test_keeps_newest_pull_per_dataset(self):
        import tempfile
        from pathlib import Path

        from nfl_ats.data import prune_raw_cache
        with tempfile.TemporaryDirectory() as d:
            names = ["schedules_2015_2026_20261026.parquet", "schedules_2015_2026_20261027.parquet",
                     "pbp_2026_2026_20261022.parquet", "pbp_2026_2026_20261027.parquet", "pbp_2025_2025_20261013.parquet"]
            for n in names:
                (Path(d) / n).touch()
            gone = {p.name for p in prune_raw_cache(folder=Path(d))}
            self.assertEqual(gone, {"schedules_2015_2026_20261026.parquet", "pbp_2026_2026_20261022.parquet"})
            self.assertEqual(len(list(Path(d).glob("*.parquet"))), 3)


if __name__ == "__main__":
    unittest.main()
