"""Auto-pilot week detection: lock inside the 30-hour window only, skip locked weeks, Thanksgiving locks a day early."""

import unittest

import pandas as pd

from nfl_ats.auto import next_lock_week, next_qb_week, unlocked_soon

ET = "America/New_York"


def _sched(games):
    """games: (week, gameday, gametime) tuples → a minimal nflverse-style schedule."""
    return pd.DataFrame([{"season": 2026, "game_type": "REG", "week": w, "gameday": d, "gametime": t,
                          "home_team": "A", "away_team": "B"} for w, d, t in games])


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

    def test_qb_week_needs_a_lock_and_no_qb_file(self):
        sat = pd.Timestamp("2026-10-31 08:00", tz=ET)
        self.assertEqual(next_qb_week(WEEKS, 2026, {8}, set(), sat), 8)
        self.assertIsNone(next_qb_week(WEEKS, 2026, {8}, {8}, sat))       # already re-locked
        self.assertIsNone(next_qb_week(WEEKS, 2026, set(), set(), sat))   # never locked: nothing to re-lock

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
