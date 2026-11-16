"""Which week the scheduled GitHub runs should act on (decision #25). Pure functions, so they're unit-tested.

- lock: the earliest regular-season week with no live ledger file, once its next kickoff is within LOCK_WINDOW.
  A daily run then catches normal Thursdays, Thanksgiving (early kickoff, locked Wednesday) and odd holiday days.
- qb (decision #27): per GAME, not per week. A game is due for its final QB lock once it is in a locked week, hasn't
  started, kicks off within QB_WINDOW, and has no row in ledger/qb yet. Whether its teams' final report and roster
  are published is checked at run time (ledger.qb_ready_teams); not-ready games wait for the next run.
"""

from pathlib import Path

import pandas as pd

from nfl_ats.ledger import LEDGER_DIR, kickoff_times

LOCK_WINDOW = pd.Timedelta(hours=30)


def ledger_weeks(season: int, folder: Path) -> set[int]:
    """Weeks with at least one ledger file in `folder` (names look like 2026_wk08_....csv)."""
    return {int(f.name.split("_wk")[1][:2]) for f in folder.glob(f"{season}_wk*.csv")}


def _upcoming(schedules: pd.DataFrame, season: int, now: pd.Timestamp) -> pd.DataFrame:
    reg = schedules[(schedules["season"] == season) & (schedules["game_type"] == "REG")].copy()
    reg["kickoff"] = kickoff_times(reg)
    return reg[reg["kickoff"] > now]


def next_lock_week(schedules: pd.DataFrame, season: int, locked: set[int], now: pd.Timestamp,
                   window: pd.Timedelta = LOCK_WINDOW) -> int | None:
    """The week to lock now, or None. Only the earliest unlocked week with games left is considered."""
    up = _upcoming(schedules, season, now)
    up = up[~up["week"].isin(locked)]
    if up.empty:
        return None
    week = int(up["week"].min())
    return week if up.loc[up["week"] == week, "kickoff"].min() - now <= window else None


QB_WINDOW = pd.Timedelta(hours=36)  # final statuses come out 1–2 days before kickoff (Friday for Sunday games)


def qb_due_games(schedules: pd.DataFrame, season: int, locked: set[int], qb_done: set[str], now: pd.Timestamp,
                 window: pd.Timedelta = QB_WINDOW) -> dict[int, set[str]]:
    """{week: game_ids} due for their final QB lock: in a locked week, not started, kickoff within `window`,
    and no ledger/qb row yet. May span two weeks (e.g. a Monday game and next Thursday's)."""
    up = _upcoming(schedules, season, now)
    due = up[up["week"].isin(locked) & ~up["game_id"].isin(qb_done) & (up["kickoff"] - now <= window)]
    return {int(w): set(g["game_id"]) for w, g in due.groupby("week")}


def qb_locked_games(season: int, ledger_dir: Path = LEDGER_DIR) -> set[str]:
    """game_ids that already have a row in a ledger/qb file."""
    files = sorted((ledger_dir / "qb").glob(f"{season}_wk*.csv"))
    return set(pd.concat([pd.read_csv(f, usecols=["game_id"]) for f in files])["game_id"]) if files else set()


WATCH_MARGIN = pd.Timedelta(hours=6)


def unlocked_soon(schedules: pd.DataFrame, season: int, locked: set[int], now: pd.Timestamp,
                  margin: pd.Timedelta = WATCH_MARGIN) -> list[int]:
    """The watchdog: weeks with a game kicking off within `margin` that have no live lock file.

    Deliberately independent of next_lock_week, so a bug there (or dropped scheduled runs) still raises the alarm.
    """
    up = _upcoming(schedules, season, now)
    soon = up[(up["kickoff"] - now <= margin) & ~up["week"].isin(locked)]
    return sorted(int(w) for w in soon["week"].unique())


def locked_weeks(season: int, ledger_dir: Path = LEDGER_DIR) -> tuple[set[int], set[int]]:
    """(live-locked weeks, weeks with a Saturday QB file)."""
    return ledger_weeks(season, ledger_dir), ledger_weeks(season, ledger_dir / "qb")
