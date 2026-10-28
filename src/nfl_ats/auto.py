"""Which week the scheduled GitHub runs should act on (decision #25). Pure functions, so they're unit-tested.

- lock: the earliest regular-season week with no live ledger file, once its next kickoff is within LOCK_WINDOW.
  A daily run then catches normal Thursdays, Thanksgiving (early kickoff, locked Wednesday) and odd holiday days.
- qb: the earliest locked week that still has games to play and no Saturday QB file yet.
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


def next_qb_week(schedules: pd.DataFrame, season: int, locked: set[int], qb_done: set[int],
                 now: pd.Timestamp) -> int | None:
    """The week to re-lock with Saturday QB news, or None: earliest locked week with games left and no QB file."""
    up = _upcoming(schedules, season, now)
    weeks = sorted(w for w in up["week"].unique() if w in locked and w not in qb_done)
    return int(weeks[0]) if weeks else None


def locked_weeks(season: int, ledger_dir: Path = LEDGER_DIR) -> tuple[set[int], set[int]]:
    """(live-locked weeks, weeks with a Saturday QB file)."""
    return ledger_weeks(season, ledger_dir), ledger_weeks(season, ledger_dir / "qb")
