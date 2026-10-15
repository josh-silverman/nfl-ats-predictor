"""The 2026 ledger: locked weekly predictions, and scoring them afterward. Rules: CLAUDE.md "Live season".

- Predictions for a week use only games from EARLIER weeks: scores from the target week onward are
  masked before Elo runs, so the same code serves the live weekly run and the pseudo-live backfill.
- A game is never predicted after its kickoff. Games that have already started are skipped and counted.
- Ledger files are never overwritten.
"""

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from nfl_ats.data import PROJECT_ROOT, load_schedules
from nfl_ats.win_model import WIN_FEATURES, build_win_table, fit_fold

LEDGER_DIR = PROJECT_ROOT / "ledger"
EASTERN = ZoneInfo("America/New_York")
FIRST_SEASON = 2015  # Elo warm-up
MODEL_COLUMNS = {"p_elo": "raw Elo (the deliverable)", "p_v2": "v2 (Elo + logistic)"}


def kickoff_times(games: pd.DataFrame) -> pd.Series:
    """nflverse gameday + gametime are US Eastern local time."""
    return pd.to_datetime(games["gameday"] + " " + games["gametime"]).dt.tz_localize(EASTERN)


def as_of(schedules: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """Schedules as they were known before `week` of `season`: every score from that week on is hidden."""
    out = schedules.copy()
    future = (out["season"] > season) | ((out["season"] == season) & (out["week"] >= week))
    out.loc[future, ["home_score", "away_score"]] = np.nan
    return out


def check_prior_weeks_final(schedules: pd.DataFrame, season: int, week: int) -> None:
    """Elo silently skips unscored games, so refuse to predict if any earlier regular-season game lacks a score."""
    reg = schedules[(schedules["game_type"] == "REG") & (schedules["season"] == season) & (schedules["week"] < week)]
    missing = reg.loc[reg["home_score"].isna() | reg["away_score"].isna(), "game_id"]
    if not missing.empty:
        raise ValueError(f"Earlier-week games without final scores (stale data?): {missing.tolist()}")


def predict_week(schedules: pd.DataFrame, season: int, week: int, now: datetime, mode: str = "live") -> pd.DataFrame:
    """Predictions for every not-yet-started game in `week`, using only results from earlier weeks.

    mode="live": `now` is the real clock. mode="backfill": `now` is that week's first kickoff (pseudo-live).
    """
    check_prior_weeks_final(schedules, season, week)
    known = as_of(schedules[schedules["season"].between(FIRST_SEASON, season)], season, week)
    table = build_win_table(known)
    model = fit_fold(table, WIN_FEATURES, test_season=season)  # trained on 2016 .. season-1 only

    games = table[(table["season"] == season) & (table["week"] == week)].copy()
    reg_week = schedules[(schedules["season"] == season) & (schedules["week"] == week) & (schedules["game_type"] == "REG")]
    games = games.merge(reg_week[["game_id", "gametime"]], on="game_id", how="left", validate="one_to_one")
    games["kickoff_et"] = kickoff_times(games)
    started = games["kickoff_et"] <= now
    games = games[~started]
    if games.empty:
        raise ValueError(f"Every game in {season} week {week} has already kicked off; nothing to predict.")

    return pd.DataFrame({
        "mode": mode,
        "season": season, "week": week, "game_id": games["game_id"],
        "kickoff_et": games["kickoff_et"].dt.strftime("%Y-%m-%d %H:%M"),
        "home_team": games["home_team"], "away_team": games["away_team"], "location": games["location"],
        "home_elo": games["home_elo"].round(1), "away_elo": games["away_elo"].round(1),
        "elo_spread": games["elo_spread"].round(1),
        "p_elo": games["p_home_win"].round(4),
        "p_v2": model.predict_proba(games[WIN_FEATURES])[:, 1].round(4),
        "home_moneyline": games["home_moneyline"], "away_moneyline": games["away_moneyline"],
        "p_market_at_pull": games["p_market"].round(4),
        "spread_line_at_pull": games["spread_line"],
        "pulled_at_et": now.astimezone(EASTERN).strftime("%Y-%m-%d %H:%M:%S"),
        "games_skipped_already_started": int(started.sum()),
    }).reset_index(drop=True)


def ledger_path(season: int, week: int, mode: str, now: datetime) -> Path:
    if mode == "backfill":
        return LEDGER_DIR / "backfill" / f"{season}_wk{week:02d}_backfill.csv"
    return LEDGER_DIR / f"{season}_wk{week:02d}_{now.astimezone(EASTERN):%Y%m%dT%H%M%S}.csv"


def write_ledger(df: pd.DataFrame, path: Path) -> Path:
    """Write once. Refuses to overwrite: a locked prediction is never edited."""
    if path.exists():
        raise FileExistsError(f"{path} already exists; ledger files are never overwritten.")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_ledger(mode: str = "live", season: int = 2026) -> pd.DataFrame:
    """All locked predictions for a mode. If a week was run more than once, the EARLIEST lock per game counts."""
    folder = LEDGER_DIR / "backfill" if mode == "backfill" else LEDGER_DIR
    files = sorted(folder.glob(f"{season}_wk*.csv"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(f).assign(file=f.name) for f in files], ignore_index=True)
    return df.sort_values("pulled_at_et").drop_duplicates("game_id", keep="first").reset_index(drop=True)


def score_ledger(ledger: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """Join locked predictions to final results. Ties and unplayed games stay in the file but get no outcome."""
    results = schedules[["game_id", "home_score", "away_score"]]
    out = ledger.merge(results, on="game_id", how="left", validate="one_to_one")
    margin = out["home_score"] - out["away_score"]
    out["is_tie"] = margin == 0
    out["home_win"] = (margin > 0).astype(float).where(margin.notna() & ~out["is_tie"])
    return out


def load_schedules_for_ledger(season: int, refresh: bool = True) -> pd.DataFrame:
    """2015 .. season schedules. Live runs always refresh so results and lines are current."""
    return load_schedules(list(range(FIRST_SEASON, season + 1)), refresh=refresh)
