"""Load nflverse data, convert Polars -> pandas, and cache raw pulls as parquet."""

from datetime import date
from pathlib import Path

import nflreadpy as nfl
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

HISTORY_SEASONS = list(range(2016, 2026))  # 2016-2025; 2026 is live
PRIOR_SEASON = 2015  # loaded only to give 2016 a prior-season carryover

# Relocated franchises -> current abbreviation, so each team keeps one ID across seasons.
TEAM_MAP = {"OAK": "LV", "SD": "LAC", "STL": "LA"}

# 2022 BUF @ CIN (Week 17) was canceled after Damar Hamlin's injury.
CANCELED_GAMES = {2022: 1}


def _cache_stem(name: str, seasons: list[int]) -> str:
    return f"{name}_{min(seasons)}_{max(seasons)}"


def _latest_cache(stem: str) -> Path | None:
    """Most recent cached pull for this stem. Filenames end in YYYYMMDD, so they sort by date."""
    files = sorted(RAW_DIR.glob(f"{stem}_*.parquet"))
    return files[-1] if files else None


def _cached_pull(name: str, seasons: list[int], loader, refresh: bool) -> pd.DataFrame:
    """Read the newest cached pull, or call `loader` and save it with today's date.

    Each pull keeps its date in the filename, because nflverse revises past data.
    """
    stem = _cache_stem(name, seasons)
    cached = _latest_cache(stem)
    if cached is not None and not refresh:
        return pd.read_parquet(cached)

    df = loader(seasons).to_pandas()  # Polars -> pandas at the boundary

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RAW_DIR / f"{stem}_{date.today():%Y%m%d}.parquet", index=False)
    return df


def load_schedules(seasons: list[int] = HISTORY_SEASONS, refresh: bool = False) -> pd.DataFrame:
    """One row per game: scores, spread_line, rest days, game IDs, kickoff dates."""
    return _cached_pull("schedules", seasons, nfl.load_schedules, refresh)


def load_pbp(seasons: list[int], refresh: bool = False) -> pd.DataFrame:
    """One row per play, with EPA. About 50k rows and 370+ columns per season, so pull only what you need."""
    return _cached_pull("pbp", seasons, nfl.load_pbp, refresh)


def regular_season(schedules: pd.DataFrame) -> pd.DataFrame:
    """Regular-season games only, with relocated teams mapped to one abbreviation."""
    reg = schedules[schedules["game_type"] == "REG"].copy()
    for col in ["home_team", "away_team"]:
        reg[col] = reg[col].replace(TEAM_MAP)
    return reg.reset_index(drop=True)


def expected_games(season: int) -> int:
    """256 games before the 17-game schedule (2021), 272 after, minus canceled games."""
    return (256 if season < 2021 else 272) - CANCELED_GAMES.get(season, 0)


def validate_schedules(reg: pd.DataFrame) -> None:
    """Fail loudly if the regular-season schedule looks wrong. Run before any feature work."""
    counts = reg.groupby("season").size()
    bad = {s: n for s, n in counts.items() if n != expected_games(s)}
    assert not bad, f"Unexpected game counts: {bad}"

    assert not reg["game_id"].duplicated().any(), "Duplicate game_id"

    # Only completed games need scores and lines; unplayed future games are allowed.
    played = reg[reg["home_score"].notna()]
    for col in ["spread_line", "away_score", "home_rest", "away_rest"]:
        n = played[col].isna().sum()
        assert n == 0, f"{n} played games missing {col}"

    # Stack home and away into one long column of team IDs, then count distinct teams.
    team_ids = reg.set_index("season")[["home_team", "away_team"]].stack()
    per_season = team_ids.groupby(level="season").nunique()
    assert (per_season == 32).all(), f"Seasons without 32 teams: {per_season[per_season != 32].to_dict()}"
    assert team_ids.nunique() == 32, f"Team mapping left extra IDs: {sorted(team_ids.unique())}"


def check_spread_sign(reg: pd.DataFrame) -> None:
    """Assert nflverse's convention: spread_line is the expected HOME margin (positive = home favored).

    Sportsbooks write it the other way ("DEN +3" = Denver is the underdog). Verified by hand on
    2016_01_CAR_DEN: spread_line = -3, and Carolina closed as the 3-point favorite.
    """
    played = reg[reg["result"].notna() & (reg["spread_line"] != 0)]
    home_fav = played["spread_line"] > 0
    home_won = played["result"] > 0
    fav_win_rate = (home_won == home_fav).mean()
    assert fav_win_rate > 0.6, f"Favorites won only {fav_win_rate:.1%}: is spread_line's sign flipped?"
    assert played["spread_line"].corr(played["result"]) > 0, "spread_line should rise with home margin"


def load_injuries(seasons: list[int], refresh: bool = False) -> pd.DataFrame:
    """Official weekly injury reports (one row per player per report week), cached like the others."""
    return _cached_pull("injuries", seasons, nfl.load_injuries, refresh)


def load_rosters_weekly(seasons: list[int], refresh: bool = False) -> pd.DataFrame:
    """Weekly team rosters with status (ACT, RES = reserve list such as injured reserve, ...), cached like the others."""
    return _cached_pull("rosters_weekly", seasons, nfl.load_rosters_weekly, refresh)


def load_depth_charts(seasons: list[int], refresh: bool = False) -> pd.DataFrame:
    """Weekly depth charts, cached like the others. The format changes around 2025, so check before use."""
    return _cached_pull("depth_charts", seasons, nfl.load_depth_charts, refresh)
