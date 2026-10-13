"""The prediction target: did the home team cover the closing spread?"""

import numpy as np
import pandas as pd


def add_home_cover(reg: pd.DataFrame) -> pd.DataFrame:
    """Add `margin`, `is_push`, and `home_cover` (1 = covered, 0 = didn't, NaN = push or unplayed).

    spread_line is the expected home margin (see data.check_spread_sign), so the home team
    covers when its actual margin beats it. Pushes stay in the frame, flagged, so they can be
    counted; the model drops them via home_cover being NaN.
    """
    out = reg.copy()
    out["margin"] = out["home_score"] - out["away_score"]
    played = out["margin"].notna()

    out["is_push"] = played & (out["margin"] == out["spread_line"])
    out["home_cover"] = np.where(
        played & ~out["is_push"],
        (out["margin"] > out["spread_line"]).astype(float),
        np.nan,
    )
    return out


def push_summary(games: pd.DataFrame) -> pd.DataFrame:
    """Per-season counts of played games and pushes, plus the home cover rate excluding pushes."""
    played = games[games["margin"].notna()]
    return played.groupby("season").agg(
        games=("game_id", "size"),
        pushes=("is_push", "sum"),
        push_rate=("is_push", "mean"),
        home_cover_rate=("home_cover", "mean"),  # mean() skips NaN, so pushes are excluded
    )
