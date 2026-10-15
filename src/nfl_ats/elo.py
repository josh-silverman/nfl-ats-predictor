"""Elo ratings: the simplest useful NFL model. Team strength from final scores only.

FiveThirtyEight-STYLE: their published core parameters, but regular-season games only (playoffs never
update ratings) and none of their later QB / travel / rest adjustments.
Parameters are fixed up front and never tuned on our data:
K = 20, home field = 48 Elo points (0 at neutral sites), a margin-of-victory multiplier, and
a 1/3 regression toward the mean between seasons.

Every game's probability is computed from ratings BEFORE that game, then the ratings update.
So a prediction never uses its own result. That's the same as-of rule as shift(1).
"""

import numpy as np
import pandas as pd

K = 20
HOME_FIELD = 48
MEAN_RATING = 1505
SEASON_REGRESSION = 1 / 3
ELO_PER_POINT = 25  # FiveThirtyEight's conversion: 25 Elo points ≈ 1 point of spread


def win_prob(rating_diff: float) -> float:
    """P(team A wins) from (A's rating + home field − B's rating)."""
    return 1 / (1 + 10 ** (-rating_diff / 400))


def _mov_multiplier(margin: float, winner_rating_diff: float) -> float:
    """Bigger wins move ratings more, damped when the favorite wins (so blowouts by good teams count less)."""
    return np.log(abs(margin) + 1) * 2.2 / (winner_rating_diff * 0.001 + 2.2)


def trailing_home_field(games: pd.DataFrame, window: int = 3) -> dict[int, float]:
    """Home field (Elo points) for each season, set BEFORE the season from the trailing `window` seasons:
    mean home margin in decided-or-tied, non-neutral games × ELO_PER_POINT. Experiment H1.

    Uses only earlier seasons, so it's as-of. Seasons with no earlier data get the published HOME_FIELD.
    """
    played = games[games["home_score"].notna() & (games["location"] != "Neutral")]
    margin_by_season = (played["home_score"] - played["away_score"]).groupby(played["season"]).agg(["sum", "size"])
    out = {}
    for season in sorted(games["season"].unique()):
        prior = margin_by_season.loc[(margin_by_season.index >= season - window) & (margin_by_season.index < season)]
        out[int(season)] = (float(prior["sum"].sum() / prior["size"].sum()) * ELO_PER_POINT
                            if len(prior) else float(HOME_FIELD))
    return out


def run_elo(games: pd.DataFrame, home_field: float | dict[int, float] = HOME_FIELD) -> pd.DataFrame:
    """Pregame Elo for every game, in kickoff order. Unplayed games get a prediction but no update.

    `games` needs season, week, gameday, game_id, home_team, away_team, home_score, away_score, location.
    home_field: Elo points for a home team, either one number or a {season: points} dict (H1).
    Returns game_id, home_elo, away_elo, elo_diff (incl. home field), p_home_win, elo_spread.
    """
    ratings: dict[str, float] = {}
    season = None
    rows = []
    for g in games.sort_values(["season", "week", "gameday", "game_id"]).itertuples(index=False):
        if g.season != season:  # new season: regress everyone 1/3 toward the mean
            ratings = {t: MEAN_RATING * SEASON_REGRESSION + r * (1 - SEASON_REGRESSION) for t, r in ratings.items()}
            season = g.season
        home, away = ratings.get(g.home_team, MEAN_RATING), ratings.get(g.away_team, MEAN_RATING)
        season_hfa = home_field[g.season] if isinstance(home_field, dict) else home_field
        hfa = 0 if g.location == "Neutral" else season_hfa
        diff = home + hfa - away
        p = win_prob(diff)
        rows.append({"game_id": g.game_id, "home_elo": home, "away_elo": away, "elo_diff": diff,
                     "p_home_win": p, "elo_spread": diff / ELO_PER_POINT})

        if pd.notna(g.home_score):  # update only after the game is played
            margin = g.home_score - g.away_score
            result = 1.0 if margin > 0 else 0.5 if margin == 0 else 0.0
            winner_diff = diff if margin > 0 else -diff
            shift = K * _mov_multiplier(margin, winner_diff) * (result - p)
            ratings[g.home_team] = home + shift
            ratings[g.away_team] = away - shift
    return pd.DataFrame(rows)
