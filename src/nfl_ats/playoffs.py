"""Playoff odds: play out the rest of the season many times with Elo win chances (decision #23).

- Played games keep their real results (a tie counts half a win).
- This week's unplayed games use the LOCKED headline pick (QB-adjusted from Week 8), so the odds match the site.
- Later games are "hot" Elo, as FiveThirtyEight did: each simulation updates ratings after every simulated game.
  Margins aren't simulated, so the update uses the historical average margin-of-victory multiplier.
- Playoff field per conference: the 4 division winners + the best other records (3 wild cards from 2020, 2 before).
  Tiebreakers are simplified: equal records are broken at random.
"""

import numpy as np
import pandas as pd

from nfl_ats.data import regular_season
from nfl_ats.elo import HOME_FIELD, K, _mov_multiplier, rating_history, run_elo

DIVISIONS = {
    "BUF": ("AFC", "AFC East"), "MIA": ("AFC", "AFC East"), "NE": ("AFC", "AFC East"), "NYJ": ("AFC", "AFC East"),
    "BAL": ("AFC", "AFC North"), "CIN": ("AFC", "AFC North"), "CLE": ("AFC", "AFC North"), "PIT": ("AFC", "AFC North"),
    "HOU": ("AFC", "AFC South"), "IND": ("AFC", "AFC South"), "JAX": ("AFC", "AFC South"), "TEN": ("AFC", "AFC South"),
    "DEN": ("AFC", "AFC West"), "KC": ("AFC", "AFC West"), "LAC": ("AFC", "AFC West"), "LV": ("AFC", "AFC West"),
    "DAL": ("NFC", "NFC East"), "NYG": ("NFC", "NFC East"), "PHI": ("NFC", "NFC East"), "WAS": ("NFC", "NFC East"),
    "CHI": ("NFC", "NFC North"), "DET": ("NFC", "NFC North"), "GB": ("NFC", "NFC North"), "MIN": ("NFC", "NFC North"),
    "ATL": ("NFC", "NFC South"), "CAR": ("NFC", "NFC South"), "NO": ("NFC", "NFC South"), "TB": ("NFC", "NFC South"),
    "ARI": ("NFC", "NFC West"), "LA": ("NFC", "NFC West"), "SF": ("NFC", "NFC West"), "SEA": ("NFC", "NFC West"),
}
TEAMS = sorted(DIVISIONS)


def average_mov_multiplier(reg: pd.DataFrame) -> float:
    """Mean margin-of-victory multiplier over decided games, using each game's pregame Elo gap."""
    elo = run_elo(reg).merge(reg[["game_id", "home_score", "away_score"]], on="game_id")
    margin = elo["home_score"] - elo["away_score"]
    d = elo[margin.notna() & (margin != 0)]
    m = (d["home_score"] - d["away_score"]).to_numpy()
    winner_gap = np.where(m > 0, d["elo_diff"], -d["elo_diff"])
    return float(np.mean([_mov_multiplier(a, b) for a, b in zip(m, winner_gap)]))


def wild_cards(season: int) -> int:
    return 3 if season >= 2020 else 2


def simulate_season(reg: pd.DataFrame, season: int, picks: dict | None = None, n_sims: int = 10_000,
                    seed: int = 0, mov: float | None = None) -> dict:
    """Simulate the unplayed games of `season`. reg: regular-season schedule (as of now: future scores NaN).

    picks: {game_id: p_home} used for those games instead of Elo (this week's locked picks).
    Returns {"wins": (n_sims, 32) array incl. played results, "games": games per team, "teams": TEAMS}.
    """
    picks = picks or {}
    rng = np.random.default_rng(seed)
    hist = rating_history(reg[reg["season"] <= season])
    current = hist[hist["season"] == season].sort_values("week").groupby("team")["rating"].last()
    mov = average_mov_multiplier(reg[reg["season"] < season]) if mov is None else mov
    idx = {t: i for i, t in enumerate(TEAMS)}

    games = reg[reg["season"] == season].sort_values(["week", "gameday", "game_id"])
    played = games[games["home_score"].notna()]
    wins = np.zeros((n_sims, len(TEAMS)))
    for g in played.itertuples(index=False):
        h, a = idx[g.home_team], idx[g.away_team]
        res = 1.0 if g.home_score > g.away_score else 0.0 if g.home_score < g.away_score else 0.5
        wins[:, h] += res
        wins[:, a] += 1 - res

    ratings = np.tile(current.reindex(TEAMS).to_numpy(dtype=float), (n_sims, 1))
    for g in games[games["home_score"].isna()].itertuples(index=False):
        h, a = idx[g.home_team], idx[g.away_team]
        hfa = 0 if g.location == "Neutral" else HOME_FIELD
        p = 1 / (1 + 10 ** (-(ratings[:, h] + hfa - ratings[:, a]) / 400))
        p_game = np.full(n_sims, picks[g.game_id]) if g.game_id in picks else p
        home_won = (rng.random(n_sims) < p_game).astype(float)
        wins[:, h] += home_won
        wins[:, a] += 1 - home_won
        shift = K * mov * (home_won - p)
        ratings[:, h] += shift
        ratings[:, a] -= shift

    n_games = pd.concat([games["home_team"], games["away_team"]]).value_counts().reindex(TEAMS).fillna(0).to_numpy()
    return {"wins": wins, "games": n_games, "teams": TEAMS, "season": season, "rng": rng}


def playoff_odds(sim: dict) -> pd.DataFrame:
    """Per team: mean projected wins/losses and the share of simulations making the playoffs, winning the
    division, and earning the #1 seed. Equal records are broken at random."""
    wins, teams, rng = sim["wins"], sim["teams"], sim["rng"]
    n_sims = wins.shape[0]
    score = wins + rng.random(wins.shape) * 1e-3  # random tiebreak
    conf = np.array([DIVISIONS[t][0] for t in teams])
    div = np.array([DIVISIONS[t][1] for t in teams])
    playoff = np.zeros_like(wins, dtype=bool)
    div_win = np.zeros_like(wins, dtype=bool)
    top_seed = np.zeros_like(wins, dtype=bool)
    rows = np.arange(n_sims)
    for c in ["AFC", "NFC"]:
        winners = []
        for d in sorted(set(div[conf == c])):
            cols = np.where(div == d)[0]
            w = cols[np.argmax(score[:, cols], axis=1)]
            div_win[rows, w] = True
            winners.append(w)
        winners = np.stack(winners, axis=1)                        # (n_sims, 4)
        best = winners[rows, np.argmax(score[rows[:, None], winners], axis=1)]
        top_seed[rows, best] = True
        cols = np.where(conf == c)[0]
        others = np.where(div_win[:, cols], -np.inf, score[:, cols])
        wc = cols[np.argsort(-others, axis=1)[:, :wild_cards(sim["season"])]]
        playoff[rows[:, None], wc] = True
        playoff[rows[:, None], winners] = True
    mean_wins = wins.mean(axis=0)
    return pd.DataFrame({"team": teams, "wins": mean_wins, "losses": sim["games"] - mean_wins,
                         "playoff": playoff.mean(axis=0), "division": div_win.mean(axis=0),
                         "top_seed": top_seed.mean(axis=0)})


def season_odds(schedules: pd.DataFrame, season: int, picks: dict | None = None, n_sims: int = 10_000,
                seed: int = 0) -> pd.DataFrame:
    """End to end from nflverse schedules (current knowledge): simulate, then odds per team."""
    reg = regular_season(schedules)
    return playoff_odds(simulate_season(reg[reg["season"] <= season], season, picks, n_sims, seed))
