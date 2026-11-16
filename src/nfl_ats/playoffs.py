"""Playoff odds: play out the rest of the season many times with Elo win chances (decision #23).

- Played games keep their real results (a tie counts half a win).
- This week's unplayed games use the LOCKED headline pick (QB-adjusted from Week 8), so the odds match the site.
- Later games are "hot" Elo, as FiveThirtyEight did: each simulation updates ratings after every simulated game.
  Margins aren't simulated, so the update uses the historical average margin-of-victory multiplier.
- Playoff field per conference: the 4 division winners + the best other records (3 wild cards from 2020, 2 before).
- Tiebreakers (decision #28), applied to teams with equal records in each simulated season:
    division title:     head-to-head among the tied teams → division record → conference record → coin flip
    wild card, #1 seed: head-to-head among the tied teams → conference record → coin flip
  Simplified from the NFL's 12-step procedure: no common games, strength of victory or schedule, and no
  "settle same-division ties first" rule for wild cards. Head-to-head counts games among ALL teams tied on wins
  (a team that played none of them gets 50%).
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
CONF = np.array([DIVISIONS[t][0] for t in TEAMS])
DIV = np.array([DIVISIONS[t][1] for t in TEAMS])
SAME_CONF = CONF[:, None] == CONF[None, :]
SAME_DIV = DIV[:, None] == DIV[None, :]


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
    n = len(TEAMS)
    # Results in DOUBLED points (win 2, tie 1, loss 0) so ties stay integers. h2h[s, i, j] = i's points vs j.
    h2h = np.zeros((n_sims, n, n), dtype=np.int8)
    div_pts = np.zeros((n_sims, n), dtype=np.int16)
    conf_pts = np.zeros((n_sims, n), dtype=np.int16)
    div_games, conf_games = np.zeros(n, dtype=int), np.zeros(n, dtype=int)

    def record(h, a, home_pts):  # home_pts: doubled points for the home team (scalar or per-sim array)
        h2h[:, h, a] += home_pts
        h2h[:, a, h] += 2 - home_pts
        for same, pts, cnt in ((SAME_DIV, div_pts, div_games), (SAME_CONF, conf_pts, conf_games)):
            if same[h, a]:
                pts[:, h] += home_pts
                pts[:, a] += 2 - home_pts
                cnt[[h, a]] += 1

    wins = np.zeros((n_sims, n))
    for g in games[games["home_score"].notna()].itertuples(index=False):
        h, a = idx[g.home_team], idx[g.away_team]
        res = 1.0 if g.home_score > g.away_score else 0.0 if g.home_score < g.away_score else 0.5
        wins[:, h] += res
        wins[:, a] += 1 - res
        record(h, a, int(2 * res))

    ratings = np.tile(current.reindex(TEAMS).to_numpy(dtype=float), (n_sims, 1))
    for g in games[games["home_score"].isna()].itertuples(index=False):
        h, a = idx[g.home_team], idx[g.away_team]
        hfa = 0 if g.location == "Neutral" else HOME_FIELD
        p = 1 / (1 + 10 ** (-(ratings[:, h] + hfa - ratings[:, a]) / 400))
        p_game = np.full(n_sims, picks[g.game_id]) if g.game_id in picks else p
        home_won = (rng.random(n_sims) < p_game).astype(float)
        wins[:, h] += home_won
        wins[:, a] += 1 - home_won
        record(h, a, (2 * home_won).astype(np.int8))
        shift = K * mov * (home_won - p)
        ratings[:, h] += shift
        ratings[:, a] -= shift

    n_games = pd.concat([games["home_team"], games["away_team"]]).value_counts().reindex(TEAMS).fillna(0).to_numpy()
    return {"wins": wins, "games": n_games, "teams": TEAMS, "season": season, "rng": rng,
            "h2h": h2h, "div_pct": div_pts / np.maximum(2 * div_games, 1), "conf_pct": conf_pts / np.maximum(2 * conf_games, 1)}


def head_to_head(h2h: np.ndarray, wins: np.ndarray, pool: np.ndarray, group: np.ndarray) -> np.ndarray:
    """Each team's win share in games against the OTHER teams of `pool` in its `group` with the same record.

    h2h: (n_sims, n, n) doubled points. pool: (n_sims, n) bool, who is competing. group: (n, n) bool, who can tie
    with whom (same division, or same conference). Teams with no such games get 0.5. Returns (n_sims, n).
    """
    n = wins.shape[1]
    tied = ((wins[:, :, None] == wins[:, None, :]) & pool[:, :, None] & pool[:, None, :]
            & group[None] & ~np.eye(n, dtype=bool)[None])
    pts = (h2h * tied).sum(axis=2, dtype=np.int32)
    games = ((h2h + h2h.transpose(0, 2, 1)) * tied).sum(axis=2, dtype=np.int32)  # doubled: 2 per game
    return np.where(games > 0, pts / np.maximum(games, 1), 0.5)


def tiebreak_key(wins: np.ndarray, *steps: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One sortable number per team: record first, then each tiebreak step (a share in [0, 1]), then a coin flip."""
    key = np.round(wins * 2)
    for s in steps:
        key = key * 1001 + np.round(s * 1000)
    return key + rng.random(key.shape) * 0.5


def playoff_odds(sim: dict) -> pd.DataFrame:
    """Per team: mean projected wins/losses and the share of simulations making the playoffs, winning the
    division, and earning the #1 seed, with the tiebreakers in the module docstring."""
    wins, teams, rng = sim["wins"], sim["teams"], sim["rng"]
    h2h, div_pct, conf_pct = sim["h2h"], sim["div_pct"], sim["conf_pct"]
    n_sims = wins.shape[0]
    everyone = np.ones_like(wins, dtype=bool)
    div_key = tiebreak_key(wins, head_to_head(h2h, wins, everyone, SAME_DIV), div_pct, conf_pct, rng=rng)
    playoff = np.zeros_like(wins, dtype=bool)
    div_win = np.zeros_like(wins, dtype=bool)
    top_seed = np.zeros_like(wins, dtype=bool)
    rows = np.arange(n_sims)
    for d in sorted(set(DIV)):
        cols = np.where(DIV == d)[0]
        div_win[rows, cols[np.argmax(div_key[:, cols], axis=1)]] = True
    seed_key = tiebreak_key(wins, head_to_head(h2h, wins, div_win, SAME_CONF), conf_pct, rng=rng)
    wc_key = tiebreak_key(wins, head_to_head(h2h, wins, ~div_win, SAME_CONF), conf_pct, rng=rng)
    for c in ["AFC", "NFC"]:
        cols = np.where(CONF == c)[0]
        winners = np.where(div_win[:, cols], seed_key[:, cols], -np.inf)
        top_seed[rows, cols[np.argmax(winners, axis=1)]] = True
        others = np.where(div_win[:, cols], -np.inf, wc_key[:, cols])
        wc = cols[np.argsort(-others, axis=1)[:, :wild_cards(sim["season"])]]
        playoff[rows[:, None], wc] = True
    playoff |= div_win
    mean_wins = wins.mean(axis=0)
    return pd.DataFrame({"team": teams, "wins": mean_wins, "losses": sim["games"] - mean_wins,
                         "playoff": playoff.mean(axis=0), "division": div_win.mean(axis=0),
                         "top_seed": top_seed.mean(axis=0)})


def season_odds(schedules: pd.DataFrame, season: int, picks: dict | None = None, n_sims: int = 10_000,
                seed: int = 0) -> pd.DataFrame:
    """End to end from nflverse schedules (current knowledge): simulate, then odds per team."""
    reg = regular_season(schedules)
    return playoff_odds(simulate_season(reg[reg["season"] <= season], season, picks, n_sims, seed))


def playoff_teams(schedules: pd.DataFrame, season: int) -> set[str]:
    """Who actually made the playoffs: every team in a wild-card or divisional-round game (byes join in round 2)."""
    from nfl_ats.data import TEAM_MAP
    po = schedules[(schedules["season"] == season) & schedules["game_type"].isin(["WC", "DIV"])]
    return set(pd.concat([po["home_team"], po["away_team"]]).replace(TEAM_MAP))


def odds_check(schedules: pd.DataFrame, seasons: list[int], week: int, n_sims: int = 10_000) -> pd.DataFrame:
    """Backtest: playoff odds as of `week` (only results before it, via ledger.as_of) vs who made it.

    One row per team-season: season, team, p (playoff odds), made (1/0). Score it with evaluate.brier.
    """
    from nfl_ats.ledger import as_of
    rows = []
    for s in seasons:
        odds = season_odds(as_of(schedules[schedules["season"] <= s], s, week), s, n_sims=n_sims)
        made = playoff_teams(schedules, s)
        rows.append(odds.assign(season=s, made=odds["team"].isin(made).astype(int))[["season", "team", "playoff", "made"]])
    return pd.concat(rows, ignore_index=True).rename(columns={"playoff": "p"})
