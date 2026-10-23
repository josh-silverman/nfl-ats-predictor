"""The 2026 ledger: locked weekly predictions, and scoring them afterward. Rules: docs/project-rules.md "Live season".

- Predictions for a week use only games from EARLIER weeks: scores from the target week onward are
  masked before Elo runs, so the same code serves the live weekly run and the pseudo-live backfill.
- A game is never predicted after its kickoff. Games that have already started are skipped and counted.
- Ledger files are never overwritten.
- From 2026 Week 8 the deliverable is the QB-adjusted Elo (H4, decision #21). Thursday's live file carries it with
  Thursday's injury news; a Saturday `ledger/qb/` file re-locks it for unstarted games once final statuses are out.
  Per game, the LATEST QB lock made before kickoff counts (fixed before any Week 8 result). Raw Elo stays recorded.
"""

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from nfl_ats.data import PROJECT_ROOT, load_schedules
from nfl_ats.elo import ELO_PER_POINT
from nfl_ats.win_model import WIN_FEATURES, build_win_table, fit_fold, qb_adjusted_p

LEDGER_DIR = PROJECT_ROOT / "ledger"
EASTERN = ZoneInfo("America/New_York")
FIRST_SEASON = 2015  # Elo warm-up
MODEL_COLUMNS = {"p_elo": "raw Elo (baseline; the deliverable through Week 7)", "p_v2": "v2 (Elo + logistic)",
                 "p_elo_qb": "QB-adjusted Elo (the deliverable from Week 8)"}
QB_COLUMNS = ["home_qb", "away_qb", "home_qb_out", "away_qb_out", "qb_pts", "elo_spread_qb", "p_elo_qb"]
QB_PULL_COLUMNS = ["qb_spread_line_at_pull", "qb_p_market_at_pull", "qb_pulled_at_et", "qb_file"]  # the QB lock's own pull
MOVE_MIN_GAP = 1.0  # market-move check: only games where we and Vegas differed by 1+ point at lock (pre-registered)


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


def predict_week(schedules: pd.DataFrame, season: int, week: int, now: datetime, mode: str = "live",
                 qb: tuple[pd.DataFrame, int] | None = None) -> pd.DataFrame:
    """Predictions for every not-yet-started game in `week`, using only results from earlier weeks.

    mode="live": `now` is the real clock. mode="backfill": `now` is that week's first kickoff (pseudo-live).
    qb: (this week's qb.build_usual_flags rows, QB_PTS fit on earlier seasons) adds the QB-adjusted columns.
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

    out = pd.DataFrame({
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
    return out if qb is None else add_qb_columns(out, games.reset_index(drop=True), *qb)


def add_qb_columns(out: pd.DataFrame, games: pd.DataFrame, flags: pd.DataFrame, qb_pts: int) -> pd.DataFrame:
    """H4 columns: each side's No. 1 QB (name), whether he's out, QB_PTS, and the adjusted spread and probability."""
    out = out.copy()
    for side in ["home", "away"]:
        f = flags[["game_id", "team", "player", "qb_out"]].rename(
            columns={"team": f"{side}_team", "player": f"{side}_qb", "qb_out": f"{side}_qb_out"})
        out = out.merge(f, on=["game_id", f"{side}_team"], how="left", validate="one_to_one")
        out[f"{side}_qb_out"] = out[f"{side}_qb_out"].fillna(0).astype(int)
    adj = games[["game_id", "elo_diff"]].merge(out[["game_id", "home_qb_out", "away_qb_out"]], on="game_id")
    adj_diff = adj["elo_diff"] - qb_pts * adj["home_qb_out"] + qb_pts * adj["away_qb_out"]
    out["qb_pts"] = qb_pts
    out["elo_spread_qb"] = (adj_diff / ELO_PER_POINT).round(1).to_numpy()
    out["p_elo_qb"] = qb_adjusted_p(adj, qb_pts).round(4).to_numpy()
    return out


def ledger_path(season: int, week: int, mode: str, now: datetime) -> Path:
    if mode == "backfill":
        return LEDGER_DIR / "backfill" / f"{season}_wk{week:02d}_backfill.csv"
    if mode == "qb":
        return LEDGER_DIR / "qb" / f"{season}_wk{week:02d}_qb_{now.astimezone(EASTERN):%Y%m%dT%H%M%S}.csv"
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


def load_qb_ledger(season: int = 2026, ledger_dir: Path = LEDGER_DIR) -> pd.DataFrame:
    """The QB-adjusted lock per game: from Thursday live files and Saturday ledger/qb files, the LATEST pull counts.

    Every pull is before its game's kickoff by construction (predict_week skips started games).
    """
    files = sorted(ledger_dir.glob(f"{season}_wk*.csv")) + sorted((ledger_dir / "qb").glob(f"{season}_wk*.csv"))
    frames = [pd.read_csv(f).assign(qb_file=f.name) for f in files]
    frames = [f for f in frames if "p_elo_qb" in f.columns]
    if not frames:
        return pd.DataFrame(columns=["game_id", *QB_COLUMNS, *QB_PULL_COLUMNS])
    df = pd.concat(frames, ignore_index=True).rename(columns={
        "pulled_at_et": "qb_pulled_at_et", "spread_line_at_pull": "qb_spread_line_at_pull",
        "p_market_at_pull": "qb_p_market_at_pull"})
    df = df.sort_values("qb_pulled_at_et").drop_duplicates("game_id", keep="last")
    return df.reindex(columns=["game_id", *QB_COLUMNS, *QB_PULL_COLUMNS]).reset_index(drop=True)


def qb_for_week(schedules: pd.DataFrame, season: int, week: int, refresh: bool = True) -> tuple[pd.DataFrame, int]:
    """(this week's No. 1 QB flags as of now, QB_PTS fit on 2016 .. season-1) for predict_week(qb=...)."""
    from nfl_ats.qb import add_qb_feature, build_usual_flags
    from nfl_ats.win_model import fit_qb_points
    past = schedules[schedules["season"].between(FIRST_SEASON, season - 1)]
    table = build_win_table(past)
    table = add_qb_feature(table[table["season"] >= 2016], build_usual_flags(past, list(range(2016, season))))
    flags = build_usual_flags(schedules, [season], refresh=refresh)
    return flags[flags["week"] == week], fit_qb_points(table, test_season=season)


def attach_qb(ledger: pd.DataFrame, season: int = 2026, ledger_dir: Path = LEDGER_DIR) -> pd.DataFrame:
    """The live ledger with the QB-adjusted columns from load_qb_ledger (latest QB lock per game; NaN before Week 8)."""
    if ledger.empty:
        return ledger
    base = ledger.drop(columns=[c for c in QB_COLUMNS + QB_PULL_COLUMNS if c in ledger.columns])
    return base.merge(load_qb_ledger(season, ledger_dir), on="game_id", how="left", validate="one_to_one")


def check_final_injury_report(injuries: pd.DataFrame, season: int, week: int, teams: list[str]) -> None:
    """Saturday lock guard: refuse if most teams still lack game statuses for `week` (nflverse not updated yet)."""
    wk = injuries[(injuries["season"] == season) & (injuries["week"] == week) & injuries["report_status"].notna()]
    have = set(wk["team"]) & set(teams)
    if len(have) < len(teams) / 2:
        raise ValueError(f"Only {len(have)} of {len(teams)} teams have final injury statuses for week {week}. "
                         "The Friday report isn't in the data yet; try again later.")


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


def live_report(scored: pd.DataFrame, n_sims: int = 1000, seed: int = 0) -> dict:
    """The checks pre-registered in docs/project-rules.md for 2026, on decided games of a scored ledger.

    For raw Elo (p_elo) and v2 (p_v2): log loss / Brier / accuracy with bootstrap CIs; paired log loss vs.
    raw Elo and vs. the pull-time market; calibration-in-the-large with a CI; and Brier reliability
    (5 equal-count bins) vs. the 95th percentile for a perfectly calibrated model with the same predictions.
    """
    from nfl_ats.evaluate import bootstrap_ci, brier_decomposition, paired_diff, score

    d = scored[scored["home_win"].notna()].reset_index(drop=True)
    info = {"games_locked": len(scored), "games_decided": len(d), "ties": int(scored["is_tie"].sum()),
            "weeks": sorted(scored["week"].unique().tolist())}
    if len(d) < 10:
        return {**info, "note": "Too few decided games to score yet."}

    y = d["home_win"].to_numpy()
    market_ok = d["p_market_at_pull"].notna().to_numpy()
    rng = np.random.default_rng(seed)
    models = {}
    for col, label in {**MODEL_COLUMNS, "p_market_at_pull": "market at pull time"}.items():
        p = d[col].to_numpy(dtype=float) if col in d else np.full(len(d), np.nan)
        ok = ~np.isnan(p)
        if ok.sum() < 10:
            continue
        s = score(y[ok], p[ok])
        citl = bootstrap_ci(lambda yy, pp: pp.mean(axis=-1) - yy.mean(axis=-1), y[ok], p[ok])
        n_bins = 5
        rel = brier_decomposition(y[ok], p[ok], n_bins)["reliability"]
        sims = [brier_decomposition((rng.uniform(size=ok.sum()) < p[ok]).astype(float), p[ok], n_bins)["reliability"]
                for _ in range(n_sims)]
        null95 = float(np.quantile(sims, 0.95))
        models[label] = {
            "games": int(ok.sum()),
            "log_loss": tuple(s.loc["log_loss"].round(4)), "brier": tuple(s.loc["brier"].round(4)),
            "accuracy": tuple(s.loc["accuracy"].round(3)),
            "calibration_in_the_large_pts": tuple(round(v * 100, 1) for v in citl),
            "reliability": round(float(rel), 5), "reliability_null95": round(null95, 5),
            "consistent_with_calibration": bool(citl[1] <= 0 <= citl[2] and rel <= null95),
        }
    paired = {}
    for col, label in MODEL_COLUMNS.items():
        if label not in models:
            continue
        ok = d[col].notna().to_numpy()
        if col != "p_elo":
            paired[f"{label} − raw Elo"] = tuple(paired_diff(y[ok], d.loc[ok, "p_elo"], d.loc[ok, col]).loc["log_loss"].round(4))
        both = ok & market_ok
        paired[f"{label} − market"] = tuple(
            paired_diff(y[both], d.loc[both, "p_market_at_pull"], d.loc[both, col]).loc["log_loss"].round(4))
    return {**info, "models": models, "paired_log_loss (diff, ci_low, ci_high)": paired}


def market_moves(scored: pd.DataFrame, schedules: pd.DataFrame, use_qb: bool = True) -> pd.DataFrame:
    """Market-move check (pre-registered in docs/project-rules.md): did the line move from our lock toward our number?

    Per played game: O = our expected home margin, L = Vegas's at the time that pick was locked, C = the schedule's
    final line. d = O − L (kept only if |d| >= MOVE_MIN_GAP), m = C − L. toward = 1 / 0 for moved games, NaN if
    no move. dir_move = m·sign(d) points; clv_prob = the same idea in devigged moneyline probability.
    use_qb=False scores the raw-Elo Thursday lock (the baseline).
    """
    from nfl_ats.diagnostics import moneyline_prob
    close = schedules[["game_id", "spread_line", "home_moneyline", "away_moneyline"]].rename(
        columns={"spread_line": "close_line"})
    close["p_close"] = moneyline_prob(close["home_moneyline"], close["away_moneyline"])
    d = scored[scored["home_score"].notna()].merge(close[["game_id", "close_line", "p_close"]], on="game_id",
                                                   how="left", validate="one_to_one")
    qb = (d["p_elo_qb"].notna() if use_qb and "p_elo_qb" in d else pd.Series(False, index=d.index)).to_numpy()
    pick = lambda qb_col, col: np.where(qb, d[qb_col], d[col]) if qb.any() else d[col].to_numpy(dtype=float)
    out = pd.DataFrame({
        "game_id": d["game_id"], "week": d["week"], "qb_lock": qb,
        "ours": pick("elo_spread_qb", "elo_spread"), "lock_line": pick("qb_spread_line_at_pull", "spread_line_at_pull"),
        "close_line": d["close_line"], "p_ours": pick("p_elo_qb", "p_elo"),
        "p_lock": pick("qb_p_market_at_pull", "p_market_at_pull"), "p_close": d["p_close"],
    }).astype({c: float for c in ["ours", "lock_line", "close_line", "p_ours", "p_lock", "p_close"]})
    out["gap"] = out["ours"] - out["lock_line"]
    out["move"] = out["close_line"] - out["lock_line"]
    out = out[(out["gap"].abs() >= MOVE_MIN_GAP) & out["move"].notna()].copy()
    sign = np.sign(out["gap"])
    out["dir_move"] = out["move"] * sign
    out["toward"] = (out["dir_move"] > 0).astype(float).where(out["move"] != 0)
    out["clv_prob"] = (out["p_close"] - out["p_lock"]) * np.sign(out["p_ours"] - out["p_lock"])
    return out.reset_index(drop=True)


def market_move_summary(moves: pd.DataFrame) -> dict:
    """Counts, share toward us among moved games, and mean moves with bootstrap 95% CIs."""
    from nfl_ats.evaluate import bootstrap_ci
    moved = moves[moves["toward"].notna()]
    out = {"games": len(moves), "toward": int(moved["toward"].sum()), "away": int(len(moved) - moved["toward"].sum()),
           "no_move": int(moves["toward"].isna().sum())}
    if len(moved) >= 2:
        out["share_toward"] = bootstrap_ci(lambda y, p: y.mean(axis=-1), moved["toward"], moved["toward"])
    if len(moves) >= 2:
        out["mean_dir_move_pts"] = bootstrap_ci(lambda y, p: y.mean(axis=-1), moves["dir_move"], moves["dir_move"])
        clv = moves["clv_prob"].dropna()
        if len(clv) >= 2:
            out["mean_clv_prob_pts"] = tuple(100 * v for v in bootstrap_ci(lambda y, p: y.mean(axis=-1), clv, clv))
    return out
