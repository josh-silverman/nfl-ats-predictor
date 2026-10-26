"""Data for the "Josh's NFL Elo" static site (site/). Everything the page shows comes from here.

Plain-language rule (Josh): the site shows simple numbers. The detailed metrics stay in reports/ and the README.
Live and backfill weeks are kept separate, and only live (locked-before-kickoff) weeks count in the record.
"""

import subprocess
from datetime import datetime

import numpy as np
import pandas as pd

from nfl_ats.data import PROJECT_ROOT, regular_season
from nfl_ats.elo import HOME_FIELD, rating_history
from nfl_ats.evaluate import accuracy, bootstrap_ci
from nfl_ats.playoffs import season_odds
from nfl_ats.ledger import (EASTERN, LEDGER_DIR, attach_qb, kickoff_times, load_ledger, market_move_summary, market_moves,
                            score_ledger)

REPO_URL = "https://github.com/josh-silverman/nfl-ats-predictor"
BIG_GAP = 0.15  # flag games where we and Vegas differ by 15+ points of win chance

TEAMS = {  # abbreviation: (name, color). Colors are team primaries; the CSS adds a light ring so dark ones show.
    "ARI": ("Arizona Cardinals", "#97233F"), "ATL": ("Atlanta Falcons", "#A71930"),
    "BAL": ("Baltimore Ravens", "#241773"), "BUF": ("Buffalo Bills", "#00338D"),
    "CAR": ("Carolina Panthers", "#0085CA"), "CHI": ("Chicago Bears", "#C83803"),
    "CIN": ("Cincinnati Bengals", "#FB4F14"), "CLE": ("Cleveland Browns", "#FF3C00"),
    "DAL": ("Dallas Cowboys", "#003594"), "DEN": ("Denver Broncos", "#FB4F14"),
    "DET": ("Detroit Lions", "#0076B6"), "GB": ("Green Bay Packers", "#203731"),
    "HOU": ("Houston Texans", "#03202F"), "IND": ("Indianapolis Colts", "#002C5F"),
    "JAX": ("Jacksonville Jaguars", "#006778"), "KC": ("Kansas City Chiefs", "#E31837"),
    "LA": ("Los Angeles Rams", "#003594"), "LAC": ("Los Angeles Chargers", "#0080C6"),
    "LV": ("Las Vegas Raiders", "#A5ACAF"), "MIA": ("Miami Dolphins", "#008E97"),
    "MIN": ("Minnesota Vikings", "#4F2683"), "NE": ("New England Patriots", "#002244"),
    "NO": ("New Orleans Saints", "#D3BC8D"), "NYG": ("New York Giants", "#0B2265"),
    "NYJ": ("New York Jets", "#125740"), "PHI": ("Philadelphia Eagles", "#004C54"),
    "PIT": ("Pittsburgh Steelers", "#FFB612"), "SEA": ("Seattle Seahawks", "#69BE28"),
    "SF": ("San Francisco 49ers", "#AA0000"), "TB": ("Tampa Bay Buccaneers", "#D50A0A"),
    "TEN": ("Tennessee Titans", "#4B92DB"), "WAS": ("Washington Commanders", "#5A1414"),
}


def _num(x):
    """JSON-safe number: NaN/None -> None, numpy -> Python."""
    return None if x is None or (isinstance(x, float) and np.isnan(x)) or pd.isna(x) else float(x)


def first_commit(path) -> tuple[str, datetime] | None:
    """The commit that first added `path` (hash, commit time), or None if it isn't committed yet."""
    rel = path.relative_to(PROJECT_ROOT)
    out = subprocess.run(["git", "log", "--diff-filter=A", "--format=%H %cI", "--", str(rel)],
                         cwd=PROJECT_ROOT, capture_output=True, text=True).stdout.strip().splitlines()
    if not out:
        return None
    sha, when = out[-1].split(" ", 1)  # git log is newest-first, so the last line is the first add
    return sha, datetime.fromisoformat(when)


def lock_status(commit: tuple[str, datetime] | None, first_kickoff: pd.Timestamp) -> dict:
    """What the lock badge says for a live week."""
    if commit is None:
        return {"locked": False, "text": "Not committed yet", "url": None, "before_kickoff": False}
    sha, when = commit
    before = pd.Timestamp(when) < first_kickoff
    stamp = pd.Timestamp(when).tz_convert(EASTERN).strftime("%a %b %-d, %-I:%M %p ET")
    return {"locked": True, "before_kickoff": bool(before), "url": f"{REPO_URL}/commit/{sha}",
            "text": f"Locked {stamp}" + (", before kickoff" if before else ", AFTER first kickoff")}


def headline(df: pd.DataFrame) -> pd.DataFrame:
    """Add the pick the site shows: QB-adjusted Elo where it was locked (Week 8 on, decision #21), else raw Elo."""
    out = df.copy()
    has_qb = "p_elo_qb" in out.columns
    out["p_pick"] = out["p_elo_qb"].fillna(out["p_elo"]) if has_qb else out["p_elo"]
    if "elo_spread" in out.columns:
        out["pick_spread"] = out["elo_spread_qb"].fillna(out["elo_spread"]) if has_qb else out["elo_spread"]
    out["qb_adjusted"] = out["p_elo_qb"].notna() if has_qb else False
    return out


def qb_lock_status(season: int, week: int) -> dict | None:
    """Badge for the week's latest Saturday QB lock (ledger/qb/), judged against the first game in that file."""
    files = sorted((LEDGER_DIR / "qb").glob(f"{season}_wk{week:02d}_*.csv"))
    if not files:
        return None
    first_kick = pd.to_datetime(pd.read_csv(files[-1])["kickoff_et"]).min().tz_localize(EASTERN)
    lock = lock_status(first_commit(files[-1]), first_kick)
    lock["text"] = lock["text"].replace("Locked", "QB update locked", 1)
    return lock


def _earliest_files(mode: str, season: int) -> dict[int, object]:
    folder = LEDGER_DIR / "backfill" if mode == "backfill" else LEDGER_DIR
    files = {}
    for f in sorted(folder.glob(f"{season}_wk*.csv")):  # names sort by time within a week
        files.setdefault(int(f.name.split("_wk")[1][:2]), f)
    return files


MINUS = "\u2212"


def format_line(home_margin, home: str, away: str) -> str | None:
    """A spread in the usual 'FAVORITE −x' form. Input is the expected HOME margin (positive = home favored)."""
    if home_margin is None or pd.isna(home_margin):
        return None
    m = float(home_margin)
    if abs(m) < 0.05:
        return "Pick'em"
    return f"{home if m > 0 else away} {MINUS}{abs(m):.1f}"


def confidence(p_home: float) -> str:
    """Plain label for how sure we are about the favorite."""
    c = max(p_home, 1 - p_home)
    return "Toss-up" if c < 0.55 else "Lean" if c < 0.65 else "Favored" if c < 0.75 else "High confidence"


def day_group(kick: pd.Timestamp) -> str:
    """Broadcast-style slot from the Eastern kickoff time."""
    day, hour = kick.day_name(), kick.hour
    if day == "Sunday":
        return "International" if hour < 12 else "Sunday early" if hour < 15 else "Sunday late" if hour < 19 else "Sunday night"
    if hour < 12:
        return "International"
    return {"Thursday": "Thursday night", "Monday": "Monday night"}.get(day, day)


def team_context(reg: pd.DataFrame, history: pd.DataFrame, season: int, week: int) -> dict:
    """Each team's rank, record, and last game using ONLY weeks before `week` (what was known at lock)."""
    h = history[(history["season"] == season) & (history["week"] < week)]
    snap = h[h["week"] == h["week"].max()].set_index("team")["rating"] if not h.empty else pd.Series(dtype=float)
    rank = snap.rank(ascending=False, method="first").astype(int).to_dict()

    past = reg[(reg["season"] == season) & (reg["week"] < week) & reg["home_score"].notna()].sort_values("week")
    ctx = {t: {"rank": rank.get(t), "w": 0, "l": 0, "t": 0, "last": None} for t in TEAMS}
    for g in past.itertuples(index=False):
        for team, opp, pf, pa, at_home in [(g.home_team, g.away_team, g.home_score, g.away_score, True),
                                           (g.away_team, g.home_team, g.away_score, g.home_score, False)]:
            c = ctx.setdefault(team, {"rank": rank.get(team), "w": 0, "l": 0, "t": 0, "last": None})
            res = "W" if pf > pa else "L" if pf < pa else "T"
            c[{"W": "w", "L": "l", "T": "t"}[res]] += 1
            c["last"] = f"{res} {int(pf)}{chr(8211)}{int(pa)} {'vs' if at_home else '@'} {opp}"
    return {t: {"rank": c["rank"], "record": f"{c['w']}-{c['l']}" + (f"-{c['t']}" if c["t"] else ""),
                "last": c["last"]} for t, c in ctx.items()}


REST_EDGE_DAYS = 3  # show the rest gauge only when the teams' rest differs by this many days or more


def _rest(days) -> dict | None:
    """{"home": d, "away": d} for the rest gauge when the gap is REST_EDGE_DAYS+, else None."""
    if days is None or any(pd.isna(d) for d in days):
        return None
    home, away = int(days[0]), int(days[1])
    return {"home": home, "away": away} if abs(home - away) >= REST_EDGE_DAYS else None


def qb_notes(flags: pd.DataFrame | None) -> dict:
    """{game_id: [{team, player, reason}]} for games where a team's usual starting QB is out (display only)."""
    if flags is None or flags.empty:
        return {}
    out = flags[flags["qb_out"] == 1]
    return {gid: [{"team": r.team, "player": None if pd.isna(r.player) else r.player, "reason": r.reason}
                  for r in g.itertuples(index=False)] for gid, g in out.groupby("game_id")}


def game_cards(scored: pd.DataFrame, context: dict | None = None, closing: dict | None = None,
               now: pd.Timestamp | None = None, notes: dict | None = None, rest: dict | None = None) -> list[dict]:
    """One plain dict per game for the cards.

    context: team_context() for this week (as of lock). closing: {game_id: current schedule spread_line};
    the closing line is only shown once the game has kicked off or has a result. notes: qb_notes() (news, not
    a prediction input: the locked probability never changes). rest: {game_id: (home_rest, away_rest)} days since
    each team's last game, from the schedule (display only: H5 found rest doesn't improve the model).
    """
    context, closing, notes, rest = context or {}, closing or {}, notes or {}, rest or {}
    cards = []
    for g in headline(scored).sort_values(["kickoff_et", "game_id"]).itertuples(index=False):
        p_home, p_mkt = float(g.p_pick), _num(g.p_market_at_pull)
        qb_teams = [t for t, side in [(g.home_team, "home"), (g.away_team, "away")]
                    if g.qb_adjusted and getattr(g, f"{side}_qb_out", 0) == 1]
        qb_pts = int(g.qb_pts) if qb_teams else 0
        fav = g.home_team if p_home >= 0.5 else g.away_team
        result = None
        if pd.notna(g.home_score):
            winner = g.home_team if g.home_score > g.away_score else g.away_team if g.away_score > g.home_score else None
            result = {"home_score": int(g.home_score), "away_score": int(g.away_score), "winner": winner,
                      "we_were_right": None if winner is None else bool(fav == winner),
                      "vegas_was_right": None if winner is None or p_mkt is None
                      else bool((g.home_team if p_mkt >= 0.5 else g.away_team) == winner)}
        kick = pd.Timestamp(g.kickoff_et).tz_localize(EASTERN)
        started = result is not None or (now is not None and kick <= now)
        hfa = 0 if g.location == "Neutral" else HOME_FIELD
        side = lambda t: {"team": t, **context.get(t, {"rank": None, "record": None, "last": None})}
        cards.append({
            "game_id": g.game_id, "home": g.home_team, "away": g.away_team, "neutral": g.location == "Neutral",
            "kickoff": kick.strftime("%a %b %-d · %-I:%M %p ET"), "day_group": day_group(kick),
            "p_home": round(p_home, 3), "p_vegas_home": None if p_mkt is None else round(p_mkt, 3),
            "favorite": fav, "confidence": confidence(p_home),
            "our_line": format_line(g.pick_spread, g.home_team, g.away_team),
            "expected_margin": None if pd.isna(g.pick_spread) else round(abs(float(g.pick_spread)), 1),
            "vegas_line_at_lock": format_line(_num(getattr(g, "spread_line_at_pull", np.nan)), g.home_team, g.away_team),
            "closing_line": format_line(closing.get(g.game_id), g.home_team, g.away_team) if started else None,
            "big_gap": p_mkt is not None and abs(p_home - p_mkt) >= BIG_GAP,
            "home_side": side(g.home_team), "away_side": side(g.away_team),
            "why": {"home_rating": round(float(g.home_elo)), "away_rating": round(float(g.away_elo)), "home_field": hfa,
                    "qb_pts": qb_pts, "qb_teams": qb_teams,
                    "edge": round(float(g.home_elo) + hfa - float(g.away_elo)
                                  - qb_pts * (g.home_team in qb_teams) + qb_pts * (g.away_team in qb_teams))},
            "result": result,
            "qb_adjusted": bool(g.qb_adjusted),
            "rest": _rest(rest.get(g.game_id)),
            "qb_out": [{**q, "in_pick": q["team"] in qb_teams} for q in notes.get(g.game_id, [])],
        })
    return cards


def move_line(moves: dict | None) -> str | None:
    """One plain sentence for the pre-registered market-move check (market_move_summary), once there are 5+ games."""
    if not moves or moves["games"] < 5:
        return None
    moved = moves["toward"] + moves["away"]
    return (f"When we disagreed with Vegas by a point or more, the betting line later moved our way in {moves['toward']} "
            f"of {moved} games where it moved ({moves['no_move']} didn't move). About half would mean no edge.")


def record_summary(scored_live: pd.DataFrame, moves: dict | None = None) -> dict:
    """W-L for us and Vegas on live weeks only, week by week, plus up to four plain extra lines."""
    d = headline(scored_live[scored_live["home_win"].notna()])
    if d.empty:
        return {"games": 0, "by_week": [], "extra": []}
    d["ours"] = ((d["p_pick"] >= 0.5) == (d["home_win"] == 1)).astype(int)
    has_mkt = d["p_market_at_pull"].notna()
    d["vegas"] = ((d["p_market_at_pull"] >= 0.5) == (d["home_win"] == 1)).astype(float).where(has_mkt)
    weekly = d.groupby("week").agg(games=("ours", "size"), ours=("ours", "sum"), vegas=("vegas", "sum"),
                                   vegas_games=("vegas", "count")).cumsum()
    by_week = [{"week": int(w), "games": int(r.games), "ours": int(r.ours), "vegas": int(r.vegas),
                "vegas_games": int(r.vegas_games)} for w, r in weekly.iterrows()]
    extra = [line] if (line := move_line(moves)) else []
    if len(d) >= 10:
        _, lo, hi = bootstrap_ci(accuracy, d["home_win"].to_numpy(), (d["p_pick"] >= 0.5).astype(float).to_numpy())
        extra.append(f"Honest range for our hit rate so far: {lo:.0%} to {hi:.0%} (small samples swing a lot).")
        conf = np.maximum(d["p_pick"], 1 - d["p_pick"])
        sure = conf >= 0.6
        if sure.sum() >= 5:
            expected = conf[sure].mean()
            actual = d.loc[sure, "ours"].mean()
            extra.append(f"When we gave a team 60%+ to win, we expected about {expected:.0%} of them to win. "
                         f"{actual:.0%} did ({int(d.loc[sure, 'ours'].sum())} of {int(sure.sum())}).")
    extra.append("Full details: reports/live_2026.md on GitHub.")
    return {"games": int(len(d)), "by_week": by_week, "extra": extra[:4]}


def rankings(history: pd.DataFrame, schedules: pd.DataFrame, current_cards: list[dict], season: int,
             odds: pd.DataFrame | None = None) -> list[dict]:
    """All teams by current rating, with change since the previous week, W-L-T, next game from this week's locks,
    and playoff odds (playoffs.season_odds) when given."""
    h = history[history["season"] == season]
    weeks = sorted(h["week"].unique())
    now = h[h["week"] == weeks[-1]].set_index("team")["rating"]
    prev = h[h["week"] == (weeks[-2] if len(weeks) > 1 else weeks[-1])].set_index("team")["rating"]
    rank_now, rank_prev = now.rank(ascending=False, method="first"), prev.rank(ascending=False, method="first")

    played = schedules[(schedules["season"] == season) & (schedules["game_type"] == "REG") & schedules["home_score"].notna()]
    rec = {t: [0, 0, 0] for t in now.index}
    for g in played.itertuples(index=False):
        if g.home_score == g.away_score:
            rec[g.home_team][2] += 1; rec[g.away_team][2] += 1
        else:
            w, l = (g.home_team, g.away_team) if g.home_score > g.away_score else (g.away_team, g.home_team)
            rec[w][0] += 1; rec[l][1] += 1

    upcoming = {}
    for c in current_cards:
        if c["result"] is None:
            upcoming[c["home"]] = {"opp": c["away"], "home": True, "p_win": c["p_home"]}
            upcoming[c["away"]] = {"opp": c["home"], "home": False, "p_win": round(1 - c["p_home"], 3)}
    playing = {t for c in current_cards for t in (c["home"], c["away"])}
    odds = odds.set_index("team") if odds is not None else None
    out = []
    for team in rank_now.sort_values().index:
        w, l, t = rec.get(team, [0, 0, 0])
        out.append({"rank": int(rank_now[team]), "team": team, "rating": round(float(now[team]), 1),
                    "rank_change": int(rank_prev.get(team, rank_now[team]) - rank_now[team]),
                    "record": f"{w}-{l}" + (f"-{t}" if t else ""), "next": upcoming.get(team),
                    "bye": bool(current_cards) and team not in playing,
                    "odds": None if odds is None or team not in odds.index else {
                        "wins": round(float(odds.loc[team, "wins"]), 1), "losses": round(float(odds.loc[team, "losses"]), 1),
                        "playoff": round(float(odds.loc[team, "playoff"]), 3),
                        "division": round(float(odds.loc[team, "division"]), 3),
                        "top_seed": round(float(odds.loc[team, "top_seed"]), 3)}})
    return out


def worked_example(scored: pd.DataFrame, history: pd.DataFrame) -> dict | None:
    """One decided game, step by step, using the numbers locked before it."""
    d = headline(scored[scored["home_win"].notna()])
    d = d[~d["qb_adjusted"] | (d["p_pick"] == d["p_elo"])]  # keep the plain rating example: no QB adjustment
    if d.empty:
        return None
    g = d.sort_values(["week", "p_elo"], ascending=[False, False]).iloc[0]  # latest week's most lopsided call
    hfa = 0 if g["location"] == "Neutral" else HOME_FIELD
    after = history[(history["season"] == g["season"]) & (history["week"] == g["week"])].set_index("team")["rating"]
    return {"week": int(g["week"]), "home": g["home_team"], "away": g["away_team"],
            "home_rating": round(float(g["home_elo"])), "away_rating": round(float(g["away_elo"])),
            "home_field": hfa, "gap": round(float(g["home_elo"]) + hfa - float(g["away_elo"])),
            "p_home": round(float(g["p_elo"]), 3), "home_score": int(g["home_score"]), "away_score": int(g["away_score"]),
            "home_after": round(float(after.get(g["home_team"], np.nan))),
            "away_after": round(float(after.get(g["away_team"], np.nan)))}


def build_site_data(schedules: pd.DataFrame, season: int = 2026, backtest: dict | None = None,
                    qb_flags: pd.DataFrame | None = None) -> dict:
    """Everything the site needs, as plain JSON-ready Python. qb_flags: qb.build_usual_flags (optional)."""
    reg = regular_season(schedules)  # maps OAK/SD/STL to LV/LAC/LA, same as the model
    history = rating_history(reg[reg["season"].between(2015, season)])
    first_kick = kickoff_times(reg[reg["season"] == season]).groupby(reg.loc[reg["season"] == season, "week"]).min()

    closing = reg.loc[reg["season"] == season].set_index("game_id")["spread_line"].to_dict()
    this_season = reg.loc[reg["season"] == season]
    rest = ({gid: (h, a) for gid, h, a in zip(this_season["game_id"], this_season["home_rest"], this_season["away_rest"])}
            if {"home_rest", "away_rest"} <= set(reg.columns) else {})
    now = pd.Timestamp.now(tz=EASTERN)
    notes = qb_notes(qb_flags)
    weeks, scored_by_mode = [], {}
    for mode in ["backfill", "live"]:
        ledger = load_ledger(mode, season)
        if mode == "live":
            ledger = attach_qb(ledger, season)
        scored_by_mode[mode] = score_ledger(ledger, schedules) if not ledger.empty else pd.DataFrame()
        for week, path in _earliest_files(mode, season).items():
            s = scored_by_mode[mode]
            weeks.append({"week": week, "mode": mode,
                          "lock": lock_status(first_commit(path), first_kick[week]) if mode == "live" else None,
                          "qb_lock": qb_lock_status(season, week) if mode == "live" else None,
                          "games": game_cards(s[s["week"] == week], team_context(reg, history, season, week),
                                              closing, now, notes, rest)})
    weeks.sort(key=lambda w: w["week"])
    live_weeks = [w for w in weeks if w["mode"] == "live"]
    current = (live_weeks or weeks)[-1] if weeks else None

    live = scored_by_mode.get("live", pd.DataFrame())
    any_scored = pd.concat([x for x in scored_by_mode.values() if not x.empty]) if weeks else pd.DataFrame()
    hist_season = history[history["season"] == season]
    return {
        "meta": {"title": "Josh's NFL Elo", "season": season, "repo": REPO_URL,
                 "generated": datetime.now(EASTERN).strftime("%b %-d, %Y %-I:%M %p ET"),
                 "current_week": current["week"] if current else None},
        "teams": {t: {"name": n, "color": c} for t, (n, c) in TEAMS.items()},
        "weeks": weeks,
        "record": record_summary(live, market_move_summary(market_moves(live, schedules)))
                  if not live.empty else {"games": 0, "by_week": [], "extra": []},
        "rankings": rankings(history, reg.assign(game_type="REG"), current["games"] if current else [], season,
                             season_odds(schedules, season, {c["game_id"]: c["p_home"] for c in (current["games"] if current else [])
                                                             if c["result"] is None})),
        "elo_history": {t: [[int(w), round(float(r), 1)] for w, r in zip(g["week"], g["rating"])]
                        for t, g in hist_season.groupby("team")},
        "backtest": backtest or {},
        "example": worked_example(any_scored, history) if not any_scored.empty else None,
    }


def backtest_summary(schedules: pd.DataFrame) -> dict:
    """Pick-the-winner rates 2019–2025 for the one-line summary (same numbers as experiment W3)."""
    from nfl_ats.evaluate import TEST_SEASONS
    from nfl_ats.win_model import TARGET, benchmark_predictions, build_win_table
    b = benchmark_predictions(build_win_table(schedules[schedules["season"] <= 2025]), TEST_SEASONS)
    y = b["elo"][TARGET].to_numpy()
    hit = lambda p: float(((p.to_numpy() >= 0.5) == (y == 1)).mean())
    return {"games": int(len(y)), "elo": round(hit(b["elo"]["p"]), 3), "vegas": round(hit(b["market"]["p"]), 3),
            "always_home": round(float((y == 1).mean()), 3)}
