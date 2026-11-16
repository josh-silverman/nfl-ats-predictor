"""Experiment H2: is the expected starting QB ruled out before kickoff? (Pre-registered in reports/experiments.md.)

As-of rules:
- The EXPECTED starter for a game is whoever threw the most passes in the team's PREVIOUS game that season
  (shift(1) within team-season), so the game's own participation never leaks in.
- "Out" comes from the official injury report for that game week. Statuses are finalized before kickoff
  (audit: 100% in 2016–2024, median ~51 h before). Never use the schedule's *_qb_id (who actually started).
"""

import numpy as np
import pandas as pd

from nfl_ats.data import TEAM_MAP, load_injuries, load_pbp

OUT_STATUSES = {"Out", "Doubtful"}


def game_starters(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per regular-season team-game: the QB with the most pass attempts (ties: first listed)."""
    passes = pbp.loc[(pbp["season_type"] == "REG") & (pbp["pass_attempt"] == 1) & pbp["passer_player_id"].notna(),
                     ["season", "week", "game_id", "posteam", "passer_player_id"]]
    counts = passes.groupby(["season", "week", "game_id", "posteam", "passer_player_id"]).size().rename("attempts")
    top = counts.reset_index().sort_values("attempts", ascending=False).drop_duplicates(["game_id", "posteam"])
    top = top.rename(columns={"posteam": "team", "passer_player_id": "starter_id"})
    top["team"] = top["team"].replace(TEAM_MAP)
    return top[["season", "week", "game_id", "team", "starter_id"]].reset_index(drop=True)


def load_game_starters(seasons: list[int]) -> pd.DataFrame:
    """game_starters for several seasons, one season of pbp at a time."""
    return pd.concat([game_starters(load_pbp([s])) for s in seasons], ignore_index=True)


def expected_starters(starters: pd.DataFrame) -> pd.DataFrame:
    """Each team-game's expected starter = last game's starter (shift(1) within team-season). Week 1 → NaN."""
    s = starters.sort_values(["season", "team", "week"]).copy()
    s["expected_id"] = s.groupby(["season", "team"])["starter_id"].shift(1)
    return s[["season", "week", "game_id", "team", "expected_id"]]


def qb_out_flags(expected: pd.DataFrame, injuries: pd.DataFrame) -> pd.DataFrame:
    """qb_out = 1 if the expected starter is listed Out/Doubtful on that team's report for that week."""
    inj = injuries.loc[(injuries["game_type"] == "REG") & injuries["report_status"].isin(OUT_STATUSES),
                       ["season", "week", "team", "gsis_id"]].copy()
    inj["team"] = inj["team"].replace(TEAM_MAP)
    inj[["season", "week"]] = inj[["season", "week"]].astype(int)
    inj = inj.drop_duplicates().assign(qb_out=1)
    out = expected.merge(inj, left_on=["season", "week", "team", "expected_id"],
                         right_on=["season", "week", "team", "gsis_id"], how="left", validate="many_to_one")
    out["qb_out"] = out["qb_out"].fillna(0).astype(int)
    return out[["season", "week", "game_id", "team", "expected_id", "qb_out"]]


def add_qb_feature(table: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    """Add home_qb_out, away_qb_out, and qb_out_diff to a win table. Games with no flag row get 0."""
    out = table.copy()
    for side in ["home", "away"]:
        f = flags[["game_id", "team", "qb_out"]].rename(columns={"team": f"{side}_team", "qb_out": f"{side}_qb_out"})
        out = out.merge(f, on=["game_id", f"{side}_team"], how="left", validate="one_to_one")
        out[f"{side}_qb_out"] = out[f"{side}_qb_out"].fillna(0).astype(int)
    out["qb_out_diff"] = out["home_qb_out"] - out["away_qb_out"]
    return out


def build_qb_flags(seasons: list[int]) -> pd.DataFrame:
    """End to end: pbp starters → expected starters → injury-report flags, for `seasons`."""
    return qb_out_flags(expected_starters(load_game_starters(seasons)), load_injuries(seasons))


# ---------- H4: the team's USUAL starter, and whether he is unavailable (injury report or reserve list) ----------

USUAL_WINDOW = 8  # the No. 1 QB = most starts in the team's previous 8 regular-season games (crossing seasons)


def usual_starters(starters: pd.DataFrame, games: pd.DataFrame, window: int = USUAL_WINDOW) -> pd.DataFrame:
    """Each team-game's No. 1 QB, from games strictly BEFORE it (so unplayed games work too).

    starters: game_starters output (played games). games: season, week, game_id, team (may include unplayed).
    Ties in the start count go to the most recent starter.
    """
    s = starters.sort_values(["season", "week"])
    rows = []
    for g in games.itertuples(index=False):
        prior = s[(s["team"] == g.team) & ((s["season"] < g.season) | ((s["season"] == g.season) & (s["week"] < g.week)))]
        recent = prior["starter_id"].tail(window).tolist()
        usual = None
        if recent:
            counts = pd.Series(recent).value_counts()
            tied = set(counts[counts == counts.max()].index)
            usual = next(q for q in reversed(recent) if q in tied)
        rows.append({"season": g.season, "week": g.week, "game_id": g.game_id, "team": g.team, "usual_id": usual})
    return pd.DataFrame(rows)


def qb_unavailable(usual: pd.DataFrame, injuries: pd.DataFrame, rosters: pd.DataFrame) -> pd.DataFrame:
    """qb_out = 1 if the No. 1 QB is Out/Doubtful on that week's injury report OR on the team's reserve list.

    Adds `reason` ("injured reserve", "reserve list", "out", "doubtful") and `player` (full name) for display.

    Reserve list: the game week's roster, or, if nflverse hasn't published that team's roster for the week yet
    (it lands Tue–Thu, so a Wednesday lock can miss it), the team's latest EARLIER roster week in the same season
    (decision #27). `roster_week` records which week was used (NaN = no roster yet this season). Historical
    weeks always have their own roster, so the backtest is unchanged.
    """
    inj = injuries.loc[(injuries["game_type"] == "REG") & injuries["report_status"].isin(OUT_STATUSES),
                       ["season", "week", "team", "gsis_id", "report_status"]].copy()
    inj["reason_inj"] = inj.pop("report_status").str.lower()
    res = rosters.loc[rosters["status"] == "RES", ["season", "week", "team", "gsis_id", "status_description_abbr"]].copy()
    res["reason_res"] = (res.pop("status_description_abbr") == "R01").map({True: "injured reserve", False: "reserve list"})
    names = rosters.drop_duplicates(["season", "gsis_id"])[["season", "gsis_id", "full_name"]]
    for src in (inj, res):
        src["team"] = src["team"].replace(TEAM_MAP)
        src[["season", "week"]] = src[["season", "week"]].astype(int)

    # Which roster week each game uses: the latest published week <= the game's week, per team-season.
    published = rosters[["season", "week", "team"]].drop_duplicates().copy()
    published["team"] = published["team"].replace(TEAM_MAP)
    published = published.astype({"season": int, "week": int}).rename(columns={"week": "roster_week"})
    out = usual.copy().astype({"season": int, "week": int})
    out["_order"] = range(len(out))
    out = pd.merge_asof(out.sort_values("week"), published.sort_values("roster_week"), left_on="week",
                        right_on="roster_week", by=["season", "team"], direction="backward")
    out = out.sort_values("_order").drop(columns="_order").reset_index(drop=True)

    out = out.merge(inj.drop_duplicates(["season", "week", "team", "gsis_id"]),
                    left_on=["season", "week", "team", "usual_id"], right_on=["season", "week", "team", "gsis_id"],
                    how="left", validate="many_to_one").drop(columns="gsis_id")
    out = out.merge(res.drop_duplicates(["season", "week", "team", "gsis_id"]).rename(columns={"week": "roster_week"}),
                    left_on=["season", "roster_week", "team", "usual_id"],
                    right_on=["season", "roster_week", "team", "gsis_id"],
                    how="left", validate="many_to_one").drop(columns="gsis_id")
    out["reason"] = out["reason_res"].fillna(out["reason_inj"])
    out["qb_out"] = out["reason"].notna().astype(int)
    out = out.merge(names, left_on=["season", "usual_id"], right_on=["season", "gsis_id"], how="left").drop(columns="gsis_id")
    out = out.rename(columns={"full_name": "player"})
    return out[["season", "week", "game_id", "team", "usual_id", "player", "qb_out", "reason", "roster_week"]]


# ---------- H8: the usual starter keeps his title while he's out ----------

def _unavailable_lookup(games: pd.DataFrame, injuries: pd.DataFrame, rosters: pd.DataFrame):
    """A function (season, week, team, qb_id) → reason or None, matching qb_unavailable exactly: reserve list on the
    game week's roster (or the latest earlier roster week, decision #27) first, then Out/Doubtful on the injury report."""
    inj = injuries.loc[(injuries["game_type"] == "REG") & injuries["report_status"].isin(OUT_STATUSES),
                       ["season", "week", "team", "gsis_id", "report_status"]].copy()
    inj["team"] = inj["team"].replace(TEAM_MAP)
    inj_map = {(int(s), int(w), t, g): r.lower() for s, w, t, g, r in inj.drop_duplicates(["season", "week", "team", "gsis_id"]).itertuples(index=False)}
    res = rosters.loc[rosters["status"] == "RES", ["season", "week", "team", "gsis_id", "status_description_abbr"]].copy()
    res["team"] = res["team"].replace(TEAM_MAP)
    res_map = {(int(s), int(w), t, g): ("injured reserve" if a == "R01" else "reserve list")
               for s, w, t, g, a in res.drop_duplicates(["season", "week", "team", "gsis_id"]).itertuples(index=False)}
    published = rosters[["season", "week", "team"]].drop_duplicates().copy()
    published["team"] = published["team"].replace(TEAM_MAP)
    published = published.astype({"season": int, "week": int}).rename(columns={"week": "roster_week"})
    g = games[["season", "week", "team"]].drop_duplicates().astype({"season": int, "week": int})
    g = pd.merge_asof(g.sort_values("week"), published.sort_values("roster_week"), left_on="week", right_on="roster_week",
                      by=["season", "team"], direction="backward")
    roster_week = {(s, w, t): rw for s, w, t, rw in g[["season", "week", "team", "roster_week"]].itertuples(index=False)}

    def lookup(season, week, team, qb):
        rw = roster_week.get((season, week, team))
        hit = res_map.get((season, int(rw), team, qb)) if pd.notna(rw) else None
        return hit or inj_map.get((season, week, team, qb)), rw
    return lookup


def sequential_usual_flags(starters: pd.DataFrame, games: pd.DataFrame, injuries: pd.DataFrame, rosters: pd.DataFrame,
                           sticky: bool = True, window: int = USUAL_WINDOW, warmup_seasons: set[int] | None = None) -> pd.DataFrame:
    """H8 flags, team by team in game order. The No. 1 QB = most starts among the team's previous `window` COUNTED
    games (ties → most recent). sticky=True: a start counts only if the team was not flagged that game, so a fill-in
    never inherits the title while the usual starter is out. sticky=False reproduces H4 (every start counts).

    games: season, week, game_id, team for every game to walk through (include the warm-up season first).
    warmup_seasons: seasons whose games just build history (every start counts, no flags returned).
    Returns qb_unavailable's columns for the non-warm-up games.
    """
    warmup_seasons = warmup_seasons or set()
    lookup = _unavailable_lookup(games, injuries, rosters)
    starter_of = dict(zip(zip(starters["game_id"], starters["team"]), starters["starter_id"]))
    rows = []
    for team, tg in games.sort_values(["season", "week"]).groupby("team", sort=False):
        history: list[str] = []
        for g in tg.itertuples(index=False):
            recent = history[-window:]
            usual = None
            if recent:
                counts = pd.Series(recent).value_counts()
                tied = set(counts[counts == counts.max()].index)
                usual = next(q for q in reversed(recent) if q in tied)
            reason, rw = lookup(int(g.season), int(g.week), team, usual) if usual is not None else (None, np.nan)
            warm = g.season in warmup_seasons
            if not warm:
                rows.append({"season": g.season, "week": g.week, "game_id": g.game_id, "team": team, "usual_id": usual,
                             "qb_out": int(reason is not None), "reason": reason, "roster_week": rw})
            started = starter_of.get((g.game_id, team))
            if started is not None and (warm or not sticky or reason is None):
                history.append(started)
    out = pd.DataFrame(rows)
    names = rosters.drop_duplicates(["season", "gsis_id"])[["season", "gsis_id", "full_name"]]
    out = out.merge(names, left_on=["season", "usual_id"], right_on=["season", "gsis_id"], how="left").drop(columns="gsis_id")
    return out.rename(columns={"full_name": "player"})[
        ["season", "week", "game_id", "team", "usual_id", "player", "qb_out", "reason", "roster_week"]]


# ---------- H7: how good is the missing QB? EPA per dropback over his recent games, as of each game ----------

VALUE_WINDOW = 20   # his previous 20 regular-season games with a dropback (crossing seasons)
VALUE_SHRINK = 200  # dropbacks of shrinkage toward 0, so short histories count for little


def dropback_games(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per QB per regular-season game: summed qb_epa and dropbacks. The QB is the passer, else the
    rusher (scrambles)."""
    d = pbp.loc[(pbp["season_type"] == "REG") & (pbp["qb_dropback"] == 1) & pbp["qb_epa"].notna(),
                ["season", "week", "game_id", "passer_player_id", "rusher_player_id", "qb_epa"]]
    d = d.assign(qb_id=d["passer_player_id"].fillna(d["rusher_player_id"])).dropna(subset=["qb_id"])
    return (d.groupby(["season", "week", "game_id", "qb_id"])["qb_epa"].agg(epa="sum", dropbacks="size")
            .reset_index())


def qb_values(db_games: pd.DataFrame, targets: pd.DataFrame, window: int = VALUE_WINDOW,
              shrink: int = VALUE_SHRINK) -> pd.Series:
    """Each target's QB value from games STRICTLY before it: sum(epa) / (dropbacks + shrink) over his last
    `window` games. targets: season, week, qb_id (qb_id may be missing). No history → 0. Index-aligned to targets."""
    g = db_games.sort_values(["qb_id", "season", "week"]).copy()
    roll = g.groupby("qb_id")[["epa", "dropbacks"]].rolling(window, min_periods=1).sum().reset_index(level=0, drop=True)
    g[["epa_w", "db_w"]] = roll.loc[g.index].to_numpy()          # through this game, used only for LATER games
    g["key"] = (g["season"] * 100 + g["week"]).astype("int64")  # pbp ints are int32; schedules are int64
    t = targets[["season", "week", "qb_id"]].copy()
    t["key"] = (t["season"] * 100 + t["week"]).astype("int64")
    t["_row"] = range(len(t))
    has = t["qb_id"].notna()
    m = pd.merge_asof(t[has].sort_values("key"), g[["qb_id", "key", "epa_w", "db_w"]].sort_values("key"),
                      on="key", by="qb_id", allow_exact_matches=False, direction="backward")
    v = pd.Series(0.0, index=range(len(t)))
    v.loc[m["_row"].to_numpy()] = (m["epa_w"] / (m["db_w"] + shrink)).fillna(0).to_numpy()
    return pd.Series(v.to_numpy(), index=targets.index)


def add_qb_value(table: pd.DataFrame, flags: pd.DataFrame, db_games: pd.DataFrame) -> pd.DataFrame:
    """Add home_qb_value / away_qb_value: the value of each side's USUAL starter (from usual-starter flags) as of
    the game. Needs add_qb_feature's columns too. Games without a flag row get 0."""
    f = flags[["season", "week", "game_id", "team", "usual_id"]].rename(columns={"usual_id": "qb_id"})
    f = f.assign(qb_value=qb_values(db_games, f))
    out = table.copy()
    for side in ["home", "away"]:
        out = out.merge(f[["game_id", "team", "qb_value"]].rename(columns={"team": f"{side}_team", "qb_value": f"{side}_qb_value"}),
                        on=["game_id", f"{side}_team"], how="left", validate="one_to_one")
        out[f"{side}_qb_value"] = out[f"{side}_qb_value"].fillna(0.0)
    return out


def build_usual_flags(schedules: pd.DataFrame, seasons: list[int], refresh: bool = False) -> pd.DataFrame:
    """End to end for `seasons`: pbp starters (plus the season before, for the window) → No. 1 QB → unavailable flags."""
    from nfl_ats.data import load_rosters_weekly, regular_season
    reg = regular_season(schedules)
    reg = reg[reg["season"].isin(seasons)]
    games = pd.concat([reg[["season", "week", "game_id"]].assign(team=reg[side]) for side in ["home_team", "away_team"]],
                      ignore_index=True)
    starters = pd.concat([game_starters(load_pbp([s], refresh=refresh and s == max(seasons)))
                          for s in range(min(seasons) - 1, max(seasons) + 1)], ignore_index=True)
    return qb_unavailable(usual_starters(starters, games), load_injuries(seasons, refresh=refresh),
                          load_rosters_weekly(seasons, refresh=refresh))
