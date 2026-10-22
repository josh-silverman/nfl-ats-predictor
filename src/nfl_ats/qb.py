"""Experiment H2: is the expected starting QB ruled out before kickoff? (Pre-registered in reports/experiments.md.)

As-of rules:
- The EXPECTED starter for a game is whoever threw the most passes in the team's PREVIOUS game that season
  (shift(1) within team-season), so the game's own participation never leaks in.
- "Out" comes from the official injury report for that game week. Statuses are finalized before kickoff
  (audit: 100% in 2016–2024, median ~51 h before). Never use the schedule's *_qb_id (who actually started).
"""

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
    """
    inj = injuries.loc[(injuries["game_type"] == "REG") & injuries["report_status"].isin(OUT_STATUSES),
                       ["season", "week", "team", "gsis_id", "report_status"]].copy()
    inj["reason_inj"] = inj.pop("report_status").str.lower()
    res = rosters.loc[rosters["status"] == "RES", ["season", "week", "team", "gsis_id", "status_description_abbr"]].copy()
    res["reason_res"] = (res.pop("status_description_abbr") == "R01").map({True: "injured reserve", False: "reserve list"})
    names = rosters.drop_duplicates(["season", "gsis_id"])[["season", "gsis_id", "full_name"]]

    out = usual.copy()
    for src in (inj, res):
        src["team"] = src["team"].replace(TEAM_MAP)
        src[["season", "week"]] = src[["season", "week"]].astype(int)
        out = out.merge(src.drop_duplicates(["season", "week", "team", "gsis_id"]),
                        left_on=["season", "week", "team", "usual_id"], right_on=["season", "week", "team", "gsis_id"],
                        how="left", validate="many_to_one").drop(columns="gsis_id")
    out["reason"] = out["reason_res"].fillna(out["reason_inj"])
    out["qb_out"] = out["reason"].notna().astype(int)
    out = out.merge(names, left_on=["season", "usual_id"], right_on=["season", "gsis_id"], how="left").drop(columns="gsis_id")
    out = out.rename(columns={"full_name": "player"})
    return out[["season", "week", "game_id", "team", "usual_id", "player", "qb_out", "reason"]]


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
