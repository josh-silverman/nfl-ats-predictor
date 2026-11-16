"""The weekly routine on auto-pilot (run by .github/workflows/weekly.yml; decision #25). Safe to run by hand.

    python scripts/auto_week.py --task lock            # twice daily: lock the next week once its first kickoff is < 30 h
                                                       # away; if no week was locked, also run the QB step (soft)
    python scripts/auto_week.py --task qb              # QB step only: final QB lock for each due game (decision #27)
    python scripts/auto_week.py --task qb --soft       # ... and "report not in yet" isn't an error
    python scripts/auto_week.py --task score           # Tuesday: score the ledger (reports/live_2026.md)
    python scripts/auto_week.py --task site            # daily: just ask the workflow for a site rebuild
    python scripts/auto_week.py --task watch           # every run: fail if a game kicks off within 6 h unlocked
    python scripts/auto_week.py --task lock --dry-run  # show what would happen, write nothing
    python scripts/auto_week.py --task lock --dry-run --week 8   # test the full lock path early

Every guard from predict_week still applies (write-once, skip started games, stale-data refusal). Committing and
pushing is the workflow's job; this script writes `message=...` to $GITHUB_OUTPUT for the commit message.

This script never builds the site. The workflow commits the new ledger file FIRST and only then runs build_site.py,
because the site's lock badge reads each ledger file's first commit (built earlier, it said "Not committed yet").
By hand: run this, commit, then `python scripts/build_site.py`.
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from nfl_ats.auto import WATCH_MARGIN, locked_weeks, next_lock_week, qb_due_games, qb_locked_games, unlocked_soon
from nfl_ats.data import PROJECT_ROOT
from nfl_ats.ledger import EASTERN, InjuryReportNotReady, load_schedules_for_ledger, run_week

sys.path.insert(0, str(Path(__file__).parent))
from predict_week import report  # noqa: E402

SEASON = 2026


def output(message: str) -> None:
    print(message)
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"message={message}\n")


def script(name: str) -> None:
    subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / name)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=["lock", "qb", "score", "site", "watch"], required=True)
    parser.add_argument("--soft", action="store_true", help="qb: exit quietly if the Friday injury report isn't in yet")
    parser.add_argument("--dry-run", action="store_true", help="predict and print, but write nothing")
    parser.add_argument("--week", type=int, help="dry runs only: force this week (to test the full path early)")
    args = parser.parse_args()

    if args.task == "watch":
        hours = f"{WATCH_MARGIN.total_seconds() / 3600:g} hours"
        late = unlocked_soon(load_schedules_for_ledger(SEASON, refresh=True), SEASON, locked_weeks(SEASON)[0],
                             pd.Timestamp.now(tz=EASTERN))
        if late:
            sys.exit(f"ALARM: week {late[0]} has a kickoff within {hours} and no lock. Lock it now: "
                     f"run the workflow with task=lock, or `python scripts/predict_week.py --week {late[0]}` and commit.")
        return print(f"Watchdog OK: every game kicking off in the next {hours} is locked.")
    today = f"{datetime.now(EASTERN):%b %-d}"
    if args.task == "site":
        if args.dry_run:
            return print("Dry run: the workflow would rebuild the site.")
        return output(f"Refresh the site ({today})")
    if args.task == "score":
        if args.dry_run:
            return print("Dry run: would run score_ledger.py (the workflow then rebuilds the site).")
        script("score_ledger.py")
        return output(f"Score results ({today})")

    if args.week and not args.dry_run:
        parser.error("--week is only for dry runs; real locks always use the detected week")
    schedules = load_schedules_for_ledger(SEASON, refresh=True)
    now = pd.Timestamp.now(tz=EASTERN)
    messages = []
    if args.task == "lock":
        week = args.week or next_lock_week(schedules, SEASON, locked_weeks(SEASON)[0], now)
        if week is None:
            print("Lock: nothing to lock right now.")
        else:
            print(f"{'Dry run: ' if args.dry_run else ''}lock week {week}")
            preds, path, qb_pts = run_week(schedules, SEASON, week, "live", now.to_pydatetime(), write=not args.dry_run)
            report(preds, path, qb_pts)
            messages.append(f"Lock week {week}")
    if args.task == "qb" or not messages:  # a run that just locked a week skips the QB step (same news, same minute)
        messages += qb_step(schedules, now, args.dry_run, args.week if args.task == "qb" else None,
                            soft=args.soft or args.task == "lock")
    if messages and not args.dry_run:
        output("; ".join(messages))


def qb_step(schedules: pd.DataFrame, now: pd.Timestamp, dry_run: bool, forced_week: int | None, soft: bool) -> list[str]:
    """Final QB lock for every due game (auto.qb_due_games), one ledger/qb file per week. Returns commit-message parts."""
    due = ({forced_week: None} if forced_week
           else qb_due_games(schedules, SEASON, locked_weeks(SEASON)[0], qb_locked_games(SEASON), now))
    if not due:
        print("QB: no game is due for its final QB lock right now.")
        return []
    parts = []
    for week, games in sorted(due.items()):
        print(f"{'Dry run: ' if dry_run else ''}QB lock week {week}: due {sorted(games) if games else 'all unstarted'}")
        try:
            preds, path, qb_pts = run_week(schedules, SEASON, week, "qb", now.to_pydatetime(), write=not dry_run, games=games)
        except InjuryReportNotReady as e:
            if not soft:
                raise
            print(f"QB not yet: {e}")
            continue
        report(preds, path, qb_pts)
        if games and len(preds) < len(games):
            print(f"Waiting for reports: {sorted(games - set(preds['game_id']))}")
        parts.append(f"QB lock week {week} ({len(preds)} game{'' if len(preds) == 1 else 's'})")
    return parts


if __name__ == "__main__":
    main()
