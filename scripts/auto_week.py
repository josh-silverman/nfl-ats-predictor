"""The weekly routine on auto-pilot (run by .github/workflows/weekly.yml; decision #25). Safe to run by hand.

    python scripts/auto_week.py --task lock            # daily: lock the next week once its first kickoff is < 30 h away
    python scripts/auto_week.py --task qb --soft       # Saturday: QB-aware re-lock; --soft = "not ready yet" isn't an error
    python scripts/auto_week.py --task score           # Tuesday: score the ledger and rebuild the site
    python scripts/auto_week.py --task lock --dry-run  # show what would happen, write nothing

Every guard from predict_week still applies (write-once, skip started games, stale-data refusal). Committing and
pushing is the workflow's job; this script writes `message=...` to $GITHUB_OUTPUT for the commit message.
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from nfl_ats.auto import locked_weeks, next_lock_week, next_qb_week
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
    parser.add_argument("--task", choices=["lock", "qb", "score"], required=True)
    parser.add_argument("--soft", action="store_true", help="qb: exit quietly if the Friday injury report isn't in yet")
    parser.add_argument("--dry-run", action="store_true", help="predict and print, but write nothing")
    args = parser.parse_args()

    if args.task == "score":
        if args.dry_run:
            return print("Dry run: would run score_ledger.py and build_site.py.")
        script("score_ledger.py")
        script("build_site.py")
        return output(f"Score results and refresh the site ({datetime.now(EASTERN):%b %-d})")

    schedules = load_schedules_for_ledger(SEASON, refresh=True)
    now = pd.Timestamp.now(tz=EASTERN)
    locked, qb_done = locked_weeks(SEASON)
    if args.task == "lock":
        week, mode = next_lock_week(schedules, SEASON, locked, now), "live"
    else:
        week, mode = next_qb_week(schedules, SEASON, locked, qb_done, now), "qb"
    if week is None:
        return print(f"Nothing to do for '{args.task}' right now.")

    print(f"{'Dry run: ' if args.dry_run else ''}{args.task} week {week}")
    try:
        preds, path, qb_pts = run_week(schedules, SEASON, week, mode, now.to_pydatetime(), write=not args.dry_run)
    except InjuryReportNotReady as e:
        if args.soft:
            return print(f"Not yet: {e}")
        raise
    report(preds, path, qb_pts)
    if args.dry_run:
        return
    script("build_site.py")
    output(f"Lock week {week}" if mode == "live" else f"QB lock week {week}")


if __name__ == "__main__":
    main()
