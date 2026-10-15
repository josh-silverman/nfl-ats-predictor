"""Lock this week's predictions. Run BEFORE the week's first kickoff, then commit the new ledger file.

    conda run -n nfl-ats python scripts/predict_week.py --week 6
    conda run -n nfl-ats python scripts/predict_week.py --week 3 --backfill   # pseudo-live, weeks already played

Live runs refresh the data, refuse if earlier weeks lack final scores, skip games that already kicked off,
and never overwrite an existing ledger file.
"""

import argparse
from datetime import datetime

import pandas as pd

from nfl_ats.ledger import (EASTERN, kickoff_times, ledger_path, load_schedules_for_ledger, predict_week,
                            write_ledger)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--backfill", action="store_true", help="pseudo-live: predict a played week as of its first kickoff")
    args = parser.parse_args()

    schedules = load_schedules_for_ledger(args.season, refresh=True)
    mode = "backfill" if args.backfill else "live"
    if args.backfill:
        week_games = schedules[(schedules["season"] == args.season) & (schedules["week"] == args.week)
                               & (schedules["game_type"] == "REG")]
        now = kickoff_times(week_games).min() - pd.Timedelta(minutes=1)
    else:
        now = datetime.now(EASTERN)

    preds = predict_week(schedules, args.season, args.week, now, mode)
    path = write_ledger(preds, ledger_path(args.season, args.week, mode, now))
    skipped = int(preds["games_skipped_already_started"].iloc[0])
    print(f"Wrote {len(preds)} predictions to {path} ({skipped} game(s) skipped: already kicked off).")
    print(preds[["game_id", "kickoff_et", "p_elo", "p_v2", "p_market_at_pull", "elo_spread"]].to_string(index=False))
    if mode == "live":
        print("\nNext: git add ledger/ && git commit -m 'Lock week {}' -- before the first kickoff.".format(args.week))


if __name__ == "__main__":
    main()
