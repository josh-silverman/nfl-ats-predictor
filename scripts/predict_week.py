"""Lock this week's predictions. Run BEFORE the week's first kickoff, then commit the new ledger file.

    conda run -n nfl-ats python scripts/predict_week.py --week 8          # Thursday, before the first kickoff
    conda run -n nfl-ats python scripts/predict_week.py --week 8 --qb     # Saturday: QB-aware re-lock (ledger/qb/)
    conda run -n nfl-ats python scripts/predict_week.py --week 3 --backfill   # pseudo-live, weeks already played

Live runs refresh the data, refuse if earlier weeks lack final scores, skip games that already kicked off,
and never overwrite an existing ledger file. Live and --qb runs add the QB-adjusted Elo (H4, decision #21);
--qb refuses until the Friday injury report (game statuses) is in the data.
"""

import argparse

from nfl_ats.ledger import load_schedules_for_ledger, run_week


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--backfill", action="store_true", help="pseudo-live: predict a played week as of its first kickoff")
    parser.add_argument("--qb", action="store_true", help="Saturday QB-aware re-lock of not-yet-started games")
    args = parser.parse_args()

    mode = "backfill" if args.backfill else "qb" if args.qb else "live"
    preds, path, qb_pts = run_week(load_schedules_for_ledger(args.season, refresh=True), args.season, args.week, mode)
    report(preds, path, qb_pts)
    if mode != "backfill":
        what = "Lock week {}" if mode == "live" else "QB lock week {}"
        print(f"\nNext: git add ledger/ && git commit -m '{what.format(args.week)}' && git push -- before the next kickoff.")


def report(preds, path, qb_pts) -> None:
    """Print what was written (shared with scripts/auto_week.py)."""
    skipped = int(preds["games_skipped_already_started"].iloc[0])
    print(f"Wrote {len(preds)} predictions to {path} ({skipped} game(s) skipped: already kicked off).")
    cols = ["game_id", "kickoff_et", "p_elo", "p_v2", "p_market_at_pull", "elo_spread"]
    if qb_pts is not None:
        cols += ["p_elo_qb", "home_qb_out", "away_qb_out"]
        out = preds[(preds["home_qb_out"] == 1) | (preds["away_qb_out"] == 1)]
        print(f"QB_PTS = {qb_pts} Elo points. Usual starter out: "
              + (", ".join(f"{r.home_team if r.home_qb_out else r.away_team} ({r.home_qb if r.home_qb_out else r.away_qb})"
                           for r in out.itertuples()) or "none"))
    print(preds[cols].to_string(index=False))


if __name__ == "__main__":
    main()
