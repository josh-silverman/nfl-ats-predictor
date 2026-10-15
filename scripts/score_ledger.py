"""Score the 2026 ledger and write reports/live_2026.md. Safe to run any time; it never touches ledger files.

    conda run -n nfl-ats python scripts/score_ledger.py
"""

from datetime import datetime

import pandas as pd

from nfl_ats.data import PROJECT_ROOT
from nfl_ats.ledger import EASTERN, live_report, load_ledger, load_schedules_for_ledger, score_ledger

SEASON = 2026
OUT = PROJECT_ROOT / "reports" / f"live_{SEASON}.md"


def section(title: str, rep: dict) -> list[str]:
    lines = [f"## {title}", "",
             f"Weeks: {rep['weeks']} · games locked: {rep['games_locked']} · decided: {rep['games_decided']} · ties: {rep['ties']}", ""]
    if "note" in rep:
        return lines + [rep["note"], ""]
    lines += ["| Model | Games | Log loss [95% CI] | Brier [95% CI] | Accuracy [95% CI] | Mean p − actual (pts) [CI] | Reliability (5 bins) vs. null 95th pct | Consistent with calibration? |",
              "|---|---|---|---|---|---|---|---|"]
    for name, m in rep["models"].items():
        fmt = lambda t, k=4: f"{t[0]:.{k}f} [{t[1]:.{k}f}, {t[2]:.{k}f}]"
        c = m["calibration_in_the_large_pts"]
        lines.append(f"| {name} | {m['games']} | {fmt(m['log_loss'])} | {fmt(m['brier'])} | {fmt(m['accuracy'], 3)} | "
                     f"{c[0]:+.1f} [{c[1]:+.1f}, {c[2]:+.1f}] | {m['reliability']:.5f} vs. {m['reliability_null95']:.5f} | "
                     f"{'yes' if m['consistent_with_calibration'] else 'no'} |")
    lines += ["", "**Paired log loss (negative = first model better):**", ""]
    lines += [f"- {k}: {v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}]" for k, v in rep["paired_log_loss (diff, ci_low, ci_high)"].items()]
    return lines + [""]


def main() -> None:
    schedules = load_schedules_for_ledger(SEASON, refresh=True)
    now = datetime.now(EASTERN).strftime("%Y-%m-%d %H:%M %Z")
    lines = [f"# {SEASON} live record", "",
             f"Generated {now} by `scripts/score_ledger.py`. The checks were pre-registered in `CLAUDE.md` before any 2026 outcome was scored.",
             "Raw Elo is the deliverable; v2 is recorded alongside. The market is the moneyline at our pull time, so it may have later information than ours and isn't comparable with the backtest's schedule moneyline.", ""]
    for mode, title in [("live", "Live: locked before kickoff (the clean test)"),
                        ("backfill", "Pseudo-live backfill (weeks predicted after they were played; NOT part of the live record)")]:
        ledger = load_ledger(mode, SEASON)
        if ledger.empty:
            lines += [f"## {title}", "", "No locked predictions yet.", ""]
            continue
        lines += section(title, live_report(score_ledger(ledger, schedules)))
        if mode == "backfill":
            lines += ["*Backfill note: these weeks were predicted on 2026-10-15, after they were played. Their \"market at pull time\" "
                      "is the schedule moneyline as of that date (likely near-closing), not a pre-game snapshot. With about 80 games, "
                      "every interval is wide, so treat this as a smoke test, not evidence.*", ""]
    OUT.write_text("\n".join(lines))
    print(f"Wrote {OUT}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
