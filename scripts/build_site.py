"""Rebuild the site's data (site/data/site-data.js). Run AFTER committing a new ledger file (the lock badge reads
each file's first commit), then commit + push.

    conda run -n nfl-ats python scripts/build_site.py
    open site/index.html          # view locally

If nothing but the "Updated" time would change, the file is left alone (so the daily rebuild makes no empty commits).
"""

from nfl_ats.data import PROJECT_ROOT
from nfl_ats.ledger import load_schedules_for_ledger
from nfl_ats.qb import build_usual_flags
from nfl_ats.site import backtest_summary, build_site_data, changed_beyond_timestamp, site_data_js

SEASON = 2026
OUT = PROJECT_ROOT / "site" / "data" / "site-data.js"


def main() -> None:
    schedules = load_schedules_for_ledger(SEASON, refresh=True)
    qb_flags = build_usual_flags(schedules, [SEASON], refresh=True)
    data = build_site_data(schedules, SEASON, backtest=backtest_summary(schedules), qb_flags=qb_flags)
    live = [w["week"] for w in data["weeks"] if w["mode"] == "live"]
    summary = (f"live weeks {live}, record games {data['record']['games']}, current week {data['meta']['current_week']}, "
               f"backtest {data['backtest']}")
    if not changed_beyond_timestamp(OUT.read_text() if OUT.exists() else None, data):
        return print(f"No change beyond the timestamp; left {OUT} as is ({summary})")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(site_data_js(data))
    print(f"Wrote {OUT}: {summary}")


if __name__ == "__main__":
    main()
