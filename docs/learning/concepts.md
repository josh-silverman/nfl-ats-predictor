# Key things to know

At the end of each phase, Claude adds 3–6 bullets with the points worth remembering from it.

<!-- Template
## Phase N: <name>
- <point>: <why it matters for this project>
-->

## Phase 1: Problem framing
- **Pushes and the hook:** a push is when the home margin exactly equals the spread. We exclude pushes from the target but count them. Half-point spreads can't push, so excluding pushes drops whole-number spreads more often, and the games left aren't a random sample.
- **Key numbers:** final margins cluster at 3 and 7, with smaller bumps at 10, 6, 4, and 14. Moving a line across 3 or 7 matters far more than moving it across 5 or 9. A model that treats margin as a smooth normal curve misprices those games.
- **52.4% breakeven:** at -110 you risk $110 to win $100, so you need 110/210 = 52.4% to break even. Every backtest result is measured against that bar.
- **Sample size:** one season (about 272 games) gives a 95% CI of about ±6 points. The 2019–2025 walk-forward (about 1,900 games) gives about ±2.2. Never judge a model on one season, and even 55% over 1,900 games only just clears 52.4%.
- **No random splits:** validation is walk-forward by season, so the model never trains on games from the future.

## Phase 2: Data pull and validation
- **Convert Polars to pandas at the load boundary:** `_cached_pull()` in `data.py` calls `.to_pandas()` right after the pull, so nothing downstream ever sees Polars.
- **Dated parquet caches:** each pull is saved as `name_seasons_YYYYMMDD.parquet`, and reads use the newest one. nflverse revises past data, so this shows exactly which data a result was built on. Use `refresh=True` to re-pull.
- **Sign convention:** `spread_line` is the expected **home margin**, so a positive value means the home team is favored. That's the opposite of how sportsbooks write it ("DEN +3" = Denver is the underdog). `check_spread_sign()` enforces it. Use the **closing** line, not the opening line, which can differ by a lot.
- **Target and pushes:** `home_cover = margin > spread_line`. Pushes are flagged with `is_push` and get a NaN target, so they're counted but never trained on. 2016–2025: 65 pushes (2.5%), all on whole-number spreads. Home cover rate is 49.1%, so "always home" won't beat 52.4%.
- **Validate before building:** `validate_schedules()` checks game counts (256 or 272 per season, minus the canceled 2022 BUF@CIN game), duplicates, missing lines, and 32 teams after mapping (OAK→LV, SD→LAC). Apply the same `TEAM_MAP` to play-by-play in Phase 4.
- **Play-by-play size:** one season is about 49k plays by 372 columns, roughly 200 MB in memory. Filter to regular-season pass and run plays and select only the columns you need. Mean EPA per play is about 0 (0.007 in 2025), which confirms the data is sane.
