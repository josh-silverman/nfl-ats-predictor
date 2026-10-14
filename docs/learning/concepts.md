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

## Phase 3: EDA with seaborn
- **Thin notebook, logic in `src/`:** `03-eda.ipynb` only loads, validates, and plots. `pip install -e .` makes `nfl_ats` importable from anywhere, so nothing in the notebook duplicates library code.
- **The closing line is unbiased and hard to beat:** mean residual (margin − spread) is +0.04 points, and the slope of margin on spread is 1.04. Home teams cover 49.1%, and every season's CI includes 50% except 2019 (43%). With 10 seasons, about one interval missing 50% is what chance alone produces (multiple testing).
- **Games are noisy:** residual SD is 12.7 points, and the spread explains only about 20% of the margin variance. Any edge will be small, so judge it with CIs over about 1,900 games, never by eye.
- **Key numbers are real:** 14.4% of margins land on exactly 3 and 8.5% on 7. 3 is the most common spread, and it pushes 10% of the time. A smooth model (logistic regression) can't see this, so watch performance near 3 and 7.
- **The market already prices obvious factors:** a home rest edge of 4+ days moves the spread about 2 points, and the cover rate stays flat (about 48–50%) in every rest bucket. What EDA *can't* show is whether a feature adds signal beyond the line *combined with others*. Only the walk-forward backtest answers that.
- **seaborn tools used:** `pointplot(errorbar=("ci", 95), seed=0)` for bootstrap CIs, `regplot(x_bins=...)` for binned calibration, and `histplot(stat="density")` with a normal-curve overlay. Run notebooks top to bottom before committing: that's how we caught the `.style` → jinja2 dependency.

## Phase 4: Feature engineering and leakage
- **Lag the cumulative sums, not the averages:** `groupby(["season","team"]).cumsum()`, then `groupby(...).shift(1)` *within* team-season. Sum EPA × plays and plays separately, so the per-play rate is play-weighted. Week 1 has no history, and the next bullet fills it.
- **Shrinkage for early weeks:** prior = last season pulled toward last season's league mean by `r`. Blend weight `w = k0 / (k0 + games_played)`. Fit (r, k0) on **training seasons only**, and in Phase 7 **refit inside every fold**. Out of sample it cuts early-season error by 35%, and it is neutral late in the season.
- **Perturbation tests beat spot checks:** wreck one game's stats and assert that nothing at or before it moves. That catches leakage you didn't think of. We test it per function, end to end, and for "the fit ignores non-training seasons".
- **Game rows are home − away diffs:** `off_epa_diff`, `def_epa_diff` (EPA *allowed*, so negative favors home), `pace_diff`, and `rest_diff`. They track the spread strongly (off 0.74) but not the cover (about 0). The market already prices them, so any edge must come from combinations.
- **One route into the model:** select model inputs only with the `FEATURES` list, because `margin` and `home_cover` live in the same table. `build_game_features` raises if any feature is missing rather than passing NaN along.
- **Skeptic review passed:** it checked 2020 COVID reshuffles (week order equals date order), a real-data wreck from Week 9 onward (nothing earlier moved), and the target logic on all 2,639 games. Its should-fix items are now requirements in the roadmap for Phases 5, 7, 8, and 9.
