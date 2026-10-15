# 01 — Roadmap

Claude writes the code in every phase and explains the key patterns as it goes. Josh reviews (decision #11). Every phase ends with a **log**: its key things to know go into `docs/learning/concepts.md`, Claude posts a 3-line recap, and work stops until Josh says go. The **Log:** line under each phase lists the topics to capture.

## Phase 0: Setup ✅
CLAUDE.md, the design and roadmap docs, the learning files, the skeptic agent, the slash commands, and the env install.

## Phase 1: Problem framing ✅
1. What "covering" means, how closing lines form, and why closing lines are hard to beat.
2. Why the target is home-cover and not favorite-cover, and how pushes are handled.
3. The -110 math: where 52.4% comes from.
4. Sample-size reality: confidence-interval width at n = 272 vs. n = 1,900.

**Log:** breakeven, pushes, why random splits are wrong here.

## Phase 2: Data pull and validation
1. Write `src/nfl_ats/data.py`: load schedules, convert Polars to pandas, cache to parquet.
2. Validate: game counts, duplicates, missing lines, team-abbreviation mapping.
3. **Spread sign check:** Josh looks up one real game, we confirm the `spread_line` direction together, then assert it with the favorite win rate.
4. Build the `home_cover` target, then count pushes per season. Compare against Josh's Phase 1 guesses (55% home cover, 2% pushes).
5. Load play-by-play for one season and check its size, columns, and EPA sanity (league average is about 0).

**Log:** the sign convention, the Polars conversion, push counts.

## Phase 3: EDA with seaborn
1. Home cover rate by season.
2. Spread distribution, plus key numbers (3 and 7).
3. Margin vs. spread residuals.
4. Rest-day differences vs. cover rate.

Before this phase, decide whether to add Jupyter (it needs Josh's agreement).

**Log:** what EDA can and can't tell us about ATS.

## Phase 4: Feature engineering and leakage
1. Team-game aggregation from play-by-play: offensive EPA/play, defensive EPA/play allowed, pace.
2. Cumulative means with `shift(1)`, then a unit test proving that Week *k* never sees week *k* or later.
3. Prior-season carryover with shrinkage for the early weeks.
4. Join to games as home − away differences.
5. 🔍 **Skeptic review.** Josh decides on the findings.

**Log:** the lag, the Week 1 plan, shrinkage.

## Phase 5: Baselines
1. Always home, always favorite, and coin flip on 2019–2025.
2. Bootstrap confidence-interval helper.
3. Report the count of neutral-site games (46) alongside "always home". Their "home" team has no real home field.

**Log:** bootstrap confidence intervals, the edge rule.

## Phase 6: Logistic regression ✅
1. Build the Pipeline (scaler + logistic regression). Fit it on one fold and inspect the coefficients.
2. Fit and predict, then interpret the coefficient signs.
3. Log the first ATS result in `reports/experiments.md` (the log was started in Phase 4).
4. Single-change experiment: add `spread_line` as a feature. Josh decides keep or reject.

**Log:** coefficient interpretation, scaling.

## Phase 7: Walk-forward backtest and calibration ✅ (ATS: rigorous negative result; then the diagnostic session D1–D7)
1. Walk-forward loop (2019–2025). **Requirements from the Phase 4 skeptic review:**
   - Inside each fold, call `fit_shrinkage` → `add_shrunk_features` → `build_game_features` using only that fold's training seasons. If an inner split ever tunes C, refit shrinkage inside it too.
   - Build `X` only as `df[FEATURES]`. Never select columns by dtype, because `margin` and `home_cover` sit in the same table.
   - Tests: `X.columns == FEATURES`, and wrecking the test season leaves the fold's params and training features unchanged.
2. Metrics with confidence intervals vs. baselines.
3. Reliability plot and Brier score decomposition.
4. Single-change experiments, each one logged. Josh decides keep or reject.
5. 🔍 **Skeptic review.**

**Log:** calibration, multiple testing.

## Pivot (decision #19): win-probability model, milestones M0–M5
The ATS study ends as a documented negative result. Phases 8–9 are replaced by these milestones. Success criteria S1–S5 are pre-registered in `CLAUDE.md`. Each milestone ends with a stop for Josh's go.

### M0: Pivot paperwork ✅
CLAUDE.md (new question, success criteria, archived ATS rules), design section 9, this roadmap, decision #19.

### M1: Win-model code
1. `src/nfl_ats/win_model.py`: `build_win_table` (runs Elo over 2015–2025, adds the `home_win` target, flags and counts ties), `WIN_FEATURES = ["elo_diff"]`, `win_walk_forward` (reuses `make_model`, `check_features`).
2. `feature_eval.compare_predictions(target=...)`, so the keep rule works for `home_win`.
3. `tests/test_win_model.py`: ties counted, fold isolation, and Elo's as-of property.

**Log:** Elo as a feature, why fixed parameters avoid fold leakage.

### M2: Evaluate v2 against S1–S3
1. `notebooks/10-win-model.ipynb`: per-season and pooled metrics with CIs, paired vs. base rate and raw Elo, gap to the market, Brier decomposition, recalibration slope, reliability plot. PASS or FAIL printed per criterion.
2. Log win experiment #3 in `reports/experiments.md`.
3. If S2 or S3 fails, raw Elo is the deliverable.
4. 🔍 **Skeptic review.** Josh decides on the findings.

**Log:** non-inferiority, calibration slope.

### M3: Live 2026 ledger
1. Check that `load_schedules([2026], refresh=True)` has scores for played weeks and moneylines for upcoming games.
2. `src/nfl_ats/ledger.py` (`predict_week`, `score_ledger`) and `scripts/predict_week.py`. Guards: no overwrite, no started games.
   - **Records the line used and its pull time.** A line pulled before kickoff is not the closing line.
3. Backfill the played weeks into `ledger/backfill/` (pseudo-live).
4. Weekly routine: Josh runs one command before Thursday's kickoff, then commits.
5. `tests/test_ledger.py`.

**Log:** what makes a ledger trustworthy.

### M4: One improvement, time-boxed: quarterback availability
1. **Data audit with a go/no-go.** Do `load_injuries` and `load_depth_charts` cover 2016–2025, with pre-game timing? Schedule QB IDs stay banned (post-hoc).
2. **Go:** build `qb_out_diff`, run `paired_power` first, then one single-change test vs. v2. Josh decides keep or reject.
3. **No-go:** log why. Either way M4 is done.

**Log:** as-of data sourcing.

### M5: Write-up and final review
1. README case study:
   - the question
   - the ATS negative result
   - the win model vs. base rate, Elo, and market
   - the live 2026 record
   - limitations (EPA model provenance, unverified closing lines, viewed holdout seasons, moneyline source change)
   - what I'd do next
2. Figures: reliability, per-season accuracy with CIs, and the live record.
3. 🔍 **Final skeptic review.**

**Log:** final review.
