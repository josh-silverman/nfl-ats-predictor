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

**Log:** bootstrap confidence intervals, the edge rule.

## Phase 6: Logistic regression
1. Build the Pipeline (scaler + logistic regression). Fit it on one fold and inspect the coefficients.
2. Fit and predict, then interpret the coefficient signs.
3. Experiment-log template (`reports/experiments.md`).
4. Single-change experiment: add `spread_line` as a feature. Josh decides keep or reject.

**Log:** coefficient interpretation, scaling.

## Phase 7: Walk-forward backtest and calibration
1. Walk-forward loop (2019–2025).
2. Metrics with confidence intervals vs. baselines.
3. Reliability plot and Brier score decomposition.
4. Single-change experiments, each one logged. Josh decides keep or reject.
5. 🔍 **Skeptic review.**

**Log:** calibration, multiple testing.

## Phase 8: Weekly live predictions with a locked ledger
1. Weekly script: pull the latest data, build features as of kickoff, predict, write a timestamped ledger file.
2. Lock procedure: commit the ledger before the first kickoff.
3. Pseudo-live backfill for the weeks already played, labeled separately.
4. Weekly scoring of locked predictions.
5. 🔍 **Skeptic review.**

**Log:** what makes a ledger trustworthy.

## Phase 9: Portfolio write-up
1. README case study: question, data, method, honest results, limits.
2. Key figures: calibration, accuracy with confidence intervals vs. baselines, the live record.
3. "What I'd do next" section.

**Log:** final review.
