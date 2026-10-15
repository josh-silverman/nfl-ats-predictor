# Progress

**Current phase:** Phase 7 paused for a **diagnostic session** (`docs/diagnostic-prompt.md`, plan approved). Steps 0–3 done (Step 3 skeptic: no blockers, wording corrected).
**Next:** Diagnostic Step 4, model diagnostics (learning curves, C sweep **with the dispersion guard**, calibration, coefficients). Josh predicts first. Exp #6 stays pending until Step 5. Session rules: stop after each step, Josh predicts before reveals, skeptic after Steps 3 and 6.
**Last updated:** 2026-10-15

## Done
- Planning docs, CLAUDE.md, learning files, skeptic agent, slash commands.
- Requirements installed in the `nfl-ats` conda env.
- pyarrow 25.0.1 installed in `nfl-ats` and the import verified (via `conda run -n nfl-ats python -m pip`).

- Phase 2 step 1: `src/nfl_ats/data.py` with `load_schedules()`. Pulled 2016–2025 (2,761 rows incl. playoffs), cached to `data/raw/`.
- Phase 2 step 2: `regular_season()` + `validate_schedules()`. 2,639 REG games, all counts match (2022 = 271, canceled BUF@CIN), no dups or missing lines, OAK→LV and SD→LAC. 46 neutral-site games, noted for later.
- Phase 2 step 3: spread sign confirmed (`spread_line` = expected home margin, positive = home favored). Asserted in `check_spread_sign()`.
- Phase 2 step 4: `src/nfl_ats/target.py` (`add_home_cover`, `push_summary`). 2,639 games, 65 pushes (2.5%), home cover 49.1% excl. pushes.
- Phase 2 step 5: `load_pbp()` added, with caching shared via `_cached_pull()`. 2025 has 48,771 plays × 372 cols, and REG pass/run mean EPA is 0.007. Older seasons will need `TEAM_MAP` in Phase 4.
- Phase 3 setup: ipykernel + editable install (decision #13). Notebook `notebooks/03-eda.ipynb`.
- Phase 3 step 1: home cover by season with bootstrap CIs, saved to `reports/figures/home_cover_by_season.png`. Every season is about 50% except 2019 (43%, CI 37–49%).
- Phase 3 step 2: key numbers, saved to `reports/figures/key_numbers.png`. Margin = 3 in 14.4% of games and 7 in 8.5%. 3 is the most common spread (14.2%). Push rate at a spread of 3 is 10.1%, and at 1 and 5 it is 0%.
- Phase 3 step 3: residuals, saved to `reports/figures/margin_vs_spread.png`. Mean resid +0.04, SD 12.7, slope 1.04, R² 0.20. The line is unbiased, and no spread bucket stands out (home dogs 7.5+ at 56%, n = 164, is noise).
- Phase 3 step 4: rest diff, saved to `reports/figures/rest_vs_cover.png`. The spread rises 1.3 → 3.4 pts with home rest edge, and the cover rate is flat (48–50%) in all buckets. `rest_diff` is in v1 per the design doc (decision #15).
- Notebook verified end to end. Replaced `.style` (needs jinja2) with `.round(3)`.
- Phase 4 step 1: `src/nfl_ats/features.py` (`team_game_stats`, `load_team_games`). 5,278 team-games (2 × 2,639), matching the schedule exactly, no nulls, 32 teams. League off EPA/play is about 0 each season, and pace is about 60–63 plays/game.
- Phase 4 step 2: `season_to_date()` (play-weighted cumulative sums, `groupby.shift(1)`). `tests/test_features.py` has 4 unittest tests, including the wreck-week-k perturbation test. Real-data checks pass (game 2 = game 1 stats, last game excludes itself). 320 week-1 NaNs, one per team-season. Early-season values are extreme (±0.65 after 1 game), which motivates shrinkage.
- Phase 4 step 3: `prior_season`, `add_shrunk_features`, `fit_shrinkage` (decision #14). 2015 pbp added as the 2016 prior, with STL → LA. 7 tests pass, including "fit ignores non-training seasons". Out-of-sample off-EPA MSE after 1–2 games: 0.063 raw vs. 0.041 shrunk. In Phase 7, refit params inside each fold.
- Phase 4 step 4: `build_game_features()` + `FEATURES` list (decision #15). 2,639 games × 23 cols, no feature nulls. Corr with spread: off 0.74, def −0.46. Corr with home_cover is about 0 for all four (expected, since the market prices them). 8 tests pass, including an end-to-end "own game never reaches its features" test.
- Phase 4 step 5: skeptic PASS, no blockers. Applied: decision #14 corrected (mistake #5), `reports/experiments.md` started (2 design entries), `build_game_features` raises on missing features (9 tests). Phase 7, 8, and 9 requirements added to the roadmap.
- Phase 5: `src/nfl_ats/evaluate.py` (accuracy, log loss, Brier, vectorized `bootstrap_ci`, `baseline_table`). `tests/test_evaluate.py` cross-checks against sklearn and the CI-width formula (12 tests total). Baselines 2019–25: home 48.7%, favorite 48.2%, coin 50.8%, all far from an edge. Logged in experiments.md.
- Phase 6 step 1: `src/nfl_ats/model.py` (`make_model`, `fold_data` with per-fold shrinkage, `coefficients`). Fold 1 (train 2016–18, 746 games): coefs per SD are def +0.12, rest +0.07, off +0.06, pace +0.02. Train probabilities 35–63%, SD 0.04. The model is barely confident.
- Phase 6 step 2: `evaluate.score()`. 2019 out-of-sample: 49.6% [43.1, 55.7], log loss 0.698 (worse than 0.693), Brier 0.2525. In-sample 52.7%. Logged as ATS experiment #3 in experiments.md (count 1).
- Phase 6 step 4: `model.walk_forward` + `evaluate.paired_diff` (15 tests). v1 walk-forward 2019–25: 51.3% [48.9, 53.5], log loss 0.6947 (worse than 0.6931). Adding `spread_line` changes nothing (paired diffs about 0). Experiment #4: rejected by Josh (decision #16).
- Phase 7 step 1: `OUTCOME_COLUMNS` + `check_features()` guard in `walk_forward`. `tests/test_model.py` (5 tests): wrecking the test season leaves the fold's shrinkage params and training rows unchanged (and the wreck is confirmed to change the test rows), leaky features are refused, and X columns == FEATURES. 20 tests total.
- Phase 7 step 2: `evaluate.season_table()` + `notebooks/07-backtest.ipynb` (verified end to end), figure `reports/figures/v1_backtest_by_season.png`. Seasons range 49.6–52.7%, every CI includes 50%, pooled 51.3% [48.9, 53.5]. The model picks away 68–82% in 2020–24 because mean p equals the training home-cover rate (about 48.3% after 2019 joins) and the features move p by only about ±2 points. Preds saved to `data/processed/v1_walk_forward_preds.parquet`.
- Phase 7 step 3: `reliability_table`, `brier_decomposition` (22 tests, including the resolution noise-floor test). Figure `reports/figures/v1_reliability.png`. v1: reliability 0.0018, resolution 0.0013 vs. noise floor 0.0014, uncertainty 0.2498. Roughly calibrated, zero resolution.
- Phase 7 step 4, exp #5: garbage-time filter (`drop_garbage_time`, wp in [0.1, 0.9]). Paired vs. v1: −0.4 pts [−2.1, +1.3], log loss −0.0001 [−0.0015, +0.0013]. Rejected (decision #17), and C tuning skipped.
- Exp #6: opponent-adjusted EPA (`adjust_for_opponent`, `opponent_adjust=True` in fold_data/walk_forward, 24 tests). Paired: +0.2 pts [−0.9, +1.4], log loss −0.0006 [−0.0014, +0.0002]. Features barely move (r = 0.99 off). Decision pending.

## Open items
- The 2026 live ledger starts with the next unplayed week. Weeks already played are a pseudo-live backfill, always labeled as backfill. As of mid-October that is more than weeks 1–4. Revisit in Phase 8.

## Josh's predictions (check against data)
- Home cover rate vs. closing spread, 2016–2025, pushes excluded: **55%** (Phase 1, step 1). **Actual: 49.1%** (range 43.1% in 2019 to 52.4% in 2017).
- Push rate vs. closing spread: **2%** of games. Whole-number spreads push more than half-point spreads (Phase 1, step 2). **Actual: 2.5%** (65 of 2,639). Whole-number spreads 5.1%, half-point 0%, as expected.

## Session log
- 2026-10-02: Session 0, planning and setup.
- 2026-10-02: Approved pyarrow, held Jupyter, started Phase 1.
- 2026-10-02: Added the conda-run rule to CLAUDE.md, installed pyarrow. Back to the step 2 push question.
- 2026-10-02: Changed teaching focus to coding/pandas/sklearn (decision #10). Summarized steps 3–4 and ran the Phase 1 gate.
- 2026-10-13: Switched to build mode (decision #11). Claude writes the code, and phase-end key-points logs replace the quizzes. Phase 1 closed.
- 2026-10-13: Phase 2 step 1 done (`data.py`, schedules cached).
- 2026-10-13: Phase 2 step 2 done (schedule validation).
- 2026-10-13: Phase 2 step 3 done (spread sign confirmed with Josh on CAR@DEN 2016).
- 2026-10-13: Phase 2 step 4 done (target + push counts).
- 2026-10-13: Phase 2 done (pbp loader, sanity checks). Key points logged in concepts.md.
- 2026-10-13: First commit (be1bf46). Added Jupyter, started Phase 3, step 1 done.
- 2026-10-13: Phase 3 step 2 done (key numbers).
- 2026-10-13: Phase 3 step 3 done (residuals).
- 2026-10-13: Phase 3 done (rest step, notebook verified, key points logged).
- 2026-10-13: Phase 4 started. Step 1 done (team-game aggregation, all 10 pbp seasons cached, 190 MB).
- 2026-10-13: Phase 4 step 2 done (lagged season-to-date + leakage tests).
- 2026-10-13: Phase 4 step 3 done (shrinkage + tests).
- 2026-10-13: Phase 4 step 4 done (game-level features).
- 2026-10-13: Phase 4 skeptic review: PASS, no blockers. Awaiting Josh's call on should-fix items.
- 2026-10-14: Applied skeptic findings and closed Phase 4. Note: the shell lost `CONDA_EXE` this session, so use `/Users/joshsilverman/opt/anaconda3/bin/conda run -n nfl-ats` if `conda run` errors.
- 2026-10-14: Phase 5 done (baselines + bootstrap).
- 2026-10-14: Phase 6 step 1 done (pipeline + fold-1 coefficients).
- 2026-10-14: Phase 6 step 2 done (first out-of-sample score).
- 2026-10-14: Phase 6 step 4 run. Awaiting Josh's keep or reject.
- 2026-10-14: Phase 6 done. `spread_line` rejected, and v1 stays kept.
- 2026-10-15: Phase 7 started. Step 1 done (walk-forward guards + fold-isolation tests).
- 2026-10-15: Phase 7 step 2 done (backtest report card).
- 2026-10-15: Phase 7 step 3 done (calibration).
- 2026-10-15: Exp #5 (garbage time) run. Awaiting Josh.
- 2026-10-15: Exp #5 rejected. Exp #6 (opponent adjustment) run, awaiting Josh.
- 2026-10-15: Started the diagnostic session. Step 0 summary given. Waiting for Josh's leak prediction and go-ahead for Step 1.
- 2026-10-15: Diagnostic Step 1 done. No bugs. Leak demo: 57.2% (Josh guessed 60%). `src/nfl_ats/diagnostics.py`, `notebooks/08-diagnostics.ipynb`, 25 tests.
- 2026-10-15: Diagnostic Step 2 done. Accuracy MDE is 1.0–2.4 pts (Josh guessed 1.5), and the observed effects are far below it. Log loss is the better yardstick. Proving profit needs a true rate of about 55.7%. 27 tests.
- 2026-10-15: Diagnostic Step 3 done. Features add no detectable signal beyond the spread (margin MSE +0.35%, partial r about 0). Skeptic: overclaim corrected (mistake #7), D3 logged. 29 tests.
