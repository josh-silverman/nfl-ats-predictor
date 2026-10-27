# Progress

**Current phase:** Pivot to win probability (decision #19). **M0–M1 done.** `src/nfl_ats/win_model.py` (`build_win_table`, `win_walk_forward`, `benchmark_predictions`), `compare_predictions(target=...)`, `tests/test_win_model.py`. 46 tests pass. Smoke test: 2,895 games, 10 ties, 1,865 test games, no scores looked at.
**M2 scored:** S1 PASS, S2 PASS, **S3 FAIL** (reliability 0.00234 > 0.002). By the pre-set rule, **raw Elo is the deliverable**. Post-hoc: raw Elo is 0.00208, and the threshold sat at about the 93rd percentile of noise (mistake #10). Logged as W3.
**M2 skeptic:** no leakage, no blockers. Wording corrected: both models fail S3, the miss is partly a real home-team bias (+2.4 to +3.0 pts), and S1/S2 were near-certain. 2026 calibration checks and the M4 baseline (raw Elo) were pre-registered in docs/project-rules.md before any 2026 outcome was viewed. Notebook outputs saved via `scripts/run_notebook.py`.
**M3 in progress:** `src/nfl_ats/ledger.py` + `scripts/predict_week.py` (as-of masking, skip-started, stale-data guard, no overwrite). 55 tests. Backfill weeks 1–5 (79 games, pseudo-live) in `ledger/backfill/`. **Week 6 locked live** 2026-10-15 14:51 ET (14 games, before the 20:15 TNF kickoff).
**Live record:** Week 6 (first live week) scored 2026-10-22: Elo picked 8 of 14 winners, Vegas 9 of 14. One week is noise; home teams went 6-8 vs. a predicted ~9. **Week 7 locked** 2026-10-22 15:50 ET (14 games, commit 233dda2, before the 20:15 TNF kickoff).
**Hardening sprint done (decision #20):** H1, H2, and H3 all rejected by the pre-set keep rule. **Raw Elo stays the deliverable**, with no shadow models and no Saturday lock. Win-model experiment count: 6.
**M5 drafted:** README case study written. `scripts/score_ledger.py` writes `reports/live_2026.md` using the pre-registered 2026 checks. Week 6 is locked twice (14:51 and 15:07: identical, and the earliest lock counts).
**Site live:** https://josh-silverman.github.io/nfl-ats-predictor/ (repo github.com/josh-silverman/nfl-ats-predictor, public). "Josh's NFL Elo" in `site/` (This Week, Record, Rankings, How It Works). Light redesign (one game per screen), plain language, Chart.js. `scripts/build_site.py`, Pages workflow. 72 tests.
**Weekly routine:** Thursday before kickoff: `predict_week.py --week N` → `build_site.py` → commit → push. Saturday (from Week 8, before the first Sunday kickoff): `predict_week.py --week N --qb` → `build_site.py` → commit → push. After Monday night: `score_ledger.py` → `build_site.py` → commit → push. Next: score Week 7 after MNF (Mon 2026-10-26); lock Week 8 before Thu 2026-10-29 kickoff. If `conda run` errors, call `~/opt/anaconda3/envs/nfl-ats/bin/python` directly.
**H4 kept (decision #21, 2026-10-22):** the QB-adjusted Elo (usual starter out via injury report or injured reserve, 90 Elo ≈ 3.6 pts) is the deliverable from Week 8. −0.0041 [−0.0076, −0.0004] vs. raw Elo, 6/7 seasons. Win-model experiment count: 7. The site shows a QB heads-up note on cards; from Week 8 the headline % is QB-adjusted. New **Saturday step**: `predict_week.py --week N --qb` → `build_site.py` → commit → push. 84 tests.
**Week 7 scored 2026-10-27:** Elo 8/14 (Vegas 9/14). Live total after 2 weeks: Elo 16/28 (57%), Vegas 18/28 (64%); home teams over-predicted by +18 pts (wide CI). Market moves: 5 toward / 4 away / 14 no move. Week 8 lock due Thu 2026-10-29 before 20:15 ET, QB lock Sat 10/31.
**Easy wins queue (Josh, 2026-10-23):** 1) market-move tracker ✅ done (decision #22: Record tab + `reports/live_2026.md`; first read 3 toward / 2 away / 8 no move), 2) playoff odds ✅ done (decision #23: Rankings tab, Brier 0.136 at Week 8 in 2019–2025), 3) rest/bye-week ✅ tested and rejected (H5, decision #24: +0.0004 vs. H4, 3/7; experiment count 8). Shipped a display-only rest gauge on cards with a 3+ day gap, 4) disagreement watch. 88 tests.
**Remaining:** S4 needs ≥ 8 locked weeks (week 13, about Dec 3), then the final README numbers and the final skeptic review (M5).
**Last updated:** 2026-10-27

## Done
- Planning docs, project rules, learning files, skeptic review checklist.
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
- 2026-10-02: Added the conda-run rule to docs/project-rules.md, installed pyarrow. Back to the step 2 push question.
- 2026-10-02: Changed teaching focus to coding/pandas/sklearn (decision #10). Summarized steps 3–4 and ran the Phase 1 gate.
- 2026-10-13: Switched to build mode (decision #11). Code is written in small reviewed steps, and phase-end key-points logs replace the quizzes. Phase 1 closed.
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
- 2026-10-15: Diagnostic Step 4 done. Learning curve and C sweep both converge on "always 50%" (a no-signal problem). Def EPA coef is positive in all folds, an unconfirmed lead. 31 tests.
- 2026-10-15: Diagnostic Step 5 done. `feature_eval.py` (keep rule with an AUC guard; mistake #8). 0/4 features earn their place (Josh predicted 0). v1 ≈ base rate. Exp #6 near-miss. 35 tests.
- 2026-10-15: Diagnostic Step 6 done. Win model 63.2% vs. base 53.2%. Skeptic: Elo matches it, so I built `elo.py` (D7: 63.6%, log loss 0.639). Ranked list in `docs/plans/02-next-steps.md`. Mistake #9. QB columns blocked as post-hoc. 40 tests.
- 2026-10-15: Josh approved the pivot. M0 done: docs rewritten, S1–S5 pre-registered before scoring v2.
- 2026-10-15: M1 done (win-model code + 6 tests). No metrics viewed before M2.
- 2026-10-15: M2 run once. v2 passes S1 and S2 and fails S3 by 0.00034, so raw Elo is the deliverable per the rule. `calibration_slope` added (48 tests).
- 2026-10-15: M2 skeptic review applied (honest S3 wording, 2026 checks pre-registered, stronger fold test, sklearn deprecation fix). 49 tests.
- 2026-10-15: M3 built. Backfilled weeks 1–5. Locked week 6 live before TNF.
- 2026-10-15: Hardening sprint: H1, H2 (QB, after a data audit), and H3 each pre-registered, run once, and rejected. Raw Elo stays. 61 tests.
- 2026-10-15: Live report script plus README case study. Backfill smoke test: Elo 65.4% on 78 games (wide CIs, not evidence). 63 tests.
- 2026-10-15: Built the static site and Pages workflow. Commit emails rewritten to the GitHub noreply address (Josh's choice; dates and contents verified identical). Public repo created, and the site is live at https://josh-silverman.github.io/nfl-ats-predictor/ . Repo-local git config: http.postBuffer plus the gh credential helper (the push failed without it).
- 2026-10-15: Redesigned the site: light theme with its own identity (cool neutrals, one blue accent, team-colored bars) and one game per screen with scroll snap. Cards are head-to-head with as-of team stats (record, rank, Elo, last game), our line + Vegas line at lock + closing line after kickoff, a confidence pill, and "Why this pick". 72 tests.
