# Experiment log

Every result goes here, including failures (CLAUDE.md, single-change rule). Each entry records the date, the change, the metrics with CIs, kept or rejected, and why.

**Types**
- **ATS:** a walk-forward backtest of the model. These are the only results that can support an edge claim, and they count toward the multiple-testing total.
- **Design:** a feature-construction check, not an ATS result. Recorded for honesty, but it makes no claim about beating the spread.

- **Baseline:** a reference bar, not an experiment. Not counted.
- **Diagnostic:** a deliberate check of the pipeline (e.g. an intentional leak). Never a candidate model. Not counted.

**Running count:** ATS experiments: **4**. Design checks: **2**.

---

## Baselines · 2026-10-14 · 2019–2025 walk-forward test window
1,871 games: **43 pushes excluded**, leaving 1,828 graded. 34 are neutral-site and 0 are pick'em. Bootstrap 95% CIs, 2,000 resamples, seed 0 (`evaluate.baseline_table`).

| Baseline | ATS accuracy | 95% CI | Edge (lower bound > 52.4%)? |
|---|---|---|---|
| Always home | 48.7% | 46.5–51.0% | No |
| Always favorite | 48.2% | 45.9–50.6% | No |
| Coin flip (seeded random picks) | 50.8% | 48.5–53.1% | No |
| Constant p = 0.5 | log loss 0.6931, Brier 0.2500 | (constant) | n/a |

- **Underdogs covered 51.8%** (the mirror of "always favorite"), and the CI includes 50%. That's not an angle. By season, favorites ranged from 44.5% (2020) to 53.4% (2024).
- **The coin flip landing at 50.8%** is a reminder that random picks wander ±2.3 points over 1,800 games.
- **Probability bar for the model:** log loss must beat 0.693 and Brier must beat 0.250. Those are the "know nothing" scores.
- **Neutral-site games:** 34 in the test window, where the "home" team covered 58.8% (n = 34, pure noise).

---

## #1 · 2026-10-13 · Design · Shrinkage vs. raw season-to-date
- **Change:** blend the prior season (regressed toward the league mean) into season-to-date stats with `w = k0 / (k0 + games_played)`, compared with raw season-to-date. See decision #14.
- **Setup:** (r, k0) grid-searched on 2016–2018 only. Graded on 2019–2025, which the fit never saw. Metric: MSE predicting each team's actual next-game offensive EPA/play.
- **Result (off EPA):**

  | Games played | Raw MSE | Shrunk MSE | Change |
  |---|---|---|---|
  | 0 (Week 1) | n/a (no data) | 0.0390 | raw has no value |
  | 1–2 | 0.0630 | 0.0411 | −35% |
  | 3–6 | 0.0487 | 0.0431 | −11% |
  | 7–12 | 0.0399 | 0.0382 | −4% |
  | 13–17 | 0.0403 | 0.0402 | ≈ 0 |

  Pace showed the same pattern (MSE after 1–2 games: 120.2 raw vs. 76.9 shrunk). No CIs, because this is a design check on about 3,700 team-games, not an ATS claim.
- **Decision:** **Kept.** It fills Week 1 and sharply reduces early-season noise. Late in the season it neither helps nor hurts.
- **Note:** the late-season numbers were first overstated in decision #14 and corrected after the skeptic review (mistake #5).

## #2 · 2026-10-13 · Design · Play filter (decision #15)
- **Change:** use regular-season `pass` and `run` plays with EPA. Drop kneels, spikes, no-play penalties, and 2-pt tries. No garbage-time filter.
- **Result:** **not tested.** Chosen by convention. Recorded so it's clear this choice was never measured.
- **Decision:** kept for v1. A garbage-time filter is a candidate single-change ATS experiment for Phase 7.

## #3 · 2026-10-14 · ATS · v1 model, single fold (train 2016–18 → test 2019)
- **Change:** first fit of `make_model()` (StandardScaler + LogisticRegression, C = 1) on `FEATURES`. Shrinkage refit inside the fold. This is a sanity check on one season, **not** the evaluation. Phase 7 runs all seven folds.
- **Result, 2019 (246 graded games, 10 pushes excluded):**

  | Metric | Model | 95% CI | Reference |
  |---|---|---|---|
  | ATS accuracy | 49.6% | 43.1–55.7% | coin 48.8%, home 43.1%, favorite 45.9% |
  | Log loss | 0.6982 | 0.6870–0.7101 | 0.6931 (constant 0.5) |
  | Brier | 0.2525 | 0.2470–0.2584 | 0.2500 (constant 0.5) |

  In-sample on 2016–18: 52.7% accuracy, log loss 0.6899. That's the usual train-vs-test gap: the fit looks better on data it has seen.
- **Decision:** no keep or reject (no change was compared). The **out-of-sample log loss and Brier are slightly worse than predicting 0.5**: the features as weighted here add no usable signal on 2019. One season is ±6 points, so this is not a verdict on the model. That comes from the full walk-forward in Phase 7.

## #4 · 2026-10-14 · ATS · Add `spread_line` as a feature (single change vs. v1)
- **Change:** v1 `FEATURES` + `spread_line`. Everything else identical: C = 1, shrinkage refit per fold.
- **Setup:** walk-forward 2019–2025 (`model.walk_forward`). Each season is predicted by a model trained on 2016 through the prior season. 1,828 graded games, both versions scored on the same games.
- **Result:**

  | | v1 (kept) | v1 + spread | New − v1 (paired 95% CI) |
  |---|---|---|---|
  | ATS accuracy | 51.3% [48.9, 53.5] | 51.3% [48.9, 53.5] | +0.0 pts [−0.7, +0.7] |
  | Log loss | 0.6947 [0.6920, 0.6975] | 0.6949 [0.6921, 0.6977] | +0.0002 [−0.0001, +0.0005] |
  | Brier | 0.2508 [0.2494, 0.2521] | 0.2509 [0.2495, 0.2523] | +0.0001 [−0.0000, +0.0002] |

  The `spread_line` coefficient in the 2025 fold is −0.027 per SD, essentially 0. Per-season accuracy moves by at most 1.6 points either way.
- **Also learned (the v1 baseline):** over 7 seasons, v1 hits 51.3%, with a CI that includes 50% and sits below 52.4%. Its log loss of 0.6947 is **worse than a constant 0.5 (0.6931)**. No edge.
- **Decision:** **Rejected** (Josh, 2026-10-14). The paired difference is 0 on every metric, so it adds a feature for no gain. v1 stays the kept version. See decision #16.

## #5 · 2026-10-15 · ATS · Garbage-time filter (single change vs. v1)
- **Change:** drop plays where the offense's win probability is outside [10%, 90%] (`GARBAGE_WP`, a common convention fixed in advance, not tuned). It applies to all play-based stats (EPA and pace). Everything else as in v1. Keeps 76.6% of plays, and every team-game keeps at least 6 plays.
- **Setup:** walk-forward 2019–2025, same 1,828 games as v1, paired bootstrap.
- **Result:**

  | | v1 (kept) | + garbage filter | New − v1 (paired 95% CI) |
  |---|---|---|---|
  | ATS accuracy | 51.3% [48.9, 53.5] | 50.9% [48.6, 53.2] | −0.4 pts [−2.1, +1.3] |
  | Log loss | 0.6947 | 0.6946 | −0.0001 [−0.0015, +0.0013] |
  | Brier | 0.2508 | 0.2507 | −0.0000 [−0.0007, +0.0007] |

  Per season it moves by −3.0 to +1.2 points, with no consistent direction. Prediction SD fell from 0.027 to 0.020, so the model is even less decisive.
- **Decision:** **Rejected** (Josh, 2026-10-15). Every paired CI straddles 0, and the filter adds complexity for no measurable gain. v1 stays kept (decision #17). `drop_garbage_time` stays in the code as an off-by-default option, so the experiment can be reproduced.

## #6 · 2026-10-15 · ATS · Opponent-adjusted EPA (single change vs. v1)
- **Change:** before computing season-to-date stats, adjust each game's EPA for the opponent's strength as of kickoff (`adjust_for_opponent`). Gains against a weak defense count for less, and EPA allowed to a strong offense counts for less. Two passes inside each fold, and both shrinkage fits use training seasons only. Tests prove only later games can change.
- **How much it moves the features:** per-game adjustment SD is 0.037 (off) and 0.061 (def), against a raw game SD of 0.205. At the feature level, adjusted vs. unadjusted correlation is 0.992 (off) and 0.960 (def). Schedules largely even out over a season.
- **Result (walk-forward 2019–2025, same 1,828 games, paired bootstrap):**

  | | v1 (kept) | + opponent adjustment | New − v1 (paired 95% CI) |
  |---|---|---|---|
  | ATS accuracy | 51.3% [48.9, 53.5] | 51.5% [49.3, 53.7] | +0.2 pts [−0.9, +1.4] |
  | Log loss | 0.6947 | 0.6941 | −0.0006 [−0.0014, +0.0002] |
  | Brier | 0.2508 | 0.2505 | −0.0003 [−0.0007, +0.0001] |

  All three metrics point slightly in the right direction, but every CI includes 0. Log loss stays above the constant-0.5 bar (0.6931).
- **Decision:** **Rejected** (2026-10-15, decision #18). It fails the pre-set keep rule's CI criterion in the Step 5 scorecard, despite improving in 6/7 seasons. Logged as a near-miss.

## D1 · 2026-10-15 · Diagnostic · Deliberate leak (no `shift(1)`)
- **What:** `diagnostics.leaky_walk_forward` includes each game's own stats in its season-to-date features. Same folds, same 1,828 games. Josh predicted 60%.
- **Result:** **57.2% [54.9, 59.5]**, log loss 0.6811, Brier 0.2440, probability SD 0.114. The honest v1 gives 51.3%, 0.6947, 0.2508, SD 0.027. By week: wk 1–4 61.5%, wk 5–9 59.2%, wk 10–18 54.1%. The leak is strongest early, when one game is a big share of the average.
- **Lesson:** a leak produces a CI whose lower bound *clears 52.4%*, a fake edge. (Corrected after the skeptic review: this demo shows what leakage looks like. It doesn't prove there's none, since leaks inflate scores and subtle ones can be quiet. The no-leak evidence is the `shift(1)` and fold-isolation tests plus the scaler check. The demo is also a diluted leak, because shrinkage down-weights the game's own stats.)
- **Side note (not a finding yet):** honest v1 by week is 54.4% (wk 1–4), 53.3% (wk 5–9), 48.6% (wk 10–18). That's a post-hoc subgroup with n of about 400–1,000 per bucket, so treat it as a lead for Step 3, not a result.

## D2 · 2026-10-15 · Diagnostic · Noise floor (minimum detectable effect)
- **What:** for each paired comparison, MDE = 2.8 × SE (80% power, 5% two-sided). Accuracy SE ≈ √(disagreement / n). Josh predicted 1.5 points.
- **Result (n = 1,828):**

  | Exp vs. v1 | Disagree | Acc. observed | Acc. MDE | Log-loss observed | Log-loss MDE |
  |---|---|---|---|---|---|
  | #4 +spread | 2.3% | +0.0 pts | 1.0 pts | +0.0002 | 0.0004 |
  | #5 garbage | 13.0% | −0.4 pts | 2.4 pts | −0.0001 | 0.0020 |
  | #6 opp-adj | 6.5% | +0.2 pts | 1.7 pts | −0.0006 | 0.0011 |

  - **v1 vs. constant 0.5:** log loss +0.0015 [−0.0012, +0.0043] and Brier +0.0008 [−0.0006, +0.0021]. Statistically indistinguishable from knowing nothing.
  - **Proving profit vs. 52.4% with 1,828 games** requires a true rate of about 55.7%. A true 54% model needs about 7,400 games (27 seasons), and a true 53% model about 51,000.
- **Lesson:** the effects we've chased are below what this sample can see. Paired **log loss** is a more sensitive yardstick than accuracy, **but only with a dispersion guard.** v1 is worse than a constant 0.5, so any change that just pulls predictions toward 0.5 "wins" on log loss with no new information. (The skeptic showed that shrinking v1's own predictions 25% toward 0.5 gives −0.0007, beating exp #6. #6 was checked and isn't this artifact.) Always report prediction SD and Brier resolution next to paired log loss. Accuracy vs. 52.4% can only ever confirm a very large edge.

## D3 · 2026-10-15 · Diagnostic · Where is the signal? (margin and residual analysis)
- **What:** walk-forward out-of-sample margin regressions (`diagnostics.margin_walk_forward`, 1,871 games with pushes kept), plus correlations of each feature with the spread residual, per season. Josh predicted a 1% error reduction.
- **Result:**
  - **Margin MSE:** spread + FEATURES vs. fitted spread-only is **+0.35% [−0.18%, +0.91%]**, slightly *worse*. Fitted spread-only vs. spread as-is is +0.16%. Per season: −0.74% to +1.20%.
  - **Partial correlation with margin, controlling for spread:** off EPA +0.016 [−0.030, +0.061], def EPA +0.014 [−0.031, +0.060], pace −0.029 [−0.074, +0.017], rest −0.007 [−0.052, +0.039].
  - **Season outliers:** off EPA in 2024 at r = +0.185 (p = 0.002, about a 6% event across 28 feature-seasons) is unstable across seasons.
  - **Weeks 1–4:** every CI includes 0 (±0.09 wide).
- **Power caveat (skeptic simulation):** the MSE test detects a breakeven-sized effect (r ≈ 0.075, about 52.4% ATS) only about 11% of the time, and its null expectation is about +0.29%. The CIs on the correlations are the real bound: the upper bounds of about 0.06–0.07 imply at most about breakeven.
- **Conclusion:** no detectable incremental signal from these four features entered linearly. Any linear effect is at most breakeven-sized. Nonlinear effects, interactions, key numbers, and subsets were not tested.

## D4 · 2026-10-15 · Diagnostic · Model diagnostics (learning curve, C sweep, calibration, coefficients)
- **Learning curve** (test 2022–25, trained on the last k seasons; Josh predicted "improves, then flattens"): test log loss goes 0.7025 (k=1) → 0.6967 → 0.6962 → 0.6973 → 0.6950 → 0.6944 (k=6). Train log loss rises 0.6850 → 0.6917. **Test pred SD falls from 0.064 to 0.020.** The improvement comes from shrinking toward 50%, not from learning signal, and test log loss never reaches 0.6931.
- **C sweep** (walk-forward 2019–25): the best log loss is 0.69324 at C = 1e-4, with pred SD 0.005, which is effectively "always 50%" (the dispersion artifact). Resolution is 0.0005–0.0017 at every C, against a noise floor of 0.0014. **No C creates signal, so C stays 1.0** (consistent with decision #17).
- **Calibration:** 91.4% of predictions are within 45–55%, and 52.7% are within 48–52%.
- **Coefficients by fold** (per SD):
  - off EPA: −0.001 to +0.064
  - **def EPA: +0.067 to +0.123 (positive in all 7 folds)**
  - pace: −0.026 to +0.031
  - rest: −0.009 to +0.073
  The folds use overlapping expanding windows, so stability is not independent evidence. Def EPA's out-of-sample partial r in D3 is +0.014 [−0.031, +0.060]. **It's an unconfirmed lead** ("market overreacts to bad defenses").
- **Conclusion:** the model isn't bias- or variance-limited in the usual sense. With these features, more data and every regularization setting converge on "always 50%". It's a no-signal problem, not a tuning problem.

## D5 · 2026-10-15 · Diagnostic · Feature scorecard (`feature_eval.py`)
- **Keep rule (fixed before running):** paired log-loss CI < 0, **and** log loss better in ≥ 5/7 seasons, **and** higher AUC (the dispersion guard). Josh predicted 0 of 4 features would earn their place.
- **Drop-one results** (with − without; negative = the feature helps):

  | Feature | Log-loss diff [95% CI] | Seasons helped | AUC with / without | Permutation importance (mean, folds > 0) | Earns place? |
  |---|---|---|---|---|---|
  | off_epa_diff | +0.00057 [−0.0003, +0.0015] | 2/7 | 0.503 / 0.508 | −0.0006, 2/7 | No |
  | def_epa_diff | −0.00034 [−0.0025, +0.0019] | 3/7 | 0.503 / 0.469 | **+0.0017, 6/7** | No (closest) |
  | pace_diff | +0.00035 [−0.0000, +0.0007] | 3/7 | 0.503 / 0.506 | −0.0003, 3/7 | No |
  | rest_diff | +0.00062 [−0.0001, +0.0014] | 2/7 | 0.503 / 0.505 | −0.0006, 1/7 | No |

- **v1 vs. base rate (no features):** +0.0014 [−0.0010, +0.0039], better in 3/7 seasons. The whole model adds nothing.
- **Exp #6 (opponent adjustment) vs. v1:** −0.00059 [−0.0014, +0.0002], better in 6/7 seasons, AUC 0.503 → 0.508. It fails the CI criterion, so it's **not kept by the pre-set rule**. Strongest near-miss so far.
- **Result: 0 of 4 earn their place (Josh: correct).** Decisions: the kept ATS model is reset to the base rate, and exp #6 is rejected (decision #18).

## D6 · 2026-10-15 · Diagnostic · Same features, different questions (win probability, power ratings)
- **What:** walk-forward 2019–25 with no betting line as input. A logistic regression for P(home wins) (ties dropped, 1,865 games) and a linear regression for margin (`diagnostics.win_and_margin_walk_forward`). The benchmark is moneyline probabilities with the vig removed, plus the closing spread.
- **Result:**

  | Model | Accuracy [95% CI] | Log loss | Brier | AUC |
  |---|---|---|---|---|
  | Base rate | 53.2% [50.8, 55.4] | 0.6933 | 0.2500 | n/a |
  | **Our win model** | **63.2% [60.9, 65.4]** | **0.6430** | **0.2257** | **0.681** |
  | Market (moneyline) | 66.5% [64.3, 68.7] | 0.6084 | 0.2105 | 0.727 |

  - **Model vs. base, paired:** accuracy +10.0 pts [+7.2, +12.8], log loss −0.050 [−0.067, −0.035].
  - **Model vs. market, paired:** accuracy −3.4 pts [−5.3, −1.5], log loss +0.035 [+0.025, +0.044].
  - **Margin RMSE:** predict 0 gives 14.38, home field only 14.30, **power ratings 13.20**, closing spread 12.72. Our predicted margin correlates 0.85 with the spread.
- **Lesson (corrected after the skeptic review):** the features carry generic team-strength signal. The market prices it into the spread, and a simple Elo from final scores captures it equally well (D7). "Beats the base rate" is a strawman for win prediction. Detectability of *increments* on this target is not yet shown: the skeptic measured a paired log-loss CI of about ±0.0055 between two strength models, wider than the ATS MDEs.
- **Counting:** D6 is **win-model experiment #1**. 2019–2025 has now been viewed for the win target, so it is no longer a pristine holdout. The 2026 live ledger is the only clean test.

## D7 · 2026-10-15 · Win model #2 · Elo baseline (`src/nfl_ats/elo.py`)
- **What:** FiveThirtyEight-style Elo from final scores only, with published parameters fixed up front (K = 20, home field 48 and 0 at neutral sites, a margin-of-victory multiplier, 1/3 regression between seasons). It starts in 2015. A unit test proves a game's own result never changes its pregame probability. Same 1,865 graded games as D6.
- **Result:**

  | Model | Accuracy [95% CI] | Log loss | Brier | AUC |
  |---|---|---|---|---|
  | EPA win model (D6) | 63.2% [60.9, 65.4] | 0.6430 | 0.2257 | 0.681 |
  | **Elo** | **63.6% [61.4, 65.8]** | **0.6387** | **0.2235** | **0.692** |
  | Market moneyline | 66.5% [64.3, 68.7] | 0.6084 | 0.2105 | 0.727 |

  - **EPA − Elo, paired:** accuracy −0.4 pts [−2.2, +1.3], log loss +0.0043 [−0.0037, +0.0126]. Indistinguishable, and Elo is nominally better.
  - **Elo − market, paired:** accuracy −3.0 pts [−4.8, −1.1], log loss +0.030 [+0.021, +0.040].
  - **Margin RMSE:** Elo spread 13.17, EPA 13.20, closing 12.72. With the per-fold training mean, home field only is 14.32.
  - The skeptic independently found a point-differential version of our pipeline gives 63.5% / 0.6435.
- **Lesson:** our EPA pipeline adds nothing over Elo for win prediction. **The simplest useful model is Elo.** New information (e.g. QB availability, which scores don't capture) must beat Elo, not the base rate.
- **Data notes (skeptic):** moneylines are not verified as closing, and their source changes around 2023 (average vig 2.5% → 4.3%). Devigging is proportional. 13 games have p_market = 0.5.

## W3 · 2026-10-15 · Win model #3 · v2 scored once against the pre-registered criteria (M2)
- **Model:** `Pipeline(StandardScaler → LogisticRegression(C=1))` on `elo_diff`, walk-forward 2019–2025 (`notebooks/10-win-model.ipynb`). 1,865 decided games, 6 ties excluded and counted.
- **Report card (pooled):**

  | Model | Accuracy [95% CI] | Log loss | Brier | AUC |
  |---|---|---|---|---|
  | Base rate | 53.2% [50.8, 55.4] | 0.6933 | 0.2500 | n/a |
  | Raw Elo | 63.6% [61.4, 65.8] | 0.6387 | 0.2235 | 0.692 |
  | **v2 (Elo + logistic)** | **63.2% [61.0, 65.5]** | **0.6391** | **0.2237** | **0.691** |
  | Market (moneyline) | 66.5% [64.3, 68.7] | 0.6084 | 0.2105 | 0.727 |

- **Criteria:**
  - **S1 PASS:** vs. the base rate, log loss −0.0541 [−0.0698, −0.0392].
  - **S2 PASS:** vs. raw Elo, +0.0004 [−0.0011, +0.0019], under the +0.005 limit.
  - **S3 FAIL:** reliability 0.00234 > 0.002. The slope of 0.947 [0.79, 1.12] is in [0.8, 1.2].
- **Decision (pre-registered rule): raw Elo is the deliverable.**
- **Post-hoc context, not used to change the verdict:**
  - Raw Elo's reliability is also above the limit: 0.00208, slope 0.925.
  - The market's reliability is 0.00075, slope 1.07.
  - A perfectly calibrated model with raw Elo's predictions exceeds 0.002 in about 7% of simulations (median 0.0011, 95th percentile 0.0021). The threshold was set at about the 93rd percentile of pure noise (mistake #10).
- **Gap to the market:** v2 is −3.4 accuracy points [−5.2, −1.5] and +0.031 log loss [+0.021, +0.040].
- **Corrections after the M2 skeptic review:**
  - **Both models fail S3a**, so the deliverable is raw Elo *by rule only*. It is **not** shown to be better calibrated: the order flips with bin count, and the v2 − Elo reliability CI is [−0.0016, +0.0018].
  - **The S3 miss is partly a real home-team bias:**
    - Mean predicted vs. actual home-win rate: v2 +2.4 pts, Elo +3.0, market +1.4.
    - By season, v2 vs. Elo: 2019 +6.7 / +4.6, 2020 +6.7 / +6.2, 2021 +3.7 / +4.5, 2022 −1.4 / +0.1, 2023 −0.5 / +0.8, 2024 +1.5 / +2.7, 2025 +0.8 / +2.1.
    - Cause: home-field advantage learned from 2016–2018 (home win rate 0.58–0.60) didn't hold in 2019–2021.
    - v2's reliability is at p ≈ 0.04 under its own perfectly-calibrated null.
  - **S1 and S2 were near-certain at registration time:** raw Elo's 0.6387 was reported in D7 six minutes before the criteria were written. Only S3 was a genuinely uncertain test.
  - Executed notebook outputs are now saved (`scripts/run_notebook.py`) and committed.

