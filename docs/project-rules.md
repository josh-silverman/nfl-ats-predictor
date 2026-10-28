# NFL Win-Probability Model: Project Rules

**Since decision #19 (2026-10-15), the model predicts the probability that the home team wins** (ties excluded and counted). The original question, whether the home team covers the closing spread, is now the **archived ATS study**: a documented negative result (D1–D5 in `reports/experiments.md`). The design is in `docs/plans/00-design.md` (section 9 covers v2), the phases are in `docs/plans/01-roadmap.md`, and the evidence for the pivot is in `docs/plans/02-next-steps.md`.

These rules came first and override anything else in the repo. The pre-registered criteria below were written before the results they judge, and never changed after.

## Leakage rules (non-negotiable)
- Every feature for a game uses **only games played before that week's kickoff**.
- Compute rolling and cumulative stats per team, then **lag them with `shift(1)`** within team-season before joining to games. A game's own result must never be in its own features.
- **Week 1** (and early weeks) carry over prior-season values, shrunk toward the league mean. The prior's weight decays as current-season games accumulate. Shrinkage parameters are fit on training seasons only.
- Scalers and models are fit **inside the training fold only**, which the scikit-learn `Pipeline` handles. Nothing is fit on the full dataset.
- **No random train/test splits, ever.** Validation is walk-forward by season.
- **Never feed the spread or moneyline into a model that is benchmarked against them.** That makes "vs. market" circular.
- **Post-hoc columns are banned as features:** scores, margin, overtime, and the schedule's `home_qb_id`/`away_qb_id` (who *actually* started). `model.OUTCOME_COLUMNS` enforces this. Pre-game information must come from pre-game sources with known timing.
- Elo uses fixed published parameters and updates only after each game, so a game's probability never uses its own result. Anything fit (the logistic regression, scalers) is fit inside the training fold only.
- If something looks too good (beating the market's win-probability log loss, or more than about 55% ATS), assume it's leakage until proven otherwise.

## Single-change rule
- Change **one thing at a time**: a feature, a hyperparameter, a data fix.
- Every change is backtested against the **current kept version** on the same walk-forward folds.
- A change is kept only if the evidence supports it, and the decision goes in `docs/decisions.md`.
- **Log every result, including failures**, in `reports/experiments.md` (date, change, metrics with confidence intervals, kept or rejected, and why).

## Evaluation rules (win model, v2)
- Metrics: **log loss (primary), Brier score, and accuracy**, each with **bootstrap 95% confidence intervals**. Compare models with **paired** bootstraps on the same games.
- Benchmarks, always shown: **base rate** (training home-win rate), **raw Elo** (`src/nfl_ats/elo.py`), and the **schedule moneyline**, devigged. Never call it "the closing line": it isn't verified as closing, and its source changes around 2023.
- **Dispersion guard:** a change that only makes predictions less confident can win on log loss. Also require higher AUC, as in the `feature_eval.py` keep rule.
- Ties are excluded from the target and **counted and reported**.
- **2019–2025 has been viewed many times.** Every test on it adds to the experiment count. The 2026 live ledger is the only clean test.

## Success criteria (pre-registered 2026-10-15, before v2 was scored; never changed after seeing results)
The v2 model is `Pipeline(StandardScaler → LogisticRegression)` on the pregame `elo_diff`, walk-forward 2019–2025.
| # | Criterion | Pass if |
|---|---|---|
| S1 | Useful vs. knowing nothing | paired log loss vs. the base rate has its CI entirely below 0 |
| S2 | At least as good as raw Elo | paired log loss vs. raw Elo has CI upper bound < +0.005 (non-inferiority) |
| S3 | Honest probabilities | Brier reliability ≤ 0.002 **and** recalibration slope within [0.8, 1.2] |
| S4 | Real out-of-sample test | ≥ 8 weeks of 2026 predictions locked before kickoff and scored vs. outcomes, Elo, and the market |
| S5 | Professional build | all tests pass, notebooks run top to bottom, README case study done |

**If S2 or S3 fails, raw Elo becomes the deliverable.** Beating the market is **not** a criterion: report the gap honestly.

**Deliverable now (decision #21):** the **QB-adjusted Elo** (H4) from 2026 Week 8. Raw Elo was the deliverable through Week 7 and stays locked every Thursday as the baseline. Live results for the QB model count only on games it locked.

**M2 outcome (W3):** S1 PASS, S2 PASS, **S3 FAIL**, so raw Elo is the deliverable by rule. Raw Elo also exceeds the S3a threshold, so **no model passes S3 on 2019–2025.** Never describe the deliverable as calibrated on 2019–2025. Both models over-predicted home wins (+2.4 to +3.0 pts pooled).

**2026 live checks (pre-registered 2026-10-15, before any 2026 outcome was scored).** These are reported, not pass/fail, for both raw Elo and v2, on locked live weeks only (backfill reported separately):
- Log loss, Brier, and accuracy with bootstrap CIs, and paired differences vs. raw Elo and vs. the pull-time market.
- **Calibration-in-the-large:** mean(p) − actual home-win rate, with a bootstrap 95% CI. The failure mode seen in M2.
- **Reliability vs. its noise band:** observed Brier reliability (5 equal-count bins, given the small n) compared with the 95th percentile of reliability for a perfectly calibrated model with the same predictions (1,000 simulations).
- "Consistent with calibration" means the calibration-in-the-large CI includes 0 **and** reliability is ≤ that 95th percentile.
- **Market moves (pre-registered 2026-10-23, before any computation; decision #22).** "Did Vegas move toward us?" is reported, not pass/fail, on played live games.
  - **Inputs:** O = the headline pick's expected home margin (`elo_spread_qb` where a QB lock exists, else `elo_spread`). L = Vegas's home margin at the time *that* pick was locked (Thursday file, or the QB file's own pull). C = the schedule's final `spread_line`, which is not verified as the true close.
  - **Rules:** d = O − L, counted only if |d| ≥ 1.0. m = C − L. "Toward us" if m ≠ 0 and sign(m) = sign(d); "away" if the opposite sign; "no move" if m = 0.
  - **Reported:** counts, the share toward us among moved games with a bootstrap 95% CI, and the mean of m·sign(d) in points with a CI. Also the moneyline version (p_close − p_lock)·sign(p_ours − p_lock) on the same games. The raw-Elo Thursday lock is reported alongside as the baseline.
  - **Reading:** ≈ 50% toward us = no informational edge.

**M4 baseline (pre-registered):** any improvement (e.g. QB availability) is judged against **raw Elo**, the deliverable, with `compare_predictions(target="home_win")` and the same keep rule. Calibration-in-the-large is reported alongside.

## Archived ATS study
The ATS evaluation rules (52.4% breakeven, edge only if the CI lower bound clears it, pushes excluded and counted, baselines always home / always favorite / coin flip) still apply to any ATS result. The kept ATS model is the base rate (decision #18).

## Live season (2026)
- Each week, `scripts/predict_week.py` writes v2 win probabilities, raw Elo, the Elo spread, and the market probability at pull time (with the pull timestamp) to `ledger/`. **Locked before the week's first kickoff** by a git commit, and never edited after the fact. The script refuses to overwrite and refuses weeks whose games have started.
- Weeks already played when the ledger starts are a **"pseudo-live" backfill** in `ledger/backfill/`, labeled separately and never mixed into the live record.
- The market may have later information than our lock time. Say so when comparing.
- **QB lock (from Week 8, decision #21):** the Thursday file also carries `p_elo_qb` with Thursday's injury news. On Saturday, `predict_week.py --week N --qb` writes `ledger/qb/` for games not yet started, once the Friday injury report is in the data (it refuses otherwise). Per game, the **latest QB lock made before kickoff** counts. This rule was fixed before any Week 8 result.
- **Weekly routine (automated since 2026-10-27, decision #25):** `.github/workflows/weekly.yml` runs `scripts/auto_week.py` on a schedule and commits as `github-actions[bot]`. Locks happen daily at 14:00 UTC once a week's first kickoff is within 30 hours; the QB lock runs Saturday 13:00/16:00 UTC; scoring runs Tuesday 12:00 UTC. A failed run emails the repo owner. The manual steps below still work and are the backup:
  - Thursday before kickoff: `predict_week.py --week N` → `build_site.py` → commit → push.
  - Saturday before the 9:30 AM ET London game, or 1 PM ET otherwise: `predict_week.py --week N --qb` → `build_site.py` → commit → push.
  - After Monday night: `score_ledger.py` → `build_site.py` → commit → push.

## Environment and stack
- Use the conda env **`nfl-ats`** (Python 3.12). **Never create a `.venv`.**
- Before any install, run `python --version` and `which python` and confirm they point to `nfl-ats`. If they don't, **stop and tell Josh**.
- The stack is **nflreadpy, pandas, scikit-learn, seaborn**. Add nothing else unless you explain why and Josh agrees.
- nflreadpy returns **Polars** DataFrames. Convert with `.to_pandas()` at the load boundary, and keep everything downstream in pandas.


## Layout
- `src/nfl_ats/`: library code.
- `tests/`: unit tests (stdlib `unittest`). Run with `conda run -n nfl-ats python -m unittest discover tests`.
- `notebooks/`: EDA and lessons.
- `scripts/`: command-line entry points (e.g. the weekly ledger run).
- `site/`: the public static site "Josh's NFL Elo" (`scripts/build_site.py` writes `site/data/site-data.js`; GitHub Pages deploys `site/` on push). **Plain language on the site:** no stats jargon; technical depth lives in the README and reports.
- `data/raw/`, `data/processed/`: data, gitignored.
- `reports/`: experiment log and figures.
- `ledger/`: locked live predictions.
- `docs/`: these rules, plans, decisions, and learning files.
