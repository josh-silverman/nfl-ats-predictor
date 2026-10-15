# Mistakes

What broke, and what it taught me.

## 1. `.venv` built on an old Python (setup)
- **What happened:** My first setup used a `.venv` built from conda base Python, which was older than 3.10. `pip install nflreadpy` failed with "no matching distribution."
- **Lesson:** Check a package's Python requirements and your interpreter version before building an environment.
- **Fix:** I now use the dedicated `nfl-ats` conda env (Python 3.12) and confirm `python --version` and `which python` before any install.

## 2. `pip` under `conda run` hit the wrong Python (setup)
- **What happened:** `conda run -n nfl-ats pip install pyarrow` reported success, but `import pyarrow` failed. The bare `pip` resolved to the system Python 3.11 at `/Library/Frameworks/...`, not the env.
- **Lesson:** A successful install doesn't prove it went into the right interpreter. Verify with an import.
- **Fix:** Use `python -m pip`, which always installs into the interpreter you name.

## 3. Read the opening line instead of the closing line (Phase 2)
- **What happened:** For the spread-sign check, I said Denver was a 3-point favorite in the 2016 opener. That came from an early article about the opening line. The line moved, and Carolina closed at -3.
- **Lesson:** Opening and closing lines can differ, sometimes flipping the favorite. This project uses the **closing** line, so verify facts against the closing number.
- **Fix:** Cross-checked against the data's moneylines and win rates. The sign convention is now an assertion in `check_spread_sign()`.

## 4. Notebook used a feature that needs an uninstalled package (Phase 3)
- **What happened:** I wrote a push-rate table with `df.style.format(...)`. pandas' `.style` needs `jinja2`, which isn't in the env. It only surfaced when I ran the whole notebook top to bottom.
- **Lesson:** Building cells one at a time hides missing dependencies and stale state. Run the full notebook from a clean start before committing.
- **Fix:** Used `.round(3)` instead of adding a package.

## 5. Claimed more than the evidence showed (Phase 4)
- **What happened:** Decision #14 said shrinkage "still helps late in the season". The skeptic re-ran it: at 13+ games the gain was 0.0403 → 0.0402, essentially zero. My table had lumped weeks 6–16 together, which hid that.
- **Lesson:** Report results for each range of games, and don't summarize past what the numbers show. Coarse buckets can hide where an effect disappears.
- **Fix:** Corrected decision #14 with the per-range numbers. Results now go in `reports/experiments.md` with the actual table.

## 6. Wrote a test that assumed "no skill" means "resolution = 0" (Phase 7)
- **What happened:** I tested that an uninformative forecast has near-zero Brier resolution. It failed: 0.0025. Binning random outcomes into groups of 100 creates spread between bin averages by chance, about variance / games_per_bin.
- **Lesson:** Binned metrics have a noise floor. Compare a model's resolution to that floor, not to 0. Our v1 resolution (0.0013) sits right at its floor (0.0014).
- **Fix:** The test now checks the noise floor, and the notebook reports it next to the decomposition.

## 7. Overclaimed "no signal" from an underpowered test (Diagnostic Step 3)
- **What happened:** I concluded "the closing spread already contains everything these public stats know". The skeptic simulated our setup with a planted breakeven-sized signal, and the margin-MSE test caught it only about 11% of the time. I also called a p = 0.002 season "about what chance produces", which it isn't.
- **Lesson:** "Not detected" ≠ "not there". Before calling a result null, ask how often the test would catch an effect of the size you care about (its power). State the bound the CI supports, here "at most about breakeven", and name what wasn't tested (nonlinear effects, interactions, key numbers).
- **Fix:** Rewrote the Step 3 verdict and the D1/D2 lessons. Added the dispersion guard for paired log loss, and logged D3 with the power caveat.

## 8. Picked a dispersion guard that couldn't detect shrinking (Diagnostic Step 5)
- **What happened:** I guarded the keep rule with Brier resolution so that "just less confident" changes couldn't win. A unit test that shrank noisy predictions toward 50% got *kept*. Resolution with equal-count bins depends only on the predictions' rank order, and shrinking doesn't change rank order.
- **Lesson:** Know what a metric is invariant to. To catch "only less confident", use a pure ranking metric like AUC, which is unchanged by shrinking and moves only when the ranking improves. Write the adversarial test before trusting a guard.
- **Fix:** The keep rule now requires higher AUC. The test `test_shrinking_toward_half_is_not_kept` passes.

## 9. Claimed a model was "useful" against a strawman baseline (Diagnostic Step 6)
- **What happened:** I reported that the EPA win model "clearly" had signal because it beat the home-team base rate by 10 points. The skeptic built Elo from final scores alone, and it did just as well (63.6% vs. 63.2%). Our whole EPA pipeline added nothing over the simplest strength rating.
- **Lesson:** Always compare against the *strongest simple* baseline for the question, not the weakest. For ATS that was coin flip / always home. For winners it's Elo or point differential. Beating a strawman proves nothing.
- **Fix:** Built `src/nfl_ats/elo.py` (tested as-of), logged it as D7, and made Elo the benchmark every win-model idea must beat. Also: QB columns are now blocked as post-hoc in `OUTCOME_COLUMNS`, the hard-coded 1.6 is replaced by per-fold training means, and 02-next-steps.md is corrected.

## 10. Set a pass/fail threshold without checking its noise floor (M2)
- **What happened:** I pre-registered "Brier reliability ≤ 0.002" for S3. v2 scored 0.00234 and failed, and raw Elo (0.00208) would fail too. A simulation showed a *perfectly calibrated* model with these predictions exceeds 0.002 about 7% of the time. The threshold sat at about the 93rd percentile of pure noise.
- **Lesson:** Before fixing a threshold, simulate what a perfect model would score with your sample size, and set the bar relative to that noise distribution. **But don't use "the threshold was tight" to excuse a real problem:** the skeptic showed both models genuinely over-predict home wins (+2.4 to +3.0 pts, up to +6.7 in 2019–2020). Both things are true, and a write-up must say both.
- **Fix:** The verdict stands as pre-registered (raw Elo is the deliverable). I did **not** change the threshold for 2019–2025. A noise-aware calibration check can be pre-registered for the untouched 2026 live data.
