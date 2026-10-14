# Experiment log

Every result goes here, including failures (CLAUDE.md, single-change rule). Each entry records the date, the change, the metrics with CIs, kept or rejected, and why.

**Types**
- **ATS:** a walk-forward backtest of the model. These are the only results that can support an edge claim, and they count toward the multiple-testing total.
- **Design:** a feature-construction check, not an ATS result. Recorded for honesty, but it makes no claim about beating the spread.

- **Baseline:** a reference bar, not an experiment. Not counted.

**Running count:** ATS experiments: **0**. Design checks: **2**.

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
