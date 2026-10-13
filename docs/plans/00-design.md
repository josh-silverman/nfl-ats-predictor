# 00 — Design (v1)

The rules in `CLAUDE.md` override anything here. This doc covers *what* we build and *why*.

## 1. Target
- **`home_cover`** is 1 if the home team covered the closing spread and 0 if it didn't.
- Let `margin = home_score − away_score`. The cover condition depends on the sign convention of `spread_line`, which is **unverified**.
  - The working hypothesis is that nflverse's `spread_line` is positive when the home team is favored, measured in home-margin terms. If so, the home team covers when `margin > spread_line`.
  - **Verification (Phase 2):** pick a real game Josh can look up (on Pro Football Reference or a news recap), compare the published closing line to `spread_line`, and confirm the sign. Then assert it across many games: the favorite's win rate should be well above 50%.
- **Pushes** (`margin == spread_line`) are excluded from training and evaluation, and their count is reported each season.
- The scope is the **regular season only**.

## 2. Data
- **Source:** `nflreadpy`.
  - `load_schedules()`: scores, `spread_line`, `home_rest` / `away_rest`, game IDs, and kickoff dates.
  - `load_pbp()`: play-level EPA and play counts.
- **Seasons:** 2016–2025 are history. 2026 is live.
- **Polars:** nflreadpy returns Polars DataFrames. Convert with `.to_pandas()` once, at the load boundary in `src/nfl_ats/data.py`. Everything downstream is pandas. *Why:* the stack is pandas and scikit-learn, and mixing the two libraries causes subtle dtype and index bugs.
- **Open, pending Josh's approval:** `.to_pandas()` and parquet caching both require **pyarrow**, which nflreadpy does *not* install. Verified 2026-10-02: both fail with `No module named 'pyarrow'`.
- **Caching:** save raw pulls to `data/raw/` as parquet with the pull date in the filename. Data revisions from nflverse are a known risk.
- **Validation checks:**
  - Expected game counts per season (256 per season before 2021, 272 from 2021).
  - No duplicate `game_id`.
  - Missing `spread_line` counted.
  - Team-abbreviation changes mapped (OAK→LV, SD→LAC, STL→LA).

## 3. Features (home − away differences)
| Feature | Built from | Notes |
|---|---|---|
| `off_epa_diff` | EPA/play on offense | Season-to-date, lagged |
| `def_epa_diff` | EPA/play allowed | Lower is better. Watch the sign |
| `pace_diff` | Offensive plays per game, or seconds per play | Season-to-date, lagged |
| `rest_diff` | `home_rest − away_rest` | Known before kickoff, so it's safe as-is |

- **Play filter:** use plays where `play_type` is pass or run. Decide in Phase 4 whether to drop garbage time and penalties, logged as a decision.
- **Lagging:** aggregate team-game stats, take the cumulative mean within team-season, then `shift(1)` so Week *k* sees only weeks before *k*.
- **Early-season plan (shrinkage):** `feature_wk = w·prior + (1−w)·current_to_date`, where `prior` is last season's value regressed toward the league mean and `w = k₀ / (k₀ + games_played)`. Week 1 is pure prior. `k₀` and the regression factor are tuned on training folds only.
- **Spread as a feature?** Not in v1. The market line already prices team strength. Phase 6 adds it as a single-change experiment, because how much the model adds beyond the line is *the* interesting question.

## 4. Model
`Pipeline([StandardScaler(), LogisticRegression()])`. It's simple, interpretable, and gives calibrated probabilities. Regularization strength `C` is tuned within the training folds only.

## 5. Evaluation
- **Walk-forward by season:** test seasons are **2019–2025**. Each fold trains on 2016 through the season before it, so the 2019 fold trains on 2016–2018.
- **Metrics:** ATS accuracy, log loss, and Brier score, per season and pooled, with **bootstrap 95% confidence intervals** (resampling games, 2,000+ resamples).
- **Calibration:** reliability plot plus Brier score decomposition.
- **Uncertainty, stated up front:**
  - With n ≈ 272, the standard error of accuracy is √(.25/272) ≈ 3.0 points, so the 95% interval is about ±6 points.
  - Pooled over 7 seasons (n ≈ 1,900), the interval is about ±2.3 points.
  - Even pooled, a true 54% model can't be reliably told apart from 52.4%.
- **Edge rule:** call it an edge only if the lower bound of the confidence interval is above 52.4%.

## 6. Baselines
1. **Always home:** pick the home team to cover every game.
2. **Always favorite:** pick the favorite to cover. Exclude pick'em games and count them.
3. **Coin flip:** expected accuracy is 50%, with log loss of ln 2 and Brier score of 0.25 at p = 0.5.

Every model result is shown next to all three.

## 7. Live season (2026)
- Each week, predictions are written to `ledger/2026_wkNN.csv` **before the first kickoff**, with a timestamp and the model version. Files are never edited after that.
- Weeks 1–4, already played when the ledger starts, go in `ledger/backfill/` and are labeled pseudo-live. They're excluded from the live record.

## 8. Risks
| Risk | Mitigation |
|---|---|
| Spread sign error that silently flips the target | Verify by hand on a real game, plus favorite-win-rate assertion |
| Leakage through rolling features or scaler fit | `shift(1)`, Pipeline, skeptic reviews |
| Small samples causing noise to read as signal | Bootstrap confidence intervals and the edge rule |
| Multiple testing (many experiments, one gets lucky) | Log every experiment and judge by count tried |
| Market efficiency (closing lines are sharp) | Expect about 50–52% and treat that as a real result |
| nflverse data revisions | Cache raw pulls with dates |
| Team relocations and abbreviations | Explicit mapping, validated in Phase 2 |
| Polars/pandas mismatches | Convert once, at the boundary |
