# Decisions

| # | Date | Decision | Choice | Why |
|---|---|---|---|---|
| 1 | 2026-10-02 | Target | Home team covers the closing spread (binary). Pushes excluded and counted | Simple, symmetric, and closing lines are the sharpest benchmark |
| 2 | 2026-10-02 | Environment | `nfl-ats` conda env, Python 3.12, no `.venv` | nflreadpy needs Python 3.10+, and conda base is older (see mistakes #1) |
| 3 | 2026-10-02 | Model v1 | scikit-learn Pipeline: StandardScaler + LogisticRegression | Interpretable, calibrated, and hard to overfit |
| 4 | 2026-10-02 | Walk-forward test seasons | 2019–2025, each trained on 2016 through the prior season | At least 3 training seasons. About 1,900 test games for tighter confidence intervals |
| 5 | 2026-10-02 | 2026 live record | Only predictions locked before kickoff count, starting with the next unplayed week. Weeks 1–4 are a labeled pseudo-live backfill | Predictions made after the fact aren't real out-of-sample evidence |
| 6 | 2026-10-02 | Spread as a feature | Excluded from v1, then tested as a single change in Phase 6 | Isolates what the stats add beyond the market |
| 7 | 2026-10-02 | Repo | Fresh start in `~/nfl-ats-predictor`. Earlier Desktop scaffold not reused | Clean history for the portfolio |
| 8 | 2026-10-02 | pyarrow | Added to requirements | Needed for Polars `.to_pandas()` and for reading/writing parquet in `data/` |
| 9 | 2026-10-02 | Jupyter | Deferred to Phase 3 | Not needed until EDA. Decide then |
| 10 | 2026-10-02 | Teaching focus | Coding, pandas, sklearn, and ML project structure come first. Stats and betting concepts only when a code decision depends on them. Josh writes code from Phase 2 with scaffolding and senior-style review | Josh's main learning goal is engineering skill, not betting theory |
| 11 | 2026-10-13 | Working style | Claude writes the code and explains the key patterns, and Josh reviews it. Phase-end "key things to know" logs replace the quizzes. Josh still signs off between phases | Josh's priority shifted to finishing the project. This replaces the code-ownership part of #10 |
| 12 | 2026-10-13 | `spread_line` sign | Positive = home favored (expected home margin). Enforced by `check_spread_sign()` | Checked by hand on 2016 CAR@DEN (closing CAR -3, spread_line -3), plus home teams win 67% when spread_line > 0 and 34% when it's < 0 |
| 13 | 2026-10-13 | Jupyter + package install | Added `ipykernel` for notebooks in Phase 3. Added `pyproject.toml` and `pip install -e .` so notebooks and scripts import `nfl_ats` directly | Charts are faster to iterate on inline. An editable install avoids sys.path hacks. Supersedes #9 |
| 14 | 2026-10-13 | Early-season shrinkage | Prior = last season regressed toward last season's league mean by `r`. Blend weight `w = k0/(k0+games_played)`. (`r`, `k0`) grid-searched per stat by MSE vs. the game's actual value, on training seasons only. 2015 pbp loaded only as 2016's prior (STL → LA added to `TEAM_MAP`) | Out-of-sample (2019–25, fit on 2016–18), shrinkage cuts off-EPA MSE by 35% after 1–2 games, 11% at 3–6, 4% at 7–12, and about 0 at 13+ (corrected after the Phase 4 skeptic review). So it matters early, and it is harmless late. Fold-1 params: off (0.5, 6), def (0.4, 12), pace (0.4, 32) |
| 15 | 2026-10-13 | Play filter + v1 features | Regular-season `pass`/`run` plays with EPA. Drop kneels, spikes, no-play penalties, and 2-pt tries. No garbage-time filter in v1. Features are home − away: `off_epa_diff`, `def_epa_diff`, `pace_diff`, `rest_diff` | Simple, standard filter. A garbage-time filter is a possible single-change experiment later. `rest_diff` stays in v1 per the design doc (corrects my Phase 3 note) |
| 16 | 2026-10-14 | `spread_line` as a feature | **Rejected.** v1 stays `FEATURES` (4 diffs) | Experiment #4: walk-forward 2019–25 paired difference is 0.0 pts accuracy [−0.7, +0.7], +0.0002 log loss. No gain for an extra feature. Josh agreed with the recommendation |
