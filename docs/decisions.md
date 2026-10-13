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
