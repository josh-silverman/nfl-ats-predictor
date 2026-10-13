# Progress

**Current phase:** Phase 3 ✅ done. Waiting for Josh's go to start Phase 4.
**Next:** Phase 4, Feature engineering and leakage. Step 1: team-game aggregation from play-by-play (EPA/play, pace). Needs pbp for 2016–2025, and `TEAM_MAP` applied.
**Last updated:** 2026-10-13

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
- Phase 3 step 4: rest diff, saved to `reports/figures/rest_vs_cover.png`. The spread rises 1.3 → 3.4 pts with home rest edge, and the cover rate is flat (48–50%) in all buckets. Rest is a Phase 7 single-change candidate, not v1.
- Notebook verified end to end. Replaced `.style` (needs jinja2) with `.round(3)`.

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
