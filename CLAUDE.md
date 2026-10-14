# NFL ATS Predictor: Project Rules

The model predicts whether the **home team covers the closing spread** (a binary outcome). Claude builds the project and explains as it goes. **The priority is finishing the project.** Learning happens along the way, not through quizzes (decision #11). The source brief is `docs/kickoff-prompt.md`, the design is in `docs/plans/00-design.md`, and the phases are in `docs/plans/01-roadmap.md`.

## How to talk to Josh
- Act as a mentor, not a textbook. Be honest about what matters and what doesn't.
- **Keep replies short. One small step at a time.**
- Josh has an MSBA and is rusty but capable. Josh already knows leakage, as-of snapshots, backtesting, and Brier scores from a PGA Tour project, so don't re-teach those.
- When Josh says they're done for the day, update `docs/learning/progress.md` so `/status` can resume next session.

## Working style
1. **Claude writes the code**, in small steps, and Josh reviews it. After each step, give 2–4 lines on the key pandas or sklearn pattern or decision behind it, and what a pro would watch for.
2. Teach stats and betting concepts only when a code decision depends on them, in a few sentences.
3. **End of phase:** append the phase's "Key things to know" (3–6 bullets) to `docs/learning/concepts.md`, post a 3-line recap, then **stop until Josh says go**. No check questions. Never roll into the next phase on your own.
4. `/explain` and `/quiz` run only when Josh asks.
5. Josh still makes the calls that need a human: the spread-sign check in Phase 2, acting on skeptic findings, and keeping or rejecting each experiment.
6. Keep these files current:
   - `docs/learning/progress.md`: the current phase and what's next.
   - `docs/learning/concepts.md`: the key things to know from each phase.
   - `docs/decisions.md`: each decision, what was chosen, and why.
   - `docs/learning/mistakes.md`: what broke and what it taught.

## Leakage rules (non-negotiable)
- Every feature for a game uses **only games played before that week's kickoff**.
- Compute rolling and cumulative stats per team, then **lag them with `shift(1)`** within team-season before joining to games. A game's own result must never be in its own features.
- **Week 1** (and early weeks) carry over prior-season values, shrunk toward the league mean. The prior's weight decays as current-season games accumulate. Shrinkage parameters are fit on training seasons only.
- Scalers and models are fit **inside the training fold only**, which the scikit-learn `Pipeline` handles. Nothing is fit on the full dataset.
- **No random train/test splits, ever.** Validation is walk-forward by season.
- The closing spread is a feature or reference, never part of the target leakage path. The target comes from final scores against the closing line.
- If something looks too good (more than about 55% ATS over several seasons), assume it's leakage until proven otherwise.

## Single-change rule
- Change **one thing at a time**: a feature, a hyperparameter, a data fix.
- Every change is backtested against the **current kept version** on the same walk-forward folds.
- A change is kept only if the evidence supports it, and the decision goes in `docs/decisions.md`.
- **Log every result, including failures**, in `reports/experiments.md` (date, change, metrics with confidence intervals, kept or rejected, and why).

## Evaluation rules
- Metrics: **ATS accuracy, log loss, and Brier score**, each with **bootstrap 95% confidence intervals** (resampling games).
- Baselines, always shown: **always home**, **always favorite**, **coin flip**.
- The breakeven at -110 is **52.4%**. One season is about 272 games, so a single season's 95% interval on accuracy is roughly ±6 points. Say this plainly.
- **Never call something an edge unless the confidence interval's lower bound clears 52.4%.**
- Pushes are excluded from the target and **counted and reported**.

## Live season (2026)
- Predictions are **locked before kickoff** in `ledger/` and never edited after the fact.
- Weeks already played when the ledger starts are a **"pseudo-live" backfill**, labeled separately and never mixed into the live record.

## Environment and stack
- Use the conda env **`nfl-ats`** (Python 3.12). **Never create a `.venv`.**
- Before any install, run `python --version` and `which python` and confirm they point to `nfl-ats`. If they don't, **stop and tell Josh**.
- Always launch Claude Code from the nfl-ats env. Use `conda run -n nfl-ats` for any python or pip command.
- The stack is **nflreadpy, pandas, scikit-learn, seaborn**. Add nothing else unless you explain why and Josh agrees.
- nflreadpy returns **Polars** DataFrames. Convert with `.to_pandas()` at the load boundary, and keep everything downstream in pandas.

## Skeptic reviews
Run the `skeptic` subagent (`.claude/agents/skeptic.md`) at the end of **Phases 4, 7, and 8**, before the gate. Share its findings with Josh and fix any blockers.

## Layout
- `src/nfl_ats/`: library code.
- `tests/`: unit tests (stdlib `unittest`). Run with `conda run -n nfl-ats python -m unittest discover tests`.
- `notebooks/`: EDA and lessons.
- `data/raw/`, `data/processed/`: data, gitignored.
- `reports/`: experiment log and figures.
- `ledger/`: locked live predictions.
- `docs/`: plans, decisions, and learning files.
