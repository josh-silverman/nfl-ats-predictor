# NFL ATS Predictor: Kickoff Prompt

I want to build a machine learning model that predicts NFL games against the spread. I also want this project to teach me how to build an end-to-end ML project the way a professional sports analyst would. You are both the builder and my tutor. Teaching me matters as much as the finished model.

This first session is planning and setup only. Do not write any modeling code yet.

## About me

I have a Master's in Business Analytics. I know Python, pandas, SQL, stats, and ML basics, but I'm rusty. I've built a PGA Tour prediction platform before, so I know about data leakage, as-of-date snapshots, backtesting, and Brier scores. I'm working toward a job in sports analytics, so this project will go in my portfolio. Talk to me like a mentor, not a textbook. Be honest about what matters and what doesn't.

## Environment (already set up, don't change it)

- I use a conda environment called `nfl-ats` with Python 3.12. Do not create a `.venv`.
- nflreadpy needs Python 3.10 or newer. My conda base is older, which broke my first attempt.
- Before installing anything, run `python --version` and `which python` and confirm they point to the `nfl-ats` env. If they don't, stop and tell me.
- Put `python_requires >= 3.10` and the env setup steps in the README so this can't happen again.

## How to talk to me

- Keep replies short. Give me one small step at a time.
- When I say I'm done for the day, update `progress.md` so `/status` can pick up where we left off next time.

## The project (keep v1 simple)

- Target. Did the home team cover the closing spread? This is a binary yes/no. Pushes get excluded and counted.
- Data. Use nflreadpy (the nflverse Python package). Get play-by-play for EPA and pace, and schedules for scores, spread lines, and rest days.
- Features, built as home minus away differences. Offensive EPA per play, defensive EPA per play allowed, pace, and rest days. Every feature must only use games played before that week's kickoff.
- Model. Logistic regression in scikit-learn, inside a Pipeline with scaling.
- Training data. The last 10 regular seasons (2016 to 2025).
- Live season. 2026 is the out-of-sample test. Predictions get locked before kickoff each week.
- Stack. nflreadpy, pandas, scikit-learn, seaborn. Don't add anything else unless you explain why first and I agree.

## Things I want handled carefully

1. Check the spread sign convention against a real game I can look up. Don't assume which way spread_line points. Teach me how you checked it.
2. nflreadpy may return Polars dataframes. Handle the conversion and explain it.
3. Leakage. Rolling features must be lagged. Week 1 has no current-season data, so we need a plan for it, like carrying over prior-season values with shrinkage.
4. Evaluation. Use walk-forward validation by season. Never use a random train/test split. Report accuracy against the spread, log loss, and Brier score, with bootstrap confidence intervals.
5. Baselines. Compare against always picking the home team, always picking the favorite, and a coin flip. The breakeven at -110 odds is 52.4%. A season is only about 272 games, so be upfront about how wide the uncertainty is. Never call something an edge unless the confidence interval supports it.
6. Single change at a time. Every model change gets backtested against the current version before it's kept. Log every result, including the ones that didn't work.

## How I want you to teach me

- Work in phases. Each phase is a lesson.
  1. Problem framing
  2. Data pull and validation
  3. EDA with seaborn
  4. Feature engineering and leakage
  5. Baselines
  6. Logistic regression
  7. Walk-forward backtest and calibration
  8. Weekly live predictions with a locked ledger
  9. Portfolio write-up
- Before each phase, explain the concept, why it matters for this problem, and how pros approach it.
- Ask me to predict outcomes before you show them. For example, ask what I think accuracy will be before we run the backtest.
- Ownership shifts as we go. In early phases you write the code and explain it. In middle phases you write the skeleton and I write the key parts. In later phases I write the code and you review it.
- Every phase ends at a gate. You give me a short recap and three check questions, then stop and wait for my sign-off before moving on.
- Keep these files updated.
  - `docs/learning/progress.md` for the current phase and what's next
  - `docs/learning/concepts.md` for concepts I've learned, which I'll rewrite in my own words
  - `docs/decisions.md` for each decision, what we chose, and why
  - `docs/learning/mistakes.md` for what broke and what it taught me. Start it with this first entry. My first setup used a `.venv` built from conda base Python, which was older than 3.10, so `pip install nflreadpy` failed with "no matching distribution." The lesson is to check package Python requirements and your interpreter version before building an environment.

## Agents and commands

- Set up a skeptic subagent in `.claude/agents/`. Its only job is to review work for leakage, overfitting, and claims that go beyond the evidence. Run it at the end of phases 4, 7, and 8.
- Create slash commands. `/next` moves to the next step. `/explain` goes deeper on the current concept. `/quiz` checks my understanding. `/status` summarizes where we are.

## What to do in this session

1. Read this whole prompt and ask me any questions you have before you start.
2. Write `CLAUDE.md` with the teaching rules, the leakage rules, and the single-change rule.
3. Write `docs/plans/00-design.md` (target, data, features, evaluation, baselines, risks) and `docs/plans/01-roadmap.md` (the phases broken into small units).
4. Set up the folder structure, the learning files, the skeptic agent, and the slash commands. Write `requirements.txt` and install it into the `nfl-ats` env after the version check above. Make sure `nflreadpy` imports cleanly.
5. Stop and walk me through the plan. Do not start Phase 1 until I say go.
