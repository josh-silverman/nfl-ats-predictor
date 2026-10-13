# NFL ATS Predictor

A machine learning model that predicts whether the NFL home team covers the closing spread. It's built end-to-end as a sports-analytics portfolio project: leakage-safe features from nflverse play-by-play, logistic regression, walk-forward backtesting (2019–2025), and a locked live ledger for the 2026 season.

Status: setup complete. See `docs/learning/progress.md`.

## Requirements
- **Python >= 3.10** (`python_requires >= 3.10`), because nflreadpy doesn't support older versions.
- Conda, with a dedicated env called `nfl-ats`.

## Environment setup
```bash
conda create -n nfl-ats python=3.12 -y
conda activate nfl-ats

# Always confirm the interpreter before installing:
python --version     # should print Python 3.12.x
which python         # should print .../envs/nfl-ats/bin/python

python -m pip install -r requirements.txt
python -c "import nflreadpy, pandas, sklearn, seaborn; print('ok')"
```

Don't use a `.venv` built from conda base. Base Python may be older than 3.10, which makes `pip install nflreadpy` fail with "no matching distribution" (see `docs/learning/mistakes.md`).

## Layout
| Path | Contents |
|---|---|
| `src/nfl_ats/` | Library code |
| `notebooks/` | EDA and lessons |
| `data/raw/`, `data/processed/` | Cached data (gitignored) |
| `reports/` | Experiment log and figures |
| `ledger/` | Locked weekly live predictions |
| `docs/plans/` | Design and roadmap |
| `docs/learning/` | Progress, concepts, mistakes |
| `docs/decisions.md` | Decision log |
