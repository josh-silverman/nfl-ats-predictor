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
