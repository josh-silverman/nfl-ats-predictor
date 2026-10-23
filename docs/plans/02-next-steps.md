# 02 — What to try next (Diagnostic Step 6)

Written 2026-10-15 after diagnostic Steps 0–6, and corrected after the Step 6 skeptic review. Evidence: `notebooks/08-diagnostics.ipynb` and `reports/experiments.md` (D1–D7).

## The diagnosis in one paragraph
**ATS (home team covers the closing spread):**
- There's no pipeline bug (D1).
- Our test can detect gains of about 1–2.4 accuracy points, and realistic edges are smaller (D2).
- Season-to-date EPA, pace, and rest add no detectable information beyond the spread. Any linear signal is at most breakeven-sized (D3).
- More data and every regularization setting converge on "always 50%" (D4).
- 0 of 4 features earn their place (D5).

**The bottleneck is information, not the model.**

**Who wins:** the same features predict winners well above the base rate (63.2%, D6). But a **simple Elo from final scores does just as well** (63.6%, log loss 0.639, D7). The betting market is about 3 points better than both (66.5%). Our EPA features carry generic team-strength signal, which both Elo and the market already capture.

## Ranking
Each idea is scored on two things: **Info** (how likely it is to hold information the relevant benchmark lacks) and **Effort**. **For the win model, the benchmark to beat is Elo, not the base rate.** Data status: ✅ verified in the cached data, ❓ the loader exists but coverage and as-of timing are *unverified*, ❌ needs a new source.

### A. Change the question
| # | Idea | Info | Effort | Data | Why |
|---|---|---|---|---|---|
| A1 | **Elo win probability, benchmarked against the moneyline** | n/a (it *is* the baseline) | Done | ✅ | The simplest useful model: 63.6%, calibrated-looking, transparent, and it updates weekly. This is the deliverable. |
| A2 | Elo / power-rating spread ("our line") | n/a | Done | ✅ | RMSE 13.17 vs. the closing 12.72 and home field only 14.32. Differences from Vegas are **noise, not picks** (D3). Present it as a rating, never as a bet signal. |
| A3 | Totals (over/under) | Low–Med | Med | ✅ `total_line` | Same efficiency problem as ATS. |
| A4 | Grade ATS against the opening line | Med | **High** | ❌ no opening lines in nflverse (verified: schedules have one line set) | Needs a new paid or scraped source, and Josh's approval. |

### B. New information that Elo (final scores) doesn't have
| # | Idea | Info vs. Elo | Info vs. market | Effort | Data | Why / as-of risk |
|---|---|---|---|---|---|---|
| B1 | **Quarterback availability / QB-adjusted rating** | **High** | Low–Med | Med | ❓ `load_depth_charts`, `load_injuries`, `load_rosters_weekly` | Elo doesn't know when the starter is out, and the market does. This is the biggest gap. **Don't use `home_qb_id`/`away_qb_id` from schedules:** they record who *actually* started (post-hoc), so they're now blocked by `OUTCOME_COLUMNS`. Pre-game info only, e.g. the final pre-game injury status or the weekly depth chart, after verifying publication timing. |
| B2 | Other key injuries (OL, CB) | Med | Low | Med–High | ❓ `load_injuries` | Use the final pre-game report status, not who played. Coverage for 2025+ and timestamps are unverified. |
| B3 | EPA as an *add-on* to Elo (efficiency vs. results) | Low–Med | Low | Low | ✅ | D7 says EPA alone ≈ Elo. It might add a little *in combination*. One single-change test. |
| B4 | Recency weighting / Elo K tuning | Low | Low | Low | ✅ | Tuning on 2019–25, which we've already viewed, risks overfitting. Prefer published defaults. |
| B5 | Opponent adjustment (exp #6 near-miss), success rate | Low | Low | Low | ✅ | Same team-strength information Elo already has. |
| B6 | Travel, time zones, short weeks | Low | Low | Med | needs stadium coordinates | The market prices rest (Phase 3). |
| B7 | Line movement (open → close) / CLV | n/a | **High** | **High** | ❌ | The best ATS signal pros use, but it needs a new source. |

### C. Model changes and leads
| # | Idea | Note |
|---|---|---|
| C1 | Gradient boosting | Only after there's new information. D4 shows a no-signal problem, not a capacity problem. |
| C2 | Key-number handling (3 and 7) | Only relevant if a margin model ever has an edge. |
| C3 | Def EPA lead (ATS) | Coefficient positive in all 7 folds, and permutation importance positive in 6/7 held-out seasons. **Log loss improved in only 3/7 seasons** (D5). It was found by looking at 2019–25, so a "pre-registered" retest on those seasons isn't clean: count it as such, or test it only on 2026. |

## Ground rules from the skeptic review
- **2019–2025 is no longer a pristine holdout** for either target. We've looked at it many times. Every further test on it adds to the multiple-testing count (ATS: 4 experiments + 7 diagnostics; win model: 2). **The 2026 live ledger is the only clean test.**
- **Detectability on the win target is unproven.** Run `paired_power` on Elo vs. a candidate before promising that an increment can be seen. The one increment measured so far had a log-loss CI of about ±0.0055.
- **Never feed the spread or moneyline into a model benchmarked against them.** That makes "vs. market" circular.
- **Moneylines aren't verified as closing**, and their source changes around 2023. Call them "the schedule moneyline", not "the closing market".
- When predictions are locked before the line closes, say plainly that the market has later information.

## Recommendation
1. **Make Elo the main deliverable:** weekly win probabilities and Elo spreads for 2026, locked in the ledger before kickoff, scored against outcomes and the market. Simple, honest, and useful.
2. **Keep ATS as a documented, rigorous negative result:** "public team stats don't beat the closing line, and here's the evidence." It's a strong portfolio story.
3. **First improvement to try: B1, QB availability** from verified pre-game sources, as a single change on top of Elo. Check the data's timing before writing any feature.
4. **Parked:** A4 and B7 need a new data source and Josh's approval.

This changes the project's core question (`docs/project-rules.md`, design doc, roadmap). Those files change only after Josh signs off.
