# Week of Apr 2 Task Report

**Period**: Mar 30 - Apr 2, 2026  
**Theme**: Enhanced2 Refactor, Backtest Comparison, and Iterative Optimization

---

## Summary

This week, the priority was to stop using underperforming branches from earlier attempts, consolidate development on `enhanced2`, and produce a stable version with clear, reproducible improvements.

Final outcomes this week:
- Refactored and iteratively optimized the `enhanced2` model.
- Upgraded analysis tooling with detailed window-level diagnostics.
- Produced a submission-ready `enhanced2` version with significantly improved score and win rate.

---

## Work Completed

### 1) Model Branch Consolidation

- Retired weaker intermediate branches from the main optimization path.
- Focused all active optimization work on `enhanced2`.

### 2) Analysis Tooling Upgrade (`performance_analyzer.py`)

Added and improved the following capabilities:
- Parameterized model analysis: `--model xxx --compare yyy`
- Window-level diagnostics:
  - `cheap_w` vs `expensive_w`
  - `buy_price_edge%`
  - `early_half_w` vs `late_half_w`
- Top-N worst-window root-cause analysis
- Yearly weight-shape diagnostics

### 3) Multi-round Optimization on `enhanced2`

Ran multiple rounds of targeted updates to balance score and robustness.
The final version achieved the best overall result this week and was selected as the weekly submission model.

---

## Performance Comparison (This Week)

| Version / Stage | Model Score | Win Rate | Exp-decay Percentile | Mean Excess | Notes |
|---|---:|---:|---:|---:|---|
| `enhanced` (reference baseline) | 61.06% | 68.13% | 53.99% | 5.60% | Previous main reference |
| `enhanced2` (mid-week low) | 52.44% | 65.31% | 39.58% | 2.93% | Overly conservative phase |
| `enhanced2` (final weekly submission) | **67.20%** | **93.27%** | 41.13% | **6.46%** | Best result this week |

> Note: The final weekly version clearly improved total score and win rate, satisfying this week’s submission objective.

---

## Key Findings

1. Improving a single local metric does not guarantee better total score; full backtest validation is required.
2. Intra-window allocation shape (early vs late, cheap vs expensive quantiles) has a major impact on performance.
3. More granular diagnostics significantly improved iteration efficiency and helped identify underperforming windows faster.

---

## Core Files Updated

- `my_model/model_development_enhanced2.py`
- `my_model/run_enhanced2_backtest.py`
- `my_model/performance_analyzer.py`

---

## Plan for Next Week

1. Create and continue development on `enhanced3`, using this week’s best `enhanced2` as the baseline.
2. Continue targeted optimization for recent-window quality and stability.
3. Keep per-iteration logs and comparison tables to ensure traceable evidence-based improvements.

---

**Status**: Weekly objective completed and ready for submission.
