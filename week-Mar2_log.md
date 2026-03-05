# Week 2 Progress Log

**Date**: March 4, 2026  
**Project**: Bitcoin Dynamic DCA Strategy Development  
**Phase**: Model Development & Backtesting

---

## Weekly Summary

This week completed model development and validation. After understanding the project architecture (template baseline, example_1 reference, tests suite), ran comparative backtests between baseline model (MA200-only, Model Score 51.74%) and example_1 (MVRV+MA200+Polymarket, Model Score 59.54%). Based on Week 1's EDA findings—the "MVRV Paradox" where Danger zone (Z>2.5) shows high short-term returns but represents unsustainable cycle-top speculation, and Polymarket data's limited coverage (2020-2025)—developed a customized model with two key modifications: (1) removed Polymarket signal and reallocated weights to MVRV 70% + MA200 30%, and (2) doubled the Danger zone negative boost coefficient from -0.5 to -1.0 to enhance cycle-top protection. Backtest results validated the strategy's effectiveness with Model Score of 59.83% (outperforming example_1's 59.54%), excess return of +5.77%, and superior performance across most key metrics. The model passed all validation requirements (Win Rate 60.85% > 50% threshold) and is ready for final report integration. Next week will focus on parameter fine-tuning or proceeding directly to final report preparation.

---

## Key Achievements

- ✅ Completed model development based on EDA insights
- ✅ Achieved Model Score of 59.83% (best among three models)
- ✅ Validated strategy meets submission requirements (Win Rate 60.85%)
- ✅ Confirmed EDA conclusions through backtest results

---

## Model Performance Comparison

| Metric | Baseline | Example 1 | My Model | Best |
|--------|----------|-----------|----------|------|
| Model Score | 51.74% | 59.54% | **59.83%** | My Model |
| Win Rate | **61.05%** | 60.31% | 60.85% | Baseline |
| Excess Return (mean) | +1.04% | +5.70% | **+5.77%** | My Model |
| Relative Improvement | +2.61% | +16.82% | **+17.03%** | My Model |

---

## Next Steps

- Parameter optimization (optional)
- Begin final report preparation
- Prepare weekly progress update for sponsor
