# A Deep Dive into Portfolio Optimisation Techniques

A rigorous empirical comparison of portfolio optimisation models applied to an 18-asset multi-asset portfolio.

## Project Objective

This project implements and backtests multiple portfolio optimisation techniques to understand their performance characteristics, trade-offs, and practical applicability. The analysis compares classical mean-variance optimisation, Bayesian approaches, and risk-based allocation methods against a simple equally weighted benchmark.

## Asset Universe

**18-asset multi-asset portfolio (2015-2025):**

- **15 UK Equities (FTSE 100):** HSBC, Lloyds, Barclays, Shell, BP, Unilever, Tesco, Diageo, AstraZeneca, GSK, Rio Tinto, Glencore, National Grid, Vodafone, Rolls-Royce
- **1 UK Government Bonds ETF:** IGLT.L (iShares Core UK Gilts)
- **1 Precious Metals ETC:** SGLD.L (Invesco Physical Gold)
- **1 Broad Commodities ETF:** WCOG.L or AIGC.L (WisdomTree Commodities)

## Portfolio Optimisation Models

1. **Equally Weighted (1/N)** - Naive diversification baseline
2. **Markowitz Mean-Variance Optimisation** - Classic Sharpe ratio maximisation with practical constraints
3. **Black-Litterman** - Bayesian framework incorporating market equilibrium and investor views
4. **Risk Parity** - Equal risk contribution allocation

## Evaluation Framework

Models are evaluated on risk-adjusted performance (Sharpe, Sortino, Calmar ratios), drawdown characteristics, turnover, and concentration. All strategies rebalance monthly over a 10-year backtest period.

## Deliverables

- **Week 2:** Technical blog post showcasing backtesting engine and preliminary results
- **Week 4:** Comprehensive research report with full analysis and findings
