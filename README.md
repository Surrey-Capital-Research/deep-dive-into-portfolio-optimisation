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
This repository implements a monthly rebalanced equal-weight benchmark portfolio and computes standard performance metrics used in portfolio analysis.

1.1 Data preparation

- price_matrix.py - builds a clean price matrix (date × asset) using adj close
Output: price_matrix.csv 
- return_matrix.py - computes simple returns from the price matrix
Output: return_matrix.csv - all risk and performance statistics are derived from returns rather than prices

1.2 Statistical inputs

- stats.py - computes core empirical statistics from the return matrix (mean return matrix and covariance matrix)
Outputs: mean_returns.csv
         covariance_matrix.csv

1.3 Portfolio construction

- weights.py - defines portfolio weights (1/18)
Output: weights.csv - weights are stored explicitly to ensure transparency

1.4 Portfolio returns

- portfolio_returns.py - computes portfolio returns using fixed weights
Output: portfolio_returns.csv
- portfolio_returns_rebalanced.py - computes portfolio returns under monthly rebalancing
Output: portfolio_returns_rebalanced.csv

1.5 Performance and risk metrics

- cumulative_returns.py - computes cumulative portfolio wealth
Output: cumulative_returns.csv
- max_drawdown.py - computes maximum drawdown from cumulative wealth
Output: max_drawdown.csv - this captures downside risk not visible in volatility alone
- volatility.py - computes daily volatility and derives annualised volatility
Outputs: daily_volatility.csv
         annual_volatility.csv
- sharpe.py - computes the Sharpe ratio using mean returns and volatility
Output: sharpe_ratio.csv

2. **Markowitz Mean-Variance Optimisation** - Classic Sharpe ratio maximisation with practical constraints
3. **Black-Litterman** - Bayesian framework incorporating market equilibrium and investor views
4. **Risk Parity** - Equal risk contribution allocation

## Evaluation Framework

Models are evaluated on risk-adjusted performance (Sharpe, Sortino, Calmar ratios), drawdown characteristics, turnover, and concentration. All strategies rebalance monthly over a 10-year backtest period.

## Deliverables

- **Week 2:** Technical blog post showcasing backtesting engine and preliminary results
- **Week 4:** Comprehensive research report with full analysis and findings
