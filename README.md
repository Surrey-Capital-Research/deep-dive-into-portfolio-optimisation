# A Deep Dive into Portfolio Optimisation Techniques

A rigorous empirical comparison of portfolio optimisation models applied to an 18-asset multi-asset portfolio.

## Project Objective

This project implements and backtests multiple portfolio optimisation techniques to understand their performance characteristics, trade-offs, and practical applicability. The analysis compares classical mean-variance optimisation, Bayesian approaches, and risk-based allocation methods against a simple equally weighted benchmark over a ten-year period (2015–2025).

## Asset Universe

**18-asset multi-asset portfolio (2015–2025):**

| Asset Class         | Tickers                                                                                                                                 |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| UK Equities (15)    | HSBC, Lloyds, Barclays, Shell, BP, Unilever, Tesco, Diageo, AstraZeneca, GSK, Rio Tinto, Glencore, National Grid, Vodafone, Rolls-Royce |
| UK Government Bonds | IGLT.L (iShares Core UK Gilts ETF)                                                                                                      |
| Precious Metals     | SGLD.L (Invesco Physical Gold ETC)                                                                                                      |
| Broad Commodities   | WCOG.L (WisdomTree Commodities ETF)                                                                                                     |

## Portfolio Optimisation Models

| Model                         | Description                                             | Status         |
| ----------------------------- | ------------------------------------------------------- | -------------- |
| **Equal Weight (1/N)**        | Naive diversification baseline                          | Complete       |
| **Risk Parity**               | Equal risk contribution allocation using Newton-Raphson | Complete       |
| **Mean-Variance (MVO)**       | Classic Sharpe ratio maximisation from first principles | Complete       |
| **Black-Litterman**           | Bayesian framework with 12-1 momentum views             | Complete       |

## Key Results (2015–2025)

Sharpe and Sortino ratios use annualised arithmetic mean return (μ_p = mean(r_t) × 252) in the numerator, not CAGR.

| Strategy        | Total Return | CAGR  | Volatility | Sharpe | Sortino | Max Drawdown | Avg Monthly Turnover |
|-----------------|-------------|-------|------------|--------|---------|--------------|----------------------|
| Equal Weight    | 93.98%      | 6.19% | 14.55%     | 0.36   | 0.47    | -30.59%      | 2.51%                |
| Black-Litterman | 62.74%      | 4.52% | 13.48%     | 0.26   | 0.35    | -27.16%      | 12.67%               |
| Risk Parity     | 41.19%      | 3.18% | 10.00%     | 0.19   | 0.23    | -23.74%      | 9.03%                |
| MVO             | 35.66%      | 2.81% | 11.07%     | 0.15   | 0.18    | -23.74%      | 34.67%               |

**Central finding:** The naive equally weighted portfolio outperforms all optimised strategies on both total return and Sharpe ratio over the full sample, consistent with DeMiguel et al. (2009). MVO's poor performance reflects the classical error-maximiser problem (Michaud, 1989).

## Repository Structure

```
├── data/                           # Price data (git-ignored)
│   └── uk_multi_asset_prices_clean.csv
├── src/
│   ├── backtesting/
│   │   ├── backtester.py           # Core backtest engine
│   │   └── strategies.py           # Strategy wrappers
│   ├── models/
│   │   ├── mvo.py                  # MVO from first principles (Lagrange multiplier)
│   │   ├── black_litterman.py      # BL posterior calculations
│   │   ├── bl_views.py             # Momentum view builder (12-1 month)
│   │   ├── risk_parity.py          # Newton-Raphson RP solver
│   │   └── base_strategy.py        # Abstract strategy interface
│   ├── scripts/
│   │   ├── run_equal_weight_backtest.py
│   │   ├── run_risk_parity_backtest.py
│   │   ├── run_mvo_backtest.py
│   │   ├── run_bl_backtest.py
│   │   └── run_bl_sensitivity.py   # BL tau × delta sensitivity grid
│   ├── visualizations/
│   │   ├── generate_results_figures.py  # All results figures (fig3–fig9)
│   │   └── plot_efficient_frontier.py   # Illustrative efficient frontier
│   └── main.py                     # Orchestrator: loads data, runs all strategies
├── reports/
│   ├── main/
│   │   └── main.tex                # LaTeX research report (near-final)
│   ├── refs/
│   │   └── refs.bib                # Bibliography
│   ├── images/plots/
│   │   ├── results/                # fig3–fig9, fig_bl_sensitivity
│   │   └── theory/                 # Efficient frontier illustration
│   └── under the hood/             # Mid-project presentation (20 slides)
├── planning/                       # Project planning docs
├── download_raw_data.py            # Data acquisition script
├── build_clean_prices.py           # Data cleaning script
└── pyproject.toml                  # Poetry dependencies
```

## Getting Started

### Prerequisites

- Python 3.11+
- Poetry for dependency management

### Installation

```bash
poetry install
poetry shell
```

### Data Pipeline

```bash
# Download raw price data from Yahoo Finance
python download_raw_data.py

# Build clean price matrix
python build_clean_prices.py
```

### Running All Backtests

```bash
# Run all four strategies and print comparison table
poetry run python src/main.py
```

### Running Individual Backtests

```bash
poetry run python src/scripts/run_equal_weight_backtest.py
poetry run python src/scripts/run_risk_parity_backtest.py
poetry run python src/scripts/run_mvo_backtest.py
poetry run python src/scripts/run_bl_backtest.py
```

### Generating Figures

```bash
# All results figures (fig3–fig9 + BL sensitivity heatmap)
poetry run python src/visualizations/generate_results_figures.py

# BL sensitivity analysis (tau × delta grid)
poetry run python src/scripts/run_bl_sensitivity.py
```

## Methodology

### Backtesting Engine

- Daily simulation with monthly rebalancing (month-end)
- Rolling 252-day window for covariance and return estimation
- Time-varying risk-free rate (3-month UK T-bill from FRED)
- No lookahead bias — strategies only see past prices at each decision date

### MVO Implementation

Analytical Lagrange multiplier solution for max-Sharpe and global minimum variance portfolios. Unconstrained (no weight cap) — concentration into recent outperformers is the expected result, not a bug (Michaud, 1989).

### Black-Litterman Implementation

12-1 month momentum views expressed as absolute expected returns. Posterior blends equilibrium prior (CAPM-implied) with momentum views via Bayesian update. Sensitivity analysis shows Sharpe is insensitive to tau (prior scaling); only risk aversion parameter delta materially affects results.

### Risk Parity

Newton-Raphson solver finding weights where each asset contributes equally to portfolio variance. Structurally overweights IGLT.L (low volatility ~6%) relative to equities (~18–25%), which is the defining feature of the approach.

## Deliverables

| Deliverable                   | Status           |
| ----------------------------- | ---------------- |
| Mid-project presentation      | Complete         |
| Comprehensive research report | Near-final draft |
| Summary blog post             | Pending          |

## Technology Stack

- **Python 3.11+** with pandas, NumPy, SciPy
- **Optimisation:** Custom Lagrange multiplier (MVO), Newton-Raphson (RP), cvxpy
- **Data:** Yahoo Finance via yfinance; FRED for risk-free rate
- **Visualisation:** Matplotlib, Seaborn
- **Reports:** LaTeX with custom APCR template

## References

- Markowitz, H. (1952). Portfolio Selection. *Journal of Finance*
- Black, F. & Litterman, R. (1992). Global Portfolio Optimization. *Financial Analysts Journal*
- Maillard, S., Roncalli, T., & Teiletche, J. (2010). The Properties of Equally Weighted Risk Contribution Portfolios. *Journal of Portfolio Management*
- Michaud, R. (1989). The Markowitz Optimization Enigma: Is Optimized Optimal? *Financial Analysts Journal*
- DeMiguel, V., Garlappi, L., & Uppal, R. (2009). Optimal Versus Naive Diversification. *Review of Financial Studies*
- He, G. & Litterman, R. (2002). The Intuition Behind Black-Litterman Model Portfolios. *Goldman Sachs Investment Management*
- Jegadeesh, N. & Titman, S. (1993). Returns to Buying Winners and Selling Losers. *Journal of Finance*

## Licence

This project is for educational and research purposes as part of AP Capital Research at the University of Surrey.
