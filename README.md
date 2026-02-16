# A Deep Dive into Portfolio Optimisation Techniques

A rigorous empirical comparison of portfolio optimisation models applied to an 18-asset multi-asset portfolio.

## Project Objective

This project implements and backtests multiple portfolio optimisation techniques to understand their performance characteristics, trade-offs, and practical applicability. The analysis compares classical mean-variance optimisation, Bayesian approaches, and risk-based allocation methods against a simple equally weighted benchmark.

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
| **Equal Weight (1/N)**  | Naive diversification baseline                          | ✓ Implemented |
| **Risk Parity**         | Equal risk contribution allocation using Newton-Raphson | ✓ Implemented |
| **Mean-Variance (MVO)** | Classic Sharpe ratio maximisation                       | In Progress    |
| **Black-Litterman**     | Bayesian framework with quantitative views              | Scaffolded     |

## Repository Structure

```
├── data/                           # Price data (git-ignored)
│   └── uk_multi_asset_prices_clean.csv
├── src/
│   ├── backtesting/
│   │   ├── backtester.py           # Core backtest engine
│   │   └── strategies.py           # Strategy implementations
│   ├── models/
│   │   ├── black_litterman.py      # BL posterior calculations
│   │   ├── risk_parity.py          # Newton-Raphson RP solver
│   │   └── predictor.py            # FFT/Ridge predictor for BL views
│   ├── scripts/
│   │   ├── run_equal_weight_backtest.py
│   │   ├── run_risk_parity_backtest.py
│   │   └── run_bl_inference.py
│   └── visualizations/
│       └── plotting.py             # Tearsheet generator
├── reports/
│   └── under the hood/             # Mid-project presentation
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
# Install dependencies
poetry install

# Activate virtual environment
poetry shell
```

### Data Pipeline

```bash
# Download raw price data from Yahoo Finance
python download_raw_data.py

# Build clean price matrix
python build_clean_prices.py
```

### Running Backtests

```bash
# Equal Weight benchmark
python src/scripts/run_equal_weight_backtest.py

# Risk Parity
python src/scripts/run_risk_parity_backtest.py
```

## Methodology

### Backtesting Engine

- Daily simulation with monthly rebalancing (month-end)
- Rolling 252-day window for covariance estimation
- No lookahead bias — strategies only see past prices at each decision date

### Risk Parity Implementation

Uses Newton-Raphson method to find weights where each asset contributes equally to portfolio risk:

$$
\text{RC}_i = w_i \times (\Sigma w)_i = c \quad \forall \, i
$$

Converges in ~5-10 iterations from inverse-volatility initial guess.

### Black-Litterman Views

Quantitative views generated using:

- Ridge regression for price trends
- FFT decomposition for cyclical patterns
- Optional LSTM for sequential learning

## Evaluation Metrics

- **Return:** Total return, CAGR
- **Risk:** Annualised volatility, Maximum drawdown
- **Risk-Adjusted:** Sharpe ratio, Sortino ratio, Calmar ratio
- **Portfolio:** Turnover, concentration, effective number of assets

## Deliverables

| Deliverable                   | Status      |
| ----------------------------- | ----------- |
| Mid-project presentation      | ✓ Complete |
| Comprehensive research report | Pending     |
| Summary blog post             | Pending     |

## Technology Stack

- **Python 3.11+** with pandas, NumPy, SciPy
- **Optimisation:** Custom Newton-Raphson, cvxpy (for MVO)
- **Data:** Yahoo Finance via yfinance
- **Visualisation:** Matplotlib, Seaborn
- **Reports:** LaTeX with custom APCR template

## References

- Markowitz, H. (1952). Portfolio Selection. *Journal of Finance*
- Black, F. & Litterman, R. (1992). Global Portfolio Optimization. *Financial Analysts Journal*
- Maillard, S., Roncalli, T., & Teiletche, J. (2010). The Properties of Equally Weighted Risk Contribution Portfolios. *Journal of Portfolio Management*
- Chaves, D., Hsu, J., Li, F., & Shakernia, O. (2012). Efficient Algorithms for Computing Risk Parity Portfolio Weights. *Journal of Investing*

## Licence

This project is for educational and research purposes as part of Surrey Capital Research at the University of Surrey.
