import pandas as pd
from src.backtesting.backtester import Backtester
from src.backtesting.strategies import MVOStrategy

# Load your price data
prices = pd.read_csv(
    "data/uk_multi_asset_prices_clean.csv",
    index_col=0,
    parse_dates=True
)
rfr = pd.read_csv(
    "data/risk_free_rate.csv",
    index_col=0,
    parse_dates=True
)
rfr = rfr.squeeze()

# Create strategy
strategy = MVOStrategy(
    tickers=prices.columns.tolist(),               
    cov_window=252,
    risk_free_rate=rfr
)

# Run backtest
backtester = Backtester(prices=prices, strategy=strategy)
result = backtester.run()

# View results
print(result.metrics)
print(result.equity_curve)