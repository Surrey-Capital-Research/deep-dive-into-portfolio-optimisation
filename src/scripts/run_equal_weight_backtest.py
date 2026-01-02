import pandas as pd
from src.backtesting.backtester import Backtester
from src.backtesting.strategies import EqualWeightStrategy

prices = pd.read_csv(
    "data/uk_multi_asset_prices_clean.csv",
    index_col=0,
    parse_dates=True,
)

strategy = EqualWeightStrategy(tickers=prices.columns)
bt = Backtester(prices=prices, strategy=strategy, initial_capital=100_000)

result = bt.run()

print(result.metrics)
print(result.equity_curve)
