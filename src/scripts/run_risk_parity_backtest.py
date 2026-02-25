import pandas as pd
from src.backtesting.backtester import Backtester, BacktestResult
from src.backtesting.strategies import RiskParityStrategy


def run(prices: pd.DataFrame, rfr: pd.Series | None = None) -> BacktestResult:
    strategy = RiskParityStrategy(tickers=prices.columns.tolist(), cov_window=252)
    return Backtester(prices=prices, strategy=strategy).run()


if __name__ == "__main__":
    prices = pd.read_csv(
        "data/uk_multi_asset_prices_clean.csv", index_col=0, parse_dates=True
    ).ffill()
    print(run(prices).metrics)
