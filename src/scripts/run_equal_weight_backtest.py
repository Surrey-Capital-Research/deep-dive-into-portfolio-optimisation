import pandas as pd
from src.backtesting.backtester import Backtester, BacktestResult
from src.backtesting.strategies import EqualWeightStrategy


def run(prices: pd.DataFrame, rfr: pd.Series | None = None) -> BacktestResult:
    strategy = EqualWeightStrategy(tickers=prices.columns) # type: ignore
    return Backtester(prices=prices, strategy=strategy, risk_free_rate=rfr if rfr is not None else 0.0).run()


if __name__ == "__main__":
    prices = pd.read_csv(
        "data/uk_multi_asset_prices_clean.csv", index_col=0, parse_dates=True
    ).ffill()
    print(run(prices).metrics)
