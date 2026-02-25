import pandas as pd
from src.backtesting.backtester import Backtester, BacktestResult
from src.backtesting.strategies import MVOStrategy


def run(prices: pd.DataFrame, rfr: pd.Series | None = None) -> BacktestResult:
    strategy = MVOStrategy(
        tickers=prices.columns.tolist(),
        cov_window=252,
        risk_free_rate=rfr, # type: ignore
    )
    return Backtester(prices=prices, strategy=strategy).run()


if __name__ == "__main__":
    prices = pd.read_csv(
        "data/uk_multi_asset_prices_clean.csv", index_col=0, parse_dates=True
    ).ffill()
    rfr = pd.read_csv(
        "data/risk_free_rate.csv",
        index_col=0,
        parse_dates=True
    )
    rfr = rfr.squeeze() / 100 # type: ignore
    print(run(prices, rfr).metrics) # type: ignore
