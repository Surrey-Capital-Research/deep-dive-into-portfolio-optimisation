import pandas as pd
from src.backtesting.backtester import Backtester, BacktestResult
from src.backtesting.strategies import BlackLittermanStrategy
from src.models.mvo import make_mvo_optimiser
from src.models.bl_views import momentum_view_builder


def run(prices: pd.DataFrame, rfr: pd.Series | None = None) -> BacktestResult:
    tickers = list(prices.columns)
    market_weights = pd.Series(1.0 / len(tickers), index=tickers)
    optimiser = make_mvo_optimiser(risk_free_rate=rfr) # type: ignore
    strategy = BlackLittermanStrategy(
        market_weights=market_weights,
        risk_aversion=3.0,
        tau=0.05,
        view_builder=momentum_view_builder,
        optimiser=optimiser, # type: ignore
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
    print(run(prices, rfr).metrics) # type: ignore
