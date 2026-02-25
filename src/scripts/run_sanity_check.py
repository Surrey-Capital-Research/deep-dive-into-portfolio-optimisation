import pandas as pd                                                                                         
from src.backtesting.backtester import Backtester
from src.backtesting.strategies import (                                                                    
    EqualWeightStrategy,                                                                                    
    RiskParityStrategy,                                                                                     
    MVOStrategy,                                                                                            
    BlackLittermanStrategy,                                                                                 
)
from src.models.mvo import make_mvo_optimiser
from src.models.bl_views import momentum_view_builder

# Load data
prices = pd.read_csv("data/uk_multi_asset_prices_clean.csv", index_col=0, parse_dates=True)
prices = prices.ffill()
tickers = list(prices.columns)

rfr = pd.read_csv("data/risk_free_rate.csv", index_col=0, parse_dates=True)
rfr = rfr.squeeze() / 100  # type: ignore

# Strategies
market_weights = pd.Series(1.0 / len(tickers), index=tickers)
optimiser = make_mvo_optimiser(risk_free_rate=rfr)  # type: ignore

strategies = {
    "Equal Weight": EqualWeightStrategy(tickers=tickers),
    "Risk Parity": RiskParityStrategy(tickers=tickers, cov_window=252),
    "MVO": MVOStrategy(tickers=tickers, cov_window=252, risk_free_rate=rfr),  # type: ignore
    "Black-Litterman": BlackLittermanStrategy(
        market_weights=market_weights,
        risk_aversion=3.0,
        tau=0.05,
        view_builder=momentum_view_builder,
        optimiser=optimiser, # type: ignore
    ),
}

# Run all and collect metrics
results = {}
for name, strategy in strategies.items():
    print(f"Running {name}...")
    bt = Backtester(prices=prices, strategy=strategy)
    results[name] = bt.run().metrics

# Print comparison table
metrics = ["total_return", "CAGR", "volatility", "Sharpe", "Sortino", "max_drawdown", "95% VaR", "95% CVaR"]
df = pd.DataFrame(results, index=metrics).T
df["total_return"] = df["total_return"].map("{:.1%}".format)
df["CAGR"] = df["CAGR"].map("{:.1%}".format)
df["volatility"] = df["volatility"].map("{:.1%}".format)
df["Sharpe"] = df["Sharpe"].map("{:.2f}".format)
df["Sortino"] = df["Sortino"].map("{:.2f}".format)
df["max_drawdown"] = df["max_drawdown"].map("{:.1%}".format)
df["95% VaR"] = df["95% VaR"].map("{:.1%}".format)
df["95% CVaR"] = df["95% CVaR"].map("{:.1%}".format)

print("\n--- Comparative Results ---")
print(df.to_string())