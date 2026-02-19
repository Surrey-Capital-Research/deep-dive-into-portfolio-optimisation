import os
import sys
import pandas as pd

# Allow imports from src/ when running from the research/ dir
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.backtesting.backtester import Backtester
from src.backtesting.strategies import RiskParityStrategy

# Sanity check for the risk parity allocation

def main():
    # Load the cleaned prod data
    prices = pd.read_csv(
        "data/processed/uk_multi_asset_prices_clean.csv",
        index_col=0,
        parse_dates=True,
    )

    # Init the RP strategy (using 1y cov window)
    strategy = RiskParityStrategy(
        tickers=prices.columns.tolist(),
        cov_window=252
    )
    
    # Run backtest (no execution friction modeled here)
    backtester = Backtester(prices=prices, strategy=strategy, initial_capital=100_000)
    result = backtester.run()

    print("\n--- RP Baseline Metrics ---")
    for k, v in result.metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
            
    print(f"\nEnding Capital: £{result.equity_curve.iloc[-1]:,.2f}")

if __name__ == "__main__":
    main()