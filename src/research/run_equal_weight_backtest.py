import os
import sys
import pandas as pd

# Allow imports from src/ when running from the research/ dir
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.backtesting.backtester import Backtester
from src.backtesting.strategies import EqualWeightStrategy

# Sanity check for the EW baseline

def main():
    # load the cleaned prod data
    prices = pd.read_csv(
        "data/processed/uk_multi_asset_prices_clean.csv",
        index_col=0,
        parse_dates=True,
    )

    # Init 1/N strategy
    strategy = EqualWeightStrategy(tickers=prices.columns)
    
    # Run basic backtest
    bt = Backtester(prices=prices, strategy=strategy, initial_capital=100_000)
    result = bt.run()

    print("\n--- EW Baseline Metrics ---")
    for k, v in result.metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
            
    print(f"\nEnding Capital: £{result.equity_curve.iloc[-1]:,.2f}")

if __name__ == "__main__":
    main()