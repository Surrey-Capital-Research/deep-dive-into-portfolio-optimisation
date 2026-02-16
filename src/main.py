import pandas as pd
import numpy as np
import os
import sys
from dataclasses import asdict

# Make sure python can find the src folder
# otherwise throws a ModuleNotFoundError
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.backtesting.backtester import Backtester
from src.backtesting.strategies import EqualWeightStrategy, BlackLittermanStrategy, RiskParityStrategy
from src.visualizations.plotting import create_professional_tearsheet
from src.models.predictor import ProductionStockRegressor

# Config hardcoded
DATA_PATH = "data/uk_multi_asset_prices_clean.csv"
RESULTS_DIR = "results"
INITIAL_CAPITAL = 100_000.0
DATA_CUTOFF_DATE = "2025-12-31"

def load_data():
    # Sanity check to make sure we actually have the file
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Missing data file: {DATA_PATH}")
    
    prices = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    prices = prices.loc[:DATA_CUTOFF_DATE] # Just in case there are some future dates in there, 
    # we don't want them to mess with the backtest
    # Forward fill to handle random missing days
    # prevents the backtester from crashing on NaNs
    prices = prices.ffill() 
    return prices

def run_full_backtest():
    # 1. Grab data
    prices = load_data()
    all_columns = list(prices.columns) # Filter tickers
    tickers = [t for t in all_columns if t != "UK_10Y_Yield"] # Everything except gilt yield
    
    # 2. Pick benchmark for the chart
    # Just using the first asset as a baseline for now
    benchmark_prices = prices[tickers[0]]

    print("\n--- 🚀 Kicking off simulation ---")
    if "UK_10Y_Yield" in all_columns:
        print("--- 📌 Macro context: UK 10Y Yield detected for risk-adjusted metrics ---")
    # 3. Pick the Strategy
    # TODO: Swap this for BlackLittermanStrategy once the optimiser logic is ready.
    # Using Equal Weights (1/N) for now just to test the pipeline works.
    strat = RiskParityStrategy(tickers=tickers, cov_window=252) # 1 year rolling covariance for risk parity weights

    # 4. Engine setup
    bt = Backtester(
        prices=prices,
        strategy=strat,
        initial_capital=INITIAL_CAPITAL,
        rebalance_freq="ME", # Month End rebalancing
        # conservative estimate: 10bps friction per trade
        cost_model=lambda x: x.sum() * 0.0010 
    )

    # 5. Let it rip
    results = bt.run()
    
    # 6. Check how we did
    m = results.metrics
    print("\n📊 FINAL METRICS:")
    # using .get() just in case a metric is missing, avoids crash
    print(f"   Total Return: {m.get('total_return', 0):.2%}")
    print(f"   Sharpe Ratio: {m.get('Sharpe', 0):.2f}")
    print(f"   Max Drawdown: {m.get('max_drawdown', 0):.2%}")

    # 7. Make the charts
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    print("\n🎨 Saving tearsheet to results folder...")
    
    # The plotting lib expects a dictionary, but 'results' is a dataclass
    # converting it here so it doesn't break
    results_dict = asdict(results)
    
    create_professional_tearsheet(
        results=results_dict, 
        benchmark_prices=benchmark_prices, 
        title="Portfolio Strategy Audit (Ending {DATA_CUTOFF_DATE})",
    )

if __name__ == "__main__":
    run_full_backtest()