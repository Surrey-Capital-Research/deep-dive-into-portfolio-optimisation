from __future__ import annotations
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.scripts.run_equal_weight_backtest import run as run_equal_weight
from src.scripts.run_risk_parity_backtest import run as run_risk_parity
from src.scripts.run_mvo_backtest import run as run_mvo
from src.scripts.run_bl_backtest import run as run_bl

DATA_PATH = "data/uk_multi_asset_prices_clean.csv"
RFR_PATH = "data/risk_free_rate.csv"

PCT_METRICS = {"total_return", "CAGR", "volatility", "max_drawdown", "95% VaR", "95% CVaR", "avg_monthly_turnover"}


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    prices = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True).ffill()
    rfr = pd.read_csv(RFR_PATH, index_col=0, parse_dates=True).squeeze() / 100 # type: ignore
    return prices, rfr # type: ignore


def main() -> None:
    prices, rfr = load_data()

    runs = {
        "Equal Weight": lambda: run_equal_weight(prices, rfr),
        "Risk Parity": lambda: run_risk_parity(prices, rfr),
        "MVO": lambda: run_mvo(prices, rfr),
        "Black-Litterman": lambda: run_bl(prices, rfr),
    }

    results = {}
    for name, fn in runs.items():
        print(f"Running {name}...")
        results[name] = fn().metrics

    table = pd.DataFrame(results).T

    def fmt(col: str, val: float) -> str:
        return f"{val:.2%}" if col in PCT_METRICS else f"{val:.2f}"

    formatted = table.apply(lambda col: col.map(lambda v: fmt(col.name, v)))

    print("\nBacktest Results:\n")
    print(formatted.to_string())


if __name__ == "__main__":
    main()
