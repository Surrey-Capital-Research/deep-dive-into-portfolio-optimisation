import os
import sys
import warnings
import logging
import pandas as pd
import numpy as np
from dataclasses import asdict

from src.visualizations.plotting import create_professional_tearsheet
from src.models.predictor import ProductionStockRegressor
from pypfopt import EfficientFrontier, objective_functions

# Fixes relative imports when running from root dir
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Suppress TensorFlow logs and warnings for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings("ignore")

from src.backtesting.backtester import Backtester
from src.backtesting.strategies import (
    EqualWeightStrategy, 
    BlackLittermanStrategy, 
    RiskParityStrategy, 
    MVOStrategy,
    RegimeSwitchingMVOStrategy
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Config
GLOBAL_CONFIG = {
    "DATA_PATH": "data/uk_multi_asset_prices_clean.csv",
    "RESULTS_DIR": "results",
    "INITIAL_CAPITAL": 100_000.0, 
    "DATA_CUTOFF": "2025-12-31",
    "RISK_FREE_TICKER": "UK_10Y_Yield",
    "REBALANCE_FREQ": "ME",
    "TRANSACTION_COST": 0.0005, # Institutional 5 bps
}

def sequential_view_builder(past_prices: pd.DataFrame, decision_date: pd.Timestamp):
    """Generates Black-Litterman views.
    Uses regression for base signals, overriden by a 100d SMA guardrail to prevent catastrophic bets."""
    tickers = [t for t in past_prices.columns if t != GLOBAL_CONFIG["RISK_FREE_TICKER"]]
    n_assets = len(tickers)
    Q, P = [], []
    
    for i, ticker in enumerate(tickers):
        try:
            current_price = past_prices[ticker].iloc[-1]
            sma_100 = past_prices[ticker].tail(100).mean()
            
            # Gen base signal from the regression model
            reg = ProductionStockRegressor(ticker=ticker, predict_days=30)
            reg.load_data(past_prices[ticker].tail(504)) 
            reg.train_hybrid_model() 
            expected_ret = reg.get_bl_view()
            
            # Macro override
            if current_price < sma_100:
                expected_ret = -0.05 
            
            # Add views only if signal is strong enough (>1.5)
            if abs(expected_ret) > 0.015: 
                Q.append(expected_ret)
                p_row = np.zeros(n_assets)
                p_row[i] = 1.0
                P.append(p_row)
        except Exception:
            pass # Skip asset if regression fails

    return np.array(P), np.array(Q), None

def bl_optimizer_wrapper(mu: pd.Series, cov: pd.DataFrame) -> pd.Series:
    """Optimiser wrapped for BL.
    Caps weights to prevent corner solutions."""
    try:
        # Cap positions at 40% so the AI can't sink the portfolio on one bad trade
        ef = EfficientFrontier(mu, cov, weight_bounds=(0.0, 0.40))
        ef.add_objective(objective_functions.L2_reg, gamma=0.05)
        weights_dict = ef.max_sharpe(risk_free_rate=0.0) 
        return pd.Series(dict(ef.clean_weights()))
    except Exception:
        # Fallback to equal weights if optimisation fails for any reason
        return pd.Series(1.0 / len(mu), index=mu.index)

def run_production_pipeline():
    if not os.path.exists(GLOBAL_CONFIG["DATA_PATH"]):
        raise FileNotFoundError(f"Missing core dataset: {GLOBAL_CONFIG['DATA_PATH']}")
    
    full_df = pd.read_csv(GLOBAL_CONFIG["DATA_PATH"], index_col=0, parse_dates=True)
    full_df = full_df.loc[:GLOBAL_CONFIG["DATA_CUTOFF"]].ffill()
    
    tickers = [t for t in full_df.columns if t != GLOBAL_CONFIG["RISK_FREE_TICKER"]]
    benchmark_prices = full_df[tickers[0]] 

    if not os.path.exists(GLOBAL_CONFIG["RESULTS_DIR"]):
        os.makedirs(GLOBAL_CONFIG["RESULTS_DIR"])

    # Init strategies
    strategies = {
        "1_Equal_Weight": EqualWeightStrategy(tickers=tickers),
        "2_Risk_Parity": RiskParityStrategy(tickers=tickers, cov_window=252),
        "3_Institutional_MVO": MVOStrategy(tickers=tickers, rf_ticker=GLOBAL_CONFIG["RISK_FREE_TICKER"], cov_window=126),
        "4_Black_Litterman_AI": BlackLittermanStrategy(
            market_weights=pd.Series(1/len(tickers), index=tickers), 
            risk_aversion=3.0, 
            tau=0.05,
            view_builder=sequential_view_builder,
            optimizer=bl_optimizer_wrapper, 
            cov_window=126
        ),
        "5_Regime_Switching_Alpha": RegimeSwitchingMVOStrategy(
            tickers=tickers, 
            rf_ticker=GLOBAL_CONFIG["RISK_FREE_TICKER"],
            cov_window=126,
            trend_window=50
        )
    }

    # Run backtests seq
    for strat_name, strategy in strategies.items():
        logger.info(f"\n🚀 Kicking off simulation for: {strat_name.replace('_', ' ')}")
        
        bt = Backtester(
            prices=full_df,
            strategy=strategy,
            initial_capital=GLOBAL_CONFIG["INITIAL_CAPITAL"],
            rebalance_freq=GLOBAL_CONFIG["REBALANCE_FREQ"],
            cost_model=lambda trade_size: trade_size.abs().sum() * GLOBAL_CONFIG["TRANSACTION_COST"]
        )

        results = bt.run()
        metrics = results.metrics

        logger.info(f"✅ {strat_name} Complete | Sharpe: {metrics.get('Sharpe', 0):.2f} | Max DD: {metrics.get('max_drawdown', 0):.2%}")
        
        create_professional_tearsheet(
            results=asdict(results), 
            benchmark_prices=benchmark_prices, 
            title=f"Strategy Audit: {strat_name.replace('_', ' ')}"
        )
        
        # os.replace will safely overwrite the old charts without crashing Windows
        source_file = os.path.join(GLOBAL_CONFIG["RESULTS_DIR"], "tearsheet.png")
        dest_file = os.path.join(GLOBAL_CONFIG["RESULTS_DIR"], f"{strat_name}_tearsheet.png")
        
        if os.path.exists(source_file):
            os.replace(source_file, dest_file)

if __name__ == "__main__":
    run_production_pipeline()