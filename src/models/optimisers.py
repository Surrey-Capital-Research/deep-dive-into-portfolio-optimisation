import pandas as pd
import numpy as np
from pypfopt import expected_returns, risk_models, EfficientFrontier, objective_functions

def run_mvo_optimization(
    prices_df: pd.DataFrame, 
    rf_rate: float, 
    max_position_size: float = 0.20,
    gamma_reg: float = 0.1
) -> dict:
    """
    Mean-Variance Optimization engine.
    Uses Ledoit-Wolf shrinkage and EMA returns to compute the Max Sharpe portfolio 
    under linear concentration constraints.
    """
    
    # EMA historical returns give higher weight to recent volatility regimes
    mu = expected_returns.ema_historical_return(prices_df, span=252)
    
    # Covariance shrinkage improves the condit num of the matrix,
    # preventing unstable weight allocations common with sample cov
    S = risk_models.CovarianceShrinkage(prices_df).ledoit_wolf()
    
    # Init EF with long-only constraints and max concentration limits
    ef = EfficientFrontier(mu, S, weight_bounds=(0.0, max_position_size))
    
    # L2 regularization encourages a non-sparse weight distribution
    ef.add_objective(objective_functions.L2_reg, gamma=gamma_reg) 
    
    try:
        # Solve for tangent portfolio (max sharpe)
        weights = ef.max_sharpe(risk_free_rate=rf_rate)
        return dict(ef.clean_weights())
        
    except Exception as e:
        # Fallback to naive diversification if solver fails to converge
        print(f"[WARNING] MVO solver failed: {e}. Defaulting to 1/N.")
        n_assets = len(prices_df.columns)
        return {ticker: 1.0 / n_assets for ticker in prices_df.columns}