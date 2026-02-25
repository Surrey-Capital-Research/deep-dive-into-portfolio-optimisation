import numpy as np
import pandas as pd

def momentum_view_builder(past_prices: pd.DataFrame, decision_date) -> tuple:

    returns = past_prices.pct_change().dropna()

    if len(returns) < 252:
        return np.array([]), np.array([]), np.array([])
    
    momentum = returns.iloc[-252:-21].sum()

    n = len(past_prices.columns)
    P = np.eye(n)
    Q = momentum.to_numpy(dtype=float)
    Omega = np.array([]) # Built by BL

    return P, Q, Omega