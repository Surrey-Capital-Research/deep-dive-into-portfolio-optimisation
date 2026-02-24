from __future__ import annotations
import numpy as np, pandas as pd

def min_variance_weights(cov: np.ndarray) -> np.ndarray:

    n = cov.shape[0]
    active = np.arange(n)

    for _ in range(n):

        cov_a = cov[np.ix_(active, active)]
        ones = np.ones(len(active))
        
        # np.linalg.solve can fail if cov_a is near singular, which may happen with correlated assets, we add a small deviation
        cov_a = cov_a + 1e-8 * np.eye(len(active))
        z = np.linalg.solve(cov_a, ones)
        
        if z.sum() <= 0:
            w = np.zeros(n)
            w[active] = 1.0 / len(active)
            return w
        
        w_a = z / z.sum()

        neg = w_a < 0 
        if not neg.any():
            break

        active = active[~neg]

    w = np.zeros(n)
    w[active] = np.maximum(w_a, 0.0) # type: ignore
    w = w / w.sum()

    return w


def max_sharpe_weights(
    mu: np.ndarray, 
    cov: np.ndarray, 
    risk_free_rate: float, 
) -> np.ndarray:

    n = len(mu)
    active = np.arange(n)

    for _ in range(n):

        mu_a = mu[active]
        cov_a = cov[np.ix_(active, active)] # submatrix for active assets
        cov_a = cov_a + 1e-8 * np.eye(len(active)) # Add regularisation
        excess = mu_a - risk_free_rate

        z = np.linalg.solve(cov_a, excess) # solve cov_a @ z = excess
        if z.sum() <= 0:
            return min_variance_weights(cov) # Fall back to min variance if assets earn below rf
        
        w_a = z / z.sum()

        neg = w_a < 0
        if not neg.any():
            break # Solution is long-only, done

        active = active[~neg] # remove negative weight assets

    w = np.zeros(n)
    w[active] = np.maximum(w_a, 0.0)  # type: ignore
    w = w / w.sum()

    return w


def make_mvo_optimiser(risk_free_rate):

    def optimiser(mu: pd.Series, cov: pd.Series):
        tickers = list(mu.index)
        mu_annual = mu.to_numpy(dtype="float") * 252
        cov_annual = cov.to_numpy(dtype="float") * 252

        weights = max_sharpe_weights(mu_annual, cov_annual, risk_free_rate)
        return pd.Series(weights, index=tickers)
    
    return optimiser