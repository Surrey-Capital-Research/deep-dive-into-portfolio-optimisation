from __future__ import annotations
import numpy as  np, pandas as pd


def compute_risk_contributions(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    
    cov_w = cov @ w
    rc = w * cov_w
    return rc


def f_risk_parity(x: np.ndarray, cov: np.ndarray):
    """Compute the risk-parity system f(x) = 0"""
    n = cov.shape[0]
    w = x[:n] # First n elements are the weights
    c = x[n] # The last element is the constant

    rc = compute_risk_contributions(w, cov)
    f_weights = rc - c # f(x) = 0

    f_sum = np.sum(w) - 1.0

    return np.append(f_weights, f_sum)


def jacobian_risk_parity(x: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Compute the Jacobian of the risk-parity system f(x) = 0"""
    n = cov.shape[0]
    w = x[:n]

    cov_w = cov @ w

    top_left = np.diag(cov_w) + np.diag(w) @ cov
    top_right = -np.ones((n, 1))
    bottom_left = np.ones((1, n))
    bottom_right = np.array([[0.0]])

    J = np.block([
        [top_left, top_right],
        [bottom_left, bottom_right]
    ])

    return J


def compute_risk_parity_weights(cov: np.ndarray, tol: float = 1e-8, max_iter: int = 100):
    """Compute risk-parity weights using Newton-Raphson method"""
    if isinstance(cov, pd.DataFrame):
        tickers = cov.index.tolist()
        cov_matrix = cov.values
    else:
        tickers = None
        cov_matrix = cov

    n = cov_matrix.shape[0]

    # Initial Guess
    vols = np.sqrt(np.diag(cov_matrix))
    w0 =(1.0 / vols) / np.sum(1.0 / vols)
    c0 = 0.1
    x = np.append(w0, c0)

    for i in range(max_iter):
        f = f_risk_parity(x, cov_matrix)
        J = jacobian_risk_parity(x, cov_matrix)

        delta = np.linalg.solve(J, -f)
        x = x + delta

        if np.linalg.norm(delta) < tol:
            break

    w = x[:n]
    w = np.maximum(w, 0.0)
    w = w / w.sum()

    if tickers:
        return pd.Series(w, index=tickers)
    return w