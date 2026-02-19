from __future__ import annotations
import numpy as np
import pandas as pd


def implied_equilibrium_returns(
    cov: pd.DataFrame,
    market_weights: pd.Series,
    risk_aversion: float,
) -> pd.Series:
    """
    Compute implied equilibrium returns:
        mu_prior = risk_aversion * Sigma * w_m

    Parameters
    ----------
    cov : pd.DataFrame
        Covariance matrix (index/columns = tickers).
    market_weights : pd.Series
        Market-cap (or benchmark) weights (index = tickers, sum = 1).
    risk_aversion : float
        Risk-aversion parameter (lambda).

    Returns
    -------
    pd.Series
        Implied equilibrium returns (index = tickers).
    """

    # Ensure indicers align and cov is ordered correctly
    cov = cov.loc[market_weights.index, market_weights.index]
    # BL formula: mu_prior = risk_aversion * Sigma * w_m
    # Compute the vector of implied returns
    mu = risk_aversion * cov.values @ market_weights.values
    return pd.Series(mu, index=market_weights.index)

def black_litterman_posterior(
    mu_prior: pd.Series,
    cov: pd.DataFrame,
    P: np.ndarray,
    Q: np.ndarray,
    Omega: np.ndarray,
    tau: float,
) -> pd.Series:
    """
    Compute Black-Litterman posterior expected returns.

    Parameters
    ----------
    mu_prior : pd.Series
        Prior (equilibrium) expected returns (index = tickers).
    cov : pd.DataFrame
        Covariance matrix (index/columns = tickers).
    P : np.ndarray
        View matrix (n_views x n_assets).
    Q : np.ndarray
        View returns (n_views,).
    Omega : np.ndarray
        Diagonal matrix of view variances (n_views x n_views).
    tau : float
        Scalar controlling uncertainty of the prior.

    Returns
    -------
    pd.Series
        Posterior expected returns (index = tickers).
    """
    Sigma = cov.values
    mu0 = mu_prior.values
    # Scale covariance by tau to reflect uncertainty in the prior
    tauSigma = tau * Sigma
    inv_tauSigma = np.linalg.inv(tauSigma)
    inv_Omega = np.linalg.inv(Omega)
    # Combine info of matrices
    middle = inv_tauSigma + P.T @ inv_Omega @ P
    rhs = inv_tauSigma @ mu0 + P.T @ inv_Omega @ Q
    mu_bl = np.linalg.solve(middle, rhs)

    return pd.Series(mu_bl, index=mu_prior.index)


def build_diagonal_omega(Q: np.ndarray, view_uncertainty_scalar: float) -> np.ndarray:
    """
    Simple helper to build diagonal Omega as:
        Omega_ii = (view_uncertainty_scalar * |Q_i|)^2
    """
    diag = (view_uncertainty_scalar * np.abs(Q)) ** 2
    return np.diag(diag + 1e-8)
