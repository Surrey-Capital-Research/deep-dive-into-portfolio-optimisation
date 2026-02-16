from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Callable

from src.models.risk_parity import compute_risk_parity_weights
from src.models.black_litterman import (
    implied_equilibrium_returns,
    black_litterman_posterior,
    build_diagonal_omega,
)

class BaseStrategy(ABC):
    """
    Abstract base class for portfolio strategies.

    The backtester will call get_target_weights at each rebalance date.
    Implementations must ONLY use information in past_prices (no future data).
    """

    @abstractmethod
    def get_target_weights(
        self,
        decision_date: pd.Timestamp,
        past_prices: pd.DataFrame,
        current_positions: pd.Series,
        cash: float,
    ) -> pd.Series:
        """
        Parameters
        ----------
        decision_date : pd.Timestamp
            Date at which the rebalance decision is made.
        past_prices : pd.DataFrame
            Price history up to and including decision_date (no future rows).
        current_positions : pd.Series
            Current holdings in units (index = tickers).
        cash : float
            Current cash level.

        Returns
        -------
        pd.Series
            Target portfolio weights (index = tickers, values sum to 1).
        """
        raise NotImplementedError


class EqualWeightStrategy(BaseStrategy):
    """
    Simple baseline strategy: allocate equal weights to all tickers
    that have valid price data up to the decision date.
    """

    def __init__(self, tickers: list[str]):
        self.tickers = list(tickers)

    def get_target_weights(
        self,
        decision_date: pd.Timestamp,
        past_prices: pd.DataFrame,
        current_positions: pd.Series,
        cash: float,
    ) -> pd.Series:
        # Only consider tickers that are in past_prices
        available = [t for t in self.tickers if t in past_prices.columns]

        if not available:
            return pd.Series(dtype=float)

        n = len(available)
        weights = pd.Series(1.0 / n, index=available)
        return weights

class BlackLittermanStrategy(BaseStrategy):
    """
    Black-Litterman strategy:
    - Uses market weights as the prior (equilibrium) portfolio.
    - Builds views from a user-supplied view_builder callable.
    - Computes posterior expected returns via Black-Litterman.
    - Calls an optimizer (e.g. mean-variance) to get final weights.
    """

    def __init__(
        self,
        market_weights: pd.Series,
        risk_aversion: float,
        tau: float,
        view_builder: Callable[
            [pd.DataFrame, pd.Timestamp], tuple[np.ndarray, np.ndarray]
        ],
        optimizer: Callable[[pd.Series, pd.DataFrame], pd.Series],
        view_uncertainty_scalar: float = 0.5,
        cov_window: int = 252,
    ):
        """
        Parameters
        ----------
        market_weights : pd.Series
            Benchmark / market-cap weights (index = tickers, sum = 1).
        risk_aversion : float
            Risk-aversion parameter lambda.
        tau : float
            Scalar controlling prior uncertainty in Black-Litterman.
        view_builder : callable
            Function (past_prices, decision_date) -> (P, Q),
            where P is (n_views x n_assets), Q is (n_views,).
        optimizer : callable
            Function (mu, cov) -> target weights (pd.Series, sum = 1).
            Typically your MVO optimizer.
        view_uncertainty_scalar : float
            Scales the diagonal Omega (view uncertainty).
        cov_window : int
            Rolling window length (in days) to estimate covariance.
        """
        self.market_weights = market_weights / market_weights.sum()
        self.risk_aversion = risk_aversion
        self.tau = tau
        self.view_builder = view_builder
        self.optimizer = optimizer
        self.view_uncertainty_scalar = view_uncertainty_scalar
        self.cov_window = cov_window
        self.tickers = list(market_weights.index)

    def get_target_weights(
        self,
        decision_date: pd.Timestamp,
        past_prices: pd.DataFrame,
        current_positions: pd.Series,
        cash: float,
    ) -> pd.Series:
        # Restrict to our universe and a rolling window for cov estimation
        past_prices = past_prices[self.tickers].dropna(how="all")
        if past_prices.shape[0] < self.cov_window:
            # fall back to market weights
            return self.market_weights.copy()

        window_prices = past_prices.iloc[-self.cov_window :]
        returns = window_prices.pct_change().dropna()
        cov = returns.cov()

        # Equilibrium returns
        mu_prior = implied_equilibrium_returns(
            cov=cov,
            market_weights=self.market_weights,
            risk_aversion=self.risk_aversion,
        )

        # Past data up to decision_date
        # We add Omega here so it can catch all three values from your FFT builder
        P, Q, Omega = self.view_builder(window_prices, decision_date)
        
        if P.size == 0:
            # No views: just use prior + MVO
            # replace return with weights.reindex(self.tickers).fillna(0.0) when MVO done
            # weights = self.optimizer(mu_prior, cov)
            return self.market_weights.copy()

        Omega = build_diagonal_omega(Q, self.view_uncertainty_scalar)

        mu_bl = black_litterman_posterior(
            mu_prior=mu_prior,
            cov=cov,
            P=P,
            Q=Q,
            Omega=Omega,
            tau=self.tau,
        )

        weights = self.optimizer(mu_bl, cov)
        weights = weights.reindex(self.tickers).fillna(0.0)
        if weights.sum() != 0:
            weights = weights / weights.sum()

        return weights

class RiskParityStrategy(BaseStrategy):
    def __init__(self, tickers: list[str], cov_window: int = 252):
        self.tickers = list(tickers)
        self.cov_window = cov_window

    def get_target_weights(
            self,
            decision_date: pd.Timestamp,
            past_prices: pd.DataFrame,
            current_positions: pd.Series,
            cash: float
        ) -> pd.Series:
        
        prices = past_prices[self.tickers].dropna(how="all")

        if past_prices.shape[0] < self.cov_window:
            # fall back to market weights
            return pd.Series(1.0 / len(self.tickers), index=self.tickers)
        
        recent_prices = prices.iloc[-self.cov_window:]
        returns = recent_prices.pct_change().dropna()

        cov = returns.cov().values

        weights = compute_risk_parity_weights(cov)

        return pd.Series(weights, index=self.tickers)

