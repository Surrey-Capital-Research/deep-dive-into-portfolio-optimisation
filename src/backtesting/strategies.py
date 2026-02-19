from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Callable
from src.models.optimisers import run_mvo_optimization

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
        past_prices = past_prices[self.tickers].dropna(how="all")
        if past_prices.shape[0] < self.cov_window:
            return self.market_weights.copy()

        window_prices = past_prices.iloc[-self.cov_window :]
        returns = window_prices.pct_change().dropna()
        cov = returns.cov()

        mu_prior = implied_equilibrium_returns(
            cov=cov,
            market_weights=self.market_weights,
            risk_aversion=self.risk_aversion,
        )

        P, Q, Omega = self.view_builder(window_prices, decision_date)
        
        if P.size == 0:

            return self.market_weights.copy()

        if Omega is None:
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

class MVOStrategy(BaseStrategy):
    """
    Institutional Mean-Variance Optimization Strategy.
    Uses Ledoit-Wolf shrinkage, EMA returns, and position limits via the optimizers module.
    """
    def __init__(self, tickers: list[str], rf_ticker: str = "UK_10Y_Yield", cov_window: int = 252):
        self.tickers = list(tickers)
        self.rf_ticker = rf_ticker
        self.cov_window = cov_window

    def get_target_weights(
        self,
        decision_date: pd.Timestamp,
        past_prices: pd.DataFrame,
        current_positions: pd.Series,
        cash: float
    ) -> pd.Series:
        
        asset_prices = past_prices[self.tickers].dropna(how="all")
        
        if asset_prices.shape[0] < self.cov_window:
            return pd.Series(1.0 / len(self.tickers), index=self.tickers)
        
        recent_prices = asset_prices.iloc[-self.cov_window:] 

        if self.rf_ticker in past_prices.columns:

            current_rf = past_prices[self.rf_ticker].ffill().iloc[-1] / 100.0
        else:
            current_rf = 0.0

        weights_dict = run_mvo_optimization(
            prices_df=recent_prices, 
            rf_rate=current_rf,
            max_position_size=0.20, 
            gamma_reg=0.1
        )

        return pd.Series(weights_dict).reindex(self.tickers).fillna(0.0)
    
from pypfopt import expected_returns, risk_models, EfficientFrontier, objective_functions
import pandas as pd
import numpy as np

class RegimeSwitchingMVOStrategy(BaseStrategy):
    """
    Fast Momentum MVO.
    Combines a rapid 50-day crash filter with uncaged 3-month momentum.
    """
    def __init__(self, tickers: list[str], rf_ticker: str = "UK_10Y_Yield", cov_window: int = 126, trend_window: int = 50):
        self.tickers = list(tickers)
        self.rf_ticker = rf_ticker
        self.cov_window = cov_window
        self.trend_window = trend_window

    def get_target_weights(
        self,
        decision_date: pd.Timestamp,
        past_prices: pd.DataFrame,
        current_positions: pd.Series,
        cash: float
    ) -> pd.Series:
        
        asset_prices = past_prices[self.tickers].dropna(how="all")
        warmup_period = max(self.cov_window, self.trend_window)
        
        if asset_prices.shape[0] < warmup_period:
            return pd.Series(1.0 / len(self.tickers), index=self.tickers)
        
        recent_prices = asset_prices.iloc[-self.cov_window:]

        current_price = asset_prices.iloc[-1]
        sma = asset_prices.tail(self.trend_window).mean()
        uptrend_assets = current_price[current_price > sma].index.tolist()

        if len(uptrend_assets) < 3:
            try:
                S_safe = risk_models.CovarianceShrinkage(recent_prices).ledoit_wolf()
                ef_safe = EfficientFrontier(None, S_safe, weight_bounds=(0.0, 1.0))
                safe_weights = ef_safe.min_volatility()
                return pd.Series(dict(ef_safe.clean_weights())).reindex(self.tickers).fillna(0.0)
            except Exception:
                return pd.Series(1.0 / len(self.tickers), index=self.tickers)
        
        uptrend_prices = recent_prices[uptrend_assets]
        current_rf = past_prices[self.rf_ticker].ffill().iloc[-1] / 100.0 if self.rf_ticker in past_prices.columns else 0.0

        try:
            mu = expected_returns.ema_historical_return(uptrend_prices, span=63)
            S = risk_models.CovarianceShrinkage(uptrend_prices).ledoit_wolf()
            ef = EfficientFrontier(mu, S, weight_bounds=(0.0, 0.50))
            
            weights_dict = ef.max_sharpe(risk_free_rate=current_rf)
            return pd.Series(dict(ef.clean_weights())).reindex(self.tickers).fillna(0.0)
            
        except Exception:
            fallback = pd.Series(1.0 / len(uptrend_assets), index=uptrend_assets)
            return fallback.reindex(self.tickers).fillna(0.0)