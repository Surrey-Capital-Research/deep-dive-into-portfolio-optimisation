# src/backtesting/strategies.py

from abc import ABC, abstractmethod
import pandas as pd


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
