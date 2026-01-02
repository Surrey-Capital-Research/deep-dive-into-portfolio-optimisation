from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Callable
import numpy as np
import pandas as pd
from .strategies import BaseStrategy

@dataclass
class BacktestResult:
    equity_curve: pd.Series
    positions: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict

class Backtester:
    """
    Daily backtester with monthly rebalancing and class-based strategies.
    - Simulates investing an initial capital from start to end of prices.
    - Steps forward day by day.
    - At each rebalance date, calls strategy.get_target_weights using ONLY past data.
    - Rebalances portfolio to those target weights.
    - Tracks portfolio value and computes simple performance metrics.
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        strategy: BaseStrategy,
        initial_capital: float = 100_000.0,
        rebalance_freq: str = "ME",  # month-end, not "M"
        cost_model: Optional[Callable[[pd.Series], float]] = None,
    ):
        """
        Parameters
        ----------
        prices : pd.DataFrame
            Daily price data. Index = dates (ascending), columns = tickers.
        strategy : BaseStrategy
            Strategy object implementing get_target_weights.
        initial_capital : float
            Starting cash.
        rebalance_freq : str
            Pandas offset alias for rebalance frequency (e.g. 'ME' for month-end).
        cost_model : Callable
            Optional function that takes a trade notional per ticker and returns total transaction cost.
        """
        self.prices = prices.sort_index()
        self.strategy = strategy
        self.initial_capital = float(initial_capital)
        self.rebalance_freq = rebalance_freq
        self.cost_model = cost_model

        self.tickers = list(self.prices.columns)

    def _get_rebalance_dates(self) -> pd.DatetimeIndex:
        """
        Return the dates on which to rebalance.
        Includes:
        - First trading day,
        - Month-end dates.
        """
        all_dates = self.prices.index
        freq_ends = self.prices.resample(self.rebalance_freq).last().index

        # First trading day is included to invest immediately
        rebalance_dates = freq_ends.union([all_dates[0]])

        return rebalance_dates

    def run(self) -> BacktestResult:
        dates = self.prices.index
        rebalance_dates = set(self._get_rebalance_dates())
        positions = pd.Series(0.0, index=self.tickers)
        cash = self.initial_capital

        # Storing history
        equity_curve = pd.Series(index=dates, dtype=float)
        positions_history = pd.DataFrame(index=dates, columns=self.tickers, dtype=float)
        trades_records = []

        for current_date in dates:
            todays_prices = self.prices.loc[current_date]

            # Rebalance if today is a rebalance date
            if current_date in rebalance_dates:
                past_prices = self.prices.loc[:current_date]

                target_weights = self.strategy.get_target_weights(
                    decision_date=current_date,
                    past_prices=past_prices,
                    current_positions=positions.copy(),
                    cash=cash,
                )
                target_weights = target_weights.reindex(self.tickers).fillna(0.0)

                # Portfolio value BEFORE trades
                portfolio_value = cash + float((positions * todays_prices).sum())
                # Target holdings in currency
                target_notional = target_weights * portfolio_value
                # Current holdings in currency
                current_notional = positions * todays_prices
                # Trade notional per ticker
                trade_notional = target_notional - current_notional
                # Convert trade notionals to position changes (units)
                trade_units = trade_notional / todays_prices.replace(0, np.nan)
                trade_units = trade_units.fillna(0.0)
                # Transaction costs
                if self.cost_model is not None:
                    cost = self.cost_model(trade_notional)
                else:
                    cost = 0.0

                cash -= float(trade_notional.sum())
                cash -= cost
                positions += trade_units

                trades_records.append(
                    pd.DataFrame(
                        {
                            "date": current_date,
                            "ticker": self.tickers,
                            "trade_units": trade_units.values,
                            "trade_notional": trade_notional.values,
                            "cost": cost / max(len(self.tickers), 1),
                        }
                    )
                )

            # Mark-to-market at close
            equity_curve.loc[current_date] = cash + float(
                (positions * todays_prices).sum()
            )
            positions_history.loc[current_date] = positions.values

        trades_df = (
            pd.concat(trades_records, ignore_index=True) if trades_records else pd.DataFrame()
        )

        metrics = self._compute_metrics(equity_curve)

        return BacktestResult(
            equity_curve=equity_curve,
            positions=positions_history,
            trades=trades_df,
            metrics=metrics,
        )

    @staticmethod
    def _compute_metrics(equity_curve: pd.Series) -> dict:
        """
        Compute simple performance metrics from a daily equity curve.
        """
        equity_curve = equity_curve.dropna()
        if equity_curve.empty:
            return {}

        # Returns daily
        rets = equity_curve.pct_change().dropna()

        total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0
        n_days = rets.shape[0]
        annual_factor = 252.0

        cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (
            annual_factor / n_days
        ) - 1.0

        vol = rets.std() * np.sqrt(annual_factor)
        sharpe = cagr / vol if vol > 0 else np.nan

        running_max = equity_curve.cummax()
        drawdown = equity_curve / running_max - 1.0
        max_dd = drawdown.min()

        return {
            "total_return": float(total_return),
            "CAGR": float(cagr),
            "volatility": float(vol),
            "Sharpe": float(sharpe),
            "max_drawdown": float(max_dd),
        }
