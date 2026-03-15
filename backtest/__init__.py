"""
backtest/ — Backtesting Engine Package

Core types used across the backtesting system.
"""

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class BacktestResult:
    """
    Container for the output of a single backtest run.

    Attributes:
        equity_curve : Daily portfolio value indexed by date.
        trade_log    : One row per closed trade with columns:
                       symbol, entry_date, exit_date, return_pct, pnl, direction, strategy
        initial_capital : Starting portfolio value (USD).
    """
    equity_curve: pd.Series
    trade_log: pd.DataFrame
    initial_capital: float

    def __repr__(self) -> str:
        days = len(self.equity_curve)
        trades = len(self.trade_log)
        return f"BacktestResult(days={days}, trades={trades}, capital=${self.initial_capital:,.0f})"
