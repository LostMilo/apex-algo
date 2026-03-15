"""
backtest/metrics.py — Performance Metrics Calculator

Computes all performance statistics from a BacktestResult.
Pure calculation functions — no data fetching, no side effects.

Usage:
    from backtest import BacktestResult
    from backtest.metrics import MetricsCalculator

    result = BacktestResult(equity_curve=..., trade_log=..., initial_capital=1000)
    stats = MetricsCalculator.compute_all(result)
    print(stats)
"""

import math
from datetime import date

import numpy as np
import pandas as pd

from backtest import BacktestResult


# ─────────────────────────────────────────────────────────────────────────────
# MetricsCalculator — thin orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class MetricsCalculator:
    """Calls every metric function and returns a single flat dict."""

    @staticmethod
    def compute_all(result: BacktestResult) -> dict:
        """
        Compute all performance metrics from a BacktestResult.

        Args:
            result: A BacktestResult containing equity_curve, trade_log,
                    and initial_capital.

        Returns:
            dict with keys for every metric.  Values are float, date,
            int, or pd.DataFrame (monthly_returns).
        """
        eq = result.equity_curve
        tl = result.trade_log
        cap = result.initial_capital

        # ── Equity-curve metrics ─────────────────────────────────
        sr = sharpe_ratio(eq)
        so = sortino_ratio(eq)
        md_pct, md_peak, md_trough = max_drawdown(eq)
        dd_dur = drawdown_duration(eq)
        ar = annualized_return(eq, cap)
        cr = calmar_ratio(eq, cap)
        mr = monthly_returns(eq)

        # ── Trade-log metrics ────────────────────────────────────
        wr = win_rate(tl)
        pf = profit_factor(tl)
        awl = avg_win_loss_ratio(tl)

        return {
            "sharpe_ratio": sr,
            "sortino_ratio": so,
            "max_drawdown_pct": md_pct,
            "max_drawdown_peak_date": md_peak,
            "max_drawdown_trough_date": md_trough,
            "max_drawdown_duration_days": dd_dur,
            "calmar_ratio": cr,
            "annualized_return": ar,
            "win_rate": wr,
            "profit_factor": pf,
            "avg_win_loss_ratio": awl,
            "monthly_returns": mr,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 1. Sharpe Ratio
# ─────────────────────────────────────────────────────────────────────────────

def sharpe_ratio(equity_curve: pd.Series, risk_free_rate: float = 0.04) -> float:
    """
    Annualised Sharpe ratio.

    Formula:
        (mean_daily_return − daily_risk_free) / std_daily_return × √252

    Returns:
        float — the Sharpe ratio, or NaN if insufficient data.
    """
    daily_returns = equity_curve.pct_change().dropna()
    if len(daily_returns) < 2:
        return float("nan")

    daily_rf = risk_free_rate / 252.0
    excess = daily_returns - daily_rf

    std = excess.std()
    if std == 0 or np.isnan(std):
        return float("nan")

    return float((excess.mean() / std) * math.sqrt(252))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Sortino Ratio
# ─────────────────────────────────────────────────────────────────────────────

def sortino_ratio(equity_curve: pd.Series, risk_free_rate: float = 0.04) -> float:
    """
    Annualised Sortino ratio — like Sharpe but penalises only downside volatility.

    Downside deviation = std of returns where return < 0.

    Returns:
        float — the Sortino ratio, or NaN if insufficient data.
    """
    daily_returns = equity_curve.pct_change().dropna()
    if len(daily_returns) < 2:
        return float("nan")

    daily_rf = risk_free_rate / 252.0
    excess = daily_returns - daily_rf

    downside = excess[excess < 0]
    if len(downside) == 0:
        # No negative returns — infinite Sortino (return NaN to avoid inf)
        return float("nan")

    downside_std = downside.std()
    if downside_std == 0 or np.isnan(downside_std):
        return float("nan")

    return float((excess.mean() / downside_std) * math.sqrt(252))


# ─────────────────────────────────────────────────────────────────────────────
# 3. Maximum Drawdown
# ─────────────────────────────────────────────────────────────────────────────

def max_drawdown(equity_curve: pd.Series) -> tuple[float, date, date]:
    """
    Maximum peak-to-trough decline as a percentage.

    Returns:
        (max_dd_pct, peak_date, trough_date)
        max_dd_pct is a NEGATIVE float (e.g. −0.15 for a 15% drawdown).
        Dates are datetime.date objects.
        Returns (NaN, None, None) if insufficient data.
    """
    if len(equity_curve) < 2:
        return (float("nan"), None, None)

    running_max = equity_curve.cummax()
    drawdowns = (equity_curve - running_max) / running_max

    trough_idx = drawdowns.idxmin()
    trough_value = drawdowns[trough_idx]

    # Peak is the date of the running max before the trough
    peak_idx = equity_curve.loc[:trough_idx].idxmax()

    peak_date = _to_date(peak_idx)
    trough_date = _to_date(trough_idx)

    return (float(trough_value), peak_date, trough_date)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Drawdown Duration
# ─────────────────────────────────────────────────────────────────────────────

def drawdown_duration(equity_curve: pd.Series) -> int:
    """
    Longest drawdown duration in trading days.

    A drawdown period starts when the equity falls below a running max
    and ends when it recovers to a new high (or at the series end).

    Returns:
        int — number of trading days in the longest drawdown, or 0.
    """
    if len(equity_curve) < 2:
        return 0

    running_max = equity_curve.cummax()
    in_drawdown = equity_curve < running_max

    max_duration = 0
    current_duration = 0

    for is_dd in in_drawdown:
        if is_dd:
            current_duration += 1
            max_duration = max(max_duration, current_duration)
        else:
            current_duration = 0

    return max_duration


# ─────────────────────────────────────────────────────────────────────────────
# 5. Calmar Ratio
# ─────────────────────────────────────────────────────────────────────────────

def calmar_ratio(equity_curve: pd.Series, initial_capital: float = None) -> float:
    """
    Calmar ratio = annualised return / abs(max drawdown).

    Args:
        equity_curve    : Daily portfolio value.
        initial_capital : Starting value.  Defaults to first value in equity_curve.

    Returns:
        float — the Calmar ratio, or NaN if max drawdown is zero.
    """
    if initial_capital is None:
        initial_capital = equity_curve.iloc[0]

    ar = annualized_return(equity_curve, initial_capital)
    md, _, _ = max_drawdown(equity_curve)

    if np.isnan(ar) or np.isnan(md) or md == 0:
        return float("nan")

    return float(ar / abs(md))


# ─────────────────────────────────────────────────────────────────────────────
# 6. Win Rate
# ─────────────────────────────────────────────────────────────────────────────

def win_rate(trade_log: pd.DataFrame) -> float:
    """
    Percentage of trades with positive return.

    Expects trade_log to have a 'return_pct' or 'pnl' column.

    Returns:
        float in [0.0, 1.0], or NaN if no trades.
    """
    returns = _get_trade_returns(trade_log)
    if returns is None or len(returns) == 0:
        return float("nan")

    return float((returns > 0).sum() / len(returns))


# ─────────────────────────────────────────────────────────────────────────────
# 7. Profit Factor
# ─────────────────────────────────────────────────────────────────────────────

def profit_factor(trade_log: pd.DataFrame) -> float:
    """
    Profit factor = sum of winning trades / abs(sum of losing trades).

    Returns:
        float, or NaN if no losing trades or no trades at all.
    """
    returns = _get_trade_returns(trade_log)
    if returns is None or len(returns) == 0:
        return float("nan")

    gains = returns[returns > 0].sum()
    losses = returns[returns < 0].sum()

    if losses == 0:
        return float("nan")  # No losers — PF is infinite

    return float(gains / abs(losses))


# ─────────────────────────────────────────────────────────────────────────────
# 8. Average Win / Loss Ratio
# ─────────────────────────────────────────────────────────────────────────────

def avg_win_loss_ratio(trade_log: pd.DataFrame) -> float:
    """
    Average winning trade return / abs(average losing trade return).

    Returns:
        float, or NaN if no winners or no losers.
    """
    returns = _get_trade_returns(trade_log)
    if returns is None or len(returns) == 0:
        return float("nan")

    winners = returns[returns > 0]
    losers = returns[returns < 0]

    if len(winners) == 0 or len(losers) == 0:
        return float("nan")

    avg_win = winners.mean()
    avg_loss = losers.mean()

    if avg_loss == 0:
        return float("nan")

    return float(avg_win / abs(avg_loss))


# ─────────────────────────────────────────────────────────────────────────────
# 9. Annualised Return
# ─────────────────────────────────────────────────────────────────────────────

def annualized_return(
    equity_curve: pd.Series,
    initial_capital: float = None,
) -> float:
    """
    Compound annual growth rate (CAGR).

    Formula:
        (final_equity / initial_equity) ^ (252 / trading_days) − 1

    Args:
        equity_curve    : Daily portfolio value.
        initial_capital : Starting value.  Defaults to first value in equity_curve.

    Returns:
        float — annualised return, or NaN if insufficient data.
    """
    if len(equity_curve) < 2:
        return float("nan")

    if initial_capital is None:
        initial_capital = equity_curve.iloc[0]

    if initial_capital <= 0:
        return float("nan")

    final = equity_curve.iloc[-1]
    trading_days = len(equity_curve)

    if trading_days <= 1:
        return float("nan")

    total_return = final / initial_capital
    if total_return <= 0:
        return float("nan")

    return float(total_return ** (252.0 / trading_days) - 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 10. Monthly Returns
# ─────────────────────────────────────────────────────────────────────────────

def monthly_returns(equity_curve: pd.Series) -> pd.DataFrame:
    """
    Monthly returns matrix — years as rows, months as columns.

    Suitable for heatmap display in a dashboard.

    Returns:
        pd.DataFrame with index=year (int), columns=month (1–12),
        values = monthly return as a decimal (e.g. 0.05 = +5%).
        Missing months are NaN.
    """
    if len(equity_curve) < 2:
        return pd.DataFrame()

    # Ensure DatetimeIndex
    if not isinstance(equity_curve.index, pd.DatetimeIndex):
        equity_curve = equity_curve.copy()
        equity_curve.index = pd.to_datetime(equity_curve.index)

    # Monthly end-of-period values
    monthly = equity_curve.resample("ME").last()
    monthly_ret = monthly.pct_change().dropna()

    if monthly_ret.empty:
        return pd.DataFrame()

    # Build the matrix
    table = pd.DataFrame({
        "year": monthly_ret.index.year,
        "month": monthly_ret.index.month,
        "return": monthly_ret.values,
    })

    pivot = table.pivot_table(
        index="year",
        columns="month",
        values="return",
        aggfunc="first",
    )

    # Ensure all 12 month columns exist
    for m in range(1, 13):
        if m not in pivot.columns:
            pivot[m] = float("nan")
    pivot = pivot[sorted(pivot.columns)]

    pivot.index.name = "Year"
    pivot.columns.name = "Month"

    return pivot


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_trade_returns(trade_log: pd.DataFrame) -> pd.Series | None:
    """
    Extract a Series of per-trade returns from the trade log.
    Prefers 'return_pct', falls back to 'pnl'.
    """
    if trade_log is None or trade_log.empty:
        return None

    if "return_pct" in trade_log.columns:
        return trade_log["return_pct"].dropna()
    elif "pnl" in trade_log.columns:
        return trade_log["pnl"].dropna()
    else:
        return None


def _to_date(idx) -> date | None:
    """Convert a pandas Timestamp or similar to datetime.date."""
    if idx is None:
        return None
    if hasattr(idx, "date"):
        return idx.date()
    if isinstance(idx, date):
        return idx
    return None
