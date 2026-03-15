"""
risk/exits.py — Exit Manager for Autonomous Trading Algorithm

Manages when to exit open positions. Completely separate from entry logic.
Exits are checked FIRST every day before any new signals are computed.

Exit priority (highest → lowest):
  1. Kill Switch   — FULL_HALT exits everything immediately
  2. Chandelier    — Wilder-smoothed ATR trailing stop
  3. Hard Stop     — Failsafe % stop-loss from entry
  4. Time Stop     — Dead-money detector (flat P&L after N days)
  5. Signal Reversal — Current signal opposes position direction
"""

import numpy as np
import pandas as pd

import config
from logger import log

# ──────────────────────────────────────────────────────────────────────
# Exit reason codes
# ──────────────────────────────────────────────────────────────────────
EXIT_CHANDELIER = "CHANDELIER"
EXIT_HARD_STOP = "HARD_STOP"
EXIT_TIME_STOP = "TIME_STOP"
EXIT_SIGNAL_REVERSAL = "SIGNAL_REVERSAL"
EXIT_KILL_SWITCH = "KILL_SWITCH"


class ExitManager:
    """Decides which open positions should be closed and why."""

    def __init__(self):
        self.chandelier_period = config.CHANDELIER_ATR_PERIOD   # 22
        self.chandelier_mult = config.CHANDELIER_ATR_MULT       # 3.0
        self.hard_stop_pct = config.HARD_STOP_PCT               # 0.07
        self.time_stop_days = config.TIME_STOP_DAYS              # 30
        self.time_stop_min = config.TIME_STOP_MIN_PNL            # -0.02
        self.time_stop_max = config.TIME_STOP_MAX_PNL            # +0.02

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────
    def check_all_exits(
        self,
        positions: dict,
        price_data: dict[str, pd.DataFrame],
        current_signals: dict,
        current_date: str,
        kill_switch_status: str,
    ) -> dict[str, tuple[bool, str, float]]:
        """
        Check every exit rule for every open position.

        Args:
            positions: Dict of symbol → position info dict. Each must have:
                - 'side': 'long' or 'short'
                - 'entry_price': float
                - 'entry_date': str ('YYYY-MM-DD')
            price_data: Dict of symbol → OHLCV DataFrame (full history)
            current_signals: Dict of symbol → Signal (current day's signals)
            current_date: 'YYYY-MM-DD' string for today
            kill_switch_status: 'FULL_HALT' triggers immediate exit of all

        Returns:
            Dict of symbol → (should_exit: bool, reason: str, exit_price: float)
        """
        results: dict[str, tuple[bool, str, float]] = {}

        for symbol, pos in positions.items():
            df = price_data.get(symbol)
            if df is None or df.empty:
                log.warning("{}: no price data — skipping exit check", symbol)
                results[symbol] = (False, "", 0.0)
                continue

            # Use yesterday's close as exit price baseline
            exit_price = float(df["Close"].iloc[-1])
            side = pos.get("side", "long")
            entry_price = float(pos.get("entry_price", 0))
            entry_date = pos.get("entry_date", "")

            # --- Priority 1: Kill Switch ---
            if kill_switch_status == "FULL_HALT":
                log.critical(
                    "{}: KILL SWITCH — exiting at ${:.2f}", symbol, exit_price
                )
                results[symbol] = (True, EXIT_KILL_SWITCH, exit_price)
                continue

            # --- Priority 2: Chandelier Exit ---
            chandelier_exit, chandelier_price = self._check_chandelier(
                df, side
            )
            if chandelier_exit:
                log.info(
                    "{}: CHANDELIER EXIT — stop ${:.2f}, close ${:.2f}",
                    symbol, chandelier_price, exit_price,
                )
                results[symbol] = (True, EXIT_CHANDELIER, exit_price)
                continue

            # --- Priority 3: Hard Stop ---
            if self._check_hard_stop(entry_price, exit_price, side):
                log.warning(
                    "{}: HARD STOP — entry ${:.2f}, close ${:.2f} ({:.1%} loss)",
                    symbol, entry_price, exit_price,
                    abs(exit_price - entry_price) / entry_price if entry_price else 0,
                )
                results[symbol] = (True, EXIT_HARD_STOP, exit_price)
                continue

            # --- Priority 4: Time Stop ---
            if self._check_time_stop(
                entry_price, exit_price, entry_date, current_date, side
            ):
                log.info(
                    "{}: TIME STOP — held since {}, P&L flat", symbol, entry_date
                )
                results[symbol] = (True, EXIT_TIME_STOP, exit_price)
                continue

            # --- Priority 5: Signal Reversal ---
            if self._check_signal_reversal(symbol, side, current_signals):
                log.info("{}: SIGNAL REVERSAL — opposing signal detected", symbol)
                results[symbol] = (True, EXIT_SIGNAL_REVERSAL, exit_price)
                continue

            # No exit triggered
            results[symbol] = (False, "", 0.0)

        # Summary
        exits = {s: r for s, r in results.items() if r[0]}
        if exits:
            log.info(
                "Exit manager: {} exits triggered — {}",
                len(exits),
                ", ".join(f"{s}({r[1]})" for s, r in exits.items()),
            )

        return results

    # ──────────────────────────────────────────────────────────────────
    # Private exit checks
    # ──────────────────────────────────────────────────────────────────
    def _check_chandelier(
        self, df: pd.DataFrame, side: str
    ) -> tuple[bool, float]:
        """
        Chandelier Exit using Wilder-smoothed ATR.

        For longs:  stop = highest_high(N) - mult × ATR(N)
        For shorts: stop = lowest_low(N)  + mult × ATR(N)

        Trailing — only moves in the favourable direction.

        Returns (should_exit, stop_level).
        """
        n = self.chandelier_period
        if len(df) < n + 1:
            return False, 0.0

        # Compute ATR using Wilder smoothing (exponential with alpha = 1/n)
        atr = self._wilder_atr(df, n)
        if atr <= 0 or np.isnan(atr):
            return False, 0.0

        recent = df.iloc[-n:]
        current_close = float(df["Close"].iloc[-1])

        if side == "long":
            highest_high = float(recent["High"].max())
            chandelier_stop = highest_high - (self.chandelier_mult * atr)
            return current_close < chandelier_stop, chandelier_stop
        else:
            lowest_low = float(recent["Low"].min())
            chandelier_stop = lowest_low + (self.chandelier_mult * atr)
            return current_close > chandelier_stop, chandelier_stop

    def _check_hard_stop(
        self, entry_price: float, current_close: float, side: str
    ) -> bool:
        """
        Hard stop-loss — failsafe at ±HARD_STOP_PCT from entry.

        For longs:  exit if (current_close - entry) / entry < -HARD_STOP_PCT
        For shorts: exit if (entry - current_close) / entry < -HARD_STOP_PCT
        """
        if entry_price <= 0:
            return False

        if side == "long":
            pnl_pct = (current_close - entry_price) / entry_price
            return pnl_pct < -self.hard_stop_pct
        else:
            pnl_pct = (entry_price - current_close) / entry_price
            return pnl_pct < -self.hard_stop_pct

    def _check_time_stop(
        self,
        entry_price: float,
        current_close: float,
        entry_date: str,
        current_date: str,
        side: str,
    ) -> bool:
        """
        Time stop — exit if held > TIME_STOP_DAYS with P&L between
        TIME_STOP_MIN_PNL and TIME_STOP_MAX_PNL (dead money).
        """
        if not entry_date or not current_date or entry_price <= 0:
            return False

        try:
            entry_dt = pd.Timestamp(entry_date)
            current_dt = pd.Timestamp(current_date)
        except (ValueError, TypeError):
            return False

        days_held = (current_dt - entry_dt).days
        if days_held <= self.time_stop_days:
            return False

        # Calculate P&L %
        if side == "long":
            pnl_pct = (current_close - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_close) / entry_price

        # Dead money: P&L stuck in the flat band
        return self.time_stop_min <= pnl_pct <= self.time_stop_max

    def _check_signal_reversal(
        self, symbol: str, side: str, current_signals: dict
    ) -> bool:
        """
        Signal reversal — exit if the current signal opposes
        the position direction.
        """
        signal = current_signals.get(symbol)
        if signal is None:
            return False

        # Signal object has a .direction attribute ('long', 'short', 'cash')
        sig_dir = getattr(signal, "direction", None)
        if sig_dir is None:
            return False

        if side == "long" and sig_dir == "short":
            return True
        if side == "short" and sig_dir == "long":
            return True

        return False

    # ──────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _wilder_atr(df: pd.DataFrame, period: int) -> float:
        """
        Compute ATR using Wilder smoothing (EMA with alpha = 1/period).

        True Range = max(high-low, |high-prev_close|, |low-prev_close|)
        ATR = Wilder EMA of True Range over `period` days.

        Returns the most recent ATR value as a float.
        """
        high = df["High"].values
        low = df["Low"].values
        close = df["Close"].values

        # True Range
        tr = np.empty(len(df))
        tr[0] = high[0] - low[0]
        for i in range(1, len(df)):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )

        # Wilder smoothing: EMA with alpha = 1/period
        atr = np.empty(len(df))
        atr[:period] = np.nan
        atr[period - 1] = np.mean(tr[:period])  # seed with SMA

        alpha = 1.0 / period
        for i in range(period, len(df)):
            atr[i] = atr[i - 1] * (1 - alpha) + tr[i] * alpha

        return float(atr[-1]) if not np.isnan(atr[-1]) else 0.0
