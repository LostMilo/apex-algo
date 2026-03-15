"""
tsmom.py — Time Series Momentum Strategy

Implements TSMOM — the primary alpha source.
Based on Moskowitz, Ooi & Pedersen (2012).
Academic reference: compares current price to its own past and bets
on continuation.

CRITICAL RULES:
  - Zero look-ahead bias: NEVER use data from the current day or future days.
  - Stateless: same inputs always produce same outputs.
  - Uses only data provided as argument — never fetches its own data.
  - Uses only past data: all calculations use strictly historical windows
    ending at t-1.
"""

import logging

import pandas as pd

import config

logger = logging.getLogger(__name__)


class TSMOMStrategy:
    """
    Time Series Momentum strategy.

    For each symbol, computes a 12-month return (skipping the most recent
    month to reduce short-term reversal noise) and returns a directional
    signal in [-1.0, +1.0].
    """

    def __init__(self, cfg=None):
        """Store config reference.  No data loading here."""
        self.cfg = cfg  # reserved for future per-instance overrides

    # ── Primary signal generation ───────────────────────────────────────────

    def compute_signals(
        self,
        price_data: dict[str, pd.DataFrame],
        current_date: pd.Timestamp,
    ) -> dict[str, float]:
        """
        Compute raw TSMOM signals for every symbol.

        Parameters
        ----------
        price_data : dict[str, pd.DataFrame]
            Symbol → OHLCV DataFrame.  Full history up to and including
            yesterday.  Columns: open, high, low, close, volume (lowercase).
            Index: DatetimeIndex.
        current_date : pd.Timestamp
            The date we are computing signals FOR.  Data from this date
            (and any future date) is strictly forbidden.

        Returns
        -------
        dict[str, float]
            Symbol → signal in [-1.0, 1.0].
            0.0 means no signal (insufficient data).
        """
        signals: dict[str, float] = {}

        for symbol, df in price_data.items():
            signals[symbol] = self._single_signal(symbol, df, current_date)

        return signals

    # ── Volume confirmation filter ──────────────────────────────────────────

    def compute_volume_confirmation(
        self,
        price_data: dict[str, pd.DataFrame],
        current_date: pd.Timestamp,
        signals: dict[str, float],
    ) -> dict[str, float]:
        """
        Reduce signal strength when yesterday's volume is abnormally low.

        For each symbol with a non-zero signal:
          1. Compute the 20-day average volume ending at current_date − 1.
          2. Get yesterday's volume.
          3. If yesterday's volume < avg * VOLUME_CONFIRM_MULTIPLIER,
             reduce the signal by 50 %.

        Returns
        -------
        dict[str, float]
            Filtered signals dict (same keys as *signals*).
        """
        filtered: dict[str, float] = {}

        for symbol, signal in signals.items():
            if signal == 0.0:
                filtered[symbol] = 0.0
                continue

            df = price_data.get(symbol)
            if df is None or df.empty:
                filtered[symbol] = signal
                continue

            # Strict past-only slice
            hist = df.loc[df.index < current_date]
            if len(hist) < config.VOLUME_MA_PERIOD:
                filtered[symbol] = signal
                continue

            vol_window = hist["volume"].iloc[-config.VOLUME_MA_PERIOD:]
            avg_vol = vol_window.mean()
            yesterday_vol = hist["volume"].iloc[-1]

            if yesterday_vol < avg_vol * config.VOLUME_CONFIRM_MULTIPLIER:
                adjusted = signal * 0.5
                logger.info(
                    "TSMOM vol filter  |  %s  vol=%.0f  avg=%.0f  "
                    "signal %.3f → %.3f",
                    symbol, yesterday_vol, avg_vol, signal, adjusted,
                )
                filtered[symbol] = adjusted
            else:
                filtered[symbol] = signal

        return filtered

    # ── Metadata for downstream consumers ───────────────────────────────────

    def get_signal_metadata(
        self,
        price_data: dict[str, pd.DataFrame],
        current_date: pd.Timestamp,
    ) -> dict[str, dict]:
        """
        Return rich metadata for each symbol's signal computation.

        Useful for logging, dashboards, and the training agent.

        Returns
        -------
        dict[str, dict]
            Symbol → {
                "momentum_return": float | None,
                "signal_direction": float,
                "signal_strength": float,
                "final_signal": float,
                "data_points": int,
                "lookback": int,
                "skip_last": int,
            }
        """
        metadata: dict[str, dict] = {}

        for symbol, df in price_data.items():
            metadata[symbol] = self._compute_metadata(symbol, df, current_date)

        return metadata

    # ─── Private Helpers ────────────────────────────────────────────────────

    def _single_signal(
        self, symbol: str, df: pd.DataFrame, current_date: pd.Timestamp,
    ) -> float:
        """Compute the TSMOM signal for a single symbol."""
        lookback = config.TSMOM_LOOKBACK
        skip_last = config.TSMOM_SKIP_LAST
        min_rows = lookback + skip_last + 10

        # ── 1. Strict past-only slice ──────────────────────────────────────
        hist = df.loc[df.index < current_date]

        # Enforced assertion — catches any accidental look-ahead
        assert (
            hist.empty or hist.index.max() < current_date
        ), "Look-ahead bias detected"

        # ── 2. Insufficient data guard ─────────────────────────────────────
        if len(hist) < min_rows:
            logger.debug(
                "TSMOM skip  |  %s  rows=%d  need=%d",
                symbol, len(hist), min_rows,
            )
            return 0.0

        # ── 3. Compute momentum return ─────────────────────────────────────
        #
        #   Formula (from spec):
        #     return = close[-(TSMOM_SKIP_LAST+1)]
        #              / close[-(TSMOM_SKIP_LAST+1+TSMOM_LOOKBACK)]
        #              - 1
        #
        #   close[-(skip_last+1)]            → price *skip_last* days ago
        #   close[-(skip_last+1+lookback)]   → price *lookback+skip_last* days ago
        #
        close = hist["close"]
        recent_price = close.iloc[-(skip_last + 1)]
        past_price = close.iloc[-(skip_last + 1 + lookback)]
        momentum_return = (recent_price / past_price) - 1.0

        # ── 4. Direction and strength ──────────────────────────────────────
        if momentum_return > 0:
            signal_direction = 1.0
        elif momentum_return < 0:
            signal_direction = -1.0
        else:
            return 0.0

        signal_strength = abs(momentum_return)  # used for position sizing
        final_signal = signal_direction * min(signal_strength, 1.0)  # clipped to [-1, 1]

        logger.debug(
            "TSMOM signal  |  %s  ret=%.4f  dir=%.0f  str=%.4f  sig=%.4f",
            symbol, momentum_return, signal_direction, signal_strength,
            final_signal,
        )
        return final_signal

    def _compute_metadata(
        self, symbol: str, df: pd.DataFrame, current_date: pd.Timestamp,
    ) -> dict:
        """Compute detailed metadata for a single symbol."""
        lookback = config.TSMOM_LOOKBACK
        skip_last = config.TSMOM_SKIP_LAST
        min_rows = lookback + skip_last + 10

        hist = df.loc[df.index < current_date]

        base = {
            "lookback": lookback,
            "skip_last": skip_last,
            "data_points": len(hist),
        }

        if len(hist) < min_rows:
            base.update({
                "momentum_return": None,
                "signal_direction": 0.0,
                "signal_strength": 0.0,
                "final_signal": 0.0,
            })
            return base

        close = hist["close"]
        recent_price = close.iloc[-(skip_last + 1)]
        past_price = close.iloc[-(skip_last + 1 + lookback)]
        momentum_return = (recent_price / past_price) - 1.0

        if momentum_return > 0:
            direction = 1.0
        elif momentum_return < 0:
            direction = -1.0
        else:
            direction = 0.0

        strength = abs(momentum_return)
        final = direction * min(strength, 1.0)

        base.update({
            "momentum_return": momentum_return,
            "signal_direction": direction,
            "signal_strength": strength,
            "final_signal": final,
        })
        return base
