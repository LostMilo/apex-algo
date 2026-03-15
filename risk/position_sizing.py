"""
risk/position_sizing.py — GARCH + Kelly Criterion Position Sizer

Translates signal strength into actual dollar position sizes.
This is the most important risk control in the system.

Pipeline per symbol:
  1. GARCH(1,1) volatility estimate (annualised)
  2. Kelly Criterion → optimal fraction
  3. Volatility scaling (reduce in high-vol regimes)
  4. Dollar amount + direction
  5. Portfolio-level exposure constraints
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import config
from logger import log

# GARCH fitting — optional, with graceful fallback
try:
    from arch import arch_model  # type: ignore
    _HAS_ARCH = True
except ImportError:
    _HAS_ARCH = False
    log.warning("arch library not installed — GARCH disabled, using rolling-std fallback")


class PositionSizer:
    """Translate signal strength → dollar position sizes via GARCH + Kelly."""

    def __init__(
        self,
        garch_lookback: int | None = None,
        kelly_fraction: float | None = None,
        kelly_cap: float | None = None,
        max_total_exposure: float | None = None,
        max_drawdown_scale: float = 0.05,
    ):
        self.garch_lookback = garch_lookback or config.GARCH_LOOKBACK        # 252
        self.kelly_fraction = kelly_fraction or config.KELLY_FRACTION        # 0.50
        self.kelly_cap = kelly_cap or config.KELLY_CAP                       # 0.25
        self.max_total_exposure = max_total_exposure or config.MAX_TOTAL_EXPOSURE  # 1.0
        self.max_drawdown_scale = max_drawdown_scale  # halve sizes above this

    # ── PUBLIC API ────────────────────────────────────────────────────

    def compute(
        self,
        signals: dict[str, float],
        price_data: dict[str, pd.DataFrame],
        portfolio: dict,
        current_date: pd.Timestamp | str,
    ) -> dict[str, float]:
        """
        Compute dollar position sizes for every symbol with a non-zero signal.

        Args:
            signals:      {symbol → signal_strength} (positive = long, negative = short)
            price_data:   {symbol → OHLCV DataFrame}
            portfolio:    dict with keys: cash, positions, equity, peak_equity
            current_date: date used to slice price history (anti-look-ahead)

        Returns:
            {symbol → dollar_amount_to_trade}  (positive = long, negative = short)
        """
        equity = portfolio.get("equity", config.STARTING_CAPITAL)
        peak_equity = portfolio.get("peak_equity", equity)

        # Drawdown guard
        drawdown_mult = self._drawdown_multiplier(equity, peak_equity)

        raw_sizes: dict[str, float] = {}

        for symbol, signal in signals.items():
            if signal == 0.0:
                continue

            df = price_data.get(symbol)
            if df is None or df.empty:
                log.warning("{}: no price data — skipping position sizing", symbol)
                continue

            # Slice to avoid look-ahead
            df_hist = self._slice_before(df, current_date)
            if df_hist.empty or len(df_hist) < 30:
                log.debug("{}: insufficient history for sizing", symbol)
                continue

            # 1. GARCH volatility estimate
            annualized_vol = self._estimate_volatility(symbol, df_hist)

            # 2. Kelly criterion → optimal fraction
            f_final = self._kelly_fraction(signal, annualized_vol)

            # 3. Volatility scaling
            f_scaled = self._volatility_scale(f_final, annualized_vol)

            # 4. Dollar amount
            dollar_size = equity * f_scaled

            # Apply signal direction
            if signal < 0:
                dollar_size *= -1

            # Apply drawdown guard
            dollar_size *= drawdown_mult

            raw_sizes[symbol] = dollar_size

        # 5. Portfolio constraints
        final_sizes = self._apply_constraints(raw_sizes, equity)

        log.info(
            "PositionSizer: {} positions sized (equity={:.2f}, dd_mult={:.2f})",
            len(final_sizes), equity, drawdown_mult,
        )
        return final_sizes

    # ── STEP 1: GARCH VOLATILITY ─────────────────────────────────────

    def _estimate_volatility(self, symbol: str, df: pd.DataFrame) -> float:
        """Estimate annualised volatility via GARCH(1,1) with rolling-std fallback."""
        price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
        prices = df[price_col].dropna()
        returns = prices.pct_change().dropna()

        lookback = min(self.garch_lookback, len(returns))
        returns_window = returns.iloc[-lookback:]

        if _HAS_ARCH and len(returns_window) >= 60:
            try:
                vol = self._garch_vol(returns_window)
                if vol is not None and np.isfinite(vol) and vol > 0:
                    log.debug("{}: GARCH vol = {:.4f}", symbol, vol)
                    return vol
            except Exception as exc:
                log.debug("{}: GARCH failed ({}), using fallback", symbol, exc)

        # Fallback — rolling standard deviation
        daily_vol = returns_window.std()
        annualized_vol = daily_vol * np.sqrt(252)
        if not np.isfinite(annualized_vol) or annualized_vol <= 0:
            annualized_vol = 0.20  # default 20%
        log.debug("{}: fallback vol = {:.4f}", symbol, annualized_vol)
        return float(annualized_vol)

    @staticmethod
    def _garch_vol(returns: pd.Series) -> float | None:
        """Fit GARCH(1,1) and return 1-day-ahead annualised volatility."""
        # arch expects returns in percentage points
        scaled = returns * 100.0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = arch_model(scaled, vol="Garch", p=1, q=1, mean="Zero", rescale=False)
            result = model.fit(disp="off", show_warning=False)

        # 1-day-ahead forecast
        forecast = result.forecast(horizon=1)
        daily_var = forecast.variance.iloc[-1, 0]  # variance in pct^2
        daily_vol = np.sqrt(daily_var) / 100.0      # back to decimal
        annualized_vol = daily_vol * np.sqrt(252)
        return float(annualized_vol)

    # ── STEP 2: KELLY CRITERION ──────────────────────────────────────

    def _kelly_fraction(self, signal: float, annualized_vol: float) -> float:
        """
        Kelly sizing: f = edge / (vol^2).

        Edge is estimated as |signal| × 0.10 (conservative: 10% edge at
        full signal strength).  Half-Kelly applied, capped at KELLY_CAP.
        """
        edge = abs(signal) * 0.10
        vol_sq = annualized_vol ** 2

        if vol_sq <= 0:
            return 0.0

        f = edge / vol_sq

        # Half-Kelly (default KELLY_FRACTION = 0.50)
        f_half = f * self.kelly_fraction

        # Cap at KELLY_CAP (default 0.25)
        f_final = min(f_half, self.kelly_cap)

        return f_final

    # ── STEP 3: VOLATILITY SCALING ───────────────────────────────────

    @staticmethod
    def _volatility_scale(f: float, annualized_vol: float) -> float:
        """
        Scale position down when volatility is elevated.

        vol_scalar = min(0.20 / annualized_vol, 1.0)
        If annualized vol > 20%, reduce position proportionally.
        """
        if annualized_vol <= 0:
            return 0.0
        vol_scalar = min(0.20 / annualized_vol, 1.0)
        return f * vol_scalar

    # ── STEP 5: PORTFOLIO CONSTRAINTS ────────────────────────────────

    def _apply_constraints(
        self, sizes: dict[str, float], equity: float,
    ) -> dict[str, float]:
        """
        Enforce portfolio-level exposure limits.

        - Total long exposure ≤ MAX_TOTAL_EXPOSURE × equity
        - Proportional scale-down if breached
        """
        if not sizes:
            return sizes

        max_long = self.max_total_exposure * equity

        # Sum of long positions
        total_long = sum(v for v in sizes.values() if v > 0)

        if total_long > max_long and total_long > 0:
            scale = max_long / total_long
            log.info(
                "Long exposure {:.2f} exceeds cap {:.2f} — scaling by {:.3f}",
                total_long, max_long, scale,
            )
            sizes = {
                sym: (val * scale if val > 0 else val)
                for sym, val in sizes.items()
            }

        # Minimum order filter
        sizes = {
            sym: val for sym, val in sizes.items()
            if abs(val) >= config.MIN_ORDER_VALUE
        }

        return sizes

    # ── HELPERS ───────────────────────────────────────────────────────

    def _drawdown_multiplier(self, equity: float, peak_equity: float) -> float:
        """
        Halve all position sizes when the portfolio is in drawdown > 5%.

        Returns a multiplier in (0, 1].
        """
        if peak_equity <= 0:
            return 1.0
        drawdown = (peak_equity - equity) / peak_equity
        if drawdown > self.max_drawdown_scale:
            log.warning("Drawdown {:.2%} > {:.2%} — halving position sizes", drawdown, self.max_drawdown_scale)
            return 0.5
        return 1.0

    @staticmethod
    def _slice_before(df: pd.DataFrame, current_date) -> pd.DataFrame:
        """Return rows strictly before current_date to prevent look-ahead."""
        if current_date is None:
            return df

        current_date = pd.Timestamp(current_date)

        if isinstance(df.index, pd.DatetimeIndex):
            return df.loc[df.index < current_date]

        # Non-datetime index — check for a Date column
        if "Date" in df.columns:
            mask = pd.to_datetime(df["Date"]) < current_date
            return df.loc[mask]

        return df  # can't slice, return as-is
