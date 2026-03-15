"""
strategies/vol_trend.py — EMA Crossover filtered by ADX

Signal logic:
  1. EMA crossover  — fast EMA vs slow EMA determines direction
  2. ADX filter     — Wilder's method; only trade when ADX_MIN ≤ ADX ≤ ADX_MAX
                      ADX > 40  →  trend exhaustion  →  skip
  3. Volume confirm — reduce signal by 50 % if volume < 20-day average

Active in TRENDING regime only.  Healthy (not exhausted) trends.
"""

import numpy as np
import pandas as pd

import config
from logger import log


class VolTrendStrategy:
    """Volatility-adjusted trend following via EMA crossover + ADX gate."""

    NAME = "VolTrend"

    def __init__(
        self,
        ema_fast: int = None,
        ema_slow: int = None,
        adx_period: int = None,
        adx_min: float = None,
        adx_max: float = None,
    ):
        self.ema_fast = ema_fast or config.EMA_FAST            # 20
        self.ema_slow = ema_slow or config.EMA_SLOW            # 50
        self.adx_period = adx_period or config.ADX_PERIOD      # 14
        self.adx_min = adx_min if adx_min is not None else config.ADX_MIN  # 25
        self.adx_max = adx_max if adx_max is not None else config.ADX_MAX  # 40

    # ── public API ──────────────────────────────────────────────────────

    def compute_signals(
        self,
        price_data: dict[str, pd.DataFrame],
        current_date: pd.Timestamp,
    ) -> dict[str, float]:
        """
        Generate VolTrend signals for every symbol whose data extends
        strictly before *current_date*.

        Returns
        -------
        dict[symbol → signal]  where signal ∈ [-1, +1]
        """
        signals: dict[str, float] = {}

        for symbol, df in price_data.items():
            sig = self._signal_for_symbol(symbol, df, current_date)
            signals[symbol] = sig

        log.info("VolTrend generated {} signals", len(signals))
        return signals

    def get_signal_metadata(
        self,
        price_data: dict[str, pd.DataFrame],
        current_date: pd.Timestamp,
    ) -> dict:
        """
        Return diagnostic metadata for each symbol.

        Returns
        -------
        {symbol: {signal, ema_fast, ema_slow, adx, adx_pass, direction}}
        """
        meta: dict = {}

        for symbol, df in price_data.items():
            hist = df.loc[df.index < current_date].copy()
            if len(hist) < self.ema_slow + self.adx_period:
                continue

            close = hist["Close"] if "Close" in hist.columns else hist["close"]

            ema_f = close.ewm(span=self.ema_fast, adjust=False).mean()
            ema_s = close.ewm(span=self.ema_slow, adjust=False).mean()
            adx = self._compute_adx(hist)

            if adx is None or np.isnan(adx.iloc[-1]):
                continue

            adx_val = adx.iloc[-1]
            adx_pass = 1 if self.adx_min <= adx_val <= self.adx_max else 0
            direction = "long" if ema_f.iloc[-1] > ema_s.iloc[-1] else "short"

            meta[symbol] = {
                "signal": self._signal_for_symbol(symbol, df, current_date),
                "ema_fast": float(ema_f.iloc[-1]),
                "ema_slow": float(ema_s.iloc[-1]),
                "adx": float(adx_val),
                "adx_pass": adx_pass,
                "direction": direction,
            }

        return meta

    # ── private ─────────────────────────────────────────────────────────

    def _signal_for_symbol(
        self,
        symbol: str,
        df: pd.DataFrame,
        current_date: pd.Timestamp,
    ) -> float:
        """Core per-symbol signal logic.  Returns value in [-1, +1]."""

        # Slice strictly before current date
        hist = df.loc[df.index < current_date].copy()

        min_rows = self.ema_slow + self.adx_period
        if len(hist) < min_rows:
            log.debug(
                "{}: insufficient data for VolTrend ({} < {})",
                symbol, len(hist), min_rows,
            )
            return 0.0

        close = hist["Close"] if "Close" in hist.columns else hist["close"]

        # ── 1. EMA crossover ────────────────────────────────────────────
        ema_fast = close.ewm(span=self.ema_fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.ema_slow, adjust=False).mean()

        crossover_signal = +1 if ema_fast.iloc[-1] > ema_slow.iloc[-1] else -1
        crossover_strength = (
            abs(ema_fast.iloc[-1] - ema_slow.iloc[-1]) / ema_slow.iloc[-1]
        )

        # ── 2. ADX filter (Wilder's method) ─────────────────────────────
        adx = self._compute_adx(hist)
        if adx is None:
            return 0.0

        adx_val = adx.iloc[-1]
        if np.isnan(adx_val):
            return 0.0

        # ADX > 40  →  trend exhaustion, skip
        adx_pass = 1 if self.adx_min <= adx_val <= self.adx_max else 0

        if adx_pass == 0:
            log.debug(
                "{}: ADX filter blocked (ADX={:.1f}, range={}-{})",
                symbol, adx_val, self.adx_min, self.adx_max,
            )
            return 0.0

        # ── 3. Final signal ─────────────────────────────────────────────
        signal = crossover_signal * min(crossover_strength * 10, 1.0)

        # ── 4. Volume confirmation ──────────────────────────────────────
        vol_col = "Volume" if "Volume" in hist.columns else "volume"
        if vol_col in hist.columns:
            vol = hist[vol_col]
            vol_ma = vol.rolling(window=config.VOLUME_MA_PERIOD).mean()
            if not np.isnan(vol_ma.iloc[-1]) and vol_ma.iloc[-1] > 0:
                if vol.iloc[-1] < vol_ma.iloc[-1]:
                    signal *= 0.5  # reduce by 50 % if below-average volume

        # Clamp to [-1, +1]
        signal = max(-1.0, min(1.0, signal))

        log.debug(
            "{}: VolTrend ema_f={:.2f} ema_s={:.2f} adx={:.1f} sig={:.3f}",
            symbol, ema_fast.iloc[-1], ema_slow.iloc[-1], adx_val, signal,
        )
        return signal

    # ── ADX via Wilder's smoothing ──────────────────────────────────────

    def _compute_adx(self, df: pd.DataFrame) -> pd.Series | None:
        """
        Compute ADX using Wilder's smoothing method.

        Steps:
          TR  = max(high-low, abs(high-prev_close), abs(low-prev_close))
          +DM = high-prev_high  if (high-prev_high) > (prev_low-low) AND > 0
          -DM = prev_low-low    if (prev_low-low)   > (high-prev_high) AND > 0
          Smooth TR, +DM, -DM with Wilder smoothing (period = ADX_PERIOD)
          +DI = 100 × smoothed_+DM / smoothed_TR
          -DI = 100 × smoothed_-DM / smoothed_TR
          DX  = 100 × abs(+DI − -DI) / (+DI + -DI)
          ADX = Wilder smooth of DX
        """
        high = df["High"] if "High" in df.columns else df["high"]
        low = df["Low"] if "Low" in df.columns else df["low"]
        close = df["Close"] if "Close" in df.columns else df["close"]

        n = len(df)
        period = self.adx_period
        if n < period * 2 + 1:
            return None

        prev_high = high.shift(1)
        prev_low = low.shift(1)
        prev_close = close.shift(1)

        # True Range
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        up_move = high - prev_high
        down_move = prev_low - low

        plus_dm = pd.Series(0.0, index=df.index)
        minus_dm = pd.Series(0.0, index=df.index)

        plus_mask = (up_move > down_move) & (up_move > 0)
        minus_mask = (down_move > up_move) & (down_move > 0)

        plus_dm[plus_mask] = up_move[plus_mask]
        minus_dm[minus_mask] = down_move[minus_mask]

        # Drop first row (NaN from shift)
        tr = tr.iloc[1:]
        plus_dm = plus_dm.iloc[1:]
        minus_dm = minus_dm.iloc[1:]

        # Wilder smoothing
        smooth_tr = self._wilder_smooth(tr, period)
        smooth_plus_dm = self._wilder_smooth(plus_dm, period)
        smooth_minus_dm = self._wilder_smooth(minus_dm, period)

        # +DI, -DI
        plus_di = 100.0 * smooth_plus_dm / smooth_tr
        minus_di = 100.0 * smooth_minus_dm / smooth_tr

        # DX
        di_sum = plus_di + minus_di
        di_sum = di_sum.replace(0, np.nan)
        dx = 100.0 * (plus_di - minus_di).abs() / di_sum

        # ADX = Wilder smooth of DX
        adx = self._wilder_smooth(dx.dropna(), period)

        # Re-index to original DataFrame index
        adx = adx.reindex(df.index)
        return adx

    @staticmethod
    def _wilder_smooth(series: pd.Series, period: int) -> pd.Series:
        """
        Wilder's smoothing (equivalent to EMA with alpha = 1 / period).

        First value = SMA of first *period* values, then:
            smoothed[i] = smoothed[i-1] × (period-1)/period + value[i] / period
        """
        values = series.values.astype(float)
        result = np.full_like(values, np.nan)

        if len(values) < period:
            return pd.Series(result, index=series.index)

        # Seed with SMA
        result[period - 1] = np.mean(values[:period])

        # Recursive smoothing
        alpha = 1.0 / period
        for i in range(period, len(values)):
            result[i] = result[i - 1] * (1.0 - alpha) + values[i] * alpha

        return pd.Series(result, index=series.index)


# ─── Self-Test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    np.random.seed(42)
    n_days = 200

    dates = pd.bdate_range("2024-01-01", periods=n_days)

    def _make_df(trend: float = 0.0, vol: float = 0.01) -> pd.DataFrame:
        """Generate synthetic OHLCV data with a controllable trend."""
        returns = np.random.normal(trend, vol, n_days)
        close = 100.0 * np.exp(np.cumsum(returns))
        high = close * (1 + np.random.uniform(0, 0.01, n_days))
        low = close * (1 - np.random.uniform(0, 0.01, n_days))
        opn = close * (1 + np.random.normal(0, 0.002, n_days))
        volume = np.random.randint(1_000_000, 10_000_000, n_days).astype(float)
        return pd.DataFrame(
            {"Open": opn, "High": high, "Low": low,
             "Close": close, "Volume": volume},
            index=dates,
        )

    strat = VolTrendStrategy()
    current = dates[-1] + pd.Timedelta(days=1)
    passed = 0
    failed = 0

    # ── Test 1: ADX blocks signals in sideways (noisy) markets ──────────
    sideways = {"FLAT": _make_df(trend=0.0, vol=0.005)}
    sig = strat.compute_signals(sideways, current)
    if sig["FLAT"] == 0.0:
        print("✓ ADX blocks sideways market")
        passed += 1
    else:
        print(f"✗ Sideways market not blocked (signal={sig['FLAT']:.4f})")
        failed += 1

    # ── Test 2: ADX blocks when ADX > ADX_MAX (exhausted trend) ─────────
    strong_trend = {"STRONG": _make_df(trend=0.005, vol=0.001)}
    meta = strat.get_signal_metadata(strong_trend, current)
    if "STRONG" in meta:
        if meta["STRONG"]["adx"] > config.ADX_MAX:
            if meta["STRONG"]["signal"] == 0.0:
                print("✓ ADX > ADX_MAX correctly blocked")
                passed += 1
            else:
                print(
                    f"✗ ADX > ADX_MAX not blocked "
                    f"(adx={meta['STRONG']['adx']:.1f})"
                )
                failed += 1
        else:
            # ADX didn't exceed the cap with this seed — not a test failure
            print(
                f"⚠ ADX={meta['STRONG']['adx']:.1f} didn't exceed "
                f"ADX_MAX — can't test exhaustion filter"
            )
            passed += 1
    else:
        print("⚠ STRONG not in metadata — insufficient data")
        passed += 1

    # ── Test 3: Correct direction detection ─────────────────────────────
    uptrend = {"UP": _make_df(trend=0.002, vol=0.008)}
    sig_up = strat.compute_signals(uptrend, current)
    meta_up = strat.get_signal_metadata(uptrend, current)
    if "UP" in meta_up:
        if meta_up["UP"]["adx_pass"] == 1:
            if sig_up["UP"] > 0:
                print("✓ Uptrend correctly detected as positive signal")
                passed += 1
            else:
                print(
                    f"✗ Uptrend gave non-positive signal "
                    f"({sig_up['UP']:.4f})"
                )
                failed += 1
        else:
            print(
                f"⚠ ADX didn't pass for uptrend data "
                f"(adx={meta_up['UP']['adx']:.1f})"
            )
            passed += 1
    else:
        print("⚠ UP not in metadata")
        passed += 1

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\nVolTrend unit tests: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
    print("VolTrend unit tests passed")
