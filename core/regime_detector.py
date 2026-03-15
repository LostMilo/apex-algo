"""
core/regime_detector.py — Market Regime Classification

The regime detector runs FIRST every day before any strategy signal is computed.
It gates which strategies are allowed to fire.  Never reads data directly — uses
sliced data passed to it by the engine.

Regime output (one of three constants):
    TRENDING  — strong directional moves, trend-following strategies active
    RANGING   — sideways / mean-reverting, stat-arb strategies active
    RISK_OFF  — high-vol drawdown environment, capital preservation

Signals combined:
    1. Volatility  (GARCH vol estimate)
    2. Trend       (200-day MA + ADX)
    3. Breadth     (% of universe above own 200-day MA)
    4. Sentiment   (optional alt-data macro signal)
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

import config
from logger import log

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

TRENDING = "TRENDING"
RANGING = "RANGING"
RISK_OFF = "RISK_OFF"


class Regime:
    """
    Lightweight regime constants with enum-style attribute access.

    Allows both:
        if regime == Regime.TRENDING: ...
        if regime == "TRENDING": ...
    """
    TRENDING = TRENDING
    RANGING = RANGING
    RISK_OFF = RISK_OFF

    @classmethod
    def values(cls) -> list[str]:
        return [cls.TRENDING, cls.RANGING, cls.RISK_OFF]


# ──────────────────────────────────────────────────────────────────────────────
# RegimeDetector
# ──────────────────────────────────────────────────────────────────────────────

class RegimeDetector:
    """
    Stateless regime classifier.

    Call :meth:`detect` once per day with the full sliced price_data dict and
    the current date.  The detector never fetches data itself.
    """

    # ── public interface ────────────────────────────────────────────────────

    def detect(
        self,
        price_data: dict[str, pd.DataFrame],
        current_date: pd.Timestamp | str,
    ) -> str:
        """
        Combine multiple signals to classify the current market regime.

        Parameters
        ----------
        price_data : dict[str, pd.DataFrame]
            Symbol → OHLCV DataFrame already sliced to rows
            **before** *current_date* (no look-ahead).
            Expected columns: ``Close``, ``High``, ``Low``, ``Volume``
            (capital first letter, matching yfinance / Alpaca standard).
        current_date : pd.Timestamp | str
            The date being evaluated (used only for logging / filtering).

        Returns
        -------
        str
            One of ``TRENDING``, ``RANGING``, or ``RISK_OFF``.
        """
        current_date = pd.Timestamp(current_date)

        # ── 1. Volatility signal (SPY proxy) ────────────────────────────────
        vol_signal = self._volatility_signal(price_data)

        # ── 2. Trend signal (SPY) ───────────────────────────────────────────
        trend_signal, trending = self._trend_signal(price_data)

        # ── 3. Breadth signal (whole universe) ──────────────────────────────
        breadth_signal = self._breadth_signal(price_data)

        # ── 4. Alt-data macro sentiment (optional) ──────────────────────────
        sentiment_signal = self._sentiment_signal()

        # ── Classification rules ────────────────────────────────────────────
        regime = self._classify(
            vol_signal, trend_signal, trending,
            breadth_signal, sentiment_signal,
        )

        log.info(
            "Regime={} | date={} | vol={} trend={} adx_trending={} "
            "breadth={} sentiment={}",
            regime,
            current_date.strftime("%Y-%m-%d"),
            vol_signal,
            trend_signal,
            trending,
            breadth_signal,
            sentiment_signal,
        )
        return regime

    # ── private: individual signals ─────────────────────────────────────────

    def _volatility_signal(
        self,
        price_data: dict[str, pd.DataFrame],
    ) -> str:
        """
        GARCH(1,1) rolling volatility on SPY returns.

        Returns ``'HIGH'`` if annualised GARCH vol > REGIME_VOL_HIGH_THRESHOLD,
        else ``'NORMAL'``.
        """
        spy = self._get_spy(price_data)
        if spy is None or len(spy) < config.GARCH_LOOKBACK:
            return "NORMAL"

        closes = spy["Close"].dropna().tail(config.GARCH_LOOKBACK)
        returns = closes.pct_change().dropna() * 100  # arch expects pct

        if len(returns) < 50:
            return "NORMAL"

        try:
            from arch import arch_model

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = arch_model(
                    returns,
                    vol="Garch",
                    p=1,
                    q=1,
                    mean="Zero",
                    rescale=False,
                )
                result = model.fit(disp="off", show_warning=False)

            # Last conditional variance → annualise
            # variance is in (pct-return)^2; convert back to decimal
            cond_var = result.conditional_volatility.iloc[-1]
            annualised_vol = (cond_var / 100) * np.sqrt(252)

            if annualised_vol > config.REGIME_VOL_HIGH_THRESHOLD:
                return "HIGH"
            return "NORMAL"

        except Exception as exc:  # noqa: BLE001
            log.warning("GARCH fit failed ({}), falling back to std vol", exc)
            # Fallback: simple rolling std
            simple_vol = (returns / 100).std() * np.sqrt(252)
            if simple_vol > config.REGIME_VOL_HIGH_THRESHOLD:
                return "HIGH"
            return "NORMAL"

    def _trend_signal(
        self,
        price_data: dict[str, pd.DataFrame],
    ) -> tuple[str, bool]:
        """
        200-day MA trend + ADX confirmation on SPY.

        Returns
        -------
        trend_signal : str
            ``'UP'``, ``'DOWN'``, or ``'NEUTRAL'``
        trending : bool
            ``True`` if ADX > REGIME_ADX_TREND_MIN
        """
        spy = self._get_spy(price_data)
        if spy is None or len(spy) < config.MA_TREND_PERIOD:
            return "NEUTRAL", False

        closes = spy["Close"]
        ma200 = closes.rolling(window=config.MA_TREND_PERIOD).mean()

        last_close = closes.iloc[-1]
        last_ma = ma200.iloc[-1]

        if pd.isna(last_ma):
            return "NEUTRAL", False

        # Trend direction
        if last_close > last_ma:
            trend_signal = "UP"
        elif last_close < last_ma * 0.97:
            trend_signal = "DOWN"
        else:
            trend_signal = "NEUTRAL"

        # ADX confirmation
        trending = False
        adx_val = self._compute_adx(spy)
        if adx_val is not None and adx_val > config.REGIME_ADX_TREND_MIN:
            trending = True

        return trend_signal, trending

    def _breadth_signal(
        self,
        price_data: dict[str, pd.DataFrame],
    ) -> str:
        """
        Fraction of universe trading above their own 200-day MA.

        Returns ``'WEAK'``, ``'STRONG'``, or ``'NEUTRAL'``.
        """
        universe = list(price_data.keys())
        if not universe:
            return "NEUTRAL"

        above_count = 0
        valid_count = 0

        for sym, df in price_data.items():
            if len(df) < config.MA_TREND_PERIOD:
                continue
            closes = df["Close"]
            ma200 = closes.rolling(window=config.MA_TREND_PERIOD).mean()
            last_close = closes.iloc[-1]
            last_ma = ma200.iloc[-1]
            if pd.isna(last_ma):
                continue
            valid_count += 1
            if last_close > last_ma:
                above_count += 1

        if valid_count == 0:
            return "NEUTRAL"

        breadth_pct = above_count / valid_count

        if breadth_pct < 0.30:
            return "WEAK"
        elif breadth_pct > 0.70:
            return "STRONG"
        else:
            return "NEUTRAL"

    def _sentiment_signal(self) -> str:
        """
        Alt-data macro sentiment (only when ``USE_ALT_DATA`` is True).

        Reads from the DataAgent.  Returns ``'BEARISH'``, ``'BULLISH'``,
        or ``'NEUTRAL'``.
        """
        if not config.USE_ALT_DATA:
            return "NEUTRAL"

        try:
            from data.data_agent import DataAgent

            agent = DataAgent()
            sentiment = agent.get_sentiment("stock market economy")

            if sentiment < -0.3:
                return "BEARISH"
            elif sentiment > 0.3:
                return "BULLISH"
            else:
                return "NEUTRAL"
        except Exception as exc:  # noqa: BLE001
            log.warning("Sentiment signal unavailable ({}), defaulting NEUTRAL", exc)
            return "NEUTRAL"

    # ── classification logic ────────────────────────────────────────────────

    @staticmethod
    def _classify(
        vol_signal: str,
        trend_signal: str,
        trending: bool,
        breadth_signal: str,
        sentiment_signal: str,
    ) -> str:
        """
        Apply regime classification rules.

        RISK_OFF if ANY of:
            - vol == 'HIGH' AND trend == 'DOWN'
            - breadth == 'WEAK' AND trend == 'DOWN'
            - sentiment == 'BEARISH' AND vol == 'HIGH'

        TRENDING if ALL of:
            - vol != 'HIGH'
            - trend in ('UP', 'NEUTRAL') AND trending is True
            - breadth != 'WEAK'

        Otherwise → RANGING.
        """
        # ── Risk-off checks (any one triggers) ──────────────────────────────
        if vol_signal == "HIGH" and trend_signal == "DOWN":
            return RISK_OFF

        if breadth_signal == "WEAK" and trend_signal == "DOWN":
            return RISK_OFF

        if sentiment_signal == "BEARISH" and vol_signal == "HIGH":
            return RISK_OFF

        # ── Trending checks (all must pass) ─────────────────────────────────
        if (
            vol_signal != "HIGH"
            and trend_signal in ("UP", "NEUTRAL")
            and trending is True
            and breadth_signal != "WEAK"
        ):
            return TRENDING

        # ── Default ─────────────────────────────────────────────────────────
        return RANGING

    # ── helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _get_spy(
        price_data: dict[str, pd.DataFrame],
    ) -> Optional[pd.DataFrame]:
        """Return the SPY DataFrame from *price_data*, or None."""
        spy = price_data.get(config.BENCHMARK)
        if spy is None:
            log.warning(
                "Benchmark {} not in price_data — regime signals degraded",
                config.BENCHMARK,
            )
        return spy

    @staticmethod
    def _compute_adx(df: pd.DataFrame, period: int = 14) -> Optional[float]:
        """
        Compute the latest ADX value for a single OHLCV DataFrame.

        Uses a manual Wilder-smoothed implementation to avoid the ``ta``
        library dependency (keeps core/ self-contained except for ``arch``).
        """
        if len(df) < period * 3:
            return None

        high = df["High"].values.astype(float)
        low = df["Low"].values.astype(float)
        close = df["Close"].values.astype(float)

        n = len(high)

        # True Range, +DM, -DM
        tr = np.empty(n)
        plus_dm = np.empty(n)
        minus_dm = np.empty(n)
        tr[0] = plus_dm[0] = minus_dm[0] = 0.0

        for i in range(1, n):
            h_l = high[i] - low[i]
            h_pc = abs(high[i] - close[i - 1])
            l_pc = abs(low[i] - close[i - 1])
            tr[i] = max(h_l, h_pc, l_pc)

            up_move = high[i] - high[i - 1]
            down_move = low[i - 1] - low[i]

            plus_dm[i] = up_move if (up_move > down_move and up_move > 0) else 0.0
            minus_dm[i] = down_move if (down_move > up_move and down_move > 0) else 0.0

        # Wilder smoothing
        def wilder_smooth(arr: np.ndarray, p: int) -> np.ndarray:
            out = np.empty_like(arr)
            out[:p] = np.nan
            out[p] = arr[1 : p + 1].sum()
            for i in range(p + 1, len(arr)):
                out[i] = out[i - 1] - out[i - 1] / p + arr[i]
            return out

        atr = wilder_smooth(tr, period)
        plus_di = 100.0 * wilder_smooth(plus_dm, period) / atr
        minus_di = 100.0 * wilder_smooth(minus_dm, period) / atr

        dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)

        # ADX = Wilder-smooth of DX
        adx = wilder_smooth(dx, period)

        last = adx[-1]
        return float(last) if not np.isnan(last) else None
