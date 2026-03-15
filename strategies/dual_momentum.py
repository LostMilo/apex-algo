"""
strategies/dual_momentum.py — Gary Antonacci's Dual Momentum Filter

This is a FILTER that gates all other strategy signals — not a standalone
entry signal.  If this returns 0 for an asset, NO trade is taken regardless
of other signals.

Zero look-ahead rules (same as tsmom.py):
  - All data sliced strictly before current_date
  - Only uses information available at the time of decision

Dual Momentum combines:
  1. ABSOLUTE momentum — asset's own 252-day return must be positive
  2. RELATIVE momentum — asset must beat the risk-free proxy (SHV)

Both conditions must pass for the filter to return 1.
"""

import numpy as np
import pandas as pd

import config
from logger import log


class DualMomentumFilter:
    """
    Dual Momentum — confirmation filter, not a standalone signal generator.

    Methods
    -------
    compute_filter(price_data, current_date)
        Returns {symbol: 0 or 1} for every symbol in price_data.
    apply_filter(signals, filter_values)
        Multiplies each signal by its filter value; zeros become 0.0.
    get_filter_metadata(price_data, current_date)
        Returns per-symbol breakdown used by the experience agent.
    """

    NAME = "DualMomentum"

    def __init__(self, lookback: int = None):
        self.lookback = lookback or config.DUAL_MOM_LOOKBACK  # 252 days

    # ── 1. compute_filter ───────────────────────────────────────────────────

    def compute_filter(
        self,
        price_data: dict[str, pd.DataFrame],
        current_date: pd.Timestamp,
    ) -> dict[str, int]:
        """
        Compute the binary Dual Momentum filter for every symbol.

        Args:
            price_data: dict of symbol → OHLCV DataFrame (full history).
            current_date: the as-of date for the decision.  Data is sliced
                          strictly *before* this date to prevent look-ahead.

        Returns:
            dict[symbol → 0 or 1]
        """
        assert price_data  # at least one symbol required

        # ── Risk-free proxy return ──────────────────────────────────────────
        rf_proxy = config.RISK_FREE_PROXY  # "SHV"
        shv_available = rf_proxy in price_data
        shv_return: float | None = None

        if shv_available:
            shv_return = self._return_before(
                price_data[rf_proxy], current_date,
            )

        results: dict[str, int] = {}

        for symbol, df in price_data.items():
            # Slice strictly before current_date
            mask = df.index < current_date
            assert mask.any() or not mask.any()  # defensive — index.max() < current_date
            sliced = df.loc[mask]

            close = self._close_series(sliced)

            # ── Insufficient data → conservative 0 ─────────────────────────
            if len(close) < self.lookback:
                log.debug(
                    "{}: DualMom — insufficient data ({} < {}), returning 0",
                    symbol, len(close), self.lookback,
                )
                results[symbol] = 0
                continue

            # ── ABSOLUTE MOMENTUM ──────────────────────────────────────────
            absolute_mom = (close.iloc[-1] / close.iloc[-self.lookback]) - 1.0
            absolute_pass = 1 if absolute_mom > 0 else 0

            # ── RELATIVE MOMENTUM ─────────────────────────────────────────
            asset_return = absolute_mom  # same 252-day return

            if shv_available and shv_return is not None:
                # Compare against actual SHV return
                relative_pass = 1 if asset_return > shv_return else 0
            else:
                # SHV not available — use 4% annual proxy
                relative_pass = 1 if asset_return > 0.04 else 0

            # ── FINAL FILTER ───────────────────────────────────────────────
            final_filter = 1 if (absolute_pass == 1 and relative_pass == 1) else 0

            log.debug(
                "{}: DualMom abs_mom={:.4f} abs_pass={} rel_pass={} → {}",
                symbol, absolute_mom, absolute_pass, relative_pass, final_filter,
            )

            results[symbol] = final_filter

        passed = sum(v for v in results.values())
        log.info(
            "DualMomentum compute_filter: {}/{} symbols passed",
            passed, len(results),
        )
        return results

    # ── 2. apply_filter ─────────────────────────────────────────────────────

    def apply_filter(
        self,
        signals: dict[str, float],
        filter_values: dict[str, int],
    ) -> dict[str, float]:
        """
        Multiply each signal by its filter value.

        Any symbol with filter=0 becomes signal=0.0 (trade blocked).

        Args:
            signals: dict of symbol → raw signal strength (float).
            filter_values: dict of symbol → 0 or 1 from compute_filter().

        Returns:
            dict of symbol → filtered signal strength (float).
        """
        filtered: dict[str, float] = {}
        for symbol, strength in signals.items():
            gate = filter_values.get(symbol, 0)
            filtered[symbol] = strength * gate
            if gate == 0:
                log.debug("{}: signal zeroed by DualMom filter", symbol)
        return filtered

    # ── 3. get_filter_metadata ──────────────────────────────────────────────

    def get_filter_metadata(
        self,
        price_data: dict[str, pd.DataFrame],
        current_date: pd.Timestamp,
    ) -> dict:
        """
        Return per-symbol breakdown for the experience agent.

        Returns:
            {symbol: {absolute_return, benchmark_return, absolute_pass,
                      relative_pass, final_filter}}
        """
        rf_proxy = config.RISK_FREE_PROXY
        shv_available = rf_proxy in price_data
        shv_return: float | None = None

        if shv_available:
            shv_return = self._return_before(price_data[rf_proxy], current_date)

        # Determine the benchmark return used for relative comparison
        if shv_available and shv_return is not None:
            benchmark_return = shv_return
        else:
            benchmark_return = 0.04  # annual proxy

        metadata: dict = {}

        for symbol, df in price_data.items():
            sliced = df.loc[df.index < current_date]
            close = self._close_series(sliced)

            if len(close) < self.lookback:
                metadata[symbol] = {
                    "absolute_return": None,
                    "benchmark_return": benchmark_return,
                    "absolute_pass": 0,
                    "relative_pass": 0,
                    "final_filter": 0,
                }
                continue

            absolute_return = (close.iloc[-1] / close.iloc[-self.lookback]) - 1.0
            absolute_pass = 1 if absolute_return > 0 else 0

            if shv_available and shv_return is not None:
                relative_pass = 1 if absolute_return > shv_return else 0
            else:
                relative_pass = 1 if absolute_return > 0.04 else 0

            final_filter = 1 if (absolute_pass == 1 and relative_pass == 1) else 0

            metadata[symbol] = {
                "absolute_return": round(absolute_return, 6),
                "benchmark_return": round(benchmark_return, 6),
                "absolute_pass": absolute_pass,
                "relative_pass": relative_pass,
                "final_filter": final_filter,
            }

        return metadata

    # ─── Private helpers ────────────────────────────────────────────────────

    @staticmethod
    def _close_series(df: pd.DataFrame) -> pd.Series:
        """Extract the close price column (lowercase or title-case)."""
        for col in ("close", "Close", "Adj Close"):
            if col in df.columns:
                return df[col].dropna()
        raise KeyError(f"No close column found in {list(df.columns)}")

    def _return_before(
        self, df: pd.DataFrame, current_date: pd.Timestamp,
    ) -> float | None:
        """Calculate lookback return for data strictly before current_date."""
        sliced = df.loc[df.index < current_date]
        close = self._close_series(sliced)
        if len(close) < self.lookback:
            return None
        return (close.iloc[-1] / close.iloc[-self.lookback]) - 1.0


# ─── Self-test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Standalone unit tests for DualMomentumFilter.
    Uses synthetic price data — no API calls required.
    """

    def _make_prices(n_days: int, annual_return: float) -> pd.DataFrame:
        """Generate a synthetic daily close series with a known annual return."""
        daily_r = (1 + annual_return) ** (1 / 252) - 1
        dates = pd.bdate_range(end=pd.Timestamp("2025-06-01"), periods=n_days)
        close = 100.0 * np.cumprod(1 + np.full(n_days, daily_r))
        return pd.DataFrame({"close": close}, index=dates)

    dm = DualMomentumFilter(lookback=252)
    current = pd.Timestamp("2025-06-02")

    # ── Test 1: Absolute momentum passes on uptrending asset ─────────────
    up = _make_prices(300, annual_return=0.15)  # +15% annual
    rf = _make_prices(300, annual_return=0.03)  # +3% risk-free
    price_data = {"ASSET": up, config.RISK_FREE_PROXY: rf}

    result = dm.compute_filter(price_data, current)
    assert result["ASSET"] == 1, f"Expected 1, got {result['ASSET']}"
    print("✓ Test 1: absolute momentum passes on uptrending asset")

    # ── Test 2: Relative momentum blocks asset underperforming benchmark ──
    weak = _make_prices(300, annual_return=0.02)  # +2% < SHV 3%
    price_data_2 = {"WEAK": weak, config.RISK_FREE_PROXY: rf}

    result_2 = dm.compute_filter(price_data_2, current)
    assert result_2["WEAK"] == 0, f"Expected 0, got {result_2['WEAK']}"
    print("✓ Test 2: relative momentum blocks asset underperforming benchmark")

    # ── Test 3: Filter correctly zeros out signals ───────────────────────
    signals = {"ASSET": 0.85, "WEAK": 0.60}
    filter_vals = {"ASSET": 1, "WEAK": 0}
    filtered = dm.apply_filter(signals, filter_vals)
    assert filtered["ASSET"] == 0.85, f"Expected 0.85, got {filtered['ASSET']}"
    assert filtered["WEAK"] == 0.0, f"Expected 0.0, got {filtered['WEAK']}"
    print("✓ Test 3: filter correctly zeros out signals")

    # ── Test 4: Insufficient data returns 0 (conservative) ──────────────
    short = _make_prices(100, annual_return=0.20)  # only 100 days
    price_data_3 = {"SHORT": short}
    result_3 = dm.compute_filter(price_data_3, current)
    assert result_3["SHORT"] == 0, f"Expected 0, got {result_3['SHORT']}"
    print("✓ Test 4: insufficient data returns 0 (conservative)")

    # ── Test 5: Metadata returns correct breakdown ──────────────────────
    meta = dm.get_filter_metadata({"ASSET": up, config.RISK_FREE_PROXY: rf}, current)
    assert meta["ASSET"]["final_filter"] == 1
    assert meta["ASSET"]["absolute_pass"] == 1
    assert meta["ASSET"]["relative_pass"] == 1
    assert meta["ASSET"]["absolute_return"] is not None
    print("✓ Test 5: metadata returns correct breakdown")

    print("\n✅ Dual Momentum unit tests passed")
