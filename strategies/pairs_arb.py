"""
strategies/pairs_arb.py — Statistical Pairs Arbitrage

Uses Engle-Granger cointegration test to find cointegrated pairs,
then trades mean-reversion of the spread.

  - Monthly recalculation of pairs via cointegration test
  - Entry: z-score of spread exceeds ±2.0
  - Exit: z-score reverts to ±0.5
  - Active only in RANGING regime
"""

from itertools import combinations
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

import config
from logger import log
from strategies import Signal
from core.regime_detector import Regime


class PairsArbStrategy:
    """Statistical pairs arbitrage using cointegration."""

    NAME = "PairsArb"

    def __init__(
        self,
        lookback: int = None,
        p_threshold: float = None,
        zscore_entry: float = None,
        zscore_exit: float = None,
        retest_days: int = None,
    ):
        self.lookback = lookback or config.COINT_LOOKBACK        # 252
        self.p_threshold = p_threshold or config.COINT_PVALUE    # 0.05
        self.zscore_entry = zscore_entry or config.ZSCORE_ENTRY  # 2.0
        self.zscore_exit = zscore_exit or config.ZSCORE_EXIT     # 0.5
        self.retest_days = retest_days or config.COINT_RETEST_DAYS  # 30

        # Cache for cointegrated pairs (recalculated monthly)
        self._pairs_cache: list[dict] = []
        self._last_test: datetime | None = None

    def generate_signals(
        self,
        data: dict[str, pd.DataFrame],
        regimes: dict[str, Regime],
    ) -> list[Signal]:
        """
        Generate pairs arbitrage signals.

        Only considers assets in RANGING regime.

        Args:
            data: Dict of symbol → OHLCV DataFrame
            regimes: Dict of symbol → Regime

        Returns:
            List of Signal objects (pairs of long/short signals)
        """
        # Filter to RANGING assets only
        ranging_symbols = [
            sym for sym, regime in regimes.items()
            if regime == Regime.RANGING and sym in data
        ]

        if len(ranging_symbols) < 2:
            log.debug("PairsArb: fewer than 2 RANGING assets, no signals")
            return []

        # Recalculate cointegrated pairs if needed (monthly)
        self._maybe_recalculate_pairs(data, ranging_symbols)

        # Generate signals from current pairs
        signals = []
        for pair in self._pairs_cache:
            sym_a = pair["sym_a"]
            sym_b = pair["sym_b"]

            if sym_a not in data or sym_b not in data:
                continue
            if sym_a not in ranging_symbols or sym_b not in ranging_symbols:
                continue

            pair_signals = self._compute_pair_signal(
                sym_a, sym_b, data[sym_a], data[sym_b], pair["hedge_ratio"]
            )
            signals.extend(pair_signals)

        log.info("PairsArb generated {} signals from {} pairs", len(signals), len(self._pairs_cache))
        return signals

    def _maybe_recalculate_pairs(
        self,
        data: dict[str, pd.DataFrame],
        symbols: list[str],
    ):
        """Re-run cointegration tests if enough time has passed."""
        now = datetime.now()

        if self._last_test and (now - self._last_test).days < self.retest_days:
            return

        log.info("PairsArb: recalculating cointegrated pairs ({} assets)", len(symbols))
        self._pairs_cache = self._find_cointegrated_pairs(data, symbols)
        self._last_test = now

    def _find_cointegrated_pairs(
        self,
        data: dict[str, pd.DataFrame],
        symbols: list[str],
    ) -> list[dict]:
        """
        Test all pairs for cointegration using Engle-Granger test.

        Returns list of cointegrated pairs with hedge ratios.
        """
        pairs = []

        for sym_a, sym_b in combinations(symbols, 2):
            df_a = data[sym_a]
            df_b = data[sym_b]

            price_col = "Adj Close" if "Adj Close" in df_a.columns else "Close"
            prices_a = df_a[price_col].dropna()
            prices_b = df_b[price_col].dropna()

            # Align on common dates
            common_idx = prices_a.index.intersection(prices_b.index)
            if len(common_idx) < self.lookback:
                continue

            # Use most recent lookback period
            prices_a = prices_a.loc[common_idx].iloc[-self.lookback:]
            prices_b = prices_b.loc[common_idx].iloc[-self.lookback:]

            try:
                # Engle-Granger cointegration test
                t_stat, p_value, crit_values = coint(prices_a, prices_b)

                if p_value < self.p_threshold:
                    # Calculate hedge ratio via OLS
                    hedge_ratio = np.polyfit(prices_b.values, prices_a.values, 1)[0]

                    pairs.append({
                        "sym_a": sym_a,
                        "sym_b": sym_b,
                        "p_value": p_value,
                        "t_stat": t_stat,
                        "hedge_ratio": hedge_ratio,
                    })

                    log.info(
                        "Cointegrated pair: {} / {} (p={:.4f}, hedge={:.4f})",
                        sym_a, sym_b, p_value, hedge_ratio,
                    )

            except Exception as e:
                log.debug("Coint test failed for {}/{}: {}", sym_a, sym_b, str(e))
                continue

        # Sort by p-value (strongest cointegration first)
        pairs.sort(key=lambda x: x["p_value"])

        log.info("Found {} cointegrated pairs", len(pairs))
        return pairs

    def _compute_pair_signal(
        self,
        sym_a: str,
        sym_b: str,
        df_a: pd.DataFrame,
        df_b: pd.DataFrame,
        hedge_ratio: float,
    ) -> list[Signal]:
        """
        Compute z-score of the spread and generate entry/exit signals.
        """
        price_col = "Adj Close" if "Adj Close" in df_a.columns else "Close"
        prices_a = df_a[price_col].dropna()
        prices_b = df_b[price_col].dropna()

        # Align
        common_idx = prices_a.index.intersection(prices_b.index)
        if len(common_idx) < 60:  # Need enough data for z-score
            return []

        prices_a = prices_a.loc[common_idx]
        prices_b = prices_b.loc[common_idx]

        # Calculate spread
        spread = prices_a - hedge_ratio * prices_b

        # Z-score (rolling 60-day mean and std)
        spread_mean = spread.rolling(60).mean()
        spread_std = spread.rolling(60).std()

        current_zscore = (spread.iloc[-1] - spread_mean.iloc[-1]) / spread_std.iloc[-1]

        if np.isnan(current_zscore):
            return []

        signals = []

        # Entry signals
        if current_zscore > self.zscore_entry:
            # Spread is too high → short A, long B (expect mean reversion)
            strength = min(abs(current_zscore) / 3.0, 1.0)
            signals.append(Signal(sym_a, "short", strength, self.NAME))
            signals.append(Signal(sym_b, "long", strength, self.NAME))
            log.debug("{}/{}: z={:.2f} → short {}, long {}", sym_a, sym_b, current_zscore, sym_a, sym_b)

        elif current_zscore < -self.zscore_entry:
            # Spread is too low → long A, short B
            strength = min(abs(current_zscore) / 3.0, 1.0)
            signals.append(Signal(sym_a, "long", strength, self.NAME))
            signals.append(Signal(sym_b, "short", strength, self.NAME))
            log.debug("{}/{}: z={:.2f} → long {}, short {}", sym_a, sym_b, current_zscore, sym_a, sym_b)

        return signals
