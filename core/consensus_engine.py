"""
core/consensus_engine.py — Signal Consensus Engine

Aggregates signals from all strategies into one final signal per asset.
Weights are dynamic — updated by the experience agent via the memory store.

Pipeline:
  1. Dual Momentum Gate  (hard filter)
  2. Regime Routing       (select active strategies)
  3. Weighted Combination (blend signals)
  4. Signal Filtering     (noise removal + clipping)
  5. Decision Logging     (metadata for experience agent)
"""

from __future__ import annotations

import copy
from datetime import datetime

from logger import log
from core.regime_detector import Regime


# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────
MIN_WEIGHT = 0.10          # Floor per strategy weight
MAX_WEIGHT = 0.60          # Cap per strategy weight
NOISE_THRESHOLD = 0.1      # Drop signals weaker than this
MIN_TRADES_BEFORE_UPDATE = 20  # Minimum trades before weight update

DEFAULT_WEIGHTS = {
    "tsmom": 0.40,
    "vol_trend": 0.35,
    "pairs_arb": 0.25,
}


class ConsensusEngine:
    """
    Blend strategy signals into a single final signal per asset.

    Weights are persisted via a plain dict (`memory_store`) so the
    module stays decoupled and testable — the caller is responsible
    for providing and optionally persisting the dict to disk.
    """

    def __init__(self, config, memory_store: dict):
        """
        Load current weights from memory store.

        Args:
            config:       Master config module (used for future params).
            memory_store: Shared dict for weight persistence.
                          Key ``"consensus_weights"`` holds the weights
                          dict; absent key → use defaults.
        """
        self.config = config
        self.memory_store = memory_store

        # Load or initialise weights
        raw_weights = memory_store.get("consensus_weights")
        if raw_weights and isinstance(raw_weights, dict):
            self.weights = dict(raw_weights)
        else:
            self.weights = dict(DEFAULT_WEIGHTS)

        # Clamp then normalise
        self._clamp_weights()
        self._normalise_weights()

        log.info(
            "ConsensusEngine initialised — weights: tsmom={:.2f} vol_trend={:.2f} pairs_arb={:.2f}",
            self.weights["tsmom"],
            self.weights["vol_trend"],
            self.weights["pairs_arb"],
        )

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    def aggregate(
        self,
        tsmom_signals: dict[str, float],
        vol_trend_signals: dict[str, float],
        pairs_signals: dict[str, float],
        dual_mom_filter: dict[str, float],
        regime: dict[str, Regime],
    ) -> dict[str, float]:
        """
        Produce one final signal per symbol.

        Args:
            tsmom_signals:      symbol → signal strength (signed float)
            vol_trend_signals:  symbol → signal strength (signed float)
            pairs_signals:      symbol → signal strength (signed float)
            dual_mom_filter:    symbol → 0 or 1 (hard gate)
            regime:             symbol → Regime enum value

        Returns:
            dict[symbol → final_signal] clipped to [-1.0, 1.0]
        """
        # Gather the full symbol universe from all inputs
        all_symbols: set[str] = set()
        all_symbols.update(tsmom_signals)
        all_symbols.update(vol_trend_signals)
        all_symbols.update(pairs_signals)

        final_signals: dict[str, float] = {}
        decisions: list[dict] = []

        for symbol in sorted(all_symbols):
            tsmom_val = tsmom_signals.get(symbol, 0.0)
            vol_trend_val = vol_trend_signals.get(symbol, 0.0)
            pairs_val = pairs_signals.get(symbol, 0.0)

            # ── 1. DUAL MOMENTUM GATE ────────────────────────────
            dm_gate = dual_mom_filter.get(symbol, 0)
            if dm_gate == 0:
                # Hard gate — zero out everything for this symbol
                final_signals[symbol] = 0.0
                decisions.append(self._make_decision(
                    symbol, 0.0, tsmom_val, vol_trend_val, pairs_val,
                    gate="BLOCKED", regime_label="N/A",
                ))
                continue

            # ── 2. REGIME ROUTING ────────────────────────────────
            sym_regime = regime.get(symbol, Regime.RANGING)

            if sym_regime == Regime.RISK_OFF:
                # No trades in risk-off
                final_signals[symbol] = 0.0
                decisions.append(self._make_decision(
                    symbol, 0.0, tsmom_val, vol_trend_val, pairs_val,
                    gate="PASS", regime_label="RISK_OFF",
                ))
                continue

            if sym_regime == Regime.TRENDING:
                # Use tsmom + vol_trend, zero pairs
                combined = self._weighted_trending(tsmom_val, vol_trend_val)
                regime_label = "TRENDING"
            else:
                # RANGING — use pairs only (full weight)
                combined = pairs_val
                regime_label = "RANGING"

            # ── 4. SIGNAL FILTERING ──────────────────────────────
            if abs(combined) < NOISE_THRESHOLD:
                combined = 0.0

            combined = max(-1.0, min(1.0, combined))

            final_signals[symbol] = combined

            # ── 5. LOG DECISION METADATA ─────────────────────────
            decisions.append(self._make_decision(
                symbol, combined, tsmom_val, vol_trend_val, pairs_val,
                gate="PASS", regime_label=regime_label,
            ))

        # Persist decision metadata for experience agent
        self.memory_store["consensus_decisions"] = decisions

        # Summary log
        active = {s: v for s, v in final_signals.items() if v != 0.0}
        log.info(
            "ConsensusEngine: {} symbols processed, {} active signals",
            len(final_signals), len(active),
        )

        return final_signals

    def update_weights(self, new_weights: dict[str, float]) -> None:
        """
        Replace strategy weights (called by the experience agent).

        Validates:
          - All three strategy keys present
          - Each weight within [MIN_WEIGHT, MAX_WEIGHT]
          - Weights sum to ~1.0 (after normalisation)

        Args:
            new_weights: ``{"tsmom": …, "vol_trend": …, "pairs_arb": …}``

        Raises:
            ValueError: if validation fails
        """
        required_keys = {"tsmom", "vol_trend", "pairs_arb"}
        if not required_keys.issubset(new_weights):
            raise ValueError(
                f"Missing weight keys: {required_keys - set(new_weights)}"
            )

        for key in required_keys:
            w = new_weights[key]
            if w < MIN_WEIGHT or w > MAX_WEIGHT:
                raise ValueError(
                    f"Weight '{key}'={w:.4f} outside [{MIN_WEIGHT}, {MAX_WEIGHT}]"
                )

        self.weights = {k: new_weights[k] for k in required_keys}
        self._normalise_weights()

        # Persist
        self.memory_store["consensus_weights"] = dict(self.weights)

        log.info(
            "ConsensusEngine weights updated — tsmom={:.2f} vol_trend={:.2f} pairs_arb={:.2f}",
            self.weights["tsmom"],
            self.weights["vol_trend"],
            self.weights["pairs_arb"],
        )

    # ──────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────

    def _weighted_trending(self, tsmom: float, vol_trend: float) -> float:
        """
        Weighted blend for TRENDING regime.

        Re-normalises tsmom and vol_trend weights to sum to 1.0
        within the active strategy set.
        """
        w_tsmom = self.weights["tsmom"]
        w_vol = self.weights["vol_trend"]

        total = w_tsmom + w_vol
        if total <= 0:
            return 0.0

        # Normalise within active set
        w_tsmom_norm = w_tsmom / total
        w_vol_norm = w_vol / total

        return tsmom * w_tsmom_norm + vol_trend * w_vol_norm

    def _clamp_weights(self) -> None:
        """Clamp each weight to [MIN_WEIGHT, MAX_WEIGHT]."""
        for key in list(self.weights):
            self.weights[key] = max(MIN_WEIGHT, min(MAX_WEIGHT, self.weights[key]))

    def _normalise_weights(self) -> None:
        """Normalise weights to sum to 1.0."""
        total = sum(self.weights.values())
        if total <= 0:
            # Fallback to defaults
            self.weights = dict(DEFAULT_WEIGHTS)
            total = sum(self.weights.values())

        for key in self.weights:
            self.weights[key] /= total

    @staticmethod
    def _make_decision(
        symbol: str,
        final: float,
        tsmom: float,
        vol_trend: float,
        pairs: float,
        gate: str,
        regime_label: str,
    ) -> dict:
        """Build a decision metadata dict for experience-agent analysis."""
        return {
            "symbol": symbol,
            "final_signal": round(final, 6),
            "tsmom_signal": round(tsmom, 6),
            "vol_trend_signal": round(vol_trend, 6),
            "pairs_signal": round(pairs, 6),
            "dual_mom_gate": gate,
            "regime": regime_label,
            "timestamp": datetime.now().isoformat(),
        }
