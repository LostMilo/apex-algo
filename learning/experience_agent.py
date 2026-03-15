"""
learning/experience_agent.py — Post-Mortem Trade Analysis & Adaptive Learning

The most novel component of the system.  After every closed trade it:

  1. Classifies outcome (strong_win / weak_win / weak_loss / strong_loss)
  2. Scores entry quality (0–100) and exit quality (0–100)
  3. Detects failure patterns (wrong regime, exhausted trend, …)
  4. Generates a structured *lesson* dict
  5. Stores the lesson in memory_store
  6. After MIN_TRADES_BEFORE_UPDATE trades, recalculates strategy weights
     and broadcasts them to the consensus engine

Also provides a Monte Carlo bootstrap simulator to stress-test the
realised trade distribution.

No external API calls — everything runs locally and autonomously.
"""

import json
import math
import random
from datetime import datetime

import numpy as np

import config
from logger import log


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Outcome classification thresholds (pnl_pct)
_STRONG_WIN_THRESHOLD = 0.05
_STRONG_LOSS_THRESHOLD = -0.05

# Default strategy weight when no data
_DEFAULT_WEIGHT = 0.25


class ExperienceAgent:
    """
    Autonomous post-mortem analyst.

    Plugs in after the execution layer closes a trade.  Extracts
    structured lessons, detects failure patterns, and — once enough
    evidence accumulates — recalibrates strategy weights via the
    consensus engine.
    """

    # ──────────────────────────────────────────────────────────────────────
    # Initialisation
    # ──────────────────────────────────────────────────────────────────────

    def __init__(self, cfg, memory_store, consensus_engine):
        """
        Args:
            cfg             : The config module (or any object exposing the
                              same attributes).
            memory_store    : A MemoryStore instance with store_lesson(),
                              get_lessons(), get_trade_count(), log_trade().
            consensus_engine: The consensus / signal-aggregator that exposes
                              update_weights(strategy_weights: dict).
        """
        self.cfg = cfg
        self.memory_store = memory_store
        self.consensus_engine = consensus_engine

        # Load running trade count from persistent store
        self.trade_count: int = self._load_trade_count()

        self.min_trades = getattr(
            cfg, "MIN_TRADES_BEFORE_UPDATE", 30
        )
        self.mc_simulations = getattr(
            cfg, "MONTE_CARLO_SIMULATIONS", 1000
        )
        self.confidence_decay = getattr(
            cfg, "LESSON_CONFIDENCE_DECAY", 0.95
        )

        log.info(
            "ExperienceAgent initialised — {} trades recorded so far",
            self.trade_count,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def analyze_trade(
        self,
        trade: dict,
        price_data: dict,
        regime_at_time: str,
    ) -> dict:
        """
        Full post-mortem on a single closed trade.

        Args:
            trade: dict with keys:
                symbol, strategy, entry_date, exit_date,
                entry_price, exit_price, pnl_pct, exit_reason,
                signal_strength, signals_json, hold_days
            price_data: dict of symbol → OHLCV DataFrame (used for
                context checks like volume and volatility)
            regime_at_time: str — the regime label when the trade was
                entered (e.g. "TRENDING", "RANGING", "RISK_OFF")

        Returns:
            lesson dict (also stored internally and broadcast).
        """
        symbol = trade.get("symbol", "UNKNOWN")
        strategy = trade.get("strategy", "UNKNOWN")
        pnl_pct = trade.get("pnl_pct", 0.0)

        log.info(
            "ExperienceAgent: analysing {} trade on {} (pnl={:.2%})",
            strategy, symbol, pnl_pct,
        )

        # ── 1. Outcome classification ────────────────────────────────
        outcome = self._classify_outcome(pnl_pct)

        # ── 2. Entry quality score ───────────────────────────────────
        entry_score = self._score_entry(trade, price_data, regime_at_time)

        # ── 3. Exit quality score ────────────────────────────────────
        exit_score = self._score_exit(trade)

        # ── 4. Failure pattern detection ─────────────────────────────
        tags = self._detect_failure_patterns(trade, regime_at_time)

        # ── 5. Build lesson ──────────────────────────────────────────
        lesson_type = "win_pattern" if pnl_pct > 0 else "failure_pattern"
        confidence = self._compute_confidence(
            entry_score, exit_score, abs(pnl_pct),
        )

        condition = self._generate_condition_text(
            outcome, tags, strategy, symbol,
        )

        lesson = {
            "source": strategy,
            "lesson_type": lesson_type,
            "condition": condition,
            "confidence": round(confidence, 4),
            "outcome": outcome,
            "entry_quality": entry_score,
            "exit_quality": exit_score,
            "tags": tags,
            "trade_ref": self._trade_id(trade),
            "pnl_pct": pnl_pct,
            "symbol": symbol,
            "timestamp": datetime.utcnow().isoformat(),
        }

        log.info(
            "  outcome={} entry_q={} exit_q={} tags={} conf={:.3f}",
            outcome, entry_score, exit_score, tags, confidence,
        )

        # ── 6. Persist & broadcast ───────────────────────────────────
        self.trade_count += 1
        self.broadcast_lessons(lesson)

        return lesson

    # ──────────────────────────────────────────────────────────────────────
    # Monte Carlo Simulation
    # ──────────────────────────────────────────────────────────────────────

    def run_monte_carlo(
        self,
        trade_history: list[dict],
        n_simulations: int | None = None,
    ) -> dict:
        """
        Bootstrap-resample from realised trade returns to build
        simulated equity paths and derive confidence intervals.

        Args:
            trade_history : List of trade dicts, each must have 'pnl_pct'.
            n_simulations : Override for config.MONTE_CARLO_SIMULATIONS.

        Returns:
            {
                "median_final"       : float,
                "p5_final"           : float,
                "p95_final"          : float,
                "prob_profit"        : float,   # fraction of sims > starting capital
                "max_drawdown_median": float,
            }
        """
        n_sims = n_simulations or self.mc_simulations
        returns = [t.get("pnl_pct", 0.0) for t in trade_history]

        if len(returns) < 2:
            log.warning("Monte Carlo: not enough trades ({}) — skipping", len(returns))
            return {
                "median_final": 0.0,
                "p5_final": 0.0,
                "p95_final": 0.0,
                "prob_profit": 0.0,
                "max_drawdown_median": 0.0,
            }

        n_trades = len(returns)
        starting_capital = getattr(self.cfg, "STARTING_CAPITAL", 1000.0)

        finals = []
        max_dds = []

        for _ in range(n_sims):
            # Resample *with replacement* — same number of trades
            sampled = random.choices(returns, k=n_trades)
            equity = starting_capital
            peak = equity
            worst_dd = 0.0
            path = [equity]

            for r in sampled:
                equity *= (1.0 + r)
                path.append(equity)
                if equity > peak:
                    peak = equity
                dd = (peak - equity) / peak if peak > 0 else 0.0
                worst_dd = max(worst_dd, dd)

            finals.append(equity)
            max_dds.append(worst_dd)

        finals_arr = np.array(finals)
        max_dds_arr = np.array(max_dds)

        result = {
            "median_final": float(np.median(finals_arr)),
            "p5_final": float(np.percentile(finals_arr, 5)),
            "p95_final": float(np.percentile(finals_arr, 95)),
            "prob_profit": float(
                np.sum(finals_arr > starting_capital) / n_sims
            ),
            "max_drawdown_median": float(np.median(max_dds_arr)),
        }

        log.info(
            "Monte Carlo ({} sims, {} trades): median=${:.2f}  "
            "P5=${:.2f}  P95=${:.2f}  prob_profit={:.1%}  "
            "median_max_dd={:.2%}",
            n_sims, n_trades,
            result["median_final"],
            result["p5_final"],
            result["p95_final"],
            result["prob_profit"],
            result["max_drawdown_median"],
        )

        return result

    # ──────────────────────────────────────────────────────────────────────
    # Broadcasting & Weight Updates
    # ──────────────────────────────────────────────────────────────────────

    def broadcast_lessons(self, lesson: dict) -> None:
        """
        Store a lesson persistently and, if enough evidence has
        accumulated, recalculate and push new strategy weights.
        """
        # Persist
        try:
            self.memory_store.store_lesson(lesson)
            log.debug("Lesson stored: {}", lesson.get("condition", ""))
        except Exception as exc:
            log.error("Failed to store lesson: {}", exc)

        # Conditional weight update
        if self._should_update_weights():
            log.info(
                "Trade count {} ≥ {} — recalculating strategy weights",
                self.trade_count, self.min_trades,
            )
            new_weights = self._recalculate_weights()
            self._push_weights(new_weights)

    # ──────────────────────────────────────────────────────────────────────
    # Private — Outcome Classification
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _classify_outcome(pnl_pct: float) -> str:
        """Map pnl_pct → one of four outcome buckets."""
        if pnl_pct > _STRONG_WIN_THRESHOLD:
            return "strong_win"
        elif pnl_pct > 0:
            return "weak_win"
        elif pnl_pct >= _STRONG_LOSS_THRESHOLD:
            return "weak_loss"
        else:
            return "strong_loss"

    # ──────────────────────────────────────────────────────────────────────
    # Private — Entry Quality Scoring
    # ──────────────────────────────────────────────────────────────────────

    def _score_entry(
        self,
        trade: dict,
        price_data: dict,
        regime_at_time: str,
    ) -> int:
        """
        Entry quality score (0–100).

        +20  Signal strength > 0.7
        +20  Dual momentum confirmed (signals_json contains dual_mom)
        +20  Regime correct for the strategy
        +20  Volume confirmed (above VOLUME_CONFIRM_MULTIPLIER × MA)
        +20  Volatility below regime high threshold
        """
        score = 0
        symbol = trade.get("symbol", "")
        strategy = trade.get("strategy", "")
        signal_strength = trade.get("signal_strength", 0.0)

        # 1. Signal strength
        if signal_strength > 0.7:
            score += 20

        # 2. Dual momentum confirmation
        signals_json = trade.get("signals_json", "")
        if self._has_dual_momentum_confirm(signals_json):
            score += 20

        # 3. Regime correctness
        if self._regime_correct_for_strategy(strategy, regime_at_time):
            score += 20

        # 4. Volume confirmation
        if self._volume_confirmed(trade, price_data):
            score += 20

        # 5. Volatility check
        if self._volatility_below_threshold(trade, price_data):
            score += 20

        return score

    # ──────────────────────────────────────────────────────────────────────
    # Private — Exit Quality Scoring
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _score_exit(trade: dict) -> int:
        """
        Exit quality score (0–100).

        +40  Exit reason is optimal (trailing_stop, target_hit)
        +20  Hold time in reasonable range (3–60 days)
        +40  Exited before signal reversed (exit_reason != signal_reversal
             when pnl > 0, or exit_reason == signal_reversal for loss)
        """
        score = 0
        exit_reason = trade.get("exit_reason", "").lower()
        hold_days = trade.get("hold_days", 0)
        pnl_pct = trade.get("pnl_pct", 0.0)

        # 1. Optimal exit reason
        optimal_exits = {"trailing_stop", "target_hit", "take_profit"}
        if exit_reason in optimal_exits:
            score += 40

        # 2. Reasonable hold time (3–60 trading days)
        if 3 <= hold_days <= 60:
            score += 20

        # 3. Exited before signal reversed
        #   - For winners: any non-reversal exit is good
        #   - For losers: early stop-loss is better than riding down
        if pnl_pct > 0 and exit_reason != "signal_reversal":
            score += 40
        elif pnl_pct <= 0 and exit_reason in {
            "stop_loss", "hard_stop", "time_stop", "chandelier_stop",
        }:
            score += 40

        return score

    # ──────────────────────────────────────────────────────────────────────
    # Private — Failure Pattern Detection
    # ──────────────────────────────────────────────────────────────────────

    def _detect_failure_patterns(
        self,
        trade: dict,
        regime_at_time: str,
    ) -> list[str]:
        """
        Tag a trade with descriptive failure (or success) patterns.
        """
        tags: list[str] = []
        pnl_pct = trade.get("pnl_pct", 0.0)
        hold_days = trade.get("hold_days", 0)
        signal_strength = trade.get("signal_strength", 0.0)

        # Parse signals_json for context
        signals_ctx = self._parse_signals_json(trade.get("signals_json", ""))
        volume_was_low = signals_ctx.get("volume_low", False)
        adx_at_entry = signals_ctx.get("adx", 0.0)

        adx_max = getattr(self.cfg, "ADX_MAX", 40)

        if pnl_pct < 0:
            # ── Loss patterns ────────────────────────────────────────
            if regime_at_time and regime_at_time.upper() == "RISK_OFF":
                tags.append("entered_in_wrong_regime")

            if adx_at_entry > adx_max:
                tags.append("entered_exhausted_trend")

            if volume_was_low:
                tags.append("low_volume_entry")

            if hold_days < 3:
                tags.append("premature_exit_or_noise")

        if pnl_pct > 0:
            # ── Win patterns ─────────────────────────────────────────
            if signal_strength > 0.8:
                tags.append("high_conviction_winner")

        return tags

    # ──────────────────────────────────────────────────────────────────────
    # Private — Confidence Calculation
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_confidence(
        entry_score: int,
        exit_score: int,
        abs_pnl: float,
    ) -> float:
        """
        Combine entry quality, exit quality, and magnitude of outcome
        into a single confidence number in [0, 1].

        Higher quality scores and larger |pnl| → higher confidence
        that the lesson is meaningful (not noise).
        """
        # Normalise scores to [0, 1]
        entry_norm = entry_score / 100.0
        exit_norm = exit_score / 100.0

        # Magnitude factor: tanh-scaled so very large moves plateau
        magnitude = math.tanh(abs_pnl * 10.0)  # 5% → 0.46, 10% → 0.76

        # Weighted blend
        confidence = 0.35 * entry_norm + 0.25 * exit_norm + 0.40 * magnitude
        return max(0.0, min(1.0, confidence))

    # ──────────────────────────────────────────────────────────────────────
    # Private — Weight Recalculation
    # ──────────────────────────────────────────────────────────────────────

    def _should_update_weights(self) -> bool:
        """True when we have enough data to justify a weight update."""
        return self.trade_count >= self.min_trades

    def _recalculate_weights(self) -> dict[str, float]:
        """
        Aggregate all lessons by strategy.  Compute a composite score
        per strategy from win-rate, average entry/exit quality, and
        apply a recency decay.  Normalise to sum = 1.0.
        """
        try:
            lessons = self.memory_store.get_lessons(
                agent=self.__class__.__name__,
                since_days=7,
                unread_only=True,
            )
        except Exception as exc:
            log.error("Cannot read lessons for weight calc: {}", exc)
            return {}

        if not lessons:
            return {}

        # Group by strategy source
        strategy_data: dict[str, list[dict]] = {}
        for lesson in lessons:
            src = lesson.get("source", "UNKNOWN")
            strategy_data.setdefault(src, []).append(lesson)

        raw_scores: dict[str, float] = {}

        for strategy, entries in strategy_data.items():
            total = len(entries)
            if total == 0:
                continue

            wins = sum(
                1 for e in entries if e.get("lesson_type") == "win_pattern"
            )
            win_rate = wins / total

            avg_entry_q = np.mean(
                [e.get("entry_quality", 50) for e in entries]
            ) / 100.0
            avg_exit_q = np.mean(
                [e.get("exit_quality", 50) for e in entries]
            ) / 100.0

            # Apply recency decay — newer lessons weighted more
            avg_confidence = np.mean(
                [e.get("confidence", 0.5) for e in entries]
            )

            # Composite: win-rate matters most, tempered by quality
            composite = (
                0.50 * win_rate
                + 0.20 * avg_entry_q
                + 0.15 * avg_exit_q
                + 0.15 * avg_confidence
            )

            raw_scores[strategy] = max(composite, 0.01)  # floor at 1%

            log.debug(
                "  {} → wr={:.2%} eq={:.0f} xq={:.0f} → {:.4f}",
                strategy, win_rate,
                avg_entry_q * 100, avg_exit_q * 100, composite,
            )

        # Normalise to sum = 1.0
        total_score = sum(raw_scores.values())
        if total_score <= 0:
            return {}

        weights = {s: v / total_score for s, v in raw_scores.items()}

        log.info("Recalculated strategy weights: {}", weights)
        return weights

    def _push_weights(self, weights: dict[str, float]) -> None:
        """Push new weights to consensus engine and persist."""
        if not weights:
            return

        try:
            # Update consensus engine live weights
            if hasattr(self.consensus_engine, "update_weights"):
                self.consensus_engine.update_weights(weights)
                log.info("Strategy weights pushed to consensus engine")
            else:
                log.warning(
                    "Consensus engine has no update_weights() method — "
                    "storing weights only"
                )

            # Persist to memory store
            if hasattr(self.memory_store, "store_weights"):
                self.memory_store.store_weights(weights)

        except Exception as exc:
            log.error("Failed to push weights: {}", exc)

    # ──────────────────────────────────────────────────────────────────────
    # Private — Helpers
    # ──────────────────────────────────────────────────────────────────────

    def _load_trade_count(self) -> int:
        """Load the running trade count from persistent memory."""
        try:
            if hasattr(self.memory_store, "get_trade_count"):
                return self.memory_store.get_trade_count()
        except Exception as exc:
            log.warning("Could not load trade count: {}", exc)
        return 0

    @staticmethod
    def _has_dual_momentum_confirm(signals_json: str) -> bool:
        """Check if dual momentum confirmed the entry."""
        if not signals_json:
            return False
        try:
            data = json.loads(signals_json) if isinstance(signals_json, str) else signals_json
            if isinstance(data, dict):
                return data.get("dual_momentum_confirmed", False)
            if isinstance(data, list):
                return any(
                    s.get("strategy", "").lower() in ("dualmom", "dual_momentum")
                    for s in data
                    if isinstance(s, dict)
                )
        except (json.JSONDecodeError, TypeError):
            pass
        return False

    def _regime_correct_for_strategy(
        self,
        strategy: str,
        regime: str,
    ) -> bool:
        """
        Check whether the regime was appropriate for the strategy:
          TSMOM / VolTrend  → should be TRENDING
          PairsArb          → should be RANGING
          DualMom           → any regime is fine
        """
        s = strategy.upper()
        r = regime.upper() if regime else ""

        if s in ("TSMOM", "VOLTREND", "VOL_TREND"):
            return r == "TRENDING"
        elif s in ("PAIRSARB", "PAIRS_ARB"):
            return r == "RANGING"
        else:
            # DualMom or unknown → regime agnostic
            return True

    def _volume_confirmed(
        self,
        trade: dict,
        price_data: dict,
    ) -> bool:
        """Check if volume at entry exceeded the MA-based threshold."""
        symbol = trade.get("symbol", "")
        df = price_data.get(symbol)

        if df is None or "Volume" not in df.columns:
            return False

        try:
            vol_ma_period = getattr(self.cfg, "VOLUME_MA_PERIOD", 20)
            vol_mult = getattr(self.cfg, "VOLUME_CONFIRM_MULTIPLIER", 0.8)

            vol_ma = df["Volume"].rolling(vol_ma_period).mean()

            # Use the latest available value as proxy for entry
            latest_vol = df["Volume"].iloc[-1]
            latest_ma = vol_ma.iloc[-1]

            if np.isnan(latest_ma) or latest_ma <= 0:
                return False

            return latest_vol >= latest_ma * vol_mult
        except Exception:
            return False

    def _volatility_below_threshold(
        self,
        trade: dict,
        price_data: dict,
    ) -> bool:
        """Check if realised volatility was below the regime-high threshold."""
        symbol = trade.get("symbol", "")
        df = price_data.get(symbol)

        if df is None or "Close" not in df.columns:
            return False

        try:
            vol_threshold = getattr(
                self.cfg, "REGIME_VOL_HIGH_THRESHOLD", 0.20
            )
            lookback = getattr(self.cfg, "GARCH_LOOKBACK", 252)

            prices = df["Close"].dropna()
            if len(prices) < lookback:
                return False

            daily_ret = prices.pct_change().dropna().iloc[-lookback:]
            realised_vol = daily_ret.std() * np.sqrt(252)

            return realised_vol < vol_threshold
        except Exception:
            return False

    @staticmethod
    def _parse_signals_json(signals_json: str) -> dict:
        """
        Best-effort parse of the signals_json blob attached to a trade.
        Returns a flat dict with context hints.
        """
        ctx: dict = {"volume_low": False, "adx": 0.0}
        if not signals_json:
            return ctx
        try:
            data = (
                json.loads(signals_json)
                if isinstance(signals_json, str)
                else signals_json
            )
            if isinstance(data, dict):
                ctx["volume_low"] = data.get("volume_low", False)
                ctx["adx"] = float(data.get("adx", 0.0))
            elif isinstance(data, list):
                for entry in data:
                    if isinstance(entry, dict):
                        if entry.get("volume_low"):
                            ctx["volume_low"] = True
                        adx_val = entry.get("adx", 0.0)
                        if adx_val:
                            ctx["adx"] = max(ctx["adx"], float(adx_val))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
        return ctx

    @staticmethod
    def _trade_id(trade: dict) -> str:
        """Generate a simple deterministic trade reference string."""
        parts = [
            trade.get("symbol", "UNK"),
            trade.get("strategy", "UNK"),
            str(trade.get("entry_date", "")),
            str(trade.get("exit_date", "")),
        ]
        return "_".join(parts)

    @staticmethod
    def _generate_condition_text(
        outcome: str,
        tags: list[str],
        strategy: str,
        symbol: str,
    ) -> str:
        """Build a human-readable condition description for the lesson."""
        tag_str = ", ".join(tags) if tags else "no tags"
        return (
            f"{strategy} trade on {symbol}: {outcome}. "
            f"Patterns: {tag_str}."
        )
