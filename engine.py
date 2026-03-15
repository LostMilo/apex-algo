"""
engine.py — Core Backtest Engine

Simulates trading on historical data day by day.
This is the most critical file — look-ahead bias here makes ALL results
meaningless.

CRITICAL ANTI-LOOK-AHEAD RULES (enforced with assertions):
  - The engine CONTROLS what data each strategy sees — strategies never
    get full history.
  - On day N, strategies receive ONLY data from day 0 to day N−1
    (strictly less than current_date).
  - Assert before EVERY strategy call: data passed has NO dates >=
    current_date.
  - Rolling calculations are computed inside the strategy on the sliced
    data — the engine does not pre-compute indicators.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import config
from logger import log
from data.data_agent import DataAgent
from core.regime_detector import RegimeDetector, Regime
from strategies import Signal
from strategies.tsmom import TSMOMStrategy
from strategies.dual_momentum import DualMomentumFilter
from strategies.vol_trend import VolTrendStrategy
from strategies.pairs_arb import PairsArbStrategy


# ═══════════════════════════════════════════════════════════════════════
# 1. BacktestResult — frozen output container
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class BacktestResult:
    """Immutable container for all backtest output metrics."""

    # Period
    start_date: str = ""
    end_date: str = ""
    trading_days: int = 0

    # Capital
    initial_capital: float = 0.0
    final_equity: float = 0.0

    # Returns
    total_return: float = 0.0
    cagr: float = 0.0
    annual_volatility: float = 0.0

    # Risk-adjusted
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Drawdown
    max_drawdown_pct: float = 0.0
    max_drawdown_duration_days: int = 0

    # Trades
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_pnl: float = 0.0
    avg_holding_days: float = 0.0

    # Raw data
    equity_curve: list[dict] = field(default_factory=list)
    trade_log: list[dict] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════
# 2. PositionSizer — ATR-based with Kelly fraction cap
# ═══════════════════════════════════════════════════════════════════════

class PositionSizer:
    """
    ATR-based position sizing capped by half-Kelly.

    Size = equity × risk_fraction / (ATR × atr_mult)
    Then capped at MAX_POSITION_PCT of equity and KELLY_CAP.
    """

    def size(
        self,
        equity: float,
        price: float,
        atr: float,
        signal_strength: float,
    ) -> float:
        """
        Calculate number of shares to buy/sell.

        Parameters
        ----------
        equity : float
            Current total portfolio equity.
        price : float
            Current asset price.
        atr : float
            Current ATR(14) value for the asset.
        signal_strength : float
            Signal confidence in [0.0, 1.0].

        Returns
        -------
        float
            Number of shares (fractional allowed).  May be 0 if the
            calculated size is below MIN_ORDER_VALUE.
        """
        if equity <= 0 or price <= 0 or atr <= 0:
            return 0.0

        # Risk per trade = portfolio risk budget × signal strength
        risk_budget = equity * config.MAX_PORTFOLIO_RISK * signal_strength

        # ATR-based sizing: how many shares so that a 2×ATR move = risk_budget
        stop_distance = atr * config.STOP_LOSS_ATR_MULT
        if stop_distance <= 0:
            return 0.0

        shares = risk_budget / stop_distance

        # Cap at MAX_POSITION_PCT of equity
        max_shares_by_pct = (equity * config.MAX_POSITION_PCT) / price
        shares = min(shares, max_shares_by_pct)

        # Cap at Kelly fraction
        kelly_max = (equity * config.KELLY_CAP) / price
        shares = min(shares, kelly_max)

        # Floor: minimum order value
        if shares * price < config.MIN_ORDER_VALUE:
            return 0.0

        return shares


# ═══════════════════════════════════════════════════════════════════════
# 3. ExitManager — Chandelier, hard stop, time stop
# ═══════════════════════════════════════════════════════════════════════

class ExitManager:
    """
    Checks three exit conditions per open position.

    1. Chandelier trailing stop: highest-high − CHANDELIER_ATR_MULT × ATR
    2. Hard stop-loss: entry_price × (1 − HARD_STOP_PCT)
    3. Time stop: close after TIME_STOP_DAYS if P&L within dead band
    """

    def check_exits(
        self,
        positions: dict[str, dict],
        day_data: dict[str, pd.DataFrame],
        trading_day: pd.Timestamp,
        day_index: int,
    ) -> list[str]:
        """
        Return list of symbols that should be closed today.

        Parameters
        ----------
        positions : dict
            Symbol → position dict with keys: qty, entry_price, side,
            entry_day_index, highest_high.
        day_data : dict
            Symbol → OHLCV DataFrame (sliced, ending before trading_day).
        trading_day : pd.Timestamp
            Current trading day.
        day_index : int
            Integer offset from backtest start (for time stop).

        Returns
        -------
        list[str]
            Symbols to close.
        """
        exits: list[str] = []

        for symbol, pos in positions.items():
            if symbol not in day_data or day_data[symbol].empty:
                continue

            df = day_data[symbol]
            current_close = float(df["Close"].iloc[-1])

            # --- ATR for stop calculations ---
            atr = self._compute_atr(df)
            if atr is None:
                continue

            # --- 1. Chandelier trailing stop ---
            if self._chandelier_stop(pos, current_close, atr):
                log.info(
                    "EXIT Chandelier  |  {}  close={:.2f}  stop={:.2f}",
                    symbol, current_close,
                    pos["highest_high"] - config.CHANDELIER_ATR_MULT * atr,
                )
                exits.append(symbol)
                continue

            # --- 2. Hard stop-loss ---
            if self._hard_stop(pos, current_close):
                log.info(
                    "EXIT Hard stop  |  {}  close={:.2f}  entry={:.2f}  "
                    "loss={:.2%}",
                    symbol, current_close, pos["entry_price"],
                    (current_close / pos["entry_price"]) - 1,
                )
                exits.append(symbol)
                continue

            # --- 3. Time stop ---
            if self._time_stop(pos, current_close, day_index):
                log.info(
                    "EXIT Time stop  |  {}  held {} days  pnl={:.2%}",
                    symbol,
                    day_index - pos["entry_day_index"],
                    (current_close / pos["entry_price"]) - 1,
                )
                exits.append(symbol)
                continue

        return exits

    def _chandelier_stop(
        self, pos: dict, current_close: float, atr: float,
    ) -> bool:
        """Chandelier trailing stop based on highest high since entry."""
        if pos["side"] == "long":
            stop_level = pos["highest_high"] - config.CHANDELIER_ATR_MULT * atr
            return current_close < stop_level
        else:
            # For shorts: lowest low + CHANDELIER_ATR_MULT × ATR
            stop_level = pos["lowest_low"] + config.CHANDELIER_ATR_MULT * atr
            return current_close > stop_level

    def _hard_stop(self, pos: dict, current_close: float) -> bool:
        """Hard stop-loss from entry price."""
        if pos["side"] == "long":
            return current_close <= pos["entry_price"] * (1 - config.HARD_STOP_PCT)
        else:
            return current_close >= pos["entry_price"] * (1 + config.HARD_STOP_PCT)

    def _time_stop(
        self, pos: dict, current_close: float, day_index: int,
    ) -> bool:
        """Time stop: close dead-money positions after N days."""
        days_held = day_index - pos["entry_day_index"]
        if days_held < config.TIME_STOP_DAYS:
            return False

        pnl_pct = (current_close / pos["entry_price"]) - 1
        if pos["side"] == "short":
            pnl_pct = -pnl_pct

        # Close if P&L is in the dead band (not profitable enough)
        return config.TIME_STOP_MIN_PNL <= pnl_pct <= config.TIME_STOP_MAX_PNL

    @staticmethod
    def _compute_atr(df: pd.DataFrame) -> float | None:
        """Compute ATR(14) from the most recent data slice."""
        period = config.ATR_PERIOD
        if len(df) < period + 1:
            return None

        high = df["High"].iloc[-period:]
        low = df["Low"].iloc[-period:]
        close_prev = df["Close"].iloc[-(period + 1):-1]

        tr = pd.concat([
            high.values - low.values,
            abs(high.values - close_prev.values),
            abs(low.values - close_prev.values),
        ], axis=1).max(axis=1) if False else None

        # Compute true range properly
        tr_values = []
        for i in range(period):
            h = float(high.iloc[i])
            l = float(low.iloc[i])
            c = float(close_prev.iloc[i])
            tr_values.append(max(h - l, abs(h - c), abs(l - c)))

        return float(np.mean(tr_values))


# ═══════════════════════════════════════════════════════════════════════
# 4. ConsensusEngine — regime-aware vote aggregation
# ═══════════════════════════════════════════════════════════════════════

class ConsensusEngine:
    """
    Aggregate signals from multiple strategies via regime-filtered
    majority voting.

    Rules:
      - TRENDING regime: TSMOM + VolTrend active
      - RANGING regime: PairsArb active
      - DualMomentum: active in all regimes (applied as filter)
      - Require ≥ 2 agreeing signals to open a position
      - Final strength = weighted average of agreeing signal strengths
    """

    def vote(
        self,
        tsmom_signals: dict[str, float],
        vol_trend_signals: list[Signal],
        pairs_arb_signals: list[Signal],
        regimes: dict[str, Regime],
        universe: list[str],
    ) -> list[Signal]:
        """
        Aggregate signals across strategies.

        Parameters
        ----------
        tsmom_signals : dict[str, float]
            Symbol → raw signal from TSMOM [-1.0, 1.0].
        vol_trend_signals : list[Signal]
            Signals from VolTrend strategy.
        pairs_arb_signals : list[Signal]
            Signals from PairsArb strategy.
        regimes : dict[str, Regime]
            Symbol → current regime.
        universe : list[str]
            Full asset universe.

        Returns
        -------
        list[Signal]
            Consensus signals that passed the voting threshold.
        """
        # Collect all votes per symbol
        votes: dict[str, list[Signal]] = {}

        for sym in universe:
            votes[sym] = []

        # TSMOM → convert to Signal objects (only if regime allows)
        for sym, raw_signal in tsmom_signals.items():
            if raw_signal == 0.0:
                continue
            regime = regimes.get(sym, Regime.RANGING)
            if regime != Regime.TRENDING:
                continue
            direction = "long" if raw_signal > 0 else "short"
            strength = min(abs(raw_signal), 1.0)
            votes.setdefault(sym, []).append(
                Signal(sym, direction, strength, "TSMOM")
            )

        # VolTrend (already regime-filtered by the strategy)
        for sig in vol_trend_signals:
            votes.setdefault(sig.symbol, []).append(sig)

        # PairsArb (already regime-filtered by the strategy)
        for sig in pairs_arb_signals:
            votes.setdefault(sig.symbol, []).append(sig)

        # Consensus: require ≥ 2 signals agreeing on direction
        consensus: list[Signal] = []
        for sym, sym_votes in votes.items():
            if len(sym_votes) < 2:
                continue

            # Group by direction
            longs = [s for s in sym_votes if s.direction == "long"]
            shorts = [s for s in sym_votes if s.direction == "short"]

            if len(longs) >= 2:
                avg_str = np.mean([s.strength for s in longs])
                consensus.append(
                    Signal(sym, "long", float(avg_str), "Consensus")
                )
            elif len(shorts) >= 2:
                avg_str = np.mean([s.strength for s in shorts])
                consensus.append(
                    Signal(sym, "short", float(avg_str), "Consensus")
                )

        return consensus


# ═══════════════════════════════════════════════════════════════════════
# 5. BacktestEngine — the main event loop
# ═══════════════════════════════════════════════════════════════════════

class BacktestEngine:
    """
    Core backtesting engine.

    Inject all components at construction time.  Call run() to execute.
    """

    def __init__(
        self,
        strategies: dict | None = None,
        regime_detector=None,
        consensus_engine: ConsensusEngine | None = None,
        position_sizer: PositionSizer | None = None,
        exit_manager: ExitManager | None = None,
    ):
        """
        Store injected components.  No logic here.

        Parameters
        ----------
        strategies : dict, optional
            Pre-built strategy instances.  If None, creates defaults.
        regime_detector : callable, optional
            Function(df) → Regime.  Defaults to detect_regime.
        consensus_engine : ConsensusEngine, optional
        position_sizer : PositionSizer, optional
        exit_manager : ExitManager, optional
        """
        # Strategies
        strats = strategies or {}
        self.tsmom = strats.get("tsmom", TSMOMStrategy())
        self.dual_momentum = strats.get("dual_momentum", DualMomentumFilter())
        self.vol_trend = strats.get("vol_trend", VolTrendStrategy())
        self.pairs_arb = strats.get("pairs_arb", PairsArbStrategy())

        # Components
        self._regime_detector = regime_detector or RegimeDetector()
        self.consensus = consensus_engine or ConsensusEngine()
        self.sizer = position_sizer or PositionSizer()
        self.exit_manager = exit_manager or ExitManager()

    # ── Public API ──────────────────────────────────────────────────────

    def run(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        initial_capital: float = None,
        data: dict[str, pd.DataFrame] | None = None,
    ) -> BacktestResult:
        """
        Execute the backtest.

        Parameters
        ----------
        start_date : str
            YYYY-MM-DD.  Defaults to config.BACKTEST_START.
        end_date : str
            YYYY-MM-DD or "today".  Defaults to config.BACKTEST_END.
        initial_capital : float
            Starting cash.  Defaults to config.STARTING_CAPITAL.
        data : dict, optional
            Pre-loaded {symbol: DataFrame}.  If None, fetches via
            data_manager.

        Returns
        -------
        BacktestResult
        """
        start = start_date or config.BACKTEST_START
        end = end_date or config.BACKTEST_END
        if end == "today":
            end = datetime.now().strftime("%Y-%m-%d")
        capital = initial_capital if initial_capital is not None else config.STARTING_CAPITAL

        log.info("═══ BACKTEST ENGINE START ═══")
        log.info("Period: {} → {}", start, end)
        log.info("Capital: ${:,.2f}", capital)

        # ── 1. PRE-LOAD all data (warmup = start − 2 years) ────────────
        warmup_start = (
            pd.Timestamp(start) - pd.DateOffset(years=2)
        ).strftime("%Y-%m-%d")

        if data is None:
            data_agent = DataAgent()
            full_data = data_agent.get_universe_data(
                config.ASSET_UNIVERSE, start=warmup_start, end=end,
            )
        else:
            full_data = data

        if not full_data:
            log.error("No data available — aborting backtest")
            return BacktestResult()

        log.info(
            "Data loaded: {} symbols, warmup from {}",
            len(full_data), warmup_start,
        )

        # ── 2. Initialize portfolio state ───────────────────────────────
        cash: float = capital
        positions: dict[str, dict] = {}   # symbol → position dict
        equity_curve: list[dict] = []
        trade_log: list[dict] = []
        peak_equity: float = capital
        day_start_equity: float = capital
        halted: bool = False

        # ── 3. Get trading days (NYSE business days) ────────────────────
        all_dates: set[pd.Timestamp] = set()
        ts_start = pd.Timestamp(start)
        ts_end = pd.Timestamp(end)
        for df in full_data.values():
            mask = (df.index >= ts_start) & (df.index <= ts_end)
            all_dates.update(df.index[mask])
        trading_days = sorted(all_dates)

        if not trading_days:
            log.error("No trading days found in range {} → {}", start, end)
            return BacktestResult()

        log.info("Trading days: {}", len(trading_days))

        # ── 4. MAIN LOOP ───────────────────────────────────────────────
        for i, trading_day in enumerate(trading_days):

            # ─── a. SLICE DATA (CRITICAL) ───────────────────────────────
            # Strict < : current day bar NOT included
            day_data: dict[str, pd.DataFrame] = {}
            for sym, full_df in full_data.items():
                sliced = full_df.loc[full_df.index < trading_day]
                if not sliced.empty:
                    day_data[sym] = sliced

            # ─── b. ASSERT look-ahead ──────────────────────────────────
            for sym, df in day_data.items():
                assert df.index.max() < trading_day, (
                    f"Look-ahead bias on {sym} at {trading_day}"
                )

            # Skip if not enough symbols have data
            if len(day_data) < 2:
                equity = self._mark_to_market(cash, positions, day_data)
                equity_curve.append({
                    "date": trading_day.strftime("%Y-%m-%d"),
                    "equity": round(equity, 2),
                })
                continue

            # ─── c. CHECK EXITS (before entries — exits have priority) ──
            symbols_to_close = self.exit_manager.check_exits(
                positions, day_data, trading_day, i,
            )
            for sym in symbols_to_close:
                if sym in positions and sym in day_data:
                    close_price = float(day_data[sym]["Close"].iloc[-1])
                    # Apply slippage for sell
                    fill_price = close_price * (1 - config.MAX_SLIPPAGE_PCT)
                    pnl = self._close_position(
                        sym, fill_price, positions, cash,
                    )
                    cash = pnl["new_cash"]
                    trade_log.append({
                        "date": trading_day.strftime("%Y-%m-%d"),
                        "symbol": sym,
                        "side": "sell" if pnl["side"] == "long" else "buy_cover",
                        "qty": round(pnl["qty"], 4),
                        "price": round(fill_price, 4),
                        "pnl": round(pnl["pnl"], 2),
                        "reason": pnl.get("reason", "exit_signal"),
                    })

            # Update highest_high / lowest_low for trailing stops
            for sym, pos in positions.items():
                if sym in day_data and not day_data[sym].empty:
                    h = float(day_data[sym]["High"].iloc[-1])
                    l = float(day_data[sym]["Low"].iloc[-1])
                    pos["highest_high"] = max(pos.get("highest_high", h), h)
                    pos["lowest_low"] = min(pos.get("lowest_low", l), l)

            # ─── d. DAILY LOSS LIMIT check ─────────────────────────────
            current_equity = self._mark_to_market(cash, positions, day_data)

            if i == 0 or trading_day.weekday() == 0:
                # Reset reference at start or beginning of week
                day_start_equity = current_equity

            daily_pnl_pct = (
                (current_equity - day_start_equity) / day_start_equity
                if day_start_equity > 0 else 0
            )

            skip_new_entries = daily_pnl_pct < -config.DAILY_LOSS_LIMIT

            # ─── e. MAX DRAWDOWN HALT ──────────────────────────────────
            peak_equity = max(peak_equity, current_equity)
            drawdown = (peak_equity - current_equity) / peak_equity if peak_equity > 0 else 0

            if drawdown > config.MAX_DRAWDOWN_PCT:
                if not halted:
                    log.warning(
                        "MAX DRAWDOWN BREACHED ({:.2%}) — halting new entries",
                        drawdown,
                    )
                    halted = True
                skip_new_entries = True

            # ─── f. DETECT REGIME ──────────────────────────────────────
            regimes: dict[str, Regime] = {}
            for sym, df in day_data.items():
                regimes[sym] = self._regime_detector.detect(day_data, trading_day)

            # ─── g. GENERATE SIGNALS ───────────────────────────────────
            # TSMOM returns dict[str, float]
            tsmom_signals = self.tsmom.compute_signals(day_data, trading_day)

            # Also apply volume confirmation
            tsmom_signals = self.tsmom.compute_volume_confirmation(
                day_data, trading_day, tsmom_signals,
            )

            # VolTrend returns list[Signal] (already regime-filtered)
            vol_trend_signals = self.vol_trend.generate_signals(
                day_data, regimes,
            )

            # PairsArb returns list[Signal] (already regime-filtered)
            pairs_arb_signals = self.pairs_arb.generate_signals(
                day_data, regimes,
            )

            # ─── h. CONSENSUS VOTE ─────────────────────────────────────
            consensus_signals = self.consensus.vote(
                tsmom_signals,
                vol_trend_signals,
                pairs_arb_signals,
                regimes,
                list(day_data.keys()),
            )

            # Apply Dual Momentum filter
            filtered_signals = self.dual_momentum.filter_signals(
                consensus_signals, day_data,
            )

            # ─── i. SIZE & EXECUTE (if not halted / daily-limited) ─────
            if not skip_new_entries:
                total_exposure = sum(
                    pos["qty"] * pos["entry_price"]
                    for pos in positions.values()
                )
                for signal in filtered_signals:
                    sym = signal.symbol

                    # Skip if already positioned in this symbol
                    if sym in positions:
                        continue

                    # Skip if no data
                    if sym not in day_data or day_data[sym].empty:
                        continue

                    # Correlated position limit
                    if len(positions) >= config.MAX_CORRELATED_POSITIONS:
                        log.debug(
                            "Max correlated positions ({}) reached — "
                            "skipping {}",
                            config.MAX_CORRELATED_POSITIONS, sym,
                        )
                        break

                    # Check total exposure limit
                    close_price = float(day_data[sym]["Close"].iloc[-1])
                    if total_exposure / current_equity > config.MAX_TOTAL_EXPOSURE:
                        break

                    # ATR for sizing
                    atr = ExitManager._compute_atr(day_data[sym])
                    if atr is None or atr <= 0:
                        continue

                    # Position size
                    qty = self.sizer.size(
                        current_equity, close_price, atr, signal.strength,
                    )
                    if qty <= 0:
                        continue

                    # Apply buy slippage
                    fill_price = close_price * (1 + config.MAX_SLIPPAGE_PCT)
                    cost = qty * fill_price

                    if cost > cash:
                        qty = cash / fill_price
                        cost = qty * fill_price

                    if qty <= 0 or cost < config.MIN_ORDER_VALUE:
                        continue

                    # Execute
                    cash -= cost
                    positions[sym] = {
                        "qty": qty,
                        "entry_price": fill_price,
                        "side": signal.direction,
                        "entry_day_index": i,
                        "entry_date": trading_day.strftime("%Y-%m-%d"),
                        "strategy": signal.strategy,
                        "highest_high": float(day_data[sym]["High"].iloc[-1]),
                        "lowest_low": float(day_data[sym]["Low"].iloc[-1]),
                    }
                    total_exposure += cost

                    trade_log.append({
                        "date": trading_day.strftime("%Y-%m-%d"),
                        "symbol": sym,
                        "side": "buy" if signal.direction == "long" else "sell_short",
                        "qty": round(qty, 4),
                        "price": round(fill_price, 4),
                        "strategy": signal.strategy,
                        "signal_strength": round(signal.strength, 4),
                    })

                    log.debug(
                        "ENTRY {}  {}  qty={:.2f}  @{:.2f}  via {}",
                        signal.direction.upper(), sym, qty, fill_price,
                        signal.strategy,
                    )

            # ─── j. RECORD STATE ───────────────────────────────────────
            equity = self._mark_to_market(cash, positions, day_data)
            equity_curve.append({
                "date": trading_day.strftime("%Y-%m-%d"),
                "equity": round(equity, 2),
            })

            # Progress logging (yearly)
            if i > 0 and i % 252 == 0:
                log.info(
                    "  [{}/{}]  {}  equity=${:.2f}  positions={}  "
                    "trades={}",
                    i, len(trading_days),
                    trading_day.strftime("%Y-%m-%d"),
                    equity, len(positions), len(trade_log),
                )

        # ── 5. Close remaining positions at final prices ────────────────
        for sym in list(positions.keys()):
            if sym in day_data and not day_data[sym].empty:
                close_price = float(day_data[sym]["Close"].iloc[-1])
                fill_price = close_price * (1 - config.MAX_SLIPPAGE_PCT)
                pnl = self._close_position(sym, fill_price, positions, cash)
                cash = pnl["new_cash"]
                trade_log.append({
                    "date": trading_days[-1].strftime("%Y-%m-%d"),
                    "symbol": sym,
                    "side": "sell_final",
                    "qty": round(pnl["qty"], 4),
                    "price": round(fill_price, 4),
                    "pnl": round(pnl["pnl"], 2),
                    "reason": "backtest_end",
                })

        # ── 6. Calculate performance metrics ────────────────────────────
        result = self._compute_metrics(
            equity_curve, trade_log, capital, start, end,
        )

        log.info("═══ BACKTEST ENGINE COMPLETE ═══")
        self._print_summary(result)

        return result

    # ── Private helpers ─────────────────────────────────────────────────

    @staticmethod
    def _mark_to_market(
        cash: float,
        positions: dict[str, dict],
        day_data: dict[str, pd.DataFrame],
    ) -> float:
        """Calculate total equity = cash + sum of position values."""
        equity = cash
        for sym, pos in positions.items():
            if sym in day_data and not day_data[sym].empty:
                price = float(day_data[sym]["Close"].iloc[-1])
            else:
                price = pos["entry_price"]

            if pos["side"] == "long":
                equity += pos["qty"] * price
            else:
                # Short P&L: entry_price − current_price, per share
                equity += pos["qty"] * (2 * pos["entry_price"] - price)
        return equity

    @staticmethod
    def _close_position(
        symbol: str,
        fill_price: float,
        positions: dict[str, dict],
        cash: float,
    ) -> dict:
        """
        Close a position and return transaction details.

        Returns dict with: new_cash, pnl, qty, side, reason.
        """
        pos = positions.pop(symbol)
        qty = pos["qty"]
        entry = pos["entry_price"]

        if pos["side"] == "long":
            proceeds = qty * fill_price
            pnl = proceeds - (qty * entry)
            new_cash = cash + proceeds
        else:
            # Short: we sold at entry, now buy back at fill_price
            pnl = qty * (entry - fill_price)
            new_cash = cash + (qty * entry) + pnl

        return {
            "new_cash": new_cash,
            "pnl": pnl,
            "qty": qty,
            "side": pos["side"],
            "strategy": pos.get("strategy", "unknown"),
        }

    @staticmethod
    def _compute_metrics(
        equity_curve: list[dict],
        trade_log: list[dict],
        initial_capital: float,
        start: str,
        end: str,
    ) -> BacktestResult:
        """Compute all performance metrics from raw equity curve and trades."""
        if not equity_curve:
            return BacktestResult()

        df = pd.DataFrame(equity_curve)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        equities = df["equity"]
        returns = equities.pct_change().dropna()

        # Total return
        total_return = (equities.iloc[-1] / equities.iloc[0]) - 1

        # CAGR
        years = (equities.index[-1] - equities.index[0]).days / 365.25
        cagr = (
            (equities.iloc[-1] / equities.iloc[0]) ** (1 / max(years, 0.01)) - 1
            if years > 0 else 0.0
        )

        # Volatility
        ann_vol = float(returns.std() * np.sqrt(252)) if len(returns) > 1 else 0.0

        # Sharpe (risk-free ≈ 4%)
        excess = cagr - 0.04
        sharpe = excess / ann_vol if ann_vol > 0 else 0.0

        # Sortino (downside deviation)
        downside = returns[returns < 0]
        downside_std = float(downside.std() * np.sqrt(252)) if len(downside) > 1 else 0.0
        sortino = excess / downside_std if downside_std > 0 else 0.0

        # Max drawdown
        rolling_max = equities.cummax()
        drawdowns = (equities - rolling_max) / rolling_max
        max_dd = float(drawdowns.min())

        # Max drawdown duration
        in_dd = drawdowns < 0
        dd_groups = (~in_dd).cumsum()
        dd_durations = []
        for _, group in in_dd.groupby(dd_groups):
            if group.any():
                duration = (group.index[-1] - group.index[0]).days
                dd_durations.append(duration)
        max_dd_duration = max(dd_durations) if dd_durations else 0

        # Calmar
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

        # Trade statistics
        closing_trades = [t for t in trade_log if "pnl" in t]
        winning = [t for t in closing_trades if t["pnl"] > 0]
        losing = [t for t in closing_trades if t["pnl"] <= 0]

        win_rate = len(winning) / len(closing_trades) if closing_trades else 0.0
        avg_pnl = float(np.mean([t["pnl"] for t in closing_trades])) if closing_trades else 0.0

        gross_profit = sum(t["pnl"] for t in winning) if winning else 0.0
        gross_loss = abs(sum(t["pnl"] for t in losing)) if losing else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Average holding period (approximate via entry/exit date pairs)
        holding_days = []
        entry_dates: dict[str, str] = {}
        for t in trade_log:
            if "pnl" not in t:
                # Entry
                entry_dates[t["symbol"]] = t["date"]
            else:
                # Exit
                entry_dt = entry_dates.pop(t["symbol"], None)
                if entry_dt:
                    delta = (
                        pd.Timestamp(t["date"]) - pd.Timestamp(entry_dt)
                    ).days
                    holding_days.append(delta)
        avg_holding = float(np.mean(holding_days)) if holding_days else 0.0

        return BacktestResult(
            start_date=start,
            end_date=end,
            trading_days=len(equities),
            initial_capital=initial_capital,
            final_equity=round(float(equities.iloc[-1]), 2),
            total_return=round(total_return, 4),
            cagr=round(cagr, 4),
            annual_volatility=round(ann_vol, 4),
            sharpe_ratio=round(sharpe, 3),
            sortino_ratio=round(sortino, 3),
            calmar_ratio=round(calmar, 3),
            max_drawdown_pct=round(max_dd, 4),
            max_drawdown_duration_days=max_dd_duration,
            total_trades=len(trade_log),
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate=round(win_rate, 4),
            profit_factor=round(min(profit_factor, 999.0), 3),
            avg_trade_pnl=round(avg_pnl, 2),
            avg_holding_days=round(avg_holding, 1),
            equity_curve=equity_curve,
            trade_log=trade_log,
        )

    @staticmethod
    def _print_summary(result: BacktestResult) -> None:
        """Print a formatted summary table."""
        log.info("┌───────────────────────────────────────────────┐")
        log.info("│           BACKTEST RESULTS SUMMARY            │")
        log.info("├───────────────────────────────────────────────┤")
        log.info("│ Period:      {} → {}", result.start_date, result.end_date)
        log.info("│ Days:        {}", result.trading_days)
        log.info("│ Capital:     ${:>10,.2f} → ${:>10,.2f}",
                 result.initial_capital, result.final_equity)
        log.info("│ Total Ret:   {:>10.2%}", result.total_return)
        log.info("│ CAGR:        {:>10.2%}", result.cagr)
        log.info("│ Volatility:  {:>10.2%}", result.annual_volatility)
        log.info("│ Sharpe:      {:>10.3f}", result.sharpe_ratio)
        log.info("│ Sortino:     {:>10.3f}", result.sortino_ratio)
        log.info("│ Calmar:      {:>10.3f}", result.calmar_ratio)
        log.info("│ Max DD:      {:>10.2%}", result.max_drawdown_pct)
        log.info("│ DD Duration: {:>10d} days", result.max_drawdown_duration_days)
        log.info("│ Trades:      {:>10d}", result.total_trades)
        log.info("│ Win Rate:    {:>10.1%}", result.win_rate)
        log.info("│ Profit Fct:  {:>10.2f}", result.profit_factor)
        log.info("│ Avg P&L:     ${:>10.2f}", result.avg_trade_pnl)
        log.info("│ Avg Hold:    {:>10.1f} days", result.avg_holding_days)
        log.info("└───────────────────────────────────────────────┘")


# ═══════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run backtest engine")
    parser.add_argument(
        "--start", default=config.BACKTEST_START,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end", default="today",
        help="End date (YYYY-MM-DD or 'today')",
    )
    parser.add_argument(
        "--capital", type=float, default=config.STARTING_CAPITAL,
        help="Initial capital",
    )
    args = parser.parse_args()

    engine = BacktestEngine()
    result = engine.run(
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
    )

    # Save results
    if result.equity_curve:
        import json
        from pathlib import Path

        out_dir = Path("data/backtest_results")
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Metrics
        metrics = {
            k: v for k, v in result.__dict__.items()
            if k not in ("equity_curve", "trade_log")
        }
        with open(out_dir / f"metrics_{ts}.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # Equity curve
        with open(out_dir / f"equity_curve_{ts}.json", "w") as f:
            json.dump(result.equity_curve, f)

        # Trade log
        with open(out_dir / f"trades_{ts}.json", "w") as f:
            json.dump(result.trade_log, f, indent=2)

        log.info("Results saved to {}", out_dir)


if __name__ == "__main__":
    main()
