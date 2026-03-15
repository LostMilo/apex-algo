"""
backtest/walk_forward.py — Walk-Forward Optimization

Tests strategy robustness by optimizing on past data and validating on
completely unseen future data.  This prevents overfitting and gives
realistic performance estimates.

CRITICAL: The validation window NEVER touches the training data.
          Training optimizes, validation only measures.
"""

from __future__ import annotations

import itertools
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# ── Project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, ".")
import config
from logger import log


# ═══════════════════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class WindowResult:
    """Results from a single walk-forward window."""

    window_index: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str

    # Best params found on train period
    best_params: dict = field(default_factory=dict)
    all_params_tested: list[dict] = field(default_factory=list)

    # Train-period metrics (for reference / robustness ratio)
    train_sharpe: float = 0.0

    # Test-period metrics (the ONLY ones that matter)
    test_sharpe: float = 0.0
    test_max_drawdown: float = 0.0
    test_annualized_return: float = 0.0
    test_win_rate: float = 0.0

    # Stability flag
    params_stable: bool = True


@dataclass
class WalkForwardResult:
    """Aggregated results across all walk-forward windows."""

    windows: list[WindowResult] = field(default_factory=list)
    num_windows: int = 0

    # Aggregated metrics
    oos_sharpe: float = 0.0          # Out-of-sample Sharpe (avg across windows)
    stability_score: float = 0.0     # % of windows with stable params
    robustness_score: float = 0.0    # OOS Sharpe / IS Sharpe (target > 0.5)

    # Summary stats
    avg_test_return: float = 0.0
    avg_test_drawdown: float = 0.0
    avg_test_win_rate: float = 0.0

    def __repr__(self) -> str:
        return (
            f"WalkForwardResult(\n"
            f"  windows={self.num_windows},\n"
            f"  OOS Sharpe={self.oos_sharpe:.4f},\n"
            f"  stability={self.stability_score:.1%},\n"
            f"  robustness={self.robustness_score:.4f},\n"
            f"  avg_return={self.avg_test_return:.2%},\n"
            f"  avg_drawdown={self.avg_test_drawdown:.2%},\n"
            f"  avg_win_rate={self.avg_test_win_rate:.1%}\n"
            f")"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Walk-Forward Optimizer
# ═══════════════════════════════════════════════════════════════════════════════


class WalkForwardOptimizer:
    """
    Walk-Forward Optimization engine.

    Splits the full backtest period into non-overlapping train/test windows,
    optimizes strategy parameters on training data, then validates on
    completely unseen test data.  Aggregates out-of-sample performance to
    measure true strategy robustness.
    """

    # Parameter grid to optimise over
    PARAM_GRID = {
        "TSMOM_LOOKBACK": [126, 189, 252],         # 6, 9, 12 months
        "ADX_MIN": [20, 25, 30],
        "CHANDELIER_ATR_MULT": [2.5, 3.0, 3.5],
    }

    # Instability threshold: TSMOM_LOOKBACK change > 63 days = unstable
    LOOKBACK_INSTABILITY_THRESHOLD = 63

    def __init__(
        self,
        train_years: int | None = None,
        test_months: int | None = None,
        min_windows: int | None = None,
    ):
        self.train_years = train_years or config.WF_TRAIN_YEARS
        self.test_months = test_months or config.WF_TEST_MONTHS
        self.min_windows = min_windows or config.WF_MIN_WINDOWS

    # ─── Public API ─────────────────────────────────────────────────────────

    def run(
        self,
        price_data: dict[str, pd.DataFrame],
        initial_capital: float = 10_000,
    ) -> WalkForwardResult:
        """
        Execute walk-forward optimization.

        Args:
            price_data: Dict of symbol → OHLCV DataFrame (full history).
                        Must have columns: Close (or close), High/high,
                        Low/low, and a DatetimeIndex.
            initial_capital: Starting capital for each backtest window.

        Returns:
            WalkForwardResult with aggregated out-of-sample metrics.
        """
        # Normalise column names to title-case
        price_data = self._normalise_columns(price_data)

        # ── Step 1: Create windows ──────────────────────────────────────
        windows = self._create_windows(price_data)
        log.info(
            "Walk-forward: created {} windows  (train={}y, test={}mo)",
            len(windows), self.train_years, self.test_months,
        )

        if len(windows) < self.min_windows:
            log.warning(
                "Only {} windows created (need ≥ {}). "
                "Extend BACKTEST_START or shorten WF_TRAIN_YEARS/WF_TEST_MONTHS.",
                len(windows), self.min_windows,
            )

        # ── Step 2: Train + Validate each window ───────────────────────
        window_results: list[WindowResult] = []
        prev_best_params: dict | None = None

        for i, (tr_start, tr_end, te_start, te_end) in enumerate(windows):
            log.info(
                "── Window {}/{}: train {}->{} | test {}->{}",
                i + 1, len(windows), tr_start, tr_end, te_start, te_end,
            )

            wr = WindowResult(
                window_index=i,
                train_start=tr_start,
                train_end=tr_end,
                test_start=te_start,
                test_end=te_end,
            )

            # (a) Optimize on train period
            best_params, best_train_sharpe, all_tested = self._optimise_on_train(
                price_data, tr_start, tr_end, initial_capital,
            )
            wr.best_params = best_params
            wr.all_params_tested = all_tested
            wr.train_sharpe = best_train_sharpe

            log.info(
                "  Best train params: {} → Sharpe={:.4f}",
                best_params, best_train_sharpe,
            )

            # (b) Validate on test period (unseen data)
            test_metrics = self._run_single_backtest(
                price_data, te_start, te_end, best_params, initial_capital,
            )
            wr.test_sharpe = test_metrics["sharpe"]
            wr.test_max_drawdown = test_metrics["max_drawdown"]
            wr.test_annualized_return = test_metrics["annualized_return"]
            wr.test_win_rate = test_metrics["win_rate"]

            log.info(
                "  Test (OOS): Sharpe={:.4f}  DD={:.2%}  Ret={:.2%}  WR={:.1%}",
                wr.test_sharpe, wr.test_max_drawdown,
                wr.test_annualized_return, wr.test_win_rate,
            )

            # (c) Check parameter stability
            if prev_best_params is not None:
                wr.params_stable = self._check_stability(
                    prev_best_params, best_params,
                )
                if not wr.params_stable:
                    log.warning("  ⚠ Window {} flagged UNSTABLE", i + 1)

            prev_best_params = best_params
            window_results.append(wr)

        # ── Step 3: Aggregate results ──────────────────────────────────
        result = self._aggregate(window_results)
        log.info("Walk-forward complete:\n{}", result)
        return result

    # ─── Step 1: Window Creation ────────────────────────────────────────────

    def _create_windows(
        self,
        price_data: dict[str, pd.DataFrame],
    ) -> list[tuple[str, str, str, str]]:
        """
        Build non-overlapping (train, test) window tuples across the
        available data range.
        """
        # Find the common date range across all assets
        all_starts, all_ends = [], []
        for sym, df in price_data.items():
            idx = df.index
            if hasattr(idx[0], "date"):
                all_starts.append(idx[0].date() if hasattr(idx[0], "date") else idx[0])
                all_ends.append(idx[-1].date() if hasattr(idx[-1], "date") else idx[-1])
            else:
                all_starts.append(pd.Timestamp(idx[0]).date())
                all_ends.append(pd.Timestamp(idx[-1]).date())

        data_start = max(all_starts)
        data_end = min(all_ends)

        # Convert config boundaries
        cfg_start = datetime.strptime(config.BACKTEST_START, "%Y-%m-%d").date()
        cfg_end = (
            datetime.now().date()
            if config.BACKTEST_END == "today"
            else datetime.strptime(config.BACKTEST_END, "%Y-%m-%d").date()
        )

        # Effective range = intersection of config and data
        effective_start = max(data_start, cfg_start)
        effective_end = min(data_end, cfg_end)

        train_delta = timedelta(days=self.train_years * 365)
        test_delta = timedelta(days=self.test_months * 30)  # approx months

        windows = []
        cursor = effective_start

        while True:
            train_start = cursor
            train_end = train_start + train_delta
            test_start = train_end + timedelta(days=1)
            test_end = test_start + test_delta

            if test_end > effective_end:
                break

            windows.append((
                train_start.strftime("%Y-%m-%d"),
                train_end.strftime("%Y-%m-%d"),
                test_start.strftime("%Y-%m-%d"),
                test_end.strftime("%Y-%m-%d"),
            ))

            # Step forward by test_months (non-overlapping test windows)
            cursor = test_start

        return windows

    # ─── Step 2a: Train-Period Optimisation ──────────────────────────────────

    def _optimise_on_train(
        self,
        price_data: dict[str, pd.DataFrame],
        train_start: str,
        train_end: str,
        initial_capital: float,
    ) -> tuple[dict, float, list[dict]]:
        """
        Exhaustive grid search over PARAM_GRID on the training period.

        Returns (best_params, best_sharpe, all_results).
        """
        keys = list(self.PARAM_GRID.keys())
        values = list(self.PARAM_GRID.values())
        combos = list(itertools.product(*values))

        best_sharpe = -np.inf
        best_params: dict = {}
        all_results: list[dict] = []

        for combo in combos:
            params = dict(zip(keys, combo))
            metrics = self._run_single_backtest(
                price_data, train_start, train_end, params, initial_capital,
            )
            record = {**params, "sharpe": metrics["sharpe"]}
            all_results.append(record)

            log.debug(
                "    Grid: TSMOM_LB={} ADX_MIN={} CHAND={} → Sharpe={:.4f}",
                params["TSMOM_LOOKBACK"], params["ADX_MIN"],
                params["CHANDELIER_ATR_MULT"], metrics["sharpe"],
            )

            if metrics["sharpe"] > best_sharpe:
                best_sharpe = metrics["sharpe"]
                best_params = params

        log.info(
            "  Tested {} param combos on train period", len(all_results),
        )
        return best_params, best_sharpe, all_results

    # ─── Step 2c: Parameter Stability ───────────────────────────────────────

    def _check_stability(
        self,
        prev_params: dict,
        curr_params: dict,
    ) -> bool:
        """
        Return True if current optimal params are close to previous window's.

        Instability rule: TSMOM_LOOKBACK change > 63 days = unstable.
        """
        prev_lb = prev_params.get("TSMOM_LOOKBACK", 252)
        curr_lb = curr_params.get("TSMOM_LOOKBACK", 252)

        if abs(curr_lb - prev_lb) > self.LOOKBACK_INSTABILITY_THRESHOLD:
            return False
        return True

    # ─── Step 3: Aggregate ──────────────────────────────────────────────────

    def _aggregate(self, windows: list[WindowResult]) -> WalkForwardResult:
        """Combine all window results into a single WalkForwardResult."""
        if not windows:
            return WalkForwardResult()

        n = len(windows)
        oos_sharpes = [w.test_sharpe for w in windows]
        is_sharpes = [w.train_sharpe for w in windows]
        stable_count = sum(1 for w in windows if w.params_stable)

        avg_is_sharpe = np.mean(is_sharpes) if is_sharpes else 0.0
        avg_oos_sharpe = float(np.mean(oos_sharpes))

        robustness = (
            avg_oos_sharpe / avg_is_sharpe
            if avg_is_sharpe > 0
            else 0.0
        )

        return WalkForwardResult(
            windows=windows,
            num_windows=n,
            oos_sharpe=avg_oos_sharpe,
            stability_score=stable_count / n,
            robustness_score=robustness,
            avg_test_return=float(np.mean([w.test_annualized_return for w in windows])),
            avg_test_drawdown=float(np.mean([w.test_max_drawdown for w in windows])),
            avg_test_win_rate=float(np.mean([w.test_win_rate for w in windows])),
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # Self-Contained Backtest Engine
    # ═══════════════════════════════════════════════════════════════════════════

    def _run_single_backtest(
        self,
        price_data: dict[str, pd.DataFrame],
        start: str,
        end: str,
        params: dict,
        initial_capital: float,
    ) -> dict:
        """
        Run a self-contained day-by-day backtest on the given window.

        Uses TSMOM + VolTrend signal logic internally (no external engine
        dependency).  Returns dict with sharpe, max_drawdown,
        annualized_return, win_rate.

        Anti-look-ahead: only data up to the current bar is visible.
        """
        tsmom_lb = params.get("TSMOM_LOOKBACK", config.TSMOM_LOOKBACK)
        adx_min = params.get("ADX_MIN", config.ADX_MIN)
        chandelier_mult = params.get("CHANDELIER_ATR_MULT", config.CHANDELIER_ATR_MULT)

        # Slice data to window
        sliced: dict[str, pd.DataFrame] = {}
        for sym, df in price_data.items():
            mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
            window_df = df.loc[mask].copy()
            if not window_df.empty:
                sliced[sym] = window_df

        if not sliced:
            return self._empty_metrics()

        # Build a common date index
        all_dates: set[pd.Timestamp] = set()
        for df in sliced.values():
            all_dates.update(df.index.tolist())
        dates = sorted(all_dates)

        if len(dates) < tsmom_lb + 5:
            return self._empty_metrics()

        # Day-by-day simulation
        capital = initial_capital
        equity_curve: list[float] = [capital]
        positions: dict[str, float] = {}   # symbol → shares held (signed)
        entry_prices: dict[str, float] = {}
        wins = 0
        losses = 0

        for day_idx in range(tsmom_lb, len(dates)):
            current_date = dates[day_idx]

            for sym, df in sliced.items():
                if current_date not in df.index:
                    continue

                # Signal data: strict < (CANNOT see today's bar)
                visible = df.loc[df.index < current_date]
                assert visible.empty or visible.index.max() < current_date, \
                    f"Walk-forward look-ahead bias detected on {sym} at {current_date}"
                if len(visible) < tsmom_lb:
                    continue

                price_col = "Close"
                # Use today's bar for execution price only
                current_price = float(df.loc[current_date, price_col]) if current_date in df.index else float(visible[price_col].iloc[-1])

                # ── Signal: TSMOM ───────────────────────────────────────
                skip = config.TSMOM_SKIP_LAST
                if len(visible) >= tsmom_lb + skip:
                    end_idx = len(visible) - skip
                    start_idx = end_idx - tsmom_lb
                    if start_idx >= 0:
                        p_end = visible[price_col].iloc[end_idx - 1]
                        p_start = visible[price_col].iloc[start_idx]
                        if p_start > 0:
                            tsmom_return = (p_end / p_start) - 1.0
                        else:
                            tsmom_return = 0.0
                    else:
                        tsmom_return = 0.0
                else:
                    tsmom_return = 0.0

                # ── Signal: VolTrend (EMA cross + ADX filter) ──────────
                vol_signal = 0.0
                if len(visible) >= config.EMA_SLOW + 20:
                    ema_fast = visible[price_col].ewm(
                        span=config.EMA_FAST, adjust=False,
                    ).mean()
                    ema_slow = visible[price_col].ewm(
                        span=config.EMA_SLOW, adjust=False,
                    ).mean()

                    # Simple ADX approximation using directional range
                    high = visible["High"]
                    low = visible["Low"]
                    close = visible[price_col]
                    tr = pd.concat([
                        high - low,
                        (high - close.shift(1)).abs(),
                        (low - close.shift(1)).abs(),
                    ], axis=1).max(axis=1)
                    atr = tr.rolling(config.ADX_PERIOD).mean()

                    # +DM / -DM approximation
                    plus_dm = (high - high.shift(1)).clip(lower=0)
                    minus_dm = (low.shift(1) - low).clip(lower=0)
                    plus_di = 100 * (plus_dm.rolling(config.ADX_PERIOD).mean() / atr)
                    minus_di = 100 * (minus_dm.rolling(config.ADX_PERIOD).mean() / atr)
                    dx = (
                        100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10))
                    )
                    adx_approx = dx.rolling(config.ADX_PERIOD).mean()

                    latest_adx = adx_approx.iloc[-1]
                    latest_fast = ema_fast.iloc[-1]
                    latest_slow = ema_slow.iloc[-1]

                    if (
                        not np.isnan(latest_adx)
                        and adx_min <= latest_adx <= config.ADX_MAX
                    ):
                        if latest_fast > latest_slow:
                            vol_signal = 0.5
                        elif latest_fast < latest_slow:
                            vol_signal = -0.5

                # ── Composite signal ────────────────────────────────────
                composite = 0.0
                if tsmom_return > 0:
                    composite += 0.5
                elif tsmom_return < 0:
                    composite -= 0.5
                composite += vol_signal

                # ── Chandelier stop logic ───────────────────────────────
                if sym in positions and sym in entry_prices:
                    held_shares = positions[sym]
                    entry_p = entry_prices[sym]

                    # ATR for stop
                    h = visible["High"]
                    l = visible["Low"]
                    c = visible[price_col]
                    tr_stop = pd.concat([
                        h - l,
                        (h - c.shift(1)).abs(),
                        (l - c.shift(1)).abs(),
                    ], axis=1).max(axis=1)
                    atr_val = tr_stop.rolling(config.ATR_PERIOD).mean().iloc[-1]

                    if not np.isnan(atr_val):
                        if held_shares > 0:
                            highest = visible["High"].iloc[-22:].max()
                            stop_price = highest - chandelier_mult * atr_val
                            if current_price <= stop_price:
                                # Exit long
                                pnl = (current_price - entry_p) * held_shares
                                capital += pnl
                                if pnl > 0:
                                    wins += 1
                                else:
                                    losses += 1
                                del positions[sym]
                                del entry_prices[sym]
                                continue
                        elif held_shares < 0:
                            lowest = visible["Low"].iloc[-22:].min()
                            stop_price = lowest + chandelier_mult * atr_val
                            if current_price >= stop_price:
                                # Exit short
                                pnl = (entry_p - current_price) * abs(held_shares)
                                capital += pnl
                                if pnl > 0:
                                    wins += 1
                                else:
                                    losses += 1
                                del positions[sym]
                                del entry_prices[sym]
                                continue

                # ── Position management ─────────────────────────────────
                if sym not in positions:
                    # Entry
                    if composite > 0.3:
                        # Go long — allocate up to MAX_POSITION_PCT
                        alloc = capital * config.MAX_POSITION_PCT
                        shares = alloc / current_price if current_price > 0 else 0
                        if shares > 0:
                            positions[sym] = shares
                            entry_prices[sym] = current_price
                    elif composite < -0.3:
                        # Go short
                        alloc = capital * config.MAX_POSITION_PCT
                        shares = alloc / current_price if current_price > 0 else 0
                        if shares > 0:
                            positions[sym] = -shares
                            entry_prices[sym] = current_price
                else:
                    # Signal reversal → exit
                    held = positions[sym]
                    if (held > 0 and composite < -0.1) or (held < 0 and composite > 0.1):
                        pnl = (
                            (current_price - entry_prices[sym]) * held
                            if held > 0
                            else (entry_prices[sym] - current_price) * abs(held)
                        )
                        capital += pnl
                        if pnl > 0:
                            wins += 1
                        else:
                            losses += 1
                        del positions[sym]
                        del entry_prices[sym]

            # Mark-to-market equity
            mtm = capital
            for sym, shares in positions.items():
                if sym in sliced and current_date in sliced[sym].index:
                    px = sliced[sym].loc[current_date, "Close"]
                    if shares > 0:
                        mtm += (px - entry_prices[sym]) * shares
                    else:
                        mtm += (entry_prices[sym] - px) * abs(shares)
            equity_curve.append(mtm)

        # ── Calculate metrics ───────────────────────────────────────────
        return self._compute_metrics(equity_curve, wins, losses)

    # ─── Metrics Calculation ────────────────────────────────────────────────

    @staticmethod
    def _compute_metrics(
        equity_curve: list[float],
        wins: int,
        losses: int,
    ) -> dict:
        """Compute Sharpe, drawdown, annualised return, win rate from equity."""
        eq = np.array(equity_curve, dtype=np.float64)
        if len(eq) < 2 or eq[0] <= 0:
            return {
                "sharpe": 0.0,
                "max_drawdown": 0.0,
                "annualized_return": 0.0,
                "win_rate": 0.0,
            }

        # Daily returns
        daily_returns = np.diff(eq) / eq[:-1]
        daily_returns = daily_returns[np.isfinite(daily_returns)]

        if len(daily_returns) == 0:
            return {
                "sharpe": 0.0,
                "max_drawdown": 0.0,
                "annualized_return": 0.0,
                "win_rate": 0.0,
            }

        # Sharpe ratio (annualised, assume 252 trading days)
        mean_ret = np.mean(daily_returns)
        std_ret = np.std(daily_returns, ddof=1) if len(daily_returns) > 1 else 1e-10
        sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 1e-10 else 0.0

        # Max drawdown
        cummax = np.maximum.accumulate(eq)
        drawdowns = (eq - cummax) / cummax
        max_dd = float(np.min(drawdowns))  # most negative

        # Annualised return
        total_return = eq[-1] / eq[0] - 1.0
        n_years = len(daily_returns) / 252.0
        if n_years > 0 and total_return > -1.0:
            ann_return = (1.0 + total_return) ** (1.0 / n_years) - 1.0
        else:
            ann_return = total_return

        # Win rate
        total_trades = wins + losses
        win_rate = wins / total_trades if total_trades > 0 else 0.0

        return {
            "sharpe": float(sharpe),
            "max_drawdown": float(max_dd),
            "annualized_return": float(ann_return),
            "win_rate": float(win_rate),
        }

    @staticmethod
    def _empty_metrics() -> dict:
        """Return zeroed-out metrics when no data is available."""
        return {
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "annualized_return": 0.0,
            "win_rate": 0.0,
        }

    # ─── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _normalise_columns(
        price_data: dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        """
        Ensure all DataFrames have title-case OHLCV columns
        (Close, High, Low, Open, Volume).
        """
        col_map = {
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume",
            "adj close": "Adj Close",
        }
        normalised = {}
        for sym, df in price_data.items():
            renamed = df.rename(
                columns={c: col_map.get(c.lower(), c) for c in df.columns},
            )
            normalised[sym] = renamed
        return normalised
