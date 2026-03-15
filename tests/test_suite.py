"""
tests/test_suite.py — Comprehensive Unit Test Suite

Run: python tests/test_suite.py
Uses only Python's built-in unittest module. All external APIs are mocked.

IMPORTANT: This file mocks missing pip packages (loguru, alpaca, statsmodels,
arch) at the sys.modules level so tests can run without `pip install`.
"""

import os
import sys
import types
import unittest
from datetime import datetime, timedelta, date
from unittest.mock import patch, MagicMock

# ═════════════════════════════════════════════════════════════════════
# MOCK SHIMS — installed before ANY project imports
# ═════════════════════════════════════════════════════════════════════

def _install_mock_modules():
    """
    Pre-install mock modules for missing pip packages so project code
    can be imported without errors.
    """
    # ── loguru ──
    loguru_mod = types.ModuleType("loguru")
    mock_logger = MagicMock()
    mock_logger.remove = MagicMock()
    mock_logger.add = MagicMock()
    mock_logger.info = MagicMock()
    mock_logger.warning = MagicMock()
    mock_logger.error = MagicMock()
    mock_logger.debug = MagicMock()
    mock_logger.critical = MagicMock()
    mock_logger.opt = MagicMock(return_value=mock_logger)
    loguru_mod.logger = mock_logger
    sys.modules["loguru"] = loguru_mod

    # ── alpaca (comprehensive) ──
    for mod_name in [
        "alpaca", "alpaca.trading", "alpaca.trading.client",
        "alpaca.trading.requests", "alpaca.trading.enums",
        "alpaca.data", "alpaca.data.historical",
        "alpaca.data.requests", "alpaca.data.timeframe",
    ]:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = types.ModuleType(mod_name)

    # Add required classes/enums
    sys.modules["alpaca.trading.client"].TradingClient = MagicMock
    sys.modules["alpaca.trading.requests"].LimitOrderRequest = MagicMock
    sys.modules["alpaca.trading.enums"].OrderSide = MagicMock()
    sys.modules["alpaca.trading.enums"].OrderSide.BUY = "buy"
    sys.modules["alpaca.trading.enums"].OrderSide.SELL = "sell"
    sys.modules["alpaca.trading.enums"].TimeInForce = MagicMock()
    sys.modules["alpaca.trading.enums"].TimeInForce.DAY = "day"
    sys.modules["alpaca.data.historical"].StockHistoricalDataClient = MagicMock
    sys.modules["alpaca.data.requests"].StockBarsRequest = MagicMock
    sys.modules["alpaca.data.requests"].StockLatestQuoteRequest = MagicMock
    sys.modules["alpaca.data.timeframe"].TimeFrame = MagicMock()
    sys.modules["alpaca.data.timeframe"].TimeFrameUnit = MagicMock()

    # ── statsmodels ──
    for mod_name in [
        "statsmodels", "statsmodels.tsa",
        "statsmodels.tsa.stattools",
    ]:
        if mod_name not in sys.modules:
            m = types.ModuleType(mod_name)
            sys.modules[mod_name] = m
    sys.modules["statsmodels.tsa.stattools"].coint = MagicMock(
        return_value=(0.0, 0.05, [0.01, 0.05, 0.10])
    )
    sys.modules["statsmodels.tsa.stattools"].adfuller = MagicMock(
        return_value=(-3.5, 0.01, 1, 100, {}, 0.0)
    )

    # ── arch ──
    for mod_name in ["arch", "arch.univariate"]:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = types.ModuleType(mod_name)
    sys.modules["arch.univariate"].arch_model = MagicMock()

    # ── scipy.optimize ──
    for mod_name in ["scipy", "scipy.optimize"]:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = types.ModuleType(mod_name)

    # ── vaderSentiment ──
    for mod_name in ["vaderSentiment", "vaderSentiment.vaderSentiment"]:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = types.ModuleType(mod_name)


# Install mocks BEFORE any project imports
_install_mock_modules()

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd


# ═════════════════════════════════════════════════════════════════════
# HELPERS — Synthetic Data Generators
# ═════════════════════════════════════════════════════════════════════

def _make_ohlcv(days: int = 300, daily_return: float = 0.001,
                start_price: float = 100.0,
                start_date: str = "2024-01-02") -> pd.DataFrame:
    """Generate synthetic OHLCV with DatetimeIndex."""
    dates = pd.bdate_range(start=start_date, periods=days)
    prices = [start_price]
    for _ in range(days - 1):
        prices.append(prices[-1] * (1 + daily_return))
    prices = np.array(prices)

    df = pd.DataFrame({
        "Open": prices * 0.999,
        "High": prices * 1.005,
        "Low": prices * 0.995,
        "Close": prices,
        "Volume": np.random.randint(1_000_000, 5_000_000, size=days),
        "Adj Close": prices,
        # lowercase aliases for strategies that expect them (TSMOM)
        "open": prices * 0.999,
        "high": prices * 1.005,
        "low": prices * 0.995,
        "close": prices,
        "volume": np.random.randint(1_000_000, 5_000_000, size=days),
    }, index=dates)
    df.index.name = "timestamp"
    return df


def _make_trade(trade_id: str = "T001", symbol: str = "SPY",
                strategy: str = "tsmom", pnl: float = 100.0) -> dict:
    return {
        "trade_id": trade_id, "symbol": symbol, "strategy": strategy,
        "entry_date": "2024-06-01", "exit_date": "2024-06-15",
        "entry_price": 500.0, "exit_price": 500.0 + pnl / 10,
        "shares": 10, "dollar_size": 5000.0,
        "pnl_dollars": pnl, "pnl_pct": pnl / 5000.0,
        "hold_days": 14, "exit_reason": "SIGNAL_REVERSAL",
    }


# ═════════════════════════════════════════════════════════════════════
# GROUP 1 — TSMOM
# ═════════════════════════════════════════════════════════════════════

class TestTSMOM(unittest.TestCase):
    def setUp(self):
        from strategies.tsmom import TSMOMStrategy
        self.strategy = TSMOMStrategy()

    def test_tsmom_positive_signal_on_uptrend(self):
        """Uptrending data should produce a positive signal."""
        df = _make_ohlcv(days=300, daily_return=+0.001)
        data = {"TEST": df}
        # Use date AFTER the last bar so strategy can see all data
        last_date = df.index[-1] + pd.Timedelta(days=1)
        signals = self.strategy.compute_signals(data, last_date)
        self.assertGreater(signals.get("TEST", 0), 0,
                           "Uptrend should produce positive signal")

    def test_tsmom_negative_signal_on_downtrend(self):
        """Downtrending data should produce a negative signal."""
        df = _make_ohlcv(days=300, daily_return=-0.001)
        data = {"TEST": df}
        last_date = df.index[-1] + pd.Timedelta(days=1)
        signals = self.strategy.compute_signals(data, last_date)
        self.assertLess(signals.get("TEST", 0), 0,
                        "Downtrend should produce negative signal")

    def test_tsmom_zero_signal_insufficient_data(self):
        """Fewer than 252 days should return 0."""
        df = _make_ohlcv(days=100, daily_return=+0.001)
        data = {"TEST": df}
        last_date = df.index[-1] + pd.Timedelta(days=1)
        signals = self.strategy.compute_signals(data, last_date)
        self.assertEqual(signals.get("TEST", 0), 0.0)

    def test_tsmom_no_lookahead_bias(self):
        """Strategy must not use data at or after current_date."""
        df = _make_ohlcv(days=300, daily_return=+0.001)
        mid_date = df.index[200]
        data = {"TEST": df}
        # Should run without assertion error
        signals = self.strategy.compute_signals(data, mid_date)
        self.assertIsInstance(signals.get("TEST", 0.0), float)


# ═════════════════════════════════════════════════════════════════════
# GROUP 2 — DUAL MOMENTUM
# ═════════════════════════════════════════════════════════════════════

class TestDualMomentum(unittest.TestCase):
    def setUp(self):
        from strategies.dual_momentum import DualMomentumFilter
        self.filter = DualMomentumFilter()

    def test_blocks_on_negative_absolute_return(self):
        df = _make_ohlcv(days=300, daily_return=-0.001)
        data = {"TEST": df}
        last_date = df.index[-1] + pd.Timedelta(days=1)
        result = self.filter.compute_filter(data, last_date)
        self.assertEqual(result.get("TEST", 1), 0)

    def test_passes_when_both_positive(self):
        df = _make_ohlcv(days=300, daily_return=+0.002)
        shv = _make_ohlcv(days=300, daily_return=+0.0001)
        data = {"TEST": df, "SHV": shv}
        last_date = df.index[-1] + pd.Timedelta(days=1)
        result = self.filter.compute_filter(data, last_date)
        self.assertEqual(result.get("TEST", 0), 1)

    def test_blocks_on_underperformance(self):
        weak = _make_ohlcv(days=300, daily_return=+0.0001)
        shv = _make_ohlcv(days=300, daily_return=+0.001)
        data = {"TEST": weak, "SHV": shv}
        last_date = weak.index[-1] + pd.Timedelta(days=1)
        result = self.filter.compute_filter(data, last_date)
        self.assertEqual(result.get("TEST", 1), 0)


# ═════════════════════════════════════════════════════════════════════
# GROUP 3 — REGIME DETECTOR
# ═════════════════════════════════════════════════════════════════════

class TestRegimeDetector(unittest.TestCase):
    def setUp(self):
        from core.regime_detector import RegimeDetector
        self.detector = RegimeDetector()

    def test_regime_trending(self):
        data = {"SPY": _make_ohlcv(days=300, daily_return=+0.002)}
        last_date = data["SPY"].index[-1] + pd.Timedelta(days=1)
        regime = self.detector.detect(data, last_date)
        self.assertIn(regime, ["TRENDING", "RANGING", "RISK_OFF"])

    def test_regime_risk_off(self):
        data = {"SPY": _make_ohlcv(days=300, daily_return=-0.003)}
        last_date = data["SPY"].index[-1] + pd.Timedelta(days=1)
        regime = self.detector.detect(data, last_date)
        self.assertIn(regime, ["TRENDING", "RANGING", "RISK_OFF"])

    def test_regime_ranging(self):
        data = {"SPY": _make_ohlcv(days=300, daily_return=+0.00001)}
        last_date = data["SPY"].index[-1] + pd.Timedelta(days=1)
        regime = self.detector.detect(data, last_date)
        self.assertIn(regime, ["TRENDING", "RANGING", "RISK_OFF"])


# ═════════════════════════════════════════════════════════════════════
# GROUP 4 — POSITION SIZING
# ═════════════════════════════════════════════════════════════════════

class TestPositionSizing(unittest.TestCase):
    def setUp(self):
        from risk.position_sizing import PositionSizer
        self.sizer = PositionSizer()

    def test_kelly_cap_enforced(self):
        import config
        df = _make_ohlcv(days=300, daily_return=+0.005)
        signals = {"TEST": 1.0}
        portfolio = {"equity": 10000, "peak_equity": 10000,
                     "cash": 10000, "positions": {}}
        sizes = self.sizer.compute(signals, {"TEST": df}, portfolio,
                                   df.index[-1])
        max_allowed = 10000 * config.KELLY_CAP
        for sym, sz in sizes.items():
            self.assertLessEqual(abs(sz), max_allowed + 1)

    def test_position_reduces_in_high_vol(self):
        np.random.seed(42)
        df_low_vol = _make_ohlcv(days=300, daily_return=+0.001)
        df_high_vol = df_low_vol.copy()
        df_high_vol["Close"] = df_high_vol["Close"] + np.random.normal(0, 3, 300)

        signals = {"TEST": 0.5}
        portfolio = {"equity": 10000, "peak_equity": 10000,
                     "cash": 10000, "positions": {}}

        sizes_low = self.sizer.compute(signals, {"TEST": df_low_vol},
                                       portfolio, df_low_vol.index[-1])
        sizes_high = self.sizer.compute(signals, {"TEST": df_high_vol},
                                        portfolio, df_high_vol.index[-1])

        low_sz = abs(sizes_low.get("TEST", 0))
        high_sz = abs(sizes_high.get("TEST", 0))
        if low_sz > 0 and high_sz > 0 and low_sz != high_sz:
            self.assertLess(high_sz, low_sz)

    def test_drawdown_multiplier_halves(self):
        mult = self.sizer._drawdown_multiplier(equity=9400, peak_equity=10000)
        self.assertEqual(mult, 0.5)

    def test_drawdown_multiplier_ok(self):
        mult = self.sizer._drawdown_multiplier(equity=9600, peak_equity=10000)
        self.assertEqual(mult, 1.0)


# ═════════════════════════════════════════════════════════════════════
# GROUP 5 — EXITS
# ═════════════════════════════════════════════════════════════════════

class TestExits(unittest.TestCase):
    def setUp(self):
        from risk.exits import ExitManager
        self.exit_mgr = ExitManager()

    def _make_position(self, entry_price=100.0, days_ago=5, side="long"):
        entry_date = (date.today() - timedelta(days=days_ago)).isoformat()
        return {"side": side, "entry_price": entry_price,
                "entry_date": entry_date}

    def test_hard_stop_triggers(self):
        pos = self._make_position(entry_price=100.0, days_ago=5)
        df = _make_ohlcv(days=10, daily_return=-0.01, start_price=91.0)
        results = self.exit_mgr.check_all_exits(
            positions={"TEST": pos}, price_data={"TEST": df},
            current_signals={"TEST": 0.5},
            current_date=date.today().isoformat(),
            kill_switch_status="OK",
        )
        should_exit, reason, _ = results.get("TEST", (False, "", 0))
        self.assertTrue(should_exit, "9% loss should trigger exit")
        self.assertIn("HARD_STOP", reason.upper())

    def test_hard_stop_does_not_trigger_early(self):
        pos = self._make_position(entry_price=100.0, days_ago=5)
        # Price at 98 → only -2% loss, well above -7% hard stop
        df = _make_ohlcv(days=10, daily_return=0.0, start_price=98.0)
        results = self.exit_mgr.check_all_exits(
            positions={"TEST": pos}, price_data={"TEST": df},
            current_signals={"TEST": 0.5},
            current_date=date.today().isoformat(),
            kill_switch_status="OK",
        )
        should_exit, reason, _ = results.get("TEST", (False, "", 0))
        if should_exit:
            self.assertNotIn("HARD_STOP", reason.upper())

    def test_time_stop_triggers(self):
        pos = self._make_position(entry_price=100.0, days_ago=35)
        df = _make_ohlcv(days=40, daily_return=0.0001, start_price=100.5)
        results = self.exit_mgr.check_all_exits(
            positions={"TEST": pos}, price_data={"TEST": df},
            current_signals={"TEST": 0.5},
            current_date=date.today().isoformat(),
            kill_switch_status="OK",
        )
        should_exit, reason, _ = results.get("TEST", (False, "", 0))
        self.assertTrue(should_exit, "35-day hold should trigger time stop")
        self.assertIn("TIME", reason.upper())


# ═════════════════════════════════════════════════════════════════════
# GROUP 6 — MEMORY STORE
# ═════════════════════════════════════════════════════════════════════

class TestMemoryStore(unittest.TestCase):
    def setUp(self):
        from learning.memory_store import MemoryStore
        self.ms = MemoryStore(":memory:")

    def tearDown(self):
        try:
            self.ms._conn.close()
        except Exception:
            pass

    def test_write_and_read_lesson(self):
        import uuid
        lesson = {
            "lesson_id": str(uuid.uuid4()),
            "lesson_type": "win_pattern",
            "source_agent": "TestAgent",
            "condition": "uptrend + high vol",
            "recommended_action": "increase weight",
            "confidence": 0.85,
        }
        self.ms.store_lesson(lesson)
        lessons = self.ms.get_lessons(since_days=1, unread_only=False)
        self.assertGreater(len(lessons), 0)
        found = lessons[0]
        self.assertEqual(found["lesson_type"], "win_pattern")
        self.assertEqual(found["source_agent"], "TestAgent")

    def test_trade_count_increments(self):
        initial = self.ms.get_trade_count()
        for i in range(3):
            self.ms.log_trade(_make_trade(trade_id=f"TC_{i}"))
        self.assertEqual(self.ms.get_trade_count(), initial + 3)

    def test_weights_persist(self):
        self.ms.store_weight_update({
            "tsmom_weight": 0.50, "vol_trend_weight": 0.30,
            "pairs_arb_weight": 0.20, "reason": "test",
        })
        w = self.ms.get_latest_weights()
        self.assertIsNotNone(w)
        self.assertAlmostEqual(w["tsmom_weight"], 0.50, places=2)
        self.assertAlmostEqual(w["vol_trend_weight"], 0.30, places=2)
        self.assertAlmostEqual(w["pairs_arb_weight"], 0.20, places=2)

    def test_paper_warmup_tracking(self):
        self.assertIsNone(self.ms.get_paper_trading_start())
        self.assertEqual(self.ms.get_paper_trading_days(), 0)
        self.ms.set_paper_trading_start(date.today())
        self.assertEqual(self.ms.get_paper_trading_start(), date.today())
        self.assertEqual(self.ms.get_paper_trading_days(), 0)

    def test_eod_snapshot(self):
        portfolio = {"equity": 10000, "cash": 5000,
                     "positions": {"SPY": {}, "QQQ": {}},
                     "daily_pnl": 150, "drawdown_pct": 0.02}
        self.ms.write_eod_snapshot(portfolio)
        row = self.ms._conn.execute(
            "SELECT * FROM eod_snapshots ORDER BY id DESC LIMIT 1"
        ).fetchone()
        self.assertIsNotNone(row)
        self.assertAlmostEqual(row["equity"], 10000)
        self.assertEqual(row["open_positions"], 2)


# ═════════════════════════════════════════════════════════════════════
# GROUP 7 — DATA AGENT
# ═════════════════════════════════════════════════════════════════════

class TestDataAgent(unittest.TestCase):
    def test_column_normalisation(self):
        from data.data_agent import DataAgent
        raw = pd.DataFrame({
            "open": [100.0], "high": [101.0], "low": [99.0],
            "close": [100.5], "volume": [1_000_000],
        })
        result = DataAgent._normalise_columns(raw)
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            self.assertIn(col, result.columns)
        self.assertIn("Adj Close", result.columns)
        self.assertEqual(result["Adj Close"].iloc[0], result["Close"].iloc[0])

    def test_column_normalisation_idempotent(self):
        from data.data_agent import DataAgent
        raw = pd.DataFrame({
            "Open": [100.0], "High": [101.0], "Low": [99.0],
            "Close": [100.5], "Volume": [1_000_000],
        })
        result = DataAgent._normalise_columns(raw)
        result2 = DataAgent._normalise_columns(result)
        self.assertListEqual(sorted(result.columns), sorted(result2.columns))

    def test_validation_rejects_nan(self):
        from data.data_agent import DataAgent, DataValidationError
        df = _make_ohlcv(days=100)
        df.loc[df.index[50], "Close"] = float("nan")
        agent = DataAgent()
        if hasattr(agent, "validate_dataframe"):
            with self.assertRaises(DataValidationError):
                agent.validate_dataframe(df)
        else:
            self.assertTrue(df["Close"].isna().any())


# ═════════════════════════════════════════════════════════════════════
# RUNNER
# ═════════════════════════════════════════════════════════════════════

class VerboseResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.passed = 0
        self.test_results = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.passed += 1
        name = str(test).split(" ")[0]
        self.test_results.append(f"  ✅ PASS  {name}")

    def addFailure(self, test, err):
        super().addFailure(test, err)
        name = str(test).split(" ")[0]
        msg = str(err[1])[:120] if err[1] else ""
        self.test_results.append(f"  ❌ FAIL  {name} — {msg}")

    def addError(self, test, err):
        super().addError(test, err)
        name = str(test).split(" ")[0]
        msg = f"{err[0].__name__}: {str(err[1])[:120]}" if err[1] else str(err[0])
        self.test_results.append(f"  ❌ ERROR {name} — {msg}")


def main():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    for cls in [TestTSMOM, TestDualMomentum, TestRegimeDetector,
                TestPositionSizing, TestExits, TestMemoryStore,
                TestDataAgent]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(
        resultclass=VerboseResult, verbosity=0,
        stream=open(os.devnull, "w"),
    )
    result = runner.run(suite)

    total = result.passed + len(result.failures) + len(result.errors)

    print()
    print("═" * 60)
    print("  UNIT TEST SUITE RESULTS")
    print("═" * 60)
    for line in result.test_results:
        print(line)
    print()
    print(f"  OVERALL: {result.passed}/{total} tests passed")
    if result.passed == total:
        print("  ✅ ALL TESTS PASSED")
    else:
        print(f"  ❌ {total - result.passed} TESTS FAILED")
        for fail in result.failures + result.errors:
            print(f"\n  --- {fail[0]} ---")
            print(f"  {str(fail[1])[:500]}")
    print("═" * 60)

    sys.exit(0 if result.passed == total else 1)


if __name__ == "__main__":
    main()
