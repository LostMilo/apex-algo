"""
main.py — Single Entry Point for the Autonomous Trading Algorithm

Wires all NEW ARCHITECTURE components together. Runs the daily trading cycle.
Supports three modes: backtest, paper, and live.

Usage:
    python main.py --mode paper      # Paper trading with scheduler
    python main.py --mode live       # Live trading (sets USE_LIVE_EXECUTION)
    python main.py --mode backtest   # Run backtest
"""

import sys
import argparse
import time
from datetime import datetime, date

import pandas as pd

import config
from logger import log, trade_log

# ─── NEW ARCHITECTURE IMPORTS ───────────────────────────────────────────────
from data.data_agent import DataAgent
from strategies.tsmom import TSMOMStrategy
from strategies.dual_momentum import DualMomentumFilter
from strategies.vol_trend import VolTrendStrategy
from strategies.pairs_arb import PairsArbStrategy
from core.regime_detector import RegimeDetector
from core.consensus_engine import ConsensusEngine
from risk.position_sizing import PositionSizer
from risk.exits import ExitManager
from learning.memory_store import MemoryStore
from learning.experience_agent import ExperienceAgent
from execution.alpaca_client import AlpacaClient
from utils.alerting import AlertManager


# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def validate_config(mode: str) -> None:
    """
    Fail loudly if required environment variables are missing.

    Backtest mode needs no API keys.
    Paper/Live modes require Alpaca credentials.
    """
    if mode in ("paper", "live"):
        missing = []
        if not config.ALPACA_API_KEY:
            missing.append("ALPACA_API_KEY")
        if not config.ALPACA_SECRET_KEY:
            missing.append("ALPACA_SECRET_KEY")

        if missing:
            log.critical(
                "Missing required environment variables for {} mode: {}",
                mode, ", ".join(missing),
            )
            sys.exit(1)

    if mode == "live" and config.PAPER_TRADING:
        log.warning(
            "PAPER_TRADING is True but running in live mode — "
            "set PAPER_TRADING=False in config.py for real execution"
        )

    log.info("Config validated for {} mode", mode)


# ─────────────────────────────────────────────────────────────────────────────
# 2. SYSTEM INITIALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def initialize_system(mode: str) -> dict:
    """
    Instantiate all components in dependency order.

    Returns a dict of named components for use by the daily cycle.
    """
    validate_config(mode)

    log.info("═" * 60)
    log.info("  INITIALIZING SYSTEM  |  mode={}  |  {}",
             mode, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log.info("═" * 60)

    # Apply mode-specific config overrides
    if mode == "live":
        config.USE_LIVE_DATA = True
        config.USE_LIVE_EXECUTION = True
    elif mode == "paper":
        config.USE_LIVE_DATA = True
        config.USE_LIVE_EXECUTION = False

    # ── 1. Foundation (no deps) ──────────────────────────────────────
    memory_store = MemoryStore(config.MEMORY_DB_PATH)
    alert_manager = AlertManager(config, memory_store=memory_store)
    data_agent = DataAgent()

    # ── 2. Strategies (no deps) ──────────────────────────────────────
    tsmom = TSMOMStrategy()
    dual_momentum = DualMomentumFilter()
    vol_trend = VolTrendStrategy()
    pairs_arb = PairsArbStrategy()

    # ── 3. Regime detector (no deps) ─────────────────────────────────
    regime_detector = RegimeDetector()

    # ── 4. Consensus engine (depends on config + memory_store) ───────
    # ConsensusEngine expects a dict-like object for weight persistence;
    # wrap MemoryStore's latest weights or provide an empty dict.
    weights_dict = memory_store.get_latest_weights() or {}
    consensus_weights = {
        "consensus_weights": {
            "tsmom": weights_dict.get("tsmom_weight", 0.40),
            "vol_trend": weights_dict.get("vol_trend_weight", 0.35),
            "pairs_arb": weights_dict.get("pairs_arb_weight", 0.25),
        }
    }
    consensus_engine = ConsensusEngine(config, consensus_weights)

    # ── 5. Risk components (no deps) ─────────────────────────────────
    position_sizer = PositionSizer()
    exit_manager = ExitManager()

    # ── 6. Experience agent (depends on memory_store + consensus) ────
    experience_agent = ExperienceAgent(config, memory_store, consensus_engine)

    # ── 7. Execution (depends on position_sizer) ────────────────────
    alpaca_client = AlpacaClient(position_sizer=position_sizer)

    components = {
        "memory_store": memory_store,
        "alert_manager": alert_manager,
        "data_agent": data_agent,
        "tsmom": tsmom,
        "dual_momentum": dual_momentum,
        "vol_trend": vol_trend,
        "pairs_arb": pairs_arb,
        "regime_detector": regime_detector,
        "consensus_engine": consensus_engine,
        "position_sizer": position_sizer,
        "exit_manager": exit_manager,
        "experience_agent": experience_agent,
        "alpaca_client": alpaca_client,
    }

    log.info("System initialized  |  {} components wired", len(components))

    # ── Paper Trading Warmup Gate ────────────────────────────────────
    paper_days = memory_store.get_paper_trading_days()
    is_paper = not config.USE_LIVE_EXECUTION

    if is_paper and paper_days == 0:
        memory_store.set_paper_trading_start(date.today())
        log.info("Paper trading warmup started. Day 1 of {}.", config.WARMUP_DAYS)
        try:
            alert_manager.system_started("PAPER_WARMUP_DAY_1", config.STARTING_CAPITAL)
        except Exception:
            pass  # Alert manager may not have system_started method yet
    elif not is_paper:  # Live mode
        if paper_days < config.WARMUP_DAYS:
            raise RuntimeError(
                f"SAFETY BLOCK: Only {paper_days} days of paper trading completed. "
                f"Minimum {config.WARMUP_DAYS} days required before live trading. "
                f"Set mode=paper to continue warmup."
            )
        else:
            log.info(
                "Paper warmup complete ({} days). Live trading enabled.",
                paper_days,
            )
    else:
        log.info("Paper trading warmup: day {} of {}", paper_days, config.WARMUP_DAYS)

    return components


# ─────────────────────────────────────────────────────────────────────────────
# 3. DAILY TRADING CYCLE
# ─────────────────────────────────────────────────────────────────────────────

def run_daily_cycle(components: dict) -> None:
    """
    Execute one complete daily trading cycle.

    Steps:
        1. Fetch latest data for all symbols
        2. Run regime detection
        3. Determine kill switch status
        4. Process exits (exits BEFORE entries)
        5. Compute strategy signals (TSMOM, VolTrend, PairsArb, DualMom filter)
        6. Aggregate signals via consensus engine
        7. Size positions
        8. Execute orders (kill-switch gated)
        9. Log daily summary
    """
    data_agent: DataAgent = components["data_agent"]
    tsmom: TSMOMStrategy = components["tsmom"]
    dual_momentum: DualMomentumFilter = components["dual_momentum"]
    vol_trend: VolTrendStrategy = components["vol_trend"]
    pairs_arb: PairsArbStrategy = components["pairs_arb"]
    regime_detector: RegimeDetector = components["regime_detector"]
    consensus_engine: ConsensusEngine = components["consensus_engine"]
    position_sizer: PositionSizer = components["position_sizer"]
    exit_manager: ExitManager = components["exit_manager"]
    alpaca_client: AlpacaClient = components["alpaca_client"]
    alert_manager: AlertManager = components["alert_manager"]

    cycle_start = time.time()
    today = pd.Timestamp(datetime.now().strftime("%Y-%m-%d"))

    log.info("─" * 60)
    log.info("  DAILY CYCLE START  |  {}", today.strftime("%Y-%m-%d"))
    log.info("─" * 60)

    try:
        # ── 1. Fetch data ────────────────────────────────────────────
        log.info("Step 1/9: Fetching data for {} symbols",
                 len(config.ASSET_UNIVERSE))
        historical_data = data_agent.get_universe_data(
            symbols=config.ASSET_UNIVERSE,
            start=config.BACKTEST_START,
            end=today.strftime("%Y-%m-%d"),
        )

        if not historical_data:
            log.warning("No data returned — aborting cycle")
            return

        log.info("Received data for {} symbols", len(historical_data))

        # ── 2. Regime detection ──────────────────────────────────────
        log.info("Step 2/9: Running regime detection")
        regime = regime_detector.detect(historical_data, today)
        log.info("Market regime: {}", regime)

        # ── 3. Kill switch check ─────────────────────────────────────
        log.info("Step 3/9: Checking kill switch status")
        account = alpaca_client.get_account()
        current_equity = account.get("equity", config.STARTING_CAPITAL)
        peak_equity = account.get("peak_equity", current_equity)

        # Compute drawdown
        if peak_equity > 0:
            drawdown = (peak_equity - current_equity) / peak_equity
        else:
            drawdown = 0.0

        # Determine kill switch tier
        if drawdown > config.MAX_DRAWDOWN_PCT:
            kill_switch_status = "FULL_HALT"
        elif drawdown > config.MAX_DRAWDOWN_PCT * 0.7:
            kill_switch_status = "STOP_NEW"
        elif drawdown > config.MAX_DRAWDOWN_PCT * 0.5:
            kill_switch_status = "REDUCE_50"
        else:
            kill_switch_status = "OK"

        # ── KILL SWITCH TIER 3: FULL HALT ────────────────────────
        if kill_switch_status == "FULL_HALT":
            log.critical(
                "KILL SWITCH TIER 3 — FULL HALT  |  "
                "drawdown={:.2%}  equity=${:.2f}",
                drawdown, current_equity,
            )
            alert_manager.kill_switch_fired("TIER_3", drawdown, current_equity)
            _log_cycle_summary(cycle_start, 0, current_equity,
                               drawdown, kill_switch_status)
            return

        # ── KILL SWITCH TIER 2: STOP NEW ENTRIES ─────────────────
        skip_new_entries = False
        if kill_switch_status == "STOP_NEW":
            log.warning(
                "KILL SWITCH TIER 2 — STOP NEW  |  "
                "drawdown={:.2%}  equity=${:.2f}  — exits only",
                drawdown, current_equity,
            )
            alert_manager.kill_switch_fired("TIER_2", drawdown, current_equity)
            skip_new_entries = True

        # ── KILL SWITCH TIER 1: REDUCE 50% ──────────────────────
        size_multiplier = 1.0
        if kill_switch_status == "REDUCE_50":
            log.warning(
                "KILL SWITCH TIER 1 — REDUCE 50%%  |  "
                "drawdown={:.2%}  equity=${:.2f}",
                drawdown, current_equity,
            )
            size_multiplier = 0.5

        # ── 4. Process exits ─────────────────────────────────────────
        log.info("Step 4/9: Checking exit conditions")
        positions = alpaca_client.get_positions()

        if positions:
            exit_results = exit_manager.check_all_exits(
                positions=positions,
                price_data=historical_data,
                current_signals={},  # Will be filled after signal gen
                current_date=today.strftime("%Y-%m-%d"),
                kill_switch_status=kill_switch_status,
            )

            for symbol, (should_exit, reason, exit_price) in exit_results.items():
                if should_exit:
                    log.info("EXIT: {} — reason={} — price=${:.2f}",
                             symbol, reason, exit_price)
                    if config.USE_LIVE_EXECUTION:
                        alpaca_client.execute_signal(
                            symbol=symbol,
                            dollar_size=-abs(
                                positions[symbol].get("market_value", 0)
                            ),
                            signal=-1.0,  # Sell signal
                            kill_switch_status="OK",  # Always allow exits
                        )

        # ── 5. Generate strategy signals ─────────────────────────────
        log.info("Step 5/9: Computing strategy signals")

        tsmom_signals = tsmom.compute_signals(historical_data, today)
        dual_mom_filter = dual_momentum.compute_filter(historical_data, today)
        vol_trend_signals = vol_trend.compute_signals(historical_data, today)

        # PairsArb uses a different interface (returns list[Signal])
        # Build a regime dict for PairsArb using the overall regime
        from core.regime_detector import Regime
        symbol_regimes = {
            sym: Regime[regime] if isinstance(regime, str) else regime
            for sym in config.ASSET_UNIVERSE
        }
        pairs_signals_list = pairs_arb.generate_signals(
            historical_data, symbol_regimes,
        )
        # Convert list[Signal] → dict[str, float] for consensus engine
        pairs_signals = {
            sig.symbol: sig.strength * (1.0 if sig.direction == "long" else -1.0)
            for sig in pairs_signals_list
        }

        # ── 6. Aggregate via consensus engine ────────────────────────
        log.info("Step 6/9: Running consensus aggregation")
        final_signals = consensus_engine.aggregate(
            tsmom_signals=tsmom_signals,
            vol_trend_signals=vol_trend_signals,
            pairs_signals=pairs_signals,
            dual_mom_filter=dual_mom_filter,
            regime=symbol_regimes,
        )

        if not final_signals or skip_new_entries:
            msg = "no signals" if not final_signals else "skip_new_entries active"
            log.info("No new entries — {} — cycle complete", msg)
            _log_cycle_summary(cycle_start, 0, current_equity,
                               drawdown, kill_switch_status)
            return

        # ── 7. Size positions ────────────────────────────────────────
        log.info("Step 7/9: Sizing positions")
        portfolio_state = {
            "cash": account.get("cash", config.STARTING_CAPITAL),
            "positions": positions,
            "equity": current_equity,
            "peak_equity": peak_equity,
        }
        sizes = position_sizer.compute(
            signals=final_signals,
            price_data=historical_data,
            portfolio=portfolio_state,
            current_date=today,
        )

        # Apply kill switch size reduction
        if size_multiplier < 1.0:
            sizes = {sym: sz * size_multiplier for sym, sz in sizes.items()}
            log.info("Position sizes halved due to kill switch tier 1")

        # ── 8. Execute orders ────────────────────────────────────────
        log.info("Step 8/9: Executing {} signals", len(sizes))
        trades_executed = 0

        for symbol, dollar_size in sizes.items():
            if dollar_size == 0:
                continue

            signal_val = final_signals.get(symbol, 0.0)

            if config.USE_LIVE_EXECUTION:
                try:
                    result = alpaca_client.execute_signal(
                        symbol=symbol,
                        dollar_size=abs(dollar_size),
                        signal=signal_val,
                        kill_switch_status=kill_switch_status,
                    )
                    log.info(
                        "Order: {} {} ${:.2f} → {}",
                        "BUY" if signal_val > 0 else "SELL",
                        symbol, abs(dollar_size),
                        result.get("status", "unknown"),
                    )
                    trades_executed += 1
                except Exception as exc:
                    log.error("Order failed for {}: {}", symbol, exc)
            else:
                log.info(
                    "[DRY RUN] Would {} {} ${:.2f}",
                    "BUY" if signal_val > 0 else "SELL",
                    symbol, abs(dollar_size),
                )
                trades_executed += 1

        # ── 9. Summary ───────────────────────────────────────────────
        log.info("Step 9/9: Daily summary")
        _log_cycle_summary(
            cycle_start, trades_executed, current_equity,
            drawdown, kill_switch_status,
        )

    except Exception as exc:
        log.critical("DAILY CYCLE FAILED: {}", exc)
        try:
            alert_manager = components.get("alert_manager")
            if alert_manager:
                alert_manager.api_failure("daily_cycle", str(exc))
        except Exception:
            pass
        raise


def _log_cycle_summary(
    cycle_start: float,
    trades_executed: int,
    equity: float,
    drawdown: float,
    kill_switch: str,
) -> None:
    """Log end-of-cycle summary."""
    elapsed = time.time() - cycle_start

    log.info("─" * 60)
    log.info("  CYCLE COMPLETE  |  {:.1f}s elapsed", elapsed)
    log.info("  Equity: ${:.2f}  |  Drawdown: {:.2%}", equity, drawdown)
    log.info("  Trades: {}  |  Kill switch: {}", trades_executed, kill_switch)
    log.info("─" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# 4. SCHEDULER (Paper / Live modes)
# ─────────────────────────────────────────────────────────────────────────────

def start_scheduler(components: dict) -> None:
    """
    Schedule the daily trading cycle using APScheduler + NYSE calendar.

    Only fires on NYSE trading days, 35 minutes after market open.
    Automatically skips holidays, early closes, and weekends.
    """
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger
    from datetime import timedelta

    try:
        import pandas_market_calendars as mcal
        nyse = mcal.get_calendar("NYSE")
        has_calendar = True
        log.info("NYSE calendar loaded — holidays and early closes will be respected")
    except ImportError:
        has_calendar = False
        log.warning(
            "pandas_market_calendars not installed — "
            "using simple weekday schedule (install with: pip install pandas-market-calendars)"
        )

    scheduler = BlockingScheduler(timezone=config.TIMEZONE)

    def _nyse_aware_cycle():
        """Run the daily cycle only if today is an NYSE trading day."""
        from datetime import date

        today = date.today()

        if has_calendar:
            schedule = nyse.schedule(
                start_date=today.strftime("%Y-%m-%d"),
                end_date=today.strftime("%Y-%m-%d"),
            )
            if schedule.empty:
                log.info(
                    "NYSE holiday / non-trading day ({}) — skipping cycle",
                    today.strftime("%Y-%m-%d (%A)"),
                )
                return

            market_open = schedule.iloc[0]["market_open"]
            log.info(
                "NYSE trading day confirmed  |  market_open={}",
                market_open.strftime("%H:%M %Z"),
            )

        run_daily_cycle(components)

    # Schedule: Mon-Fri, 35 minutes after NYSE open (typically 09:30 → 10:05)
    open_hour, open_min = map(int, config.MARKET_OPEN.split(":"))
    run_min = open_min + 35
    run_hour = open_hour
    if run_min >= 60:
        run_hour += run_min // 60
        run_min = run_min % 60

    scheduler.add_job(
        _nyse_aware_cycle,
        trigger=CronTrigger(
            day_of_week="mon-fri",
            hour=run_hour,
            minute=run_min,
            timezone=config.TIMEZONE,
        ),
        args=[],
        id="daily_cycle",
        name="Daily Trading Cycle (NYSE-aware)",
        misfire_grace_time=300,
    )

    # Log next 5 trading days
    if has_calendar:
        upcoming = nyse.schedule(
            start_date=datetime.now().strftime("%Y-%m-%d"),
            end_date=(datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d"),
        )
        if not upcoming.empty:
            next_days = upcoming.head(5).index.strftime("%Y-%m-%d (%a)").tolist()
            log.info("Next 5 trading days: {}", ", ".join(next_days))

    # ── EOD cleanup: cancel stale orders + snapshot ───────────────────
    def _eod_cleanup():
        """End-of-day: cancel open orders, snapshot portfolio, rotate idle capital."""
        from datetime import date as _date
        try:
            from utils.market_calendar import MarketCalendar
            if not MarketCalendar().is_trading_day(_date.today()):
                return
        except ImportError:
            pass  # If no calendar, run anyway

        alpaca = components["alpaca_client"]
        ms = components["memory_store"]

        # Cancel open day orders
        cancelled = alpaca.cancel_all_day_orders()
        log.info("EOD cleanup: cancelled {} open day orders", cancelled)

        # Snapshot portfolio
        try:
            account = alpaca.get_account()
            positions = alpaca.get_positions()
            portfolio = {
                "equity": float(account.get("equity", 0)),
                "cash": float(account.get("cash", 0)),
                "positions": positions,
                "daily_pnl": 0,
                "drawdown_pct": 0,
            }
            ms.write_eod_snapshot(portfolio)

            # Rotate idle capital into SHV
            alpaca.rotate_idle_capital(portfolio)
        except Exception as e:
            log.error("EOD snapshot/rotation failed: {}", e)

    # Schedule EOD at 16:05 Eastern
    scheduler.add_job(
        _eod_cleanup,
        trigger=CronTrigger(
            day_of_week="mon-fri",
            hour=16,
            minute=5,
            timezone=config.TIMEZONE,
        ),
        args=[],
        id="eod_cleanup",
        name="EOD Cleanup (cancel orders + snapshot)",
        misfire_grace_time=300,
    )

    log.info(
        "Scheduler started  |  daily cycle at {:02d}:{:02d} + EOD at 16:05 {} (NYSE trading days only)",
        run_hour, run_min, config.TIMEZONE,
    )
    log.info("Press Ctrl+C to stop")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("Scheduler stopped by user")
        scheduler.shutdown()


# ─────────────────────────────────────────────────────────────────────────────
# 5. CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Parse CLI arguments and run the appropriate mode."""
    parser = argparse.ArgumentParser(
        description="Autonomous Trading Algorithm — Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  backtest   Run historical backtest
  paper      Paper trading with scheduled daily cycles
  live       Live trading with real order execution

Examples:
  python main.py --mode paper
  python main.py --mode backtest
  python main.py --mode live
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["backtest", "paper", "live"],
        default="paper",
        help="Trading mode (default: paper)",
    )
    parser.add_argument(
        "--run-once",
        action="store_true",
        help="Run a single cycle instead of scheduling (paper/live modes)",
    )

    args = parser.parse_args()

    log.info("═" * 60)
    log.info("  AUTONOMOUS TRADING ALGORITHM")
    log.info("  Mode: {}  |  Started: {}", args.mode,
             datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log.info("═" * 60)

    # ── Backtest mode ────────────────────────────────────────────────
    if args.mode == "backtest":
        log.info("Backtest mode selected")
        from backtest.engine import BacktestEngine
        from backtest.metrics import MetricsCalculator

        engine = BacktestEngine()
        result = engine.run(
            start_date=config.BACKTEST_START,
            end_date=config.BACKTEST_END,
            initial_capital=config.STARTING_CAPITAL,
        )
        metrics = MetricsCalculator(result)
        metrics.print_report()
        return

    # ── Paper / Live modes ───────────────────────────────────────────
    components = initialize_system(args.mode)

    if args.run_once:
        log.info("Running single cycle (--run-once)")
        run_daily_cycle(components)
    else:
        start_scheduler(components)


if __name__ == "__main__":
    main()
