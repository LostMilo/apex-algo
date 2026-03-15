"""
live_runner.py — Live Autonomous Trading Runner

Main entry point for the autonomous trading system.
Uses APScheduler for daily execution at 15:30 ET (30 min before close).

Daily cycle:
  1. Refresh data cache
  2. Detect regimes
  3. Generate strategy signals
  4. Aggregate and filter signals
  5. Risk-check proposed trades
  6. Submit LIMIT orders to Alpaca
  7. Log everything

Usage:
  python live_runner.py
"""

import signal
import sys
from datetime import datetime

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

import config
from logger import log
from broker import AlpacaBroker
from data_manager import fetch_all, get_latest_prices
from regime_detector import detect_all_regimes
from signal_aggregator import SignalAggregator
from risk_manager import RiskManager
from portfolio import Portfolio


class LiveRunner:
    """
    Autonomous trading runner.
    Executes the full pipeline daily and manages the scheduler.
    """

    def __init__(self):
        log.info("═══ INITIALIZING LIVE RUNNER ═══")

        # Core components
        self.broker = AlpacaBroker()
        self.portfolio = Portfolio(broker=self.broker)
        self.aggregator = SignalAggregator()
        self.risk_manager = RiskManager()

        # Scheduler
        self.scheduler = BlockingScheduler(timezone=config.TIMEZONE)

        # Sync initial state
        self.portfolio.sync_with_broker()
        self.risk_manager._peak_equity = self.portfolio.peak_equity

        log.info("Live runner initialized. Equity: ${:.2f}", self.portfolio.equity)

    def run_daily_cycle(self):
        """
        Execute the full daily trading cycle.
        This is called by the scheduler at 15:30 ET each trading day.
        """
        cycle_start = datetime.now()
        log.info("═══ DAILY CYCLE START — {} ═══", cycle_start.strftime("%Y-%m-%d %H:%M"))

        try:
            # Step 0: Sync with broker
            log.info("── Step 0: Sync with broker ──")
            self.portfolio.sync_with_broker()
            equity = self.portfolio.equity

            # Step 1: Circuit breaker check
            log.info("── Step 1: Circuit breaker check ──")
            if not self.risk_manager.check_circuit_breaker(equity):
                log.warning("Circuit breaker active — skipping today")
                return

            # Check daily loss limit
            daily_pnl = self.portfolio.get_daily_pnl()
            if not self.risk_manager.check_daily_loss_limit(daily_pnl, equity):
                log.warning("Daily loss limit reached — skipping today")
                return

            # Step 2: Refresh data
            log.info("── Step 2: Refresh data ──")
            data = fetch_all(force_refresh=True)
            if not data:
                log.error("No data available — aborting cycle")
                return

            # Step 3: Detect regimes
            log.info("── Step 3: Detect regimes ──")
            regimes = detect_all_regimes(data)

            # Step 4: Generate & aggregate signals
            log.info("── Step 4: Generate signals ──")
            signals = self.aggregator.generate_signals(data, regimes)

            if not signals:
                log.info("No signals generated — no trades today")
                self._log_status("no_signals")
                return

            # Step 5: Risk management & order preparation
            log.info("── Step 5: Risk management ──")
            latest_prices = get_latest_prices(data)
            current_positions = self.portfolio.get_current_symbols()
            orders = []

            for signal in signals:
                sym = signal.symbol

                # Correlation check
                if not self.risk_manager.check_correlation(
                    sym, data, current_positions
                ):
                    log.info("{}: blocked by correlation check", sym)
                    continue

                # Position sizing
                if sym not in data or sym not in latest_prices:
                    continue

                sizing = self.risk_manager.size_position(
                    signal=signal,
                    df=data[sym],
                    equity=equity,
                    current_price=latest_prices[sym],
                )

                if sizing["qty"] <= 0:
                    continue

                orders.append({
                    "symbol": sym,
                    "side": "buy" if signal.direction == "long" else "sell",
                    "qty": sizing["qty"],
                    "limit_price": sizing["limit_price"],
                    "stop_price": sizing["stop_price"],
                    "strategy": signal.strategy,
                    "strength": signal.strength,
                })

            # Step 6: Execute orders
            log.info("── Step 6: Execute {} orders ──", len(orders))
            for order in orders:
                try:
                    result = self.broker.submit_limit_order(
                        symbol=order["symbol"],
                        qty=order["qty"],
                        side=order["side"],
                        limit_price=order["limit_price"],
                    )

                    # Record trade
                    order["order_id"] = result.get("id", "")
                    order["status"] = result.get("status", "unknown")
                    self.portfolio.record_trade(order)

                    log.info(
                        "Order submitted: {} {} x{} @ ${:.2f} — status: {}",
                        order["side"].upper(), order["symbol"],
                        order["qty"], order["limit_price"],
                        order["status"],
                    )

                except Exception as e:
                    log.error("Failed to submit order for {}: {}", order["symbol"], str(e))

            # Step 7: Log cycle summary
            self._log_status("completed", orders)

        except Exception as e:
            log.critical("DAILY CYCLE FAILED: {}", str(e))
            self._log_status("error")

        finally:
            elapsed = (datetime.now() - cycle_start).total_seconds()
            log.info("═══ DAILY CYCLE END — {:.1f}s ═══", elapsed)

    def _log_status(self, status: str, orders: list = None):
        """Log cycle status summary."""
        summary = self.portfolio.get_summary()
        log.info(
            "Cycle status: {} | equity=${:.2f} | positions={} | trades_today={}",
            status,
            summary["equity"],
            summary["num_positions"],
            len(orders) if orders else 0,
        )

    def start(self):
        """Start the scheduler — runs forever until killed."""
        log.info("Starting scheduler...")

        # Schedule daily run at 15:30 ET, Monday-Friday
        self.scheduler.add_job(
            self.run_daily_cycle,
            CronTrigger(
                day_of_week="mon-fri",
                hour=15,
                minute=30,
                timezone=config.TIMEZONE,
            ),
            id="daily_trading_cycle",
            name="Daily Trading Cycle",
            misfire_grace_time=3600,  # Allow 1-hour grace period
        )

        log.info(
            "Scheduled: Daily trading at 15:30 ET (Mon-Fri) in timezone {}",
            config.TIMEZONE,
        )

        # Register shutdown handler
        signal.signal(signal.SIGTERM, self._shutdown)
        signal.signal(signal.SIGINT, self._shutdown)

        try:
            self.scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            log.info("Scheduler stopped by user")

    def _shutdown(self, signum, frame):
        """Graceful shutdown handler."""
        log.info("Received signal {} — shutting down gracefully...", signum)
        self.scheduler.shutdown(wait=False)
        log.info("Shutdown complete")
        sys.exit(0)


# ── CLI Entry Point ──────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Autonomous Trading Runner")
    parser.add_argument(
        "--once", action="store_true",
        help="Run a single cycle immediately without scheduling",
    )
    args = parser.parse_args()

    runner = LiveRunner()

    if args.once:
        log.info("Running single cycle (--once mode)")
        runner.run_daily_cycle()
    else:
        runner.start()


if __name__ == "__main__":
    main()
