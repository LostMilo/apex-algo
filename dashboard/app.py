"""
dashboard/app.py — Flask Web Dashboard

Serves the trading dashboard with pages for:
  - Overview: account equity, positions, P&L
  - Strategies: regime status, active strategies, signals
  - Trade Log: list of all executed trades
  - Backtest: equity curve and metrics from backtest results

Usage:
  python dashboard/app.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, render_template

import config
from logger import log

app = Flask(__name__)


# ── Helpers ──────────────────────────────────────────────────────────

def _get_account_data() -> dict:
    """Get account data — tries broker, falls back to defaults."""
    try:
        from broker import AlpacaBroker
        broker = AlpacaBroker()
        account = broker.get_account()
        positions = broker.get_positions()

        from portfolio import Portfolio
        portfolio = Portfolio(broker=broker)
        portfolio.sync_with_broker()
        summary = portfolio.get_summary()

        account["daily_pnl"] = summary.get("daily_pnl", 0)
        account["drawdown"] = summary.get("drawdown", 0)
        account["exposure"] = summary.get("total_exposure", 0)
        account["total_trades"] = summary.get("total_trades", 0)

        return account, positions
    except Exception as e:
        log.debug("Broker not available: {}", str(e))
        return {
            "equity": config.STARTING_CAPITAL,
            "cash": config.STARTING_CAPITAL,
            "buying_power": config.STARTING_CAPITAL,
            "portfolio_value": config.STARTING_CAPITAL,
            "daily_pnl": 0,
            "drawdown": 0,
            "exposure": 0,
            "total_trades": 0,
        }, []


def _get_trade_history() -> list:
    """Load trade history from journal."""
    journal_file = Path("data/trade_journal.json")
    if journal_file.exists():
        try:
            with open(journal_file) as f:
                return json.load(f)
        except Exception:
            pass
    return []


def _get_backtest_results() -> tuple:
    """Load most recent backtest results."""
    results_dir = Path("data/backtest_results")
    if not results_dir.exists():
        return {}, []

    # Find most recent metrics file
    metrics_files = sorted(results_dir.glob("metrics_*.json"), reverse=True)
    if not metrics_files:
        return {}, []

    try:
        with open(metrics_files[0]) as f:
            metrics = json.load(f)

        # Find corresponding equity curve
        timestamp = metrics_files[0].stem.replace("metrics_", "")
        curve_file = results_dir / f"equity_curve_{timestamp}.json"

        equity_curve = []
        if curve_file.exists():
            with open(curve_file) as f:
                equity_curve = json.load(f)

        return metrics, equity_curve
    except Exception:
        return {}, []


def _get_regime_data() -> dict:
    """Try to get current regimes."""
    try:
        from data_manager import fetch_all
        from regime_detector import detect_all_regimes
        data = fetch_all(force_refresh=False)
        if data:
            regimes = detect_all_regimes(data)
            return {sym: r.value for sym, r in regimes.items()}
    except Exception as e:
        log.debug("Cannot load regimes: {}", str(e))

    # Default
    return {sym: "UNKNOWN" for sym in config.ASSET_UNIVERSE}


# ── Routes ───────────────────────────────────────────────────────────

@app.route("/")
def overview():
    account, positions = _get_account_data()
    return render_template(
        "overview.html",
        page="overview",
        account=account,
        positions=positions,
        last_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )


@app.route("/strategies")
def strategies():
    regimes = _get_regime_data()

    strategy_info = [
        {
            "name": "TSMOM",
            "description": "Time Series Momentum (Moskowitz 2012) — 252-day rolling return",
            "regime": "TRENDING",
            "active": any(r == "TRENDING" for r in regimes.values()),
            "signal_count": 0,
        },
        {
            "name": "Dual Momentum",
            "description": "Antonacci confirmation filter — absolute + relative momentum gate",
            "regime": "ALL",
            "active": True,
            "signal_count": 0,
        },
        {
            "name": "Vol Trend",
            "description": "EMA(20)/EMA(50) crossover with ADX(14) filter [25-40]",
            "regime": "TRENDING",
            "active": any(r == "TRENDING" for r in regimes.values()),
            "signal_count": 0,
        },
        {
            "name": "Pairs Arb",
            "description": "Engle-Granger cointegration → z-score mean reversion",
            "regime": "RANGING",
            "active": any(r == "RANGING" for r in regimes.values()),
            "signal_count": 0,
        },
    ]

    return render_template(
        "strategies.html",
        page="strategies",
        regimes=regimes,
        strategies=strategy_info,
        signals=[],
    )


@app.route("/trades")
def trades():
    trade_history = _get_trade_history()

    # Calculate stats
    closing_trades = [t for t in trade_history if "pnl" in t]
    winning = [t for t in closing_trades if t.get("pnl", 0) > 0]
    total_pnl = sum(t.get("pnl", 0) for t in closing_trades)
    win_rate = len(winning) / len(closing_trades) if closing_trades else 0
    avg_pnl = total_pnl / len(closing_trades) if closing_trades else 0

    return render_template(
        "trades.html",
        page="trades",
        trades=trade_history,
        total_pnl=total_pnl,
        win_rate=win_rate,
        avg_pnl=avg_pnl,
    )


@app.route("/backtest")
def backtest():
    metrics, equity_curve = _get_backtest_results()

    return render_template(
        "backtest.html",
        page="backtest",
        metrics=metrics,
        equity_curve=equity_curve,
    )


# ── Entry Point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("Starting dashboard on {}:{}", config.DASHBOARD_HOST, config.DASHBOARD_PORT)
    app.run(
        host=config.DASHBOARD_HOST,
        port=config.DASHBOARD_PORT,
        debug=False,
    )
