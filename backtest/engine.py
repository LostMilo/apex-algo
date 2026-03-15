"""
backtest/engine.py — Canonical Import Path for the Backtest Engine

This is the standard import location for the backtesting engine.
The actual implementation lives in the root engine.py module.

Usage:
    from backtest.engine import BacktestEngine
    engine = BacktestEngine()
    result = engine.run()
"""

import sys
from pathlib import Path

# Ensure project root is importable
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Re-export from root engine.py
from engine import BacktestEngine, BacktestResult  # noqa: E402, F401

__all__ = ["BacktestEngine", "BacktestResult"]
