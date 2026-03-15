"""
risk/ — Risk Management Package

Modules:
  - exits: Exit management (Chandelier, Hard Stop, Time Stop, Signal Reversal, Kill Switch)
"""

from risk.exits import (
    ExitManager,
    EXIT_CHANDELIER,
    EXIT_HARD_STOP,
    EXIT_TIME_STOP,
    EXIT_SIGNAL_REVERSAL,
    EXIT_KILL_SWITCH,
)

__all__ = [
    "ExitManager",
    "EXIT_CHANDELIER",
    "EXIT_HARD_STOP",
    "EXIT_TIME_STOP",
    "EXIT_SIGNAL_REVERSAL",
    "EXIT_KILL_SWITCH",
]
