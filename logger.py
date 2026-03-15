"""
logger.py — Structured Logging for Autonomous Trading Algorithm

Uses loguru for structured, rotated logging to files and stdout.
All other modules import `log` from here.
"""

import sys
from pathlib import Path
from loguru import logger

import config

# ── Remove default handler ──
logger.remove()

# ── Stdout handler — human-readable ──
logger.add(
    sys.stdout,
    level=config.LOG_LEVEL,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    ),
    colorize=True,
)

# ── File handlers — with graceful fallback ──
def _setup_file_logging():
    """Set up file logging, falling back to /tmp if primary dir is not writable."""
    log_dir = Path(config.LOG_DIR)

    # Try primary log directory, fallback to /tmp/algo_logs
    for candidate in [log_dir, Path("/tmp/algo_logs")]:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            # Test write permission
            test_file = candidate / ".write_test"
            test_file.touch()
            test_file.unlink()

            # System log file — rotates daily, keeps 30 days
            logger.add(
                str(candidate / "system_{time:YYYY-MM-DD}.log"),
                level=config.LOG_LEVEL,
                format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
                rotation="00:00",
                retention="30 days",
                compression="gz",
                enqueue=True,
            )

            # Trade log file — separate file for trade events
            logger.add(
                str(candidate / "trades_{time:YYYY-MM-DD}.log"),
                level="INFO",
                format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {message}",
                rotation="00:00",
                retention="90 days",
                compression="gz",
                filter=lambda record: record["extra"].get("trade", False),
                enqueue=True,
            )

            logger.info("Log files writing to: {}", candidate)
            return

        except (PermissionError, OSError):
            continue

    # If all directories fail, only stdout logging is available
    logger.warning("File logging unavailable — using stdout only")


_setup_file_logging()

# ── Convenience aliases ──
log = logger
trade_log = logger.bind(trade=True)
