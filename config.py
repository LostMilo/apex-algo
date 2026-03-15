"""
config.py — Master Configuration for Autonomous Trading Algorithm

Rules:
- Every parameter used anywhere in the system lives HERE.
- No other file ever hardcodes a value — they all import from config.
- This file never imports from any other project file.
- Parameters are grouped into logical sections with clear comments.
"""

import os


try:
    from dotenv import load_dotenv
    # Load environment variables from .env file
    load_dotenv()
except ImportError:
    # python-dotenv not installed yet — secrets must be set as
    # real environment variables until `pip install python-dotenv`
    pass

# ──────────────────────────────────────────────────────────────────────
# 1. ENVIRONMENT — Load secrets from .env (never hardcode keys)
# ──────────────────────────────────────────────────────────────────────

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL = os.getenv(
    "ALPACA_BASE_URL", "https://paper-api.alpaca.markets"
)

ALERT_EMAIL = os.getenv("ALERT_EMAIL", "")
ALERT_EMAIL_CC = os.getenv("ALERT_EMAIL_CC", "")
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")

# ──────────────────────────────────────────────────────────────────────
# 2. TRADING MODE FLAGS
# ──────────────────────────────────────────────────────────────────────
USE_LIVE_DATA = False          # Must be explicitly set True for live
PAPER_TRADING = True
USE_ALT_DATA = False           # Enable when alt data sources connected
USE_LIVE_EXECUTION = False

# ──────────────────────────────────────────────────────────────────────
# 3. ASSET UNIVERSE
# ──────────────────────────────────────────────────────────────────────
ASSET_UNIVERSE = [
    "SPY", "QQQ", "IWM", "GLD", "TLT",
    "XLE", "XLF", "XLK", "XLV", "XLI",
]
BENCHMARK = "SPY"
RISK_FREE_PROXY = "SHV"

# ──────────────────────────────────────────────────────────────────────
# 4. TIMEFRAME AND DATA
# ──────────────────────────────────────────────────────────────────────
TIMEFRAME = "1Day"
BACKTEST_START = "2005-01-01"
BACKTEST_END = "today"
DATA_CACHE_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
MIN_HISTORY_DAYS = 300

# ──────────────────────────────────────────────────────────────────────
# 5. STRATEGY PARAMETERS
# ──────────────────────────────────────────────────────────────────────

# TSMOM — Time Series Momentum (Moskowitz 2012)
TSMOM_LOOKBACK = 252           # 12-month rolling return window
TSMOM_SKIP_LAST = 21           # Skip most recent month (reduce noise)

# Dual Momentum — Confirmation filter (Antonacci)
DUAL_MOM_LOOKBACK = 252

# Vol Trend — Volatility-adjusted trend following
EMA_FAST = 20
EMA_SLOW = 50
ADX_PERIOD = 14
ADX_MIN = 25
ADX_MAX = 40

# General volume confirmation
VOLUME_MA_PERIOD = 20
VOLUME_CONFIRM_MULTIPLIER = 0.8

# ──────────────────────────────────────────────────────────────────────
# 6. REGIME DETECTION
# ──────────────────────────────────────────────────────────────────────
GARCH_LOOKBACK = 252
REGIME_VOL_HIGH_THRESHOLD = 0.20
REGIME_ADX_TREND_MIN = 25
MA_TREND_PERIOD = 200

# ──────────────────────────────────────────────────────────────────────
# 7. RISK MANAGEMENT
# ──────────────────────────────────────────────────────────────────────
STARTING_CAPITAL = 1000        # EUR (~1,100 USD equivalent)
MAX_POSITION_PCT = 0.20        # 20% max per position
MAX_PORTFOLIO_RISK = 0.02      # 2% total portfolio risk
STOP_LOSS_ATR_MULT = 2.0       # Stop-loss at 2× ATR
TRAILING_STOP_ATR_MULT = 1.5
MAX_DRAWDOWN_PCT = 0.15        # 15% max drawdown halt
DAILY_LOSS_LIMIT = 0.03        # 3% daily loss limit
MAX_CORRELATED_POSITIONS = 3   # Max positions in correlated assets
ATR_PERIOD = 14

# Exit-specific parameters
CHANDELIER_ATR_PERIOD = 22     # ATR lookback for Chandelier exit
CHANDELIER_ATR_MULT = 3.0      # Chandelier stop distance multiplier
HARD_STOP_PCT = 0.07           # 7% hard stop-loss from entry price
TIME_STOP_DAYS = 30            # Close dead-money positions after N days
TIME_STOP_MIN_PNL = -0.02      # Min P&L band for time stop (-2%)
TIME_STOP_MAX_PNL = 0.02       # Max P&L band for time stop (+2%)

# Position Sizing — Kelly Criterion
KELLY_FRACTION = 0.50          # Half-Kelly for conservative sizing
KELLY_CAP = 0.25               # Max 25% Kelly allocation per position
MAX_TOTAL_EXPOSURE = 1.0       # Sum of long positions ≤ 100% equity

# ──────────────────────────────────────────────────────────────────────
# 8. EXECUTION RULES
# ──────────────────────────────────────────────────────────────────────
ORDER_TYPE = "limit"           # Always LIMIT, never market
TIME_IN_FORCE = "day"
MAX_SLIPPAGE_PCT = 0.001       # 0.1% max slippage
MIN_ORDER_VALUE = 1.0          # Minimum order in USD
REBALANCE_FREQUENCY = "weekly" # How often to rebalance

# Idle capital rotation
IDLE_CAPITAL_ETF = "SHV"       # iShares Short Treasury Bond ETF
IDLE_CAPITAL_MIN_PCT = 0.10    # Park idle cash when > 10% of equity

# Paper trading warmup
WARMUP_DAYS = 90               # Days of paper trading required before live

# ──────────────────────────────────────────────────────────────────────
# 9. COINTEGRATION / STAT-ARB (Pairs Arbitrage)
# ──────────────────────────────────────────────────────────────────────
COINT_LOOKBACK = 252           # Lookback for Engle-Granger test
COINT_PVALUE = 0.05            # p-value threshold
ZSCORE_ENTRY = 2.0             # Enter when z-score exceeds ±2
ZSCORE_EXIT = 0.5              # Exit when z-score reverts to ±0.5
COINT_RETEST_DAYS = 30         # Re-run cointegration test monthly

# ──────────────────────────────────────────────────────────────────────
# 10. LOGGING & ALERTS
# ──────────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
LOG_DIR = "logs"
TRADE_LOG_FILE = "logs/trades.json"
SYSTEM_LOG_FILE = "logs/system.log"
ALERT_ON_TRADE = True
ALERT_ON_ERROR = True
ALERT_ON_DRAWDOWN = True

# ──────────────────────────────────────────────────────────────────────
# 11. DASHBOARD / WEB UI
# ──────────────────────────────────────────────────────────────────────
DASHBOARD_HOST = "0.0.0.0"
DASHBOARD_PORT = 8501
DASHBOARD_REFRESH_SECONDS = 60

# ──────────────────────────────────────────────────────────────────────
# 12. SCHEDULING
# ──────────────────────────────────────────────────────────────────────
MARKET_OPEN = "09:30"          # US Eastern
MARKET_CLOSE = "16:00"
PRE_MARKET_PREP_MINUTES = 30   # Start data pull before open
POST_MARKET_ANALYSIS_MINUTES = 15
TIMEZONE = "US/Eastern"

# ──────────────────────────────────────────────────────────────────────
# 13. LEARNING / EXPERIENCE AGENT
# ──────────────────────────────────────────────────────────────────────
MEMORY_DB_PATH = "learning/memory.db"
MIN_TRADES_BEFORE_UPDATE = 30  # Min trades before weight recalculation
MONTE_CARLO_SIMULATIONS = 1000 # Bootstrap resampling iterations
LESSON_CONFIDENCE_DECAY = 0.95 # Older lessons lose weight per cycle

# ──────────────────────────────────────────────────────────────────────
# 14. WALK-FORWARD OPTIMIZATION
# ──────────────────────────────────────────────────────────────────────
WF_TRAIN_YEARS = 2              # Training window length (years)
WF_TEST_MONTHS = 6              # Test/validation window length (months)
WF_MIN_WINDOWS = 4              # Minimum windows for statistical validity
CHANDELIER_ATR_MULT = 3.0       # Default Chandelier ATR multiplier
