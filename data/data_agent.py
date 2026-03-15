"""
data_agent.py — Centralised Data Agent

This is the ONLY file in the system that touches external data sources.
Every other component requests data through this agent.

Rules:
  - Never called directly from strategies — always via get_ohlcv() or get_alt_data()
  - Validates all data before returning it — raises errors on bad data
  - Caches data locally to avoid redundant API calls
  - NEVER silently falls back to cached data if USE_LIVE_DATA is True — raise
    clear error instead
  - Every fetch is timestamped and logged
"""

import os
import time
import logging
import hashlib
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

import config

logger = logging.getLogger(__name__)


# ─── Custom Exceptions ──────────────────────────────────────────────────────────

class DataSourceError(Exception):
    """Raised when live data is required but USE_LIVE_DATA is False,
    or when an external API call fails."""


class DataValidationError(Exception):
    """Raised when fetched data fails quality checks."""


# ─── Timeframe Helper ───────────────────────────────────────────────────────────

_TIMEFRAME_MAP = {
    "1Min":  TimeFrame(1, TimeFrameUnit.Minute),
    "5Min":  TimeFrame(5, TimeFrameUnit.Minute),
    "15Min": TimeFrame(15, TimeFrameUnit.Minute),
    "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
    "1Day":  TimeFrame(1, TimeFrameUnit.Day),
}


def _parse_timeframe(tf_str: str) -> TimeFrame:
    """Convert a config-style timeframe string to an Alpaca TimeFrame."""
    if tf_str in _TIMEFRAME_MAP:
        return _TIMEFRAME_MAP[tf_str]
    raise ValueError(
        f"Unknown timeframe '{tf_str}'. "
        f"Supported: {list(_TIMEFRAME_MAP.keys())}"
    )


# ─── DataAgent ──────────────────────────────────────────────────────────────────

class DataAgent:
    """
    Centralised data gateway.

    All market data, sentiment, weather, and alternative data is fetched,
    validated, cached, and served through this single class.
    """

    def __init__(self):
        # Alpaca client (works for both live and paper)
        self._client = StockHistoricalDataClient(
            api_key=config.ALPACA_API_KEY,
            secret_key=config.ALPACA_SECRET_KEY,
        )
        # Ensure cache directories exist
        os.makedirs(config.DATA_CACHE_DIR, exist_ok=True)
        os.makedirs(config.PROCESSED_DIR, exist_ok=True)

        # In-memory caches for alt-data
        self._sentiment_cache: dict[str, tuple[float, float]] = {}  # key → (score, ts)
        self._weather_cache: dict[str, tuple[float, float]] = {}

        logger.info("DataAgent initialised  |  live_data=%s", config.USE_LIVE_DATA)

    # ── 1. get_ohlcv ────────────────────────────────────────────────────────

    def get_ohlcv(
        self,
        symbol: str,
        start: str,
        end: str,
        timeframe: str = "1Day",
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars for *symbol* between *start* and *end*.

        Primary source : Alpaca historical data API
        Columns        : open, high, low, close, volume  (lowercase)
        Index          : DatetimeIndex (UTC, sorted ascending)

        Caching:
          - Results cached as parquet in DATA_CACHE_DIR/{symbol}_{start}_{end}.parquet
          - On cache hit: refresh if data is stale (> 1 day old for daily bars)

        Raises:
          DataSourceError      – if USE_LIVE_DATA is False
          DataValidationError  – if the data fails quality checks
        """
        if not config.USE_LIVE_DATA:
            raise DataSourceError(
                f"USE_LIVE_DATA is False — cannot fetch live OHLCV for {symbol}. "
                "Set config.USE_LIVE_DATA = True or provide cached data manually."
            )

        cache_path = self._cache_path(symbol, start, end)

        # Try cache first
        if os.path.exists(cache_path):
            cached_df = pd.read_parquet(cache_path)
            if not self._is_stale(cached_df, timeframe):
                logger.info(
                    "OHLCV cache hit  |  symbol=%s  range=%s→%s  rows=%d  source=cache",
                    symbol, start, end, len(cached_df),
                )
                return self._normalise_columns(cached_df)
            logger.info("OHLCV cache stale — refreshing  |  symbol=%s", symbol)

        # Live fetch
        df = self._fetch_bars(symbol, start, end, timeframe)
        self._validate_ohlcv(df, symbol)

        # Persist to cache
        df.to_parquet(cache_path)
        logger.info(
            "OHLCV fetched    |  symbol=%s  range=%s→%s  rows=%d  source=live",
            symbol, start, end, len(df),
        )
        return self._normalise_columns(df)

    # ── 2. get_universe_data ────────────────────────────────────────────────

    def get_universe_data(
        self,
        symbols: list[str],
        start: str,
        end: str,
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch OHLCV for every symbol in *symbols*.

        Returns a dict keyed by symbol.  Raises if ANY symbol fails
        validation — partial data is worse than no data.
        """
        result: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            result[sym] = self.get_ohlcv(sym, start, end, config.TIMEFRAME)
        return result

    # ── 3. get_latest_bars ──────────────────────────────────────────────────

    def get_latest_bars(
        self, symbols: list[str],
    ) -> dict[str, pd.Series]:
        """
        Fetch the latest complete daily bar for each symbol.

        Used by the live runner to get current prices.
        Only works when USE_LIVE_DATA is True.
        """
        if not config.USE_LIVE_DATA:
            raise DataSourceError(
                "USE_LIVE_DATA is False — cannot fetch latest bars."
            )

        result: dict[str, pd.Series] = {}
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        lookback = (datetime.now(timezone.utc) - timedelta(days=5)).strftime("%Y-%m-%d")

        for sym in symbols:
            df = self._fetch_bars(sym, lookback, today, "1Day")
            if df.empty:
                logger.warning("No recent bars for %s", sym)
                continue
            df = self._normalise_columns(df)
            result[sym] = df.iloc[-1]

        return result

    # ── 4. get_sentiment ────────────────────────────────────────────────────

    def get_sentiment(self, query: str, days_back: int = 7) -> float:
        """
        Return a composite sentiment score for *query* in [-1.0, +1.0].

        Source : NewsAPI (free tier) → VADER sentiment on headlines.
        Caching: results cached for 4 hours.

        Returns 0.0 gracefully if USE_ALT_DATA is False or API fails.
        """
        if not config.USE_ALT_DATA:
            return 0.0

        # Check cache (4-hour TTL)
        cache_key = f"sentiment:{query}:{days_back}"
        cached = self._sentiment_cache.get(cache_key)
        if cached and (time.time() - cached[1]) < 4 * 3600:
            logger.info("Sentiment cache hit  |  query=%s  score=%.3f", query, cached[0])
            return cached[0]

        try:
            from newsapi import NewsApiClient
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

            newsapi_key = os.getenv("NEWSAPI_KEY", "")
            if not newsapi_key:
                logger.warning("NEWSAPI_KEY not set — returning 0.0")
                return 0.0

            client = NewsApiClient(api_key=newsapi_key)
            from_date = (
                datetime.now(timezone.utc) - timedelta(days=days_back)
            ).strftime("%Y-%m-%d")

            articles = client.get_everything(
                q=query, from_param=from_date, language="en", sort_by="relevancy",
            )
            headlines = [a["title"] for a in articles.get("articles", []) if a.get("title")]

            if not headlines:
                logger.info("No headlines for query=%s — returning 0.0", query)
                return 0.0

            analyzer = SentimentIntensityAnalyzer()
            scores = [analyzer.polarity_scores(h)["compound"] for h in headlines]
            composite = float(np.mean(scores))
            composite = max(-1.0, min(1.0, composite))

            self._sentiment_cache[cache_key] = (composite, time.time())
            logger.info(
                "Sentiment fetched |  query=%s  headlines=%d  score=%.3f",
                query, len(headlines), composite,
            )
            return composite

        except Exception as exc:
            logger.error("Sentiment fetch failed: %s — returning 0.0", exc)
            return 0.0

    # ── 5. get_weather_signal ───────────────────────────────────────────────

    def get_weather_signal(self, region: str = "US_MIDWEST") -> float:
        """
        Return a drought index in [-1.0 (severe drought), +1.0 (normal)].

        Source : NOAA Climate Data API.
        Caching: results cached for 24 hours.

        Returns 0.0 gracefully if USE_ALT_DATA is False or API fails.
        """
        if not config.USE_ALT_DATA:
            return 0.0

        cache_key = f"weather:{region}"
        cached = self._weather_cache.get(cache_key)
        if cached and (time.time() - cached[1]) < 24 * 3600:
            logger.info("Weather cache hit  |  region=%s  signal=%.3f", region, cached[0])
            return cached[0]

        try:
            import requests

            noaa_token = os.getenv("NOAA_TOKEN", "")
            if not noaa_token:
                logger.warning("NOAA_TOKEN not set — returning 0.0")
                return 0.0

            # NOAA GHCN daily — Palmer Drought Severity Index (PDSI)
            url = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"
            end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            start_date = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d")

            resp = requests.get(
                url,
                headers={"token": noaa_token},
                params={
                    "datasetid": "GHCND",
                    "datatypeid": "PDSI",
                    "locationid": f"FIPS:{self._region_fips(region)}",
                    "startdate": start_date,
                    "enddate": end_date,
                    "limit": 10,
                    "sortfield": "date",
                    "sortorder": "desc",
                },
                timeout=15,
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])

            if not results:
                logger.info("No PDSI data for region=%s — returning 0.0", region)
                return 0.0

            # PDSI typically ranges −6 → +6; normalise to −1 → +1
            raw = float(results[0]["value"])
            signal = max(-1.0, min(1.0, raw / 6.0))

            self._weather_cache[cache_key] = (signal, time.time())
            logger.info(
                "Weather fetched  |  region=%s  raw_pdsi=%.2f  signal=%.3f",
                region, raw, signal,
            )
            return signal

        except Exception as exc:
            logger.error("Weather fetch failed: %s — returning 0.0", exc)
            return 0.0

    # ─── Private Helpers ────────────────────────────────────────────────────

    def _fetch_bars(
        self, symbol: str, start: str, end: str, timeframe: str,
    ) -> pd.DataFrame:
        """Hit the Alpaca historical bars API and return a clean DataFrame."""
        tf = _parse_timeframe(timeframe)
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=tf,
            start=datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc),
            end=datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            if end != "today"
            else datetime.now(timezone.utc),
        )
        bars = self._client.get_stock_bars(request)
        df = bars.df

        if df.empty:
            return df

        # If multi-index (symbol, timestamp), drop the symbol level
        if isinstance(df.index, pd.MultiIndex):
            df = df.droplevel("symbol")

        df.columns = [c.lower() for c in df.columns]
        df = df.sort_index()
        return df

    def _validate_ohlcv(self, df: pd.DataFrame, symbol: str) -> None:
        """Run quality checks on an OHLCV DataFrame.  Raise on failure."""
        if df.empty:
            raise DataValidationError(f"No data returned for {symbol}")

        # No NaN in OHLCV
        ohlcv_cols = ["open", "high", "low", "close", "volume"]
        missing = [c for c in ohlcv_cols if c not in df.columns]
        if missing:
            raise DataValidationError(
                f"{symbol}: missing columns {missing}"
            )

        nan_counts = df[ohlcv_cols].isna().sum()
        if nan_counts.any():
            raise DataValidationError(
                f"{symbol}: NaN values found — {nan_counts.to_dict()}"
            )

        # No zero-volume bars
        if (df["volume"] == 0).any():
            raise DataValidationError(
                f"{symbol}: zero-volume bars detected"
            )

        # No price gaps > 50%
        pct_change = df["close"].pct_change().abs()
        if (pct_change > 0.50).any():
            bad_dates = df.index[pct_change > 0.50].tolist()
            raise DataValidationError(
                f"{symbol}: price gaps > 50% on {bad_dates}"
            )

        # Chronological order
        if not df.index.is_monotonic_increasing:
            raise DataValidationError(
                f"{symbol}: index is not in chronological order"
            )

    def _cache_path(self, symbol: str, start: str, end: str) -> str:
        """Build a deterministic cache file path."""
        return os.path.join(
            config.DATA_CACHE_DIR,
            f"{symbol}_{start}_{end}.parquet",
        )

    def _is_stale(self, df: pd.DataFrame, timeframe: str) -> bool:
        """Return True if cached data is too old and should be refreshed."""
        if df.empty:
            return True
        last_ts = df.index[-1]
        if hasattr(last_ts, "to_pydatetime"):
            last_ts = last_ts.to_pydatetime()
        if last_ts.tzinfo is None:
            last_ts = last_ts.replace(tzinfo=timezone.utc)
        age = datetime.now(timezone.utc) - last_ts
        # For daily bars, stale if older than 1 calendar day
        if "Day" in timeframe:
            return age > timedelta(days=1)
        # For intraday, stale if older than 2 hours
        return age > timedelta(hours=2)

    @staticmethod
    def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Rename lowercase Alpaca columns to Title-case and add 'Adj Close'.

        This ensures compatibility with all downstream consumers
        (strategies, risk, core modules) which expect Title-case columns.
        """
        rename_map = {
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
            "trade_count": "Trade_Count",
            "vwap": "VWAP",
        }
        # Only rename columns that exist
        active_renames = {k: v for k, v in rename_map.items() if k in df.columns}
        df = df.rename(columns=active_renames)

        # Add 'Adj Close' as a copy of 'Close' for legacy compatibility
        if "Adj Close" not in df.columns and "Close" in df.columns:
            df["Adj Close"] = df["Close"]

        return df

    @staticmethod
    def _region_fips(region: str) -> str:
        """Map friendly region names to FIPS codes for NOAA."""
        mapping = {
            "US_MIDWEST": "19",     # Iowa (representative)
            "US_SOUTHEAST": "13",   # Georgia
            "US_WEST": "06",        # California
            "US_NORTHEAST": "36",   # New York
            "US_PLAINS": "20",      # Kansas
        }
        return mapping.get(region, "19")
