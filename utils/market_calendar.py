"""
utils/market_calendar.py — NYSE Trading Calendar Utility

Wraps pandas_market_calendars to provide a clean interface for
determining trading days, market open/close times, and holiday checks.
"""

from datetime import date, timedelta

import pandas as pd


class MarketCalendar:
    """NYSE trading calendar wrapper."""

    def __init__(self):
        from pandas_market_calendars import get_calendar
        self.nyse = get_calendar("NYSE")

    def is_trading_day(self, dt: date) -> bool:
        """Return True if dt is an NYSE trading day."""
        schedule = self.nyse.schedule(
            start_date=dt.strftime("%Y-%m-%d"),
            end_date=dt.strftime("%Y-%m-%d"),
        )
        return not schedule.empty

    def next_trading_day(self, dt: date) -> date:
        """Return the next NYSE trading day after dt."""
        check = dt + timedelta(days=1)
        for _ in range(10):
            if self.is_trading_day(check):
                return check
            check += timedelta(days=1)
        raise ValueError(f"No trading day found within 10 days of {dt}")

    def get_trading_days(self, start: date, end: date) -> list[date]:
        """Return list of NYSE trading days in [start, end]."""
        schedule = self.nyse.schedule(
            start_date=start.strftime("%Y-%m-%d"),
            end_date=end.strftime("%Y-%m-%d"),
        )
        return [d.date() for d in schedule.index]

    def market_open_time(self, dt: date) -> pd.Timestamp:
        """Return market open time for a trading day."""
        schedule = self.nyse.schedule(
            start_date=dt.strftime("%Y-%m-%d"),
            end_date=dt.strftime("%Y-%m-%d"),
        )
        if schedule.empty:
            raise ValueError(f"{dt} is not a trading day")
        return schedule.iloc[0]["market_open"]

    def market_close_time(self, dt: date) -> pd.Timestamp:
        """Return market close time for a trading day."""
        schedule = self.nyse.schedule(
            start_date=dt.strftime("%Y-%m-%d"),
            end_date=dt.strftime("%Y-%m-%d"),
        )
        if schedule.empty:
            raise ValueError(f"{dt} is not a trading day")
        return schedule.iloc[0]["market_close"]
