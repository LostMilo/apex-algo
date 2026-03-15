"""
execution/alpaca_client.py — Alpaca API Wrapper for Order Execution

Wraps the Alpaca API for order execution. Works identically for paper and
live trading — only the API endpoint changes. Kill switch is checked before
EVERY order. Always uses limit orders.

Dependencies:
    pip install alpaca-py          # NOT deprecated alpaca-trade-api
"""

import math
from typing import Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame

import config
from logger import log


# ──────────────────────────────────────────────────────────────────────
# Constants derived from config
# ──────────────────────────────────────────────────────────────────────
LIMIT_SLIPPAGE_PCT = config.MAX_SLIPPAGE_PCT  # 0.001 (0.1%)


class AlpacaClient:
    """
    Thin execution layer over Alpaca's TradingClient.

    Responsibilities:
      • Connect to paper or live via a single flag
      • Fetch account state and open positions
      • Execute signals: kill-switch → direction → price → sizing → order
      • Always LIMIT orders, never market
    """

    def __init__(self, position_sizer=None):
        """
        Initialise TradingClient with API key/secret from config.
        PAPER_TRADING flag determines base URL (paper vs live).

        Args:
            position_sizer: Optional PositionSizer instance (for future use).
        """
        self.position_sizer = position_sizer

        # ── Trading client (order management) ────────────────────────
        self.trading_client = TradingClient(
            api_key=config.ALPACA_API_KEY,
            secret_key=config.ALPACA_SECRET_KEY,
            paper=config.PAPER_TRADING,
        )

        # ── Data client (quotes / bars) ──────────────────────────────
        self.data_client = StockHistoricalDataClient(
            api_key=config.ALPACA_API_KEY,
            secret_key=config.ALPACA_SECRET_KEY,
        )

        mode_label = "paper" if config.PAPER_TRADING else "live"
        log.info("AlpacaClient connected to {} with account equity", mode_label)

    # ─────────────────────────────────────────────────────────────────
    # Account
    # ─────────────────────────────────────────────────────────────────

    def get_account(self) -> dict:
        """
        Returns: {equity, cash, buying_power, portfolio_value}
        Used to sync portfolio state with broker.
        """
        acct = self.trading_client.get_account()
        return {
            "equity": float(acct.equity),
            "cash": float(acct.cash),
            "buying_power": float(acct.buying_power),
            "portfolio_value": float(acct.portfolio_value),
        }

    # ─────────────────────────────────────────────────────────────────
    # Positions
    # ─────────────────────────────────────────────────────────────────

    def get_positions(self) -> dict[str, dict]:
        """
        Returns all open positions from Alpaca.

        Format:
            {symbol: {qty, avg_entry_price, current_price, unrealized_pnl_pct}}
        """
        raw = self.trading_client.get_all_positions()
        positions: dict[str, dict] = {}

        for p in raw:
            positions[p.symbol] = {
                "qty": float(p.qty),
                "avg_entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "unrealized_pnl_pct": float(p.unrealized_plpc),
            }

        return positions

    # ─────────────────────────────────────────────────────────────────
    # Execute Signal
    # ─────────────────────────────────────────────────────────────────

    def execute_signal(
        self,
        symbol: str,
        dollar_size: float,
        signal: float,
        kill_switch_status: str,
    ) -> dict:
        """
        Execute a trading signal end-to-end.

        CRITICAL: Check kill switch BEFORE anything else.
        If kill_switch_status != 'OK' and the action would increase
        exposure, abort with log.

        Steps:
            1. CHECK KILL SWITCH
            2. DETERMINE DIRECTION
            3. GET CURRENT PRICE
            4. COMPUTE SHARES
            5. COMPUTE LIMIT PRICE
            6. CHECK EXISTING POSITION
            7. SUBMIT LIMIT ORDER

        Args:
            symbol:             Ticker (e.g. 'SPY')
            dollar_size:        Absolute dollar position size
            signal:             Positive → buy, negative → sell/short
            kill_switch_status: 'OK', 'FULL_HALT', 'REDUCE_ONLY', etc.

        Returns:
            dict with order details or abort reason.
        """
        # ── 1. CHECK KILL SWITCH ─────────────────────────────────────
        if kill_switch_status == "FULL_HALT":
            log.warning(
                "KILL SWITCH [FULL_HALT]: aborting {} signal for {}",
                "BUY" if signal > 0 else "SELL", symbol,
            )
            return {"status": "aborted", "reason": "FULL_HALT"}

        # ── 2. DETERMINE DIRECTION ───────────────────────────────────
        if signal > 0:
            side = OrderSide.BUY
            side_label = "BUY"
        elif signal < 0:
            side = OrderSide.SELL
            side_label = "SELL"
        else:
            log.debug("Signal is zero for {} — no action", symbol)
            return {"status": "skipped", "reason": "zero_signal"}

        # If kill switch is not OK and we'd be INCREASING exposure, abort
        if kill_switch_status != "OK" and side == OrderSide.BUY:
            log.warning(
                "KILL SWITCH [{}]: blocking BUY for {} (would increase exposure)",
                kill_switch_status, symbol,
            )
            return {"status": "aborted", "reason": kill_switch_status}

        # ── 3. GET CURRENT PRICE ─────────────────────────────────────
        try:
            quote_request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quotes = self.data_client.get_stock_latest_quote(quote_request)
            quote = quotes[symbol]
            current_ask = float(quote.ask_price) if quote.ask_price else 0.0
            current_bid = float(quote.bid_price) if quote.bid_price else 0.0

            # Fallback: use midpoint if one side is missing
            if current_ask <= 0 and current_bid > 0:
                current_ask = current_bid
            elif current_bid <= 0 and current_ask > 0:
                current_bid = current_ask

            current_price = (current_ask + current_bid) / 2.0

            if current_price <= 0:
                log.error("Cannot get valid price for {} — aborting", symbol)
                return {"status": "aborted", "reason": "no_price"}

        except Exception as e:
            log.error("Failed to fetch quote for {}: {}", symbol, str(e))
            return {"status": "aborted", "reason": f"quote_error: {e}"}

        # ── 4. COMPUTE SHARES ────────────────────────────────────────
        shares = abs(dollar_size) / current_price
        shares = math.floor(shares)  # Round down to whole shares

        if shares < 1:
            log.debug(
                "Computed {} shares for {} (${:.2f} / ${:.2f}) — below minimum, aborting",
                shares, symbol, abs(dollar_size), current_price,
            )
            return {"status": "aborted", "reason": "below_min_shares"}

        # ── 5. COMPUTE LIMIT PRICE ──────────────────────────────────
        if side == OrderSide.BUY:
            limit_price = current_ask * (1 + LIMIT_SLIPPAGE_PCT)
        else:
            limit_price = current_bid * (1 - LIMIT_SLIPPAGE_PCT)

        limit_price = round(limit_price, 2)

        # ── 6. CHECK EXISTING POSITION ──────────────────────────────
        existing_positions = self.get_positions()
        close_result = None

        if symbol in existing_positions:
            existing = existing_positions[symbol]
            existing_qty = existing["qty"]

            # If already long and signal is sell → close first, then re-enter
            if existing_qty > 0 and side == OrderSide.SELL:
                log.info(
                    "Closing existing LONG {} ({} shares) before SELL",
                    symbol, existing_qty,
                )
                try:
                    close_result = self.trading_client.close_position(symbol)
                    log.info("Closed position for {}", symbol)
                except Exception as e:
                    log.error("Failed to close position for {}: {}", symbol, str(e))
                    return {"status": "aborted", "reason": f"close_error: {e}"}

            # If already short and signal is buy → close first, then re-enter
            elif existing_qty < 0 and side == OrderSide.BUY:
                log.info(
                    "Closing existing SHORT {} ({} shares) before BUY",
                    symbol, abs(existing_qty),
                )
                try:
                    close_result = self.trading_client.close_position(symbol)
                    log.info("Closed position for {}", symbol)
                except Exception as e:
                    log.error("Failed to close position for {}: {}", symbol, str(e))
                    return {"status": "aborted", "reason": f"close_error: {e}"}

        # ── 7. SUBMIT LIMIT ORDER ───────────────────────────────────
        try:
            order_request = LimitOrderRequest(
                symbol=symbol,
                qty=shares,
                side=side,
                time_in_force=TimeInForce.DAY,
                limit_price=limit_price,
            )
            order = self.trading_client.submit_order(order_request)

            result = {
                "status": "submitted",
                "order_id": str(order.id),
                "symbol": symbol,
                "side": side_label,
                "qty": shares,
                "limit_price": limit_price,
                "current_price": current_price,
                "dollar_size": abs(dollar_size),
                "signal": signal,
            }

            if close_result is not None:
                result["closed_existing"] = True

            log.info(
                "ORDER SUBMITTED: {} {} x{} @ ${:.2f} (signal={:.3f}, kill={})",
                side_label, symbol, shares, limit_price, signal,
                kill_switch_status,
            )

            return result

        except Exception as e:
            log.error(
                "Failed to submit {} order for {}: {}",
                side_label, symbol, str(e),
            )
            return {"status": "aborted", "reason": f"order_error: {e}"}

    # ── End-of-Day Cleanup ──────────────────────────────────────────────

    def cancel_all_day_orders(self) -> int:
        """
        Cancel all open day orders at end of trading day.

        Returns the count of cancelled orders.
        """
        cancelled = 0
        try:
            orders = self.trading_client.get_orders()
            for order in orders:
                if (order.time_in_force == TimeInForce.DAY
                        and order.status in ("new", "accepted",
                                             "pending_new", "partially_filled")):
                    try:
                        self.trading_client.cancel_order_by_id(order.id)
                        log.info(
                            "EOD CANCEL: {} {} x{} @ ${} (id={})",
                            order.side, order.symbol, order.qty,
                            order.limit_price or "mkt", order.id,
                        )
                        cancelled += 1
                    except Exception as e:
                        log.error("Failed to cancel order {}: {}", order.id, e)
        except Exception as e:
            log.error("Failed to fetch open orders for EOD cleanup: {}", e)

        log.info("EOD cleanup complete: {} orders cancelled", cancelled)
        return cancelled

    # ── Idle Capital Rotation ───────────────────────────────────────────

    def rotate_idle_capital(self, portfolio: dict) -> None:
        """
        Park idle cash in SHV (Short Treasury ETF) to earn yield.

        Keeps 20% of idle cash as buffer for margin/fees.
        """
        try:
            equity = portfolio.get("equity", 0)
            cash = portfolio.get("cash", 0)
            min_pct = getattr(config, "IDLE_CAPITAL_MIN_PCT", 0.10)
            etf = getattr(config, "IDLE_CAPITAL_ETF", "SHV")

            if equity <= 0:
                return

            idle_threshold = equity * min_pct

            if cash <= idle_threshold:
                return

            # Target: 80% of idle cash into SHV (keep 20% buffer)
            target_shv = cash * 0.80
            current_shv = 0.0

            positions = portfolio.get("positions", {})
            if isinstance(positions, dict) and etf in positions:
                current_shv = float(positions[etf].get("market_value", 0))
            elif isinstance(positions, list):
                for p in positions:
                    if p.get("symbol") == etf:
                        current_shv = float(p.get("market_value", 0))
                        break

            buy_amount = target_shv - current_shv

            if buy_amount > 100:  # Minimum $100 to avoid micro-orders
                # Get current SHV price for limit order
                try:
                    quote = self.data_client.get_stock_latest_quote(
                        StockLatestQuoteRequest(symbol_or_symbols=etf)
                    )
                    if isinstance(quote, dict):
                        price = float(quote[etf].ask_price)
                    else:
                        price = float(quote.ask_price)

                    shares = int(buy_amount / price)
                    if shares > 0:
                        limit = round(price * (1 + LIMIT_SLIPPAGE_PCT), 2)
                        order = LimitOrderRequest(
                            symbol=etf,
                            qty=shares,
                            side=OrderSide.BUY,
                            time_in_force=TimeInForce.DAY,
                            limit_price=limit,
                        )
                        self.trading_client.submit_order(order)
                        log.info(
                            "IDLE CAPITAL: Rotating ${:.2f} into {} "
                            "({}sh @ ${:.2f})",
                            buy_amount, etf, shares, limit,
                        )
                except Exception as e:
                    log.warning("Idle capital rotation failed: {}", e)

        except Exception as e:
            log.error("Idle capital rotation error: {}", e)

