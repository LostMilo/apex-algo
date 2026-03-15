"""
memory_store.py — SQLite-Based Shared Knowledge Database

All agents read lessons relevant to them.  All learning is written here.
Full audit trail of every parameter change.

Rules:
  - Every insert uses parameterised queries (no string interpolation).
  - The database file + parent dirs are created automatically on __init__.
  - All timestamps are ISO-8601 UTC strings.
  - This file imports ONLY config and stdlib — no project-internal imports.
"""

import json
import os
import sqlite3
import logging
from datetime import datetime, timezone
from typing import Optional

import config

logger = logging.getLogger(__name__)


# ─── MemoryStore ────────────────────────────────────────────────────────────────

class MemoryStore:
    """
    SQLite-backed shared knowledge database.

    Tables
    ------
    1. lessons        — cross-agent knowledge entries
    2. agent_weights  — weight-change audit trail
    3. trade_log      — complete trade history
    """

    def __init__(self, db_path: str = config.MEMORY_DB_PATH):
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)

        self._db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row          # dict-like rows
        self._conn.execute("PRAGMA journal_mode=WAL")  # concurrent reads
        self._create_tables()
        logger.info("MemoryStore ready  |  db=%s", db_path)

    # ─── Table Creation ─────────────────────────────────────────────────────

    def _create_tables(self) -> None:
        """Create all tables if they don't already exist."""
        cur = self._conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS lessons (
                id                    INTEGER PRIMARY KEY AUTOINCREMENT,
                lesson_id             TEXT    UNIQUE,
                timestamp             TEXT,
                source_agent          TEXT,
                lesson_type           TEXT,
                condition             TEXT,
                market_regime         TEXT,
                recommended_action    TEXT,
                confidence            REAL,
                broadcast_to          TEXT,
                trade_count_at_creation INTEGER,
                applied               INTEGER DEFAULT 0
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS agent_weights (
                id                    INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp             TEXT,
                tsmom_weight          REAL,
                vol_trend_weight      REAL,
                pairs_arb_weight      REAL,
                reason                TEXT,
                trade_count_at_update INTEGER,
                sharpe_before         REAL,
                sharpe_after          REAL
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS trade_log (
                id                    INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id              TEXT    UNIQUE,
                symbol                TEXT,
                strategy              TEXT,
                entry_date            TEXT,
                exit_date             TEXT,
                entry_price           REAL,
                exit_price            REAL,
                shares                REAL,
                dollar_size           REAL,
                pnl_dollars           REAL,
                pnl_pct               REAL,
                hold_days             INTEGER,
                exit_reason           TEXT,
                regime_at_entry       TEXT,
                signal_strength       REAL,
                confidence_at_entry   REAL,
                notes                 TEXT
            )
        """)

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS system_state (
                key   TEXT PRIMARY KEY,
                value TEXT
            )
        """)

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS eod_snapshots (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                date            TEXT UNIQUE,
                equity          REAL,
                cash            REAL,
                open_positions  INTEGER,
                daily_pnl       REAL,
                drawdown_pct    REAL
            )
        """)

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS alerts_log (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TEXT DEFAULT (datetime('now')),
                alert_type      TEXT,
                subject         TEXT,
                recipient       TEXT,
                status          TEXT,
                error           TEXT DEFAULT '',
                acknowledged    INTEGER DEFAULT 0
            )
        """)

        self._conn.commit()

    # ─── Alerts Log ─────────────────────────────────────────────────────

    def log_alert(self, alert: dict) -> None:
        """Log an email alert attempt."""
        self._conn.execute(
            """INSERT INTO alerts_log
               (alert_type, subject, recipient, status, error)
               VALUES (?, ?, ?, ?, ?)""",
            (
                alert.get("alert_type", "NORMAL"),
                alert.get("subject", ""),
                alert.get("recipient", ""),
                alert.get("status", "unknown"),
                alert.get("error", ""),
            ),
        )
        self._conn.commit()

    def get_alert_history(self, limit: int = 100,
                          status_filter: str = None) -> list[dict]:
        """Retrieve recent alert history."""
        query = "SELECT * FROM alerts_log"
        params = []
        if status_filter:
            query += " WHERE status = ?"
            params.append(status_filter)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def acknowledge_alert(self, alert_id: int) -> None:
        """Mark an alert as acknowledged."""
        self._conn.execute(
            "UPDATE alerts_log SET acknowledged = 1 WHERE id = ?",
            (alert_id,),
        )
        self._conn.commit()

    def get_unacknowledged_count(self) -> int:
        """Return count of unacknowledged failed alerts."""
        row = self._conn.execute(
            "SELECT COUNT(*) as cnt FROM alerts_log "
            "WHERE status = 'failed' AND acknowledged = 0"
        ).fetchone()
        return row["cnt"] if row else 0

    # ─── Lessons ────────────────────────────────────────────────────────────

    def store_lesson(self, lesson: dict) -> int:
        """
        Insert a new lesson.

        Required keys in *lesson*:
            lesson_id, source_agent, lesson_type, condition,
            recommended_action, confidence

        Optional keys (defaults supplied if missing):
            timestamp, market_regime, broadcast_to,
            trade_count_at_creation, applied

        Returns the rowid of the inserted row.
        """
        now = datetime.now(timezone.utc).isoformat()
        broadcast = lesson.get("broadcast_to", [])
        if isinstance(broadcast, list):
            broadcast = json.dumps(broadcast)

        cur = self._conn.execute(
            """
            INSERT INTO lessons (
                lesson_id, timestamp, source_agent, lesson_type,
                condition, market_regime, recommended_action,
                confidence, broadcast_to, trade_count_at_creation, applied
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                lesson["lesson_id"],
                lesson.get("timestamp", now),
                lesson["source_agent"],
                lesson["lesson_type"],
                lesson["condition"],
                lesson.get("market_regime", ""),
                lesson["recommended_action"],
                lesson["confidence"],
                broadcast,
                lesson.get("trade_count_at_creation", 0),
                lesson.get("applied", 0),
            ),
        )
        self._conn.commit()
        logger.info(
            "Lesson stored  |  id=%s  source=%s  type=%s",
            lesson["lesson_id"], lesson["source_agent"], lesson["lesson_type"],
        )
        return cur.lastrowid

    def get_pending_lessons(self, agent_name: str) -> list[dict]:
        """
        Return all unapplied lessons whose broadcast_to list
        contains *agent_name*.
        """
        rows = self._conn.execute(
            "SELECT * FROM lessons WHERE applied = 0"
        ).fetchall()

        results = []
        for row in rows:
            broadcast = json.loads(row["broadcast_to"] or "[]")
            if agent_name in broadcast:
                results.append(dict(row))
        return results

    def mark_lesson_applied(self, lesson_id: str) -> None:
        """Mark a lesson as applied (applied = 1)."""
        self._conn.execute(
            "UPDATE lessons SET applied = 1 WHERE lesson_id = ?",
            (lesson_id,),
        )
        self._conn.commit()
        logger.info("Lesson applied  |  id=%s", lesson_id)

    def get_lessons(
        self,
        agent: str = None,
        since_days: int = 7,
        unread_only: bool = True,
    ) -> list[dict]:
        """
        Flexible lesson query — wraps the lessons table.

        Args:
            agent:       If provided, filter to lessons where *agent* is
                         listed in the broadcast_to JSON array.
            since_days:  Only return lessons created within the last N days.
            unread_only: If True, only return lessons where applied == 0.

        Returns:
            List of lesson dicts, most recent first.
        """
        clauses = []
        params: list = []

        if unread_only:
            clauses.append("applied = 0")

        if since_days and since_days > 0:
            from datetime import timedelta
            cutoff = (
                datetime.now(timezone.utc) - timedelta(days=since_days)
            ).isoformat()
            clauses.append("timestamp >= ?")
            params.append(cutoff)

        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        query = f"SELECT * FROM lessons{where} ORDER BY id DESC"

        rows = self._conn.execute(query, params).fetchall()

        results: list[dict] = []
        for row in rows:
            row_dict = dict(row)
            # Filter by agent if requested
            if agent:
                broadcast = json.loads(row_dict.get("broadcast_to") or "[]")
                if agent not in broadcast:
                    continue
            results.append(row_dict)

        return results

    # ─── Agent Weights ──────────────────────────────────────────────────────

    def store_weight_update(self, weights: dict) -> int:
        """
        Record a weight-change event.

        Required keys: tsmom_weight, vol_trend_weight, pairs_arb_weight, reason
        Optional keys: timestamp, trade_count_at_update, sharpe_before, sharpe_after

        Returns the rowid.
        """
        now = datetime.now(timezone.utc).isoformat()
        cur = self._conn.execute(
            """
            INSERT INTO agent_weights (
                timestamp, tsmom_weight, vol_trend_weight, pairs_arb_weight,
                reason, trade_count_at_update, sharpe_before, sharpe_after
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                weights.get("timestamp", now),
                weights["tsmom_weight"],
                weights["vol_trend_weight"],
                weights["pairs_arb_weight"],
                weights["reason"],
                weights.get("trade_count_at_update", 0),
                weights.get("sharpe_before"),
                weights.get("sharpe_after"),
            ),
        )
        self._conn.commit()
        logger.info(
            "Weights updated  |  tsmom=%.2f  vol=%.2f  arb=%.2f  reason=%s",
            weights["tsmom_weight"],
            weights["vol_trend_weight"],
            weights["pairs_arb_weight"],
            weights["reason"],
        )
        return cur.lastrowid

    def get_latest_weights(self) -> Optional[dict]:
        """Return the most recent weights row, or None if table is empty."""
        row = self._conn.execute(
            "SELECT * FROM agent_weights ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None

    def store_weights(self, weights: dict, reason: str = "experience_agent_update") -> None:
        """
        Convenience wrapper around store_weight_update().

        Accepts a simple dict like {'TSMOM': 0.4, 'VolTrend': 0.3, 'PairsArb': 0.3}
        and translates it to the column-keyed format store_weight_update() expects.

        Args:
            weights: Strategy name → weight mapping.
            reason:  Human-readable reason for the weight change.
        """
        # Map from strategy names to DB column keys
        _KEY_MAP = {
            "TSMOM": "tsmom_weight",
            "tsmom": "tsmom_weight",
            "VolTrend": "vol_trend_weight",
            "vol_trend": "vol_trend_weight",
            "PairsArb": "pairs_arb_weight",
            "pairs_arb": "pairs_arb_weight",
        }
        translated = {}
        for k, v in weights.items():
            db_key = _KEY_MAP.get(k, k)
            translated[db_key] = v

        # Ensure all required keys exist with defaults
        translated.setdefault("tsmom_weight", 0.33)
        translated.setdefault("vol_trend_weight", 0.33)
        translated.setdefault("pairs_arb_weight", 0.34)
        translated.setdefault("reason", reason)

        self.store_weight_update(translated)

    # ─── Trade Log ──────────────────────────────────────────────────────────

    def log_trade(self, trade: dict) -> int:
        """
        Record a completed trade.

        Required keys: trade_id, symbol, strategy, entry_date, exit_date,
                       entry_price, exit_price, shares, dollar_size,
                       pnl_dollars, pnl_pct
        Optional keys: hold_days, exit_reason, regime_at_entry,
                       signal_strength, confidence_at_entry, notes

        Returns the rowid.
        """
        cur = self._conn.execute(
            """
            INSERT INTO trade_log (
                trade_id, symbol, strategy, entry_date, exit_date,
                entry_price, exit_price, shares, dollar_size,
                pnl_dollars, pnl_pct, hold_days, exit_reason,
                regime_at_entry, signal_strength, confidence_at_entry, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trade["trade_id"],
                trade["symbol"],
                trade["strategy"],
                trade["entry_date"],
                trade["exit_date"],
                trade["entry_price"],
                trade["exit_price"],
                trade["shares"],
                trade["dollar_size"],
                trade["pnl_dollars"],
                trade["pnl_pct"],
                trade.get("hold_days", 0),
                trade.get("exit_reason", ""),
                trade.get("regime_at_entry", ""),
                trade.get("signal_strength"),
                trade.get("confidence_at_entry"),
                trade.get("notes", ""),
            ),
        )
        self._conn.commit()
        logger.info(
            "Trade logged  |  id=%s  sym=%s  strat=%s  pnl=$%.2f (%.2f%%)",
            trade["trade_id"], trade["symbol"], trade["strategy"],
            trade["pnl_dollars"], trade["pnl_pct"] * 100,
        )
        return cur.lastrowid

    def get_trades(
        self,
        strategy: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """
        Return recent trades, optionally filtered by strategy.
        Most recent first, capped at *limit*.
        """
        if strategy:
            rows = self._conn.execute(
                "SELECT * FROM trade_log WHERE strategy = ? "
                "ORDER BY id DESC LIMIT ?",
                (strategy, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM trade_log ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_trade_count(self) -> int:
        """Return total number of completed trades."""
        row = self._conn.execute(
            "SELECT COUNT(*) AS cnt FROM trade_log"
        ).fetchone()
        return row["cnt"]

    def get_strategy_performance(self, strategy: str) -> dict:
        """
        Compute aggregate performance stats for a given strategy.

        Returns a dict with:
            total_trades, wins, losses, win_rate,
            avg_pnl_dollars, avg_pnl_pct, total_pnl_dollars,
            avg_hold_days, best_trade_pnl, worst_trade_pnl
        """
        rows = self._conn.execute(
            "SELECT * FROM trade_log WHERE strategy = ?",
            (strategy,),
        ).fetchall()

        if not rows:
            return {
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "avg_pnl_dollars": 0.0,
                "avg_pnl_pct": 0.0,
                "total_pnl_dollars": 0.0,
                "avg_hold_days": 0.0,
                "best_trade_pnl": 0.0,
                "worst_trade_pnl": 0.0,
            }

        pnls = [r["pnl_dollars"] for r in rows]
        pnl_pcts = [r["pnl_pct"] for r in rows]
        hold_days = [r["hold_days"] for r in rows]
        wins = sum(1 for p in pnls if p > 0)
        losses = sum(1 for p in pnls if p <= 0)
        total = len(pnls)

        return {
            "total_trades": total,
            "wins": wins,
            "losses": losses,
            "win_rate": wins / total if total else 0.0,
            "avg_pnl_dollars": sum(pnls) / total,
            "avg_pnl_pct": sum(pnl_pcts) / total,
            "total_pnl_dollars": sum(pnls),
            "avg_hold_days": sum(hold_days) / total,
            "best_trade_pnl": max(pnls),
            "worst_trade_pnl": min(pnls),
        }

    # ─── Paper Trading Warmup ───────────────────────────────────────────────

    def get_paper_trading_start(self):
        """Return the date paper trading started, or None."""
        row = self._conn.execute(
            "SELECT value FROM system_state WHERE key = 'paper_trading_start'"
        ).fetchone()
        if row and row["value"]:
            from datetime import date as _d
            return _d.fromisoformat(row["value"])
        return None

    def set_paper_trading_start(self, dt) -> None:
        """Record when paper trading started (upsert)."""
        self._conn.execute(
            "INSERT OR REPLACE INTO system_state (key, value) VALUES (?, ?)",
            ("paper_trading_start", dt.isoformat()),
        )
        self._conn.commit()
        logger.info("Paper trading start recorded: %s", dt.isoformat())

    def get_paper_trading_days(self) -> int:
        """Return number of days since paper trading started. 0 if not started."""
        start = self.get_paper_trading_start()
        if start is None:
            return 0
        from datetime import date as _d
        return (_d.today() - start).days

    # ─── EOD Snapshots ──────────────────────────────────────────────────────

    def write_eod_snapshot(self, portfolio: dict) -> None:
        """
        Store end-of-day portfolio snapshot.

        portfolio should have: equity, cash, positions (list/dict),
                               daily_pnl, drawdown_pct
        """
        from datetime import date as _d
        today = _d.today().isoformat()
        positions = portfolio.get("positions", {})
        n_positions = len(positions) if isinstance(positions, (dict, list)) else 0

        self._conn.execute(
            """INSERT OR REPLACE INTO eod_snapshots
               (date, equity, cash, open_positions, daily_pnl, drawdown_pct)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                today,
                portfolio.get("equity", 0),
                portfolio.get("cash", 0),
                n_positions,
                portfolio.get("daily_pnl", 0),
                portfolio.get("drawdown_pct", 0),
            ),
        )
        self._conn.commit()
        logger.info(
            "EOD snapshot  |  equity=$%.2f  cash=$%.2f  positions=%d",
            portfolio.get("equity", 0), portfolio.get("cash", 0), n_positions,
        )

    # ─── Lifecycle ──────────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
        logger.info("MemoryStore closed  |  db=%s", self._db_path)
