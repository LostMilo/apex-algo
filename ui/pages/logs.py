"""
pages/logs.py — System Logs & Trade History Viewer

Sections: system log viewer, trade log table with export,
error log, data fetch log.
"""

import io
import os
import re
from datetime import datetime, timedelta
from pathlib import Path

import streamlit as st
import pandas as pd

import config

GREEN = "#00D97E"
RED = "#FF4B4B"
YELLOW = "#FFD93D"
MUTED = "#8B8D97"

LOG_DIR = Path("data/logs")


def _get_memory_store():
    ms = st.session_state.get("memory_store")
    if ms is None:
        try:
            from learning.memory_store import MemoryStore
            ms = MemoryStore(config.MEMORY_DB_PATH)
            st.session_state.memory_store = ms
        except Exception:
            return None
    return ms


def _read_log_tail(pattern: str, max_lines: int = 200) -> list[str]:
    """Read the tail of the most recent matching log file."""
    if not LOG_DIR.exists():
        return []
    files = sorted(LOG_DIR.glob(pattern), reverse=True)
    if not files:
        return []

    try:
        with open(files[0], "r", errors="replace") as f:
            lines = f.readlines()
        return lines[-max_lines:]
    except Exception:
        return []


def _colorize_log_line(line: str) -> str:
    """Return HTML-styled log line with color by level."""
    stripped = line.strip()
    if not stripped:
        return ""

    if "ERROR" in stripped or "CRITICAL" in stripped:
        color = RED
    elif "WARNING" in stripped:
        color = YELLOW
    elif "DEBUG" in stripped:
        color = MUTED
    else:
        color = "#E0E0E0"

    escaped = stripped.replace("<", "&lt;").replace(">", "&gt;")
    return f'<span style="color:{color};font-family:monospace;font-size:0.82em;">{escaped}</span>'


# ═════════════════════════════════════════════════════════════════════
# PAGE RENDER
# ═════════════════════════════════════════════════════════════════════

def render():
    st.header("📋 Logs & Trade History")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📜 System Log", "📊 Trade Log", "❌ Error Log", "📡 Data Fetch Log",
    ])

    # ─── 1. SYSTEM LOG VIEWER ────────────────────────────────────────
    with tab1:
        st.subheader("System Log Viewer")

        auto_refresh = st.toggle("Auto-refresh (30s)", key="log_auto_refresh")
        if auto_refresh:
            import time
            if "log_last_refresh" not in st.session_state:
                st.session_state.log_last_refresh = time.time()
            if time.time() - st.session_state.log_last_refresh > 30:
                st.session_state.log_last_refresh = time.time()
                st.rerun()

        lines = _read_log_tail("system_*.log", max_lines=200)

        if lines:
            colored = [_colorize_log_line(l) for l in lines if l.strip()]
            html_block = "<br>".join(colored)
            st.markdown(
                f'<div style="background:#0E1117;border-radius:8px;'
                f'padding:16px;max-height:500px;overflow-y:auto;">'
                f'{html_block}</div>',
                unsafe_allow_html=True,
            )

            # Download button
            full_log = "".join(lines)
            st.download_button(
                "⬇️ Download Full Log",
                data=full_log,
                file_name=f"system_log_{datetime.now():%Y%m%d_%H%M}.txt",
                mime="text/plain",
            )
        else:
            st.info("No system log files found. Logs appear in `data/logs/system_*.log`.")

    # ─── 2. TRADE LOG TABLE ──────────────────────────────────────────
    with tab2:
        st.subheader("Trade History")

        ms = _get_memory_store()
        if ms is None:
            st.error("Memory store unavailable")
            return

        all_trades = ms.get_trades(limit=5000)

        if not all_trades:
            st.info("No trades logged yet.")
            return

        trades_df = pd.DataFrame(all_trades)

        # Filters
        filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)

        with filter_col1:
            strategies = ["All"] + sorted(trades_df["strategy"].dropna().unique().tolist())
            strat_filter = st.selectbox("Strategy", strategies, key="trade_strat")

        with filter_col2:
            symbols = ["All"] + sorted(trades_df["symbol"].dropna().unique().tolist())
            sym_filter = st.selectbox("Symbol", symbols, key="trade_sym")

        with filter_col3:
            wl_filter = st.selectbox("Result", ["All", "Winners", "Losers"], key="trade_wl")

        with filter_col4:
            if "entry_date" in trades_df.columns:
                date_range = st.date_input(
                    "Date range",
                    value=[],
                    key="trade_date_range",
                )
            else:
                date_range = []

        # Apply filters
        df = trades_df.copy()
        if strat_filter != "All":
            df = df[df["strategy"] == strat_filter]
        if sym_filter != "All":
            df = df[df["symbol"] == sym_filter]
        if wl_filter == "Winners" and "pnl_dollars" in df.columns:
            df = df[df["pnl_dollars"] > 0]
        elif wl_filter == "Losers" and "pnl_dollars" in df.columns:
            df = df[df["pnl_dollars"] <= 0]
        if date_range and len(date_range) == 2 and "entry_date" in df.columns:
            df = df[
                (pd.to_datetime(df["entry_date"]) >= pd.Timestamp(date_range[0]))
                & (pd.to_datetime(df["entry_date"]) <= pd.Timestamp(date_range[1]))
            ]

        # Display columns
        display_cols = [
            "trade_id", "symbol", "strategy", "entry_date", "exit_date",
            "entry_price", "exit_price", "pnl_pct", "hold_days", "exit_reason",
        ]
        available = [c for c in display_cols if c in df.columns]
        show_df = df[available].copy()

        # Format
        if "pnl_pct" in show_df.columns:
            show_df["pnl_pct"] = show_df["pnl_pct"].apply(
                lambda x: f"{x:+.2%}" if pd.notna(x) else "—"
            )
        for col in ["entry_price", "exit_price"]:
            if col in show_df.columns:
                show_df[col] = show_df[col].apply(
                    lambda x: f"${x:,.2f}" if pd.notna(x) else "—"
                )

        st.dataframe(show_df, use_container_width=True, height=450, hide_index=True)

        # Summary
        total = len(df)
        if "pnl_dollars" in df.columns:
            wins = (df["pnl_dollars"] > 0).sum()
            total_pnl = df["pnl_dollars"].sum()
            st.caption(
                f"Showing {total} trades  |  "
                f"Win rate: {wins/total:.0%}  |  "
                f"Total P&L: ${total_pnl:,.2f}"
                if total > 0 else "No matching trades"
            )

        # Export to CSV
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        st.download_button(
            "⬇️ Export to CSV",
            data=csv_buffer.getvalue(),
            file_name=f"trade_log_{datetime.now():%Y%m%d}.csv",
            mime="text/csv",
        )

    # ─── 3. ERROR LOG ────────────────────────────────────────────────
    with tab3:
        st.subheader("Error Log (Last 50)")

        lines = _read_log_tail("system_*.log", max_lines=2000)
        error_lines = [l for l in lines if "ERROR" in l or "CRITICAL" in l]
        error_lines = error_lines[-50:]

        if error_lines:
            for i, line in enumerate(reversed(error_lines)):
                stripped = line.strip()
                with st.expander(
                    stripped[:100] + ("…" if len(stripped) > 100 else ""),
                    expanded=False,
                ):
                    st.code(stripped, language="log")
        else:
            st.success("✅ No errors found in the recent log.")

    # ─── 4. DATA FETCH LOG ───────────────────────────────────────────
    with tab4:
        st.subheader("Data Fetch Log")

        fetch_lines = _read_log_tail("system_*.log", max_lines=2000)
        data_lines = [
            l for l in fetch_lines
            if any(kw in l for kw in ["OHLCV", "fetched", "cache hit", "cache stale", "DataAgent"])
        ]
        data_lines = data_lines[-100:]

        if data_lines:
            fetch_records = []
            for line in data_lines:
                record = {"raw": line.strip()}
                # Try to parse structured info
                if "symbol=" in line:
                    m = re.search(r"symbol=(\w+)", line)
                    if m:
                        record["symbol"] = m.group(1)
                if "rows=" in line:
                    m = re.search(r"rows=(\d+)", line)
                    if m:
                        record["rows"] = int(m.group(1))
                if "source=" in line:
                    m = re.search(r"source=(\w+)", line)
                    if m:
                        record["source"] = m.group(1)
                if "range=" in line:
                    m = re.search(r"range=(\S+)", line)
                    if m:
                        record["range"] = m.group(1)

                fetch_records.append(record)

            df_fetch = pd.DataFrame(fetch_records)
            display = ["symbol", "source", "rows", "range"]
            available = [c for c in display if c in df_fetch.columns]

            if available:
                st.dataframe(df_fetch[available], use_container_width=True, hide_index=True)
            else:
                # Fallback: show raw lines
                for line in data_lines[-20:]:
                    st.text(line.strip())
        else:
            st.info("No data fetch entries found in logs.")


# ── Entry point ─────────────────────────────────────────────────────
render()
