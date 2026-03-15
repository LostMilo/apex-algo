"""
pages/positions.py — Open & Closed Positions Page

Open positions table (color-coded), manual close button with dialog,
historical closed trades table with filters.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

import config

GREEN = "#00D97E"
RED = "#FF4B4B"
MUTED = "#8B8D97"

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif"),
    margin=dict(l=40, r=20, t=40, b=30),
)


# ── Close-position dialog ────────────────────────────────────────────

@st.dialog("⚠️ Close Position")
def _confirm_close(symbol: str):
    st.warning(f"You are about to **close your entire position** in **{symbol}**.")
    st.write("This will submit a market close order to the broker.")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("✅ Confirm Close", use_container_width=True, type="primary"):
            broker = st.session_state.get("broker")
            if broker:
                try:
                    broker.close_position(symbol)
                    st.success(f"Close order submitted for {symbol}")
                except Exception as e:
                    st.error(f"Failed: {e}")
            else:
                st.error("Broker not connected")
            st.rerun()
    with c2:
        if st.button("Cancel", use_container_width=True):
            st.rerun()


# ── Page ─────────────────────────────────────────────────────────────

st.markdown("## 💼 Positions")

tab_open, tab_closed = st.tabs(["Open Positions", "Closed Trades"])

# ── Tab 1: Open Positions ────────────────────────────────────────────
with tab_open:
    positions = st.session_state.get("broker_positions", [])

    if not positions:
        st.info("No open positions. The system hasn't entered any trades yet.")
    else:
        # Build DataFrame
        rows = []
        for p in positions:
            pnl_pct = p.get("unrealized_plpc", 0) * 100
            rows.append({
                "Symbol": p["symbol"],
                "Side": p.get("side", "long").capitalize(),
                "Qty": f"{p['qty']:.2f}",
                "Entry Price": f"${p['avg_entry']:.2f}",
                "Current Price": f"${p['current_price']:.2f}",
                "Market Value": f"${p['market_value']:,.2f}",
                "P&L ($)": p["unrealized_pl"],
                "P&L (%)": pnl_pct,
            })

        df = pd.DataFrame(rows)

        # Color-coded table using Streamlit's column_config
        st.dataframe(
            df,
            column_config={
                "P&L ($)": st.column_config.NumberColumn(
                    "P&L ($)",
                    format="$%.2f",
                ),
                "P&L (%)": st.column_config.ProgressColumn(
                    "P&L (%)",
                    format="%.1f%%",
                    min_value=-20,
                    max_value=20,
                ),
            },
            hide_index=True,
            use_container_width=True,
        )

        # Close buttons
        st.markdown("#### Close a Position")
        cols = st.columns(min(len(positions), 5))
        for i, p in enumerate(positions):
            col_idx = i % min(len(positions), 5)
            with cols[col_idx]:
                pnl = p.get("unrealized_pl", 0)
                color = GREEN if pnl >= 0 else RED
                st.markdown(
                    f'<div style="text-align:center; padding:0.3rem; '
                    f'background:#161B22; border:1px solid #21262D; border-radius:8px; margin-bottom:0.3rem;">'
                    f'<span style="font-weight:700;">{p["symbol"]}</span><br>'
                    f'<span style="color:{color}; font-size:0.85rem;">'
                    f'{"+" if pnl >= 0 else ""}${pnl:.2f}</span></div>',
                    unsafe_allow_html=True,
                )
                if st.button(f"Close {p['symbol']}", key=f"close_{p['symbol']}", use_container_width=True):
                    _confirm_close(p["symbol"])

# ── Tab 2: Closed Trades ─────────────────────────────────────────────
with tab_closed:
    ms = st.session_state.get("memory_store")

    # Filters
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        strategy_filter = st.selectbox(
            "Strategy",
            ["All", "TSMOM", "VolTrend", "PairsArb", "Agg(TSMOM)", "Agg(VolTrend)", "Agg(PairsArb)"],
        )
    with col_f2:
        symbol_filter = st.text_input("Symbol (blank = all)", "")
    with col_f3:
        limit = st.number_input("Max rows", min_value=10, max_value=500, value=100, step=10)

    # Fetch trades
    trades = []
    if ms:
        strat = strategy_filter if strategy_filter != "All" else None
        trades = ms.get_trades(strategy=strat, limit=limit)

    # Also check trade journal on disk
    if not trades:
        import json
        from pathlib import Path
        journal = Path("data/trade_journal.json")
        if journal.exists():
            try:
                with open(journal) as f:
                    raw = json.load(f)
                trades = [
                    {
                        "symbol": t.get("symbol", ""),
                        "strategy": t.get("strategy", ""),
                        "entry_date": t.get("timestamp", ""),
                        "exit_date": "",
                        "entry_price": t.get("price", 0),
                        "exit_price": 0,
                        "pnl_dollars": 0,
                        "pnl_pct": 0,
                        "hold_days": 0,
                        "exit_reason": "",
                    }
                    for t in raw
                ]
            except Exception:
                trades = []

    if not trades:
        st.info("No closed trades recorded yet.")
    else:
        # Apply symbol filter
        if symbol_filter:
            trades = [t for t in trades if symbol_filter.upper() in t.get("symbol", "").upper()]

        # Build display DF
        display_rows = []
        for t in trades:
            pnl_d = t.get("pnl_dollars", 0) or 0
            pnl_p = (t.get("pnl_pct", 0) or 0) * 100
            display_rows.append({
                "Symbol": t.get("symbol", ""),
                "Strategy": t.get("strategy", ""),
                "Entry Date": t.get("entry_date", ""),
                "Exit Date": t.get("exit_date", ""),
                "Entry $": f"${t.get('entry_price', 0):.2f}",
                "Exit $": f"${t.get('exit_price', 0):.2f}" if t.get("exit_price") else "—",
                "P&L ($)": pnl_d,
                "P&L (%)": pnl_p,
                "Days": t.get("hold_days", 0),
                "Exit Reason": t.get("exit_reason", ""),
            })

        tdf = pd.DataFrame(display_rows)
        st.dataframe(
            tdf,
            column_config={
                "P&L ($)": st.column_config.NumberColumn("P&L ($)", format="$%.2f"),
                "P&L (%)": st.column_config.NumberColumn("P&L (%)", format="%.1f%%"),
            },
            hide_index=True,
            use_container_width=True,
        )

        # Summary stats
        if display_rows:
            total_pnl = sum(r["P&L ($)"] for r in display_rows)
            wins = sum(1 for r in display_rows if r["P&L ($)"] > 0)
            losses = sum(1 for r in display_rows if r["P&L ($)"] <= 0)
            wr = wins / max(wins + losses, 1)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Trades", len(display_rows))
            c2.metric("Total P&L", f"${total_pnl:,.2f}")
            c3.metric("Wins / Losses", f"{wins} / {losses}")
            c4.metric("Win Rate", f"{wr:.1%}")
