"""
pages/overview.py — Portfolio Overview Page

Equity curve, key metrics cards, regime indicator, kill switch,
daily P&L bar chart, asset allocation pie chart.
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

import config

# ── Plotly defaults ──────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif"),
    margin=dict(l=40, r=20, t=40, b=30),
)
GREEN = "#00D97E"
RED = "#FF4B4B"
BLUE = "#00B4D8"
YELLOW = "#FFD93D"
MUTED = "#8B8D97"


# ── Helpers ──────────────────────────────────────────────────────────

def _load_equity_curve() -> pd.DataFrame:
    """Load the most recent equity curve from backtest results or trade journal."""
    # Try backtest results first
    results_dir = Path("data/backtest_results")
    if results_dir.exists():
        curve_files = sorted(results_dir.glob("equity_curve_*.json"), reverse=True)
        if curve_files:
            with open(curve_files[0]) as f:
                data = json.load(f)
            if data:
                df = pd.DataFrame(data)
                df["date"] = pd.to_datetime(df["date"])
                df.set_index("date", inplace=True)
                return df

    # Fallback: build from trade journal
    journal_file = Path("data/trade_journal.json")
    if journal_file.exists():
        with open(journal_file) as f:
            trades = json.load(f)
        if trades:
            rows = []
            for t in trades:
                rows.append({
                    "date": pd.to_datetime(t.get("timestamp", "")),
                    "equity": t.get("equity_at_trade", config.STARTING_CAPITAL),
                })
            if rows:
                df = pd.DataFrame(rows).set_index("date").sort_index()
                return df

    # No data — generate placeholder
    dates = pd.date_range(end=datetime.now(), periods=30, freq="B")
    equity = config.STARTING_CAPITAL + np.cumsum(np.random.randn(30) * (config.STARTING_CAPITAL * 0.005))
    return pd.DataFrame({"equity": equity}, index=dates)


def _load_latest_metrics() -> dict:
    """Load latest backtest metrics if available."""
    results_dir = Path("data/backtest_results")
    if results_dir.exists():
        metric_files = sorted(results_dir.glob("metrics_*.json"), reverse=True)
        if metric_files:
            with open(metric_files[0]) as f:
                return json.load(f)
    return {}


def _get_current_regime() -> str:
    """Detect the current benchmark regime (best-effort)."""
    try:
        from data.data_agent import DataAgent
        from core.regime_detector import RegimeDetector
        agent = DataAgent()
        data = agent.get_ohlcv(config.BENCHMARK, start="2024-01-01", end="today")
        if not data.empty:
            detector = RegimeDetector()
            regime = detector.detect({config.BENCHMARK: data}, data.index[-1])
            return regime
    except Exception:
        pass
    return "UNKNOWN"


def _build_daily_pnl(positions: list) -> pd.DataFrame:
    """Build daily P&L from trade journal for last 30 days."""
    journal_file = Path("data/trade_journal.json")
    if journal_file.exists():
        try:
            with open(journal_file) as f:
                trades = json.load(f)
            if trades:
                rows = []
                for t in trades:
                    ts = pd.to_datetime(t.get("timestamp", ""))
                    eq = t.get("equity_at_trade", config.STARTING_CAPITAL)
                    rows.append({"date": ts.date(), "equity": eq})
                if rows:
                    df = pd.DataFrame(rows)
                    df = df.groupby("date").last().sort_index()
                    df["pnl"] = df["equity"].diff().fillna(0)
                    return df.tail(30)
        except Exception:
            pass

    # Fallback: synthetic data
    dates = pd.date_range(end=datetime.now(), periods=30, freq="B")
    pnl = np.random.randn(30) * (config.STARTING_CAPITAL * 0.01)
    return pd.DataFrame({"pnl": pnl}, index=dates)


# ── Page Render ──────────────────────────────────────────────────────

st.markdown("## 📊 Portfolio Overview")

# ── Row 1: Key Metrics ──────────────────────────────────────────────
metrics = _load_latest_metrics()
acct = st.session_state.get("broker_account")
positions = st.session_state.get("broker_positions", [])

equity = acct["equity"] if acct else config.STARTING_CAPITAL
total_pnl = equity - config.STARTING_CAPITAL
daily_pnl = sum(p.get("unrealized_pl", 0) for p in positions)

# Compute days running
start_date = pd.to_datetime(metrics.get("start_date", config.BACKTEST_START))
days_running = (datetime.now() - start_date).days

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    pnl_delta = f"{'+' if total_pnl >= 0 else ''}{total_pnl / config.STARTING_CAPITAL * 100:.1f}%"
    st.metric("Total P&L", f"${total_pnl:,.2f}", delta=pnl_delta)
with col2:
    sharpe = metrics.get("sharpe_ratio", 0)
    st.metric("Sharpe Ratio", f"{sharpe:.3f}")
with col3:
    max_dd = metrics.get("max_drawdown", 0)
    st.metric("Max Drawdown", f"{max_dd:.2%}")
with col4:
    win_rate = metrics.get("win_rate", 0)
    st.metric("Win Rate", f"{win_rate:.1%}")
with col5:
    st.metric("Days Running", f"{days_running:,}")

# ── Paper Trading Warmup ─────────────────────────────────────────
try:
    from learning.memory_store import MemoryStore
    _ms = MemoryStore(config.MEMORY_DB_PATH)
    warmup_days = _ms.get_paper_trading_days()
    warmup_target = getattr(config, "WARMUP_DAYS", 90)
    warmup_pct = min(warmup_days / warmup_target, 1.0)
    warmup_complete = warmup_days >= warmup_target

    if warmup_complete:
        st.success(f"✅ Paper trading warmup complete — {warmup_days} days. Live trading eligible.")
    else:
        from datetime import timedelta as _td
        est_live = (datetime.now() + _td(days=warmup_target - warmup_days)).strftime("%Y-%m-%d")
        st.progress(warmup_pct, text=f"📋 Paper Warmup: {warmup_days} / {warmup_target} days  |  Est. live start: {est_live}")
except Exception:
    pass  # Don't break overview if memory store unavailable

st.markdown("---")

# ── Row 2: Regime + Kill Switch ──────────────────────────────────────
col_regime, col_kill = st.columns([2, 1])

with col_regime:
    regime = _get_current_regime()
    regime_colors = {"TRENDING": "green", "RANGING": "yellow", "RISK_OFF": "red", "UNKNOWN": "yellow"}
    regime_emojis = {"TRENDING": "🟢", "RANGING": "🟡", "RISK_OFF": "🔴", "UNKNOWN": "⚪"}
    badge_cls = f"badge-{regime_colors.get(regime, 'yellow')}"

    st.markdown(f"""
    <div style="padding:1rem; background:#161B22; border:1px solid #21262D; border-radius:12px; text-align:center;">
        <p style="color:#8B8D97; font-size:0.7rem; text-transform:uppercase; letter-spacing:0.08em; margin:0;">
            Current Market Regime ({config.BENCHMARK})
        </p>
        <p style="font-size:2.2rem; font-weight:800; margin:0.3rem 0;">
            {regime_emojis.get(regime, '⚪')}
        </p>
        <span class="badge {badge_cls}" style="font-size:0.9rem; padding:0.3rem 1.2rem;">
            {regime}
        </span>
    </div>
    """, unsafe_allow_html=True)

with col_kill:
    st.markdown("""
    <div style="padding:1rem; background:#161B22; border:1px solid #21262D; border-radius:12px;">
        <p style="color:#8B8D97; font-size:0.7rem; text-transform:uppercase; letter-spacing:0.08em; margin:0 0 0.5rem 0; text-align:center;">
            Kill Switch Control
        </p>
    </div>
    """, unsafe_allow_html=True)

    @st.dialog("⚠️ Kill Switch Confirmation")
    def _kill_switch_dialog():
        if st.session_state.kill_switch:
            st.warning("The kill switch is currently **ACTIVE**. All trading is halted.")
            st.write("Do you want to **deactivate** the kill switch and resume trading?")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("✅ Deactivate", use_container_width=True, type="primary"):
                    st.session_state.kill_switch = False
                    st.rerun()
            with c2:
                if st.button("Cancel", use_container_width=True):
                    st.rerun()
        else:
            st.error("This will **immediately halt all trading** and cancel pending orders.")
            st.write("Are you sure you want to activate the kill switch?")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("⛔ Activate Kill Switch", use_container_width=True, type="primary"):
                    st.session_state.kill_switch = True
                    # Try to cancel all orders
                    broker = st.session_state.get("broker")
                    if broker:
                        try:
                            broker.cancel_all_orders()
                        except Exception:
                            pass
                    st.rerun()
            with c2:
                if st.button("Cancel", use_container_width=True):
                    st.rerun()

    btn_label = "⛔ Deactivate Kill Switch" if st.session_state.kill_switch else "🛑 Activate Kill Switch"
    btn_type = "primary" if st.session_state.kill_switch else "secondary"
    if st.button(btn_label, use_container_width=True, type=btn_type):
        _kill_switch_dialog()

st.markdown("---")

# ── Row 3: Equity Curve ─────────────────────────────────────────────
st.markdown("### Equity Curve")
eq_df = _load_equity_curve()

fig_eq = go.Figure()
fig_eq.add_trace(go.Scatter(
    x=eq_df.index,
    y=eq_df["equity"],
    mode="lines",
    line=dict(color=GREEN, width=2.5),
    fill="tozeroy",
    fillcolor="rgba(0,217,126,0.08)",
    name="Equity",
    hovertemplate="<b>%{x|%b %d, %Y}</b><br>Equity: $%{y:,.2f}<extra></extra>",
))
fig_eq.update_layout(
    **PLOTLY_LAYOUT,
    height=350,
    yaxis=dict(title="Equity ($)", gridcolor="#21262D"),
    xaxis=dict(gridcolor="#21262D"),
    showlegend=False,
)
st.plotly_chart(fig_eq, use_container_width=True)

# ── Row 4: Daily P&L + Allocation ───────────────────────────────────
col_pnl, col_alloc = st.columns([3, 2])

with col_pnl:
    st.markdown("### Daily P&L (Last 30 Days)")
    pnl_df = _build_daily_pnl(positions)

    colors = [GREEN if v >= 0 else RED for v in pnl_df["pnl"]]
    fig_pnl = go.Figure()
    fig_pnl.add_trace(go.Bar(
        x=pnl_df.index,
        y=pnl_df["pnl"],
        marker_color=colors,
        hovertemplate="<b>%{x|%b %d}</b><br>P&L: $%{y:,.2f}<extra></extra>",
    ))
    fig_pnl.update_layout(
        **PLOTLY_LAYOUT,
        height=320,
        yaxis=dict(title="P&L ($)", gridcolor="#21262D", zeroline=True, zerolinecolor="#21262D"),
        xaxis=dict(gridcolor="#21262D"),
        showlegend=False,
    )
    st.plotly_chart(fig_pnl, use_container_width=True)

with col_alloc:
    st.markdown("### Asset Allocation")
    if positions:
        symbols = [p["symbol"] for p in positions]
        values = [abs(p.get("market_value", 0)) for p in positions]

        fig_pie = go.Figure()
        fig_pie.add_trace(go.Pie(
            labels=symbols,
            values=values,
            hole=0.55,
            marker=dict(
                colors=px.colors.qualitative.Set2[:len(symbols)],
                line=dict(color="#0E1117", width=2),
            ),
            textposition="inside",
            textinfo="label+percent",
            textfont=dict(size=11, color="white"),
            hovertemplate="<b>%{label}</b><br>$%{value:,.2f}<br>%{percent}<extra></extra>",
        ))
        fig_pie.update_layout(
            **PLOTLY_LAYOUT,
            height=320,
            showlegend=False,
            annotations=[dict(
                text=f"<b>{len(positions)}</b><br>Positions",
                x=0.5, y=0.5, font_size=14, font_color="#E6EDF3",
                showarrow=False,
            )],
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("No open positions to display.")
