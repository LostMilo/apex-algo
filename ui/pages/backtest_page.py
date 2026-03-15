"""
pages/backtest_page.py — Run Backtests from the UI

Date range selector, initial capital input, run button,
progress bar, equity curve, metrics table, monthly returns heatmap,
per-strategy contribution chart.
"""

import json
from datetime import datetime
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

import config

GREEN = "#00D97E"
RED = "#FF4B4B"
BLUE = "#00B4D8"
YELLOW = "#FFD93D"

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif"),
    margin=dict(l=40, r=20, t=40, b=30),
)


# ── Page ─────────────────────────────────────────────────────────────

st.markdown("## 🧪 Backtest Engine")

# ── Controls ─────────────────────────────────────────────────────────
col_c1, col_c2, col_c3, col_c4 = st.columns([2, 2, 2, 1])

with col_c1:
    start_date = st.date_input(
        "Start Date",
        value=pd.to_datetime(config.BACKTEST_START),
        min_value=pd.to_datetime("2000-01-01"),
    )
with col_c2:
    end_date = st.date_input(
        "End Date",
        value=pd.to_datetime(datetime.now().strftime("%Y-%m-%d")),
    )
with col_c3:
    initial_capital = st.number_input(
        "Initial Capital ($)",
        min_value=100.0,
        max_value=1_000_000.0,
        value=float(config.STARTING_CAPITAL),
        step=100.0,
    )
with col_c4:
    st.markdown("<br>", unsafe_allow_html=True)
    run_clicked = st.button("▶ Run Backtest", use_container_width=True, type="primary")

st.markdown("---")

# ── Run / Load Results ───────────────────────────────────────────────

def _run_backtest(start: str, end: str, capital: float):
    """Execute the backtest engine and return results."""
    try:
        from backtest.engine import BacktestEngine
        engine = BacktestEngine(
            start_date=start,
            end_date=end,
            initial_capital=capital,
        )
        results = engine.run()
        return results, engine.equity_curve, engine.trades
    except Exception as e:
        st.error(f"Backtest failed: {e}")
        return None, None, None


def _load_latest_results():
    """Load the most recent saved backtest results."""
    results_dir = Path("data/backtest_results")
    if not results_dir.exists():
        return None, None, None

    metric_files = sorted(results_dir.glob("metrics_*.json"), reverse=True)
    curve_files = sorted(results_dir.glob("equity_curve_*.json"), reverse=True)
    trade_files = sorted(results_dir.glob("trades_*.json"), reverse=True)

    metrics, curve, trades = None, None, None

    if metric_files:
        with open(metric_files[0]) as f:
            metrics = json.load(f)
    if curve_files:
        with open(curve_files[0]) as f:
            curve = json.load(f)
    if trade_files:
        with open(trade_files[0]) as f:
            trades = json.load(f)

    return metrics, curve, trades


if run_clicked:
    progress_bar = st.progress(0, text="Initializing backtest engine...")

    progress_bar.progress(10, text="Loading market data...")
    progress_bar.progress(25, text="Running backtest — this may take a few minutes...")

    results, equity_curve, trades = _run_backtest(
        str(start_date), str(end_date), initial_capital,
    )

    progress_bar.progress(100, text="✅ Backtest complete!")

    if results:
        st.session_state["bt_results"] = results
        st.session_state["bt_equity"] = equity_curve
        st.session_state["bt_trades"] = trades
else:
    # Try loading from session or disk
    if "bt_results" not in st.session_state:
        results, equity_curve, trades = _load_latest_results()
        if results:
            st.session_state["bt_results"] = results
            st.session_state["bt_equity"] = equity_curve
            st.session_state["bt_trades"] = trades

# ── Display Results ──────────────────────────────────────────────────
results = st.session_state.get("bt_results")
equity_curve = st.session_state.get("bt_equity")
trades = st.session_state.get("bt_trades")

if not results:
    st.info(
        "No backtest results available. Configure your parameters above and click **▶ Run Backtest**, "
        "or previous results will be loaded automatically if available."
    )
    st.stop()

# ── Metrics Cards ────────────────────────────────────────────────────
st.markdown("### Performance Summary")

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Final Equity", f"${results.get('final_equity', 0):,.2f}")
m2.metric("Total Return", f"{results.get('total_return', 0):.2%}")
m3.metric("Annual Return", f"{results.get('annual_return', 0):.2%}")
m4.metric("Sharpe Ratio", f"{results.get('sharpe_ratio', 0):.3f}")
m5.metric("Max Drawdown", f"{results.get('max_drawdown', 0):.2%}")
m6.metric("Win Rate", f"{results.get('win_rate', 0):.1%}")

# Secondary metrics
m7, m8, m9, m10 = st.columns(4)
m7.metric("Total Trades", results.get("total_trades", 0))
m8.metric("Closing Trades", results.get("closing_trades", 0))
m9.metric("Avg Trade P&L", f"${results.get('avg_trade_pnl', 0):.2f}")
m10.metric("Years", f"{results.get('years', 0):.1f}")

st.markdown("---")

# ── Equity Curve ─────────────────────────────────────────────────────
st.markdown("### Equity Curve")

if equity_curve:
    eq_df = pd.DataFrame(equity_curve)
    eq_df["date"] = pd.to_datetime(eq_df["date"])

    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(
        x=eq_df["date"],
        y=eq_df["equity"],
        mode="lines",
        line=dict(color=GREEN, width=2),
        fill="tozeroy",
        fillcolor="rgba(0,217,126,0.06)",
        name="Equity",
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>$%{y:,.2f}<extra></extra>",
    ))
    # Starting capital reference line
    fig_eq.add_hline(
        y=results.get("initial_capital", config.STARTING_CAPITAL),
        line_dash="dot", line_color=YELLOW,
        annotation_text="Starting Capital",
    )
    fig_eq.update_layout(
        **PLOTLY_LAYOUT,
        height=400,
        yaxis=dict(title="Equity ($)", gridcolor="#21262D"),
        xaxis=dict(gridcolor="#21262D"),
        showlegend=False,
    )
    st.plotly_chart(fig_eq, use_container_width=True)

# ── Monthly Returns Heatmap ──────────────────────────────────────────
if equity_curve:
    st.markdown("### Monthly Returns Heatmap")

    eq_df_idx = eq_df.set_index("date")
    monthly = eq_df_idx["equity"].resample("ME").last()
    monthly_returns = monthly.pct_change().dropna()

    if len(monthly_returns) > 1:
        # Build year x month matrix
        years = sorted(monthly_returns.index.year.unique())
        months = list(range(1, 13))
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        heat_matrix = np.full((len(years), 12), np.nan)
        for dt, ret in monthly_returns.items():
            y_idx = years.index(dt.year)
            m_idx = dt.month - 1
            heat_matrix[y_idx][m_idx] = ret * 100  # percent

        fig_monthly = go.Figure(data=go.Heatmap(
            z=heat_matrix,
            x=month_names,
            y=[str(y) for y in years],
            colorscale=[
                [0.0, RED],
                [0.5, "#161B22"],
                [1.0, GREEN],
            ],
            zmid=0,
            text=np.where(
                np.isnan(heat_matrix), "",
                np.char.add(np.char.mod("%.1f", np.nan_to_num(heat_matrix)), "%"),
            ),
            texttemplate="%{text}",
            textfont=dict(size=10, color="white"),
            hovertemplate="<b>%{y} %{x}</b><br>Return: %{z:.1f}%<extra></extra>",
            colorbar=dict(title="Return %"),
        ))
        fig_monthly.update_layout(
            **PLOTLY_LAYOUT,
            height=max(200, len(years) * 35 + 60),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_monthly, use_container_width=True)

# ── Per-Strategy Contribution ────────────────────────────────────────
if trades:
    st.markdown("### Per-Strategy Contribution")

    # Group trades by strategy
    strategy_pnl = {}
    strategy_count = {}
    for t in trades:
        strat = t.get("strategy", "Unknown")
        pnl = t.get("pnl", 0)
        strategy_pnl[strat] = strategy_pnl.get(strat, 0) + pnl
        strategy_count[strat] = strategy_count.get(strat, 0) + 1

    col_bar, col_table = st.columns([3, 2])

    with col_bar:
        strats = list(strategy_pnl.keys())
        pnls = list(strategy_pnl.values())
        bar_colors = [GREEN if p >= 0 else RED for p in pnls]

        fig_strat = go.Figure()
        fig_strat.add_trace(go.Bar(
            x=strats,
            y=pnls,
            marker_color=bar_colors,
            text=[f"${p:,.2f}" for p in pnls],
            textposition="outside",
            textfont=dict(color="white", size=12),
            hovertemplate="<b>%{x}</b><br>P&L: $%{y:,.2f}<extra></extra>",
        ))
        fig_strat.update_layout(
            **PLOTLY_LAYOUT,
            height=350,
            yaxis=dict(title="Total P&L ($)", gridcolor="#21262D"),
            xaxis=dict(title="Strategy"),
            showlegend=False,
        )
        st.plotly_chart(fig_strat, use_container_width=True)

    with col_table:
        strat_df = pd.DataFrame([
            {
                "Strategy": s,
                "Trades": strategy_count.get(s, 0),
                "Total P&L": f"${strategy_pnl[s]:,.2f}",
                "Avg P&L": f"${strategy_pnl[s] / max(strategy_count.get(s, 1), 1):,.2f}",
            }
            for s in strats
        ])
        st.dataframe(strat_df, hide_index=True, use_container_width=True)
