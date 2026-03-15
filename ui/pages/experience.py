"""
pages/experience.py — Experience Agent & Learning System Dashboard

Shows: learning summary, strategy weights, lessons, Monte Carlo results,
win/loss breakdown by strategy and regime.
"""

import json
from datetime import datetime, timedelta

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
PURPLE = "#A855F7"
MUTED = "#8B8D97"

LESSON_COLORS = {
    "win_pattern": "rgba(0, 217, 126, 0.1)",
    "failure_pattern": "rgba(255, 75, 75, 0.1)",
    "weight_update": "rgba(0, 180, 216, 0.1)",
}


# ── Memory store helper ─────────────────────────────────────────────

def _get_memory_store():
    """Return the MemoryStore from session state, or create one."""
    ms = st.session_state.get("memory_store")
    if ms is None:
        try:
            from learning.memory_store import MemoryStore
            ms = MemoryStore(config.MEMORY_DB_PATH)
            st.session_state.memory_store = ms
        except Exception:
            return None
    return ms


def _get_weight_history(ms) -> pd.DataFrame:
    """Query agent_weights table for full weight history."""
    try:
        rows = ms._conn.execute(
            "SELECT * FROM agent_weights ORDER BY id ASC"
        ).fetchall()
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame([dict(r) for r in rows])
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    except Exception:
        return pd.DataFrame()


def _get_lesson_count(ms) -> int:
    """Get total lesson count."""
    try:
        row = ms._conn.execute("SELECT COUNT(*) AS cnt FROM lessons").fetchone()
        return row["cnt"] if row else 0
    except Exception:
        return 0


# ═════════════════════════════════════════════════════════════════════
# PAGE RENDER
# ═════════════════════════════════════════════════════════════════════

def render():
    st.header("🧠 Experience Agent — Learning System")

    ms = _get_memory_store()
    if ms is None:
        st.error("Memory store unavailable — check database path in config.py")
        return

    # ─── 1. SUMMARY CARDS ────────────────────────────────────────────
    st.subheader("Learning Summary")

    trade_count = ms.get_trade_count()
    lesson_count = _get_lesson_count(ms)
    latest_weights = ms.get_latest_weights()

    w_tsmom = latest_weights.get("tsmom_weight", 0.33) if latest_weights else 0.33
    w_vol = latest_weights.get("vol_trend_weight", 0.33) if latest_weights else 0.33
    w_arb = latest_weights.get("pairs_arb_weight", 0.34) if latest_weights else 0.34
    last_update = latest_weights.get("timestamp", "—") if latest_weights else "—"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Trades Analyzed", f"{trade_count:,}")
    c2.metric("Lessons Generated", f"{lesson_count:,}")
    c3.metric("Last Weight Update", str(last_update)[:10] if last_update != "—" else "—")
    c4.metric(
        "Weights (T / V / P)",
        f"{w_tsmom:.0%} / {w_vol:.0%} / {w_arb:.0%}",
    )

    st.divider()

    # ─── 2. STRATEGY WEIGHTS CHARTS ─────────────────────────────────
    st.subheader("Strategy Weights")
    wt_col1, wt_col2 = st.columns([1, 2])

    with wt_col1:
        # Pie chart
        fig_pie = go.Figure(data=[go.Pie(
            labels=["TSMOM", "VolTrend", "PairsArb"],
            values=[w_tsmom, w_vol, w_arb],
            hole=0.45,
            marker=dict(colors=[BLUE, GREEN, PURPLE]),
            textinfo="label+percent",
            textfont=dict(size=13),
        )])
        fig_pie.update_layout(
            **PLOTLY_LAYOUT,
            title="Current Allocation",
            height=320,
            showlegend=False,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with wt_col2:
        # Weight history line chart
        wh = _get_weight_history(ms)
        if not wh.empty and "timestamp" in wh.columns:
            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(
                x=wh["timestamp"], y=wh["tsmom_weight"],
                name="TSMOM", line=dict(color=BLUE, width=2),
            ))
            fig_line.add_trace(go.Scatter(
                x=wh["timestamp"], y=wh["vol_trend_weight"],
                name="VolTrend", line=dict(color=GREEN, width=2),
            ))
            fig_line.add_trace(go.Scatter(
                x=wh["timestamp"], y=wh["pairs_arb_weight"],
                name="PairsArb", line=dict(color=PURPLE, width=2),
            ))
            fig_line.update_layout(
                **PLOTLY_LAYOUT,
                title="Weight History Over Time",
                height=320,
                yaxis_title="Weight",
                yaxis=dict(range=[0, 1]),
                legend=dict(orientation="h", y=-0.15),
            )
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("No weight history available yet — weights update after sufficient trades.")

    st.divider()

    # ─── 3. RECENT LESSONS TABLE ────────────────────────────────────
    st.subheader("Recent Lessons")

    filter_type = st.selectbox(
        "Filter by type",
        ["All", "Win Patterns", "Failure Patterns", "Weight Updates"],
        key="lesson_filter",
    )

    lessons = ms.get_lessons(since_days=90, unread_only=False)

    if filter_type == "Win Patterns":
        lessons = [l for l in lessons if l.get("lesson_type") == "win_pattern"]
    elif filter_type == "Failure Patterns":
        lessons = [l for l in lessons if l.get("lesson_type") == "failure_pattern"]
    elif filter_type == "Weight Updates":
        lessons = [l for l in lessons if l.get("lesson_type") == "weight_update"]

    if lessons:
        lesson_df = pd.DataFrame(lessons)
        display_cols = [
            "timestamp", "source_agent", "lesson_type",
            "condition", "recommended_action", "confidence",
        ]
        available = [c for c in display_cols if c in lesson_df.columns]
        show_df = lesson_df[available].copy()
        if "timestamp" in show_df.columns:
            show_df["timestamp"] = pd.to_datetime(show_df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M")
        if "confidence" in show_df.columns:
            show_df["confidence"] = show_df["confidence"].apply(lambda x: f"{x:.1%}" if x else "—")

        # Color-code by type
        def _row_color(row):
            lt = row.get("lesson_type", "")
            bg = LESSON_COLORS.get(lt, "")
            return [f"background-color: {bg}"] * len(row) if bg else [""] * len(row)

        styled = show_df.style.apply(_row_color, axis=1)
        st.dataframe(styled, use_container_width=True, height=350)
    else:
        st.info("No lessons found — the experience agent generates lessons after trades complete.")

    st.divider()

    # ─── 4. MONTE CARLO RESULTS ──────────────────────────────────────
    st.subheader("Monte Carlo Skill Analysis")

    strategies = ["TSMOM", "VolTrend", "PairsArb"]
    mc_cols = st.columns(3)

    for i, strat in enumerate(strategies):
        with mc_cols[i]:
            perf = ms.get_strategy_performance(strat.lower() if strat != "TSMOM" else "tsmom")

            total = perf.get("total_trades", 0)
            wr = perf.get("win_rate", 0.0)

            if total >= 20:
                # Simulate: random trades with same win-rate distribution
                np.random.seed(42 + i)
                n_sims = 5000
                sim_returns = []
                for _ in range(n_sims):
                    random_wins = np.random.binomial(total, 0.5) / total
                    sim_returns.append(random_wins)
                skill_prob = sum(1 for s in sim_returns if s < wr) / n_sims
                worst_dd_95 = np.percentile(
                    [min(0, perf.get("worst_trade_pnl", 0) * np.random.uniform(0.5, 2))
                     for _ in range(n_sims)], 5
                )
            else:
                skill_prob = 0.0
                worst_dd_95 = 0.0

            # Gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=skill_prob * 100,
                title={"text": strat, "font": {"size": 16}},
                number={"suffix": "%", "font": {"size": 28}},
                gauge=dict(
                    axis=dict(range=[0, 100]),
                    bar=dict(color=GREEN if skill_prob > 0.7 else YELLOW if skill_prob > 0.5 else RED),
                    steps=[
                        {"range": [0, 50], "color": "rgba(255,75,75,0.15)"},
                        {"range": [50, 70], "color": "rgba(255,217,61,0.15)"},
                        {"range": [70, 100], "color": "rgba(0,217,126,0.15)"},
                    ],
                ),
            ))
            fig_gauge.update_layout(
                **PLOTLY_LAYOUT, height=200,
                margin=dict(l=20, r=20, t=50, b=10),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            st.caption(
                f"Trades: {total}  |  Win rate: {wr:.0%}  |  "
                f"Worst DD (95%): {worst_dd_95:.1%}" if total > 0
                else "Insufficient trades for analysis"
            )

    st.divider()

    # ─── 5. WIN/LOSS BREAKDOWN ───────────────────────────────────────
    st.subheader("Win / Loss Breakdown")

    all_trades = ms.get_trades(limit=1000)

    if all_trades:
        trades_df = pd.DataFrame(all_trades)

        wr_col1, wr_col2 = st.columns(2)

        with wr_col1:
            st.markdown("**Win Rate by Strategy**")
            if "strategy" in trades_df.columns and "pnl_dollars" in trades_df.columns:
                strat_stats = []
                for strat_name, grp in trades_df.groupby("strategy"):
                    wins = (grp["pnl_dollars"] > 0).sum()
                    total_t = len(grp)
                    strat_stats.append({
                        "Strategy": strat_name,
                        "Win Rate": wins / total_t if total_t else 0,
                        "Trades": total_t,
                    })
                ss_df = pd.DataFrame(strat_stats)
                if not ss_df.empty:
                    fig_wr = px.bar(
                        ss_df, x="Strategy", y="Win Rate",
                        text=ss_df["Win Rate"].apply(lambda x: f"{x:.0%}"),
                        color="Win Rate",
                        color_continuous_scale=["#FF4B4B", "#FFD93D", "#00D97E"],
                        range_color=[0, 1],
                    )
                    fig_wr.update_layout(**PLOTLY_LAYOUT, height=300, showlegend=False)
                    fig_wr.update_traces(textposition="outside")
                    st.plotly_chart(fig_wr, use_container_width=True)

        with wr_col2:
            st.markdown("**Win Rate by Market Regime**")
            if "regime_at_entry" in trades_df.columns and "pnl_dollars" in trades_df.columns:
                regime_stats = []
                for regime_name, grp in trades_df.groupby("regime_at_entry"):
                    if not regime_name:
                        continue
                    wins = (grp["pnl_dollars"] > 0).sum()
                    total_t = len(grp)
                    regime_stats.append({
                        "Regime": regime_name,
                        "Win Rate": wins / total_t if total_t else 0,
                        "Trades": total_t,
                    })
                rg_df = pd.DataFrame(regime_stats)
                if not rg_df.empty:
                    fig_rg = px.bar(
                        rg_df, x="Regime", y="Win Rate",
                        text=rg_df["Win Rate"].apply(lambda x: f"{x:.0%}"),
                        color="Win Rate",
                        color_continuous_scale=["#FF4B4B", "#FFD93D", "#00D97E"],
                        range_color=[0, 1],
                    )
                    fig_rg.update_layout(**PLOTLY_LAYOUT, height=300, showlegend=False)
                    fig_rg.update_traces(textposition="outside")
                    st.plotly_chart(fig_rg, use_container_width=True)
                else:
                    st.info("No regime data in trade log yet.")
            else:
                st.info("No regime data in trade log yet.")

        # P&L by exit reason
        st.markdown("**Average P&L by Exit Reason**")
        if "exit_reason" in trades_df.columns and "pnl_dollars" in trades_df.columns:
            exit_stats = (
                trades_df.groupby("exit_reason")
                .agg(
                    avg_pnl=("pnl_dollars", "mean"),
                    count=("pnl_dollars", "count"),
                    total_pnl=("pnl_dollars", "sum"),
                )
                .reset_index()
                .sort_values("count", ascending=False)
            )
            exit_stats["avg_pnl"] = exit_stats["avg_pnl"].apply(lambda x: f"${x:,.2f}")
            exit_stats["total_pnl"] = exit_stats["total_pnl"].apply(lambda x: f"${x:,.2f}")
            st.dataframe(exit_stats, use_container_width=True, hide_index=True)
        else:
            st.info("No exit reason data in trade log yet.")

    else:
        st.info("No trades logged yet — the system will populate this after executing trades.")


# ── Entry point ─────────────────────────────────────────────────────
render()
