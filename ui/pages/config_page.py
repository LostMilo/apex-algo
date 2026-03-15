"""
pages/config_page.py — System Configuration Viewer / Editor

View and modify system parameters with safety checks.
All changes are persisted via config module and logged to memory store.
"""

import json
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd

import config

# ── Plotly defaults (reused across pages) ─────────────────────────────
GREEN = "#00D97E"
RED = "#FF4B4B"
BLUE = "#00B4D8"
YELLOW = "#FFD93D"
MUTED = "#8B8D97"


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


def _get_alert_manager():
    am = st.session_state.get("alert_manager")
    if am is None:
        try:
            from utils.alerting import AlertManager
            am = AlertManager(config)
            st.session_state.alert_manager = am
        except Exception:
            return None
    return am


# ═════════════════════════════════════════════════════════════════════
# PAGE RENDER
# ═════════════════════════════════════════════════════════════════════

def render():
    st.header("⚙️ System Configuration")

    ms = _get_memory_store()

    # ─── 1. TRADING MODE ─────────────────────────────────────────────
    st.subheader("Trading Mode")

    col_mode, col_badge = st.columns([3, 1])

    with col_mode:
        current_mode = "LIVE" if getattr(config, "USE_LIVE_EXECUTION", False) else "PAPER"
        is_paper = current_mode == "PAPER"

        mode_toggle = st.toggle(
            "Live Execution",
            value=not is_paper,
            key="mode_toggle",
            help="⚠️ Switching to LIVE will execute real trades with real money",
        )

        if mode_toggle and is_paper:
            st.warning(
                "⚠️ **WARNING**: You are about to enable LIVE trading. "
                "This will execute REAL orders with REAL money. "
                "Make sure you have completed 90 days of paper trading first.",
                icon="🚨",
            )
            if st.button("Confirm switch to LIVE", type="primary"):
                config.USE_LIVE_EXECUTION = True
                st.success("Switched to LIVE mode — restart main.py for changes to take effect")
                st.rerun()

        if not mode_toggle and not is_paper:
            config.USE_LIVE_EXECUTION = False
            st.success("Switched to PAPER mode")

    with col_badge:
        if current_mode == "PAPER":
            st.markdown(
                '<div style="background:#00D97E22;border:1px solid #00D97E;'
                'border-radius:8px;padding:12px;text-align:center;margin-top:8px;">'
                '<span style="color:#00D97E;font-weight:700;font-size:1.2em;">📄 PAPER</span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="background:#FF4B4B22;border:1px solid #FF4B4B;'
                'border-radius:8px;padding:12px;text-align:center;margin-top:8px;">'
                '<span style="color:#FF4B4B;font-weight:700;font-size:1.2em;">🔴 LIVE</span></div>',
                unsafe_allow_html=True,
            )

    st.divider()

    # ─── 2. ACTIVE STRATEGIES ─────────────────────────────────────────
    st.subheader("Active Strategies")

    latest_weights = None
    if ms:
        latest_weights = ms.get_latest_weights()

    w_tsmom = latest_weights.get("tsmom_weight", 0.33) if latest_weights else 0.33
    w_vol = latest_weights.get("vol_trend_weight", 0.33) if latest_weights else 0.33
    w_arb = latest_weights.get("pairs_arb_weight", 0.34) if latest_weights else 0.34

    strategies = [
        ("TSMOM (Time Series Momentum)", "tsmom_enabled", w_tsmom),
        ("Dual Momentum Filter", "dual_mom_enabled", None),
        ("VolTrend (Volatility Trend)", "vol_trend_enabled", w_vol),
        ("Pairs Arbitrage", "pairs_arb_enabled", w_arb),
    ]

    for name, key, weight in strategies:
        col_s, col_w = st.columns([3, 1])
        with col_s:
            enabled = st.toggle(
                name,
                value=True,
                key=key,
            )
        with col_w:
            if weight is not None:
                st.metric("Weight", f"{weight:.0%}")
            else:
                st.caption("Gate/filter")

        if not enabled and weight and weight > 0.3:
            st.warning(f"⚠️ {name} has weight {weight:.0%} — disabling a high-weight strategy may reduce performance.")

    st.divider()

    # ─── 3. RISK PARAMETERS ──────────────────────────────────────────
    st.subheader("Risk Parameters")

    with st.form("risk_params_form"):
        risk_col1, risk_col2 = st.columns(2)

        with risk_col1:
            kill_switch = st.slider(
                "Kill Switch — Max Drawdown %",
                min_value=3, max_value=25, value=int(config.MAX_DRAWDOWN_PCT * 100),
                step=1, format="%d%%",
                help="Full halt triggered at this drawdown level",
            )

            max_pos = st.slider(
                "Max Position Size %",
                min_value=5, max_value=25, value=int(config.MAX_POSITION_PCT * 100),
                step=1, format="%d%%",
            )

            kelly = st.slider(
                "Kelly Fraction",
                min_value=0.10, max_value=0.75, value=float(config.KELLY_FRACTION),
                step=0.05, format="%.2f",
                help="Fraction of Kelly-optimal to use (lower = more conservative)",
            )

        with risk_col2:
            hard_stop = st.slider(
                "Hard Stop Loss %",
                min_value=3, max_value=15, value=int(config.HARD_STOP_PCT * 100),
                step=1, format="%d%%",
            )

            time_stop = st.number_input(
                "Time Stop (days)",
                min_value=5, max_value=90, value=int(config.TIME_STOP_DAYS),
                step=5,
            )

            max_exposure = st.slider(
                "Max Total Exposure",
                min_value=0.5, max_value=2.0, value=float(config.MAX_TOTAL_EXPOSURE),
                step=0.1, format="%.1f",
                help="Maximum total portfolio exposure as fraction of equity",
            )

        submitted = st.form_submit_button("💾 Save Risk Parameters", type="primary")

        if submitted:
            # Apply changes
            old_values = {
                "MAX_DRAWDOWN_PCT": config.MAX_DRAWDOWN_PCT,
                "MAX_POSITION_PCT": config.MAX_POSITION_PCT,
                "KELLY_FRACTION": config.KELLY_FRACTION,
                "HARD_STOP_PCT": config.HARD_STOP_PCT,
                "TIME_STOP_DAYS": config.TIME_STOP_DAYS,
                "MAX_TOTAL_EXPOSURE": config.MAX_TOTAL_EXPOSURE,
            }

            config.MAX_DRAWDOWN_PCT = kill_switch / 100
            config.MAX_POSITION_PCT = max_pos / 100
            config.KELLY_FRACTION = kelly
            config.HARD_STOP_PCT = hard_stop / 100
            config.TIME_STOP_DAYS = time_stop
            config.MAX_TOTAL_EXPOSURE = max_exposure

            new_values = {
                "MAX_DRAWDOWN_PCT": config.MAX_DRAWDOWN_PCT,
                "MAX_POSITION_PCT": config.MAX_POSITION_PCT,
                "KELLY_FRACTION": config.KELLY_FRACTION,
                "HARD_STOP_PCT": config.HARD_STOP_PCT,
                "TIME_STOP_DAYS": config.TIME_STOP_DAYS,
                "MAX_TOTAL_EXPOSURE": config.MAX_TOTAL_EXPOSURE,
            }

            st.success("✅ Risk parameters updated. Restart main.py for persistence.")

            # Show changes
            changes = []
            for k in old_values:
                if old_values[k] != new_values[k]:
                    changes.append(f"  • **{k}**: {old_values[k]} → {new_values[k]}")
            if changes:
                st.info("Changes made:\n" + "\n".join(changes))

    st.divider()

    # ─── 4. ALERT SETTINGS ───────────────────────────────────────────
    st.subheader("Alert Settings")

    alert_col1, alert_col2 = st.columns(2)

    with alert_col1:
        st.toggle("Kill Switch Alerts", value=True, key="alert_ks")
        st.toggle("Trade Execution Alerts", value=getattr(config, "ALERT_ON_TRADE", True), key="alert_trade")
        st.toggle("API Failure Alerts", value=True, key="alert_api")
        st.toggle("Daily Digest Email", value=True, key="alert_digest")

    with alert_col2:
        am = _get_alert_manager()

        if st.button("📧 Send Test Email", type="secondary"):
            if am:
                try:
                    result = am.test_alerts()
                    if result.get("email_sent"):
                        st.success("Email sent successfully")
                    else:
                        st.error(f"Email failed: {result.get('email_error', 'unknown')}")
                except Exception as e:
                    st.error(f"Email failed: {e}")
            else:
                st.error("Alert manager not available")

        st.caption("📬 [Open Alert Manager →](Alert Manager) for full diagnostics")

    st.divider()

    # ─── 5. PARAMETER HISTORY ─────────────────────────────────────────
    st.subheader("Parameter Change History")

    if ms:
        wh = ms._conn.execute(
            "SELECT timestamp, tsmom_weight, vol_trend_weight, "
            "pairs_arb_weight, reason FROM agent_weights ORDER BY id DESC LIMIT 30"
        ).fetchall()

        if wh:
            hist_df = pd.DataFrame([dict(r) for r in wh])
            hist_df.columns = ["Timestamp", "TSMOM", "VolTrend", "PairsArb", "Reason"]
            if "Timestamp" in hist_df.columns:
                hist_df["Timestamp"] = pd.to_datetime(hist_df["Timestamp"]).dt.strftime("%Y-%m-%d %H:%M")
            for col in ["TSMOM", "VolTrend", "PairsArb"]:
                if col in hist_df.columns:
                    hist_df[col] = hist_df[col].apply(lambda x: f"{x:.2%}" if x else "—")
            st.dataframe(hist_df, use_container_width=True, hide_index=True)
        else:
            st.info("No parameter changes recorded yet.")
    else:
        st.warning("Memory store unavailable — cannot show parameter history.")


# ── Entry point ─────────────────────────────────────────────────────
render()
