"""
ui/pages/alert_manager.py — APEX ALGO Alert Manager Dashboard

4-tab page: Status, Config, History, Test.
Registered in dashboard.py at 📬 Alert Manager.
"""

from datetime import datetime

import streamlit as st

import config


# ── Brand colours ────────────────────────────────────────────────────
BRAND = {
    "primary": "#0D1B3E", "accent": "#2563EB", "success": "#059669",
    "warning": "#D97706", "danger": "#DC2626", "intelligence": "#7C3AED",
    "neutral": "#374151",
}


# ── Helpers ──────────────────────────────────────────────────────────

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
            ms = _get_memory_store()
            am = AlertManager(config, memory_store=ms)
            st.session_state.alert_manager = am
        except Exception:
            return None
    return am


def _status_badge(status: str) -> str:
    colors = {
        "sent": BRAND["success"],
        "failed": BRAND["danger"],
        "suppressed": BRAND["warning"],
    }
    c = colors.get(status, BRAND["neutral"])
    return (
        f'<span style="display:inline-block;padding:2px 8px;'
        f'background:{c}22;color:{c};border-radius:4px;'
        f'font-size:12px;font-weight:600;">{status.upper()}</span>'
    )


# ═════════════════════════════════════════════════════════════════════
# PAGE
# ═════════════════════════════════════════════════════════════════════

st.markdown("## 📬 Alert Manager")

ms = _get_memory_store()
am = _get_alert_manager()

tab_status, tab_config, tab_history, tab_test = st.tabs([
    "📡 Status", "⚙️ Config", "📜 History", "🧪 Test",
])


# ─── TAB 1: STATUS ──────────────────────────────────────────────────

with tab_status:
    st.subheader("Connection Status")

    smtp_configured = bool(
        getattr(config, "SMTP_HOST", "") and
        getattr(config, "SMTP_USER", "") and
        getattr(config, "SMTP_PASS", "")
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if smtp_configured:
            st.markdown(
                f'<div style="background:{BRAND["success"]}22;border:1px solid '
                f'{BRAND["success"]};border-radius:8px;padding:16px;text-align:'
                f'center;"><span style="font-size:24px;">✅</span><br/>'
                f'<span style="color:{BRAND["success"]};font-weight:700;">'
                f'SMTP Connected</span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div style="background:{BRAND["danger"]}22;border:1px solid '
                f'{BRAND["danger"]};border-radius:8px;padding:16px;text-align:'
                f'center;"><span style="font-size:24px;">❌</span><br/>'
                f'<span style="color:{BRAND["danger"]};font-weight:700;">'
                f'SMTP Not Configured</span></div>',
                unsafe_allow_html=True,
            )

    if ms:
        today_alerts = ms.get_alert_history(limit=200)
        today_str = datetime.now().strftime("%Y-%m-%d")
        today_only = [a for a in today_alerts
                      if a.get("timestamp", "").startswith(today_str)]

        sent_today = sum(1 for a in today_only if a["status"] == "sent")
        failed_today = sum(1 for a in today_only if a["status"] == "failed")
        suppressed_today = sum(1 for a in today_only if a["status"] == "suppressed")
        unacked = ms.get_unacknowledged_count()

        with col2:
            st.metric("Sent Today", sent_today)
        with col3:
            st.metric("Failed Today", failed_today)
        with col4:
            if unacked > 0:
                st.metric("⚠️ Unacknowledged", unacked)
            else:
                st.metric("Unacknowledged", 0)

        st.divider()
        st.caption(f"Daily limit: {sent_today}/50 emails used today")
    else:
        with col2:
            st.metric("Sent Today", "—")
        with col3:
            st.metric("Failed", "—")
        with col4:
            st.metric("Unacknowledged", "—")
        st.warning("Memory store unavailable — cannot show activity.")


# ─── TAB 2: CONFIG ──────────────────────────────────────────────────

with tab_config:
    st.subheader("Recipient Settings")

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.text_input(
            "Primary Email", value=getattr(config, "ALERT_EMAIL", ""),
            disabled=True, key="cfg_email",
            help="Set in .env as ALERT_EMAIL",
        )
    with col_r2:
        st.text_input(
            "CC Email", value=getattr(config, "ALERT_EMAIL_CC", ""),
            disabled=True, key="cfg_cc",
            help="Set in .env as ALERT_EMAIL_CC",
        )

    st.divider()
    st.subheader("Alert Toggles")

    toggle_col1, toggle_col2 = st.columns(2)
    with toggle_col1:
        st.toggle("Kill Switch Alerts", value=True, key="toggle_ks",
                   disabled=True)
        st.toggle("Trade Execution Alerts",
                   value=getattr(config, "ALERT_ON_TRADE", True),
                   key="toggle_trade")
        st.toggle("API Failure Alerts", value=True, key="toggle_api",
                   disabled=True)
    with toggle_col2:
        st.toggle("Daily Summary Email", value=True, key="toggle_daily")
        st.toggle("System Start/Stop Alerts", value=True,
                   key="toggle_sys")

    st.divider()
    st.subheader("Drawdown Warning Threshold")
    dd_threshold = st.slider(
        "Send warning when drawdown exceeds:",
        min_value=3, max_value=15,
        value=int(getattr(config, "MAX_DRAWDOWN_PCT", 0.15) * 100 * 0.7),
        step=1, format="%d%%",
        help="Alerts before the kill switch triggers",
    )
    st.caption(
        f"Kill switch fires at "
        f"{int(getattr(config, 'MAX_DRAWDOWN_PCT', 0.15)*100)}% — "
        f"this warns at {dd_threshold}%"
    )


# ─── TAB 3: HISTORY ─────────────────────────────────────────────────

with tab_history:
    st.subheader("Alert History")

    if ms:
        filter_col, action_col = st.columns([2, 1])
        with filter_col:
            status_filter = st.selectbox(
                "Filter by status",
                ["All", "sent", "failed", "suppressed"],
                key="history_filter",
            )

        sf = None if status_filter == "All" else status_filter
        alerts = ms.get_alert_history(limit=100, status_filter=sf)

        if alerts:
            for alert in alerts:
                with st.container():
                    hcol1, hcol2, hcol3, hcol4 = st.columns([4, 2, 1, 1])
                    with hcol1:
                        st.markdown(
                            f"**{alert.get('subject', 'N/A')[:60]}**",
                        )
                    with hcol2:
                        ts = alert.get("timestamp", "")[:16]
                        st.caption(ts)
                    with hcol3:
                        st.markdown(
                            _status_badge(alert.get("status", "unknown")),
                            unsafe_allow_html=True,
                        )
                    with hcol4:
                        aid = alert.get("id")
                        if (alert.get("status") == "failed" and
                                not alert.get("acknowledged")):
                            if st.button("✓", key=f"ack_{aid}",
                                         help="Acknowledge"):
                                ms.acknowledge_alert(aid)
                                st.rerun()

                    if alert.get("error"):
                        st.caption(f"Error: {alert['error'][:120]}")
                    st.markdown("---")
        else:
            st.info("No alert history to display.")
    else:
        st.warning("Memory store unavailable.")


# ─── TAB 4: TEST ────────────────────────────────────────────────────

with tab_test:
    st.subheader("Quick Test")

    if st.button("📧 Send Test Email", type="primary"):
        if am:
            with st.spinner("Sending test email..."):
                result = am.test_alerts()
            if result.get("email_sent"):
                st.success(
                    f"✅ Test email sent to {config.ALERT_EMAIL}"
                )
            else:
                st.error(
                    f"❌ Send failed: {result.get('email_error', 'unknown')}"
                )
        else:
            st.error("Alert manager not available")

    st.divider()
    st.subheader("SMTP Diagnostics")

    if st.button("🔍 Run SMTP Check"):
        import smtplib
        findings = []
        try:
            host = getattr(config, "SMTP_HOST", "")
            port = int(getattr(config, "SMTP_PORT", 587))
            findings.append(f"✅ SMTP_HOST: {host}")
            findings.append(f"✅ SMTP_PORT: {port}")

            with smtplib.SMTP(host, port, timeout=10) as srv:
                srv.ehlo()
                findings.append("✅ EHLO successful")
                srv.starttls()
                findings.append("✅ STARTTLS successful")

                user = getattr(config, "SMTP_USER", "")
                pwd = getattr(config, "SMTP_PASS", "")
                if user and pwd:
                    srv.login(user, pwd)
                    findings.append("✅ Login successful")
                else:
                    findings.append("⚠️ No credentials — skipping login")

            for f in findings:
                st.markdown(f)
            st.success("SMTP connection test passed!")

        except Exception as e:
            for f in findings:
                st.markdown(f)
            st.error(f"❌ SMTP test failed: {e}")

    st.divider()
    st.subheader("Recent Logs")

    if ms:
        recent = ms.get_alert_history(limit=10)
        if recent:
            for r in recent:
                badge = _status_badge(r.get("status", "unknown"))
                st.markdown(
                    f'{badge} **{r.get("subject", "N/A")[:50]}** '
                    f'— {r.get("timestamp", "")[:16]}',
                    unsafe_allow_html=True,
                )
        else:
            st.info("No logs yet.")
    else:
        st.warning("Memory store unavailable.")
