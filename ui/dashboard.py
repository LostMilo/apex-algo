"""
dashboard.py — Main Streamlit Dashboard for Autonomous Trading System

Multi-page app with sidebar navigation, auto-refresh, and live data.
Run: streamlit run ui/dashboard.py --server.port 8501

Accessible at http://VPS_IP:8501
"""

import sys
import os
import time
from pathlib import Path

# Ensure project root is on path so all imports work
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st

import config
from learning.memory_store import MemoryStore

# ── Page Config (must be first Streamlit call) ───────────────────────
st.set_page_config(
    page_title="ForexVPS Trading System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — dark premium theme ──────────────────────────────────
st.markdown("""
<style>
    /* ── Global overrides ─────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar header */
    .sidebar-header {
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }
    .sidebar-header h1 {
        font-size: 1.3rem;
        font-weight: 800;
        letter-spacing: 0.04em;
        background: linear-gradient(135deg, #00D97E 0%, #00B4D8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .sidebar-header p {
        color: #8B8D97;
        font-size: 0.72rem;
        margin: 0.2rem 0 0 0;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    /* Mode / status badges */
    .badge {
        display: inline-block;
        padding: 0.2rem 0.65rem;
        border-radius: 6px;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .badge-paper  { background: #1B2838; color: #00B4D8; border: 1px solid #00B4D8; }
    .badge-live   { background: #2D1B1B; color: #FF4B4B; border: 1px solid #FF4B4B; }
    .badge-green  { background: #0D2818; color: #00D97E; border: 1px solid #00D97E; }
    .badge-yellow { background: #2D2A0D; color: #FFD93D; border: 1px solid #FFD93D; }
    .badge-red    { background: #2D1B1B; color: #FF4B4B; border: 1px solid #FF4B4B; }

    /* Quick‐stat cards */
    .stat-card {
        background: #161B22;
        border: 1px solid #21262D;
        border-radius: 10px;
        padding: 0.7rem 0.9rem;
        margin-bottom: 0.5rem;
    }
    .stat-card .label {
        color: #8B8D97;
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin: 0;
    }
    .stat-card .value {
        font-size: 1.15rem;
        font-weight: 700;
        color: #E6EDF3;
        margin: 0;
    }
    .stat-card .value.green { color: #00D97E; }
    .stat-card .value.red   { color: #FF4B4B; }

    /* Dividers */
    .sidebar-divider {
        border: none;
        border-top: 1px solid #21262D;
        margin: 0.7rem 0;
    }

    /* Kill switch indicator */
    .kill-indicator {
        text-align: center;
        padding: 0.35rem;
        border-radius: 6px;
        font-size: 0.7rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .kill-active   { background: #3D1515; color: #FF4B4B; border: 1px solid #FF4B4B; }
    .kill-inactive { background: #0D2818; color: #00D97E; border: 1px solid #0D3620; }

    /* Hide default streamlit footer & menu */
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }

    /* Remove default padding top */
    .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)


# ── Session state defaults ───────────────────────────────────────────
def _init_session_state():
    """Initialize session state variables used across pages."""
    if "kill_switch" not in st.session_state:
        st.session_state.kill_switch = False
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = time.time()
    if "memory_store" not in st.session_state:
        try:
            st.session_state.memory_store = MemoryStore()
        except Exception:
            st.session_state.memory_store = None
    if "broker" not in st.session_state:
        try:
            from execution.alpaca_client import AlpacaClient
            st.session_state.broker = AlpacaClient()
        except Exception:
            st.session_state.broker = None
    if "broker_account" not in st.session_state:
        st.session_state.broker_account = None
    if "broker_positions" not in st.session_state:
        st.session_state.broker_positions = []


_init_session_state()


# ── Auto‐refresh logic ──────────────────────────────────────────────
def _check_auto_refresh():
    """Trigger a rerun if DASHBOARD_REFRESH_SECONDS have elapsed."""
    elapsed = time.time() - st.session_state.last_refresh
    if elapsed >= config.DASHBOARD_REFRESH_SECONDS:
        st.session_state.last_refresh = time.time()
        # Refresh broker data
        _refresh_broker_data()
        st.rerun()


def _refresh_broker_data():
    """Pull latest account + positions from broker (best-effort)."""
    broker = st.session_state.get("broker")
    if broker is None:
        return
    try:
        st.session_state.broker_account = broker.get_account()
    except Exception:
        st.session_state.broker_account = None
    try:
        st.session_state.broker_positions = broker.get_positions()
    except Exception:
        st.session_state.broker_positions = []


# ── Sidebar ──────────────────────────────────────────────────────────
def _render_sidebar():
    """Render the persistent sidebar with system info and quick stats."""
    with st.sidebar:
        # Header
        st.markdown("""
        <div class="sidebar-header">
            <h1>⚡ TRADING SYSTEM</h1>
            <p>Autonomous Algorithm v1.0</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

        # Mode badge
        if config.USE_LIVE_EXECUTION:
            mode_badge = '<span class="badge badge-live">● LIVE</span>'
        elif config.PAPER_TRADING:
            mode_badge = '<span class="badge badge-paper">● PAPER</span>'
        else:
            mode_badge = '<span class="badge badge-yellow">● BACKTEST</span>'
        st.markdown(f"**Mode** &nbsp; {mode_badge}", unsafe_allow_html=True)

        # Kill switch
        if st.session_state.kill_switch:
            st.markdown(
                '<div class="kill-indicator kill-active">⛔ KILL SWITCH ACTIVE</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="kill-indicator kill-inactive">✓ SYSTEM OPERATIONAL</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

        # Quick stats
        acct = st.session_state.broker_account
        positions = st.session_state.broker_positions or []

        equity = acct["equity"] if acct else config.STARTING_CAPITAL
        daily_pnl = sum(p.get("unrealized_pl", 0) for p in positions)
        pnl_class = "green" if daily_pnl >= 0 else "red"
        pnl_sign = "+" if daily_pnl >= 0 else ""

        st.markdown(f"""
        <div class="stat-card">
            <p class="label">Portfolio Value</p>
            <p class="value">${equity:,.2f}</p>
        </div>
        <div class="stat-card">
            <p class="label">Daily P&L</p>
            <p class="value {pnl_class}">{pnl_sign}${daily_pnl:,.2f}</p>
        </div>
        <div class="stat-card">
            <p class="label">Open Positions</p>
            <p class="value">{len(positions)}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

        # Refresh info
        elapsed = int(time.time() - st.session_state.last_refresh)
        remaining = max(0, config.DASHBOARD_REFRESH_SECONDS - elapsed)
        st.caption(f"Auto-refresh in {remaining}s")

        if st.button("🔄 Refresh Now", use_container_width=True):
            st.session_state.last_refresh = time.time()
            _refresh_broker_data()
            st.rerun()


# ── Page Definitions ─────────────────────────────────────────────────
overview_page = st.Page("pages/overview.py", title="Overview", icon="📊", default=True)
positions_page = st.Page("pages/positions.py", title="Positions", icon="💼")
signals_page = st.Page("pages/signals.py", title="Signals", icon="📡")
backtest_page = st.Page("pages/backtest_page.py", title="Backtest", icon="🧪")
alert_manager_page = st.Page("pages/alert_manager.py", title="Alert Manager", icon="📬")


# ── Main ─────────────────────────────────────────────────────────────
def main():
    _render_sidebar()

    pg = st.navigation([overview_page, positions_page, signals_page, backtest_page, alert_manager_page])
    pg.run()

    # Schedule next auto-refresh check
    _check_auto_refresh()


if __name__ == "__main__":
    main()
