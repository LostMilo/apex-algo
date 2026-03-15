"""
pages/signals.py — Current Signals & Pairs Arbitrage Page

Signal table per asset, signal strength heatmap, active strategies panel,
regime metadata, pairs arb tab with z-scores and spread charts.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

import config
from core.regime_detector import Regime

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


# ── Helpers ──────────────────────────────────────────────────────────

@st.cache_data(ttl=120)
def _load_universe_data():
    """Load data for the entire asset universe."""
    try:
        from data.data_agent import DataAgent
        agent = DataAgent()
        data = agent.get_universe_data(config.ASSET_UNIVERSE, start="2024-01-01", end="today")
        return data
    except Exception:
        return {}


@st.cache_data(ttl=120)
def _compute_regimes(_data_keys):
    """Detect regimes for all symbols. _data_keys used for caching."""
    try:
        from data.data_agent import DataAgent
        from core.regime_detector import RegimeDetector
        agent = DataAgent()
        data = agent.get_universe_data(config.ASSET_UNIVERSE, start="2024-01-01", end="today")
        if data:
            detector = RegimeDetector()
            import pandas as pd
            today = pd.Timestamp.now()
            regimes = {}
            for sym in data:
                regime = detector.detect({sym: data[sym]}, today)
                regimes[sym] = regime
            return regimes, data
    except Exception:
        pass
    return {}, {}


def _compute_all_signals(data: dict, regimes_enum: dict):
    """Run all strategies and return raw + filtered signals."""
    try:
        from strategies.tsmom import TSMOMStrategy
        from strategies.vol_trend import VolTrendStrategy
        from strategies.pairs_arb import PairsArbStrategy
        from strategies.dual_momentum import DualMomentumFilter
        from core.regime_detector import Regime as R

        # Convert string regimes back to enum
        regime_map = {k: R(v) for k, v in regimes_enum.items()}

        tsmom = TSMOMStrategy()
        vol_trend = VolTrendStrategy()
        pairs_arb = PairsArbStrategy()
        dual_mom = DualMomentumFilter()

        tsmom_sigs = tsmom.generate_signals(data, regime_map)
        vol_sigs = vol_trend.generate_signals(data, regime_map)
        pairs_sigs = pairs_arb.generate_signals(data, regime_map)

        all_raw = tsmom_sigs + vol_sigs + pairs_sigs
        filtered = dual_mom.filter_signals(all_raw, data)

        return {
            "tsmom": tsmom_sigs,
            "vol_trend": vol_sigs,
            "pairs_arb": pairs_sigs,
            "all_raw": all_raw,
            "filtered": filtered,
            "pairs_strategy": pairs_arb,
        }
    except Exception as e:
        st.warning(f"Could not compute signals: {e}")
        return None


# ── Page ─────────────────────────────────────────────────────────────

st.markdown("## 📡 Signals & Strategy Status")

tab_signals, tab_pairs = st.tabs(["Signal Overview", "Pairs Arbitrage"])

# ── Tab 1: Signal Overview ───────────────────────────────────────────
with tab_signals:
    data = _load_universe_data()
    regimes_str, _ = _compute_regimes(tuple(sorted(data.keys())) if data else ())

    if not data:
        st.warning("No market data loaded. Run the data fetch first or check your data cache.")
    else:
        # Compute signals
        signal_results = _compute_all_signals(data, regimes_str)

        # ── Active Strategies Panel ──────────────────────────────────
        st.markdown("### Active Strategies by Regime")
        regime_counts = {}
        for sym, r in regimes_str.items():
            regime_counts[r] = regime_counts.get(r, 0) + 1

        cols = st.columns(3)
        for i, (regime_name, emoji, strategies, color) in enumerate([
            ("TRENDING", "🟢", "TSMOM, VolTrend", GREEN),
            ("RANGING", "🟡", "PairsArb", YELLOW),
            ("RISK_OFF", "🔴", "None (cash)", RED),
        ]):
            with cols[i]:
                count = regime_counts.get(regime_name, 0)
                st.markdown(
                    f'<div style="background:#161B22; border:1px solid #21262D; border-radius:10px; '
                    f'padding:1rem; text-align:center;">'
                    f'<p style="font-size:1.5rem; margin:0;">{emoji}</p>'
                    f'<p style="font-weight:700; color:#E6EDF3; margin:0.2rem 0;">{regime_name}</p>'
                    f'<p style="color:#8B8D97; font-size:0.75rem; margin:0;">{strategies}</p>'
                    f'<p style="color:{color}; font-weight:600; font-size:0.9rem; margin:0.3rem 0 0 0;">'
                    f'{count} assets</p></div>',
                    unsafe_allow_html=True,
                )

        st.markdown("---")

        # ── Signal Table ─────────────────────────────────────────────
        st.markdown("### Current Signal Table")

        if signal_results:
            # Index signals by symbol and strategy
            sig_by_sym = {}
            for sig in signal_results.get("all_raw", []):
                if sig.symbol not in sig_by_sym:
                    sig_by_sym[sig.symbol] = {}
                sig_by_sym[sig.symbol][sig.strategy] = sig

            # Build consensus from filtered
            consensus_by_sym = {}
            for sig in signal_results.get("filtered", []):
                key = sig.symbol
                if key not in consensus_by_sym:
                    consensus_by_sym[key] = sig
                else:
                    # Keep strongest
                    if sig.strength > consensus_by_sym[key].strength:
                        consensus_by_sym[key] = sig

            # Dual Mom pass/fail
            filtered_symbols = {sig.symbol for sig in signal_results.get("filtered", [])}

            rows = []
            for sym in config.ASSET_UNIVERSE:
                sigs = sig_by_sym.get(sym, {})
                tsmom_sig = sigs.get("TSMOM")
                vol_sig = sigs.get("VolTrend")
                cons = consensus_by_sym.get(sym)

                # Check if passed dual mom
                has_raw = sym in sig_by_sym
                dm_status = "✅ PASS" if sym in filtered_symbols else ("❌ FAIL" if has_raw else "—")

                rows.append({
                    "Symbol": sym,
                    "Regime": regimes_str.get(sym, "—"),
                    "TSMOM": f"{tsmom_sig.direction} ({tsmom_sig.strength:.2f})" if tsmom_sig else "—",
                    "VolTrend": f"{vol_sig.direction} ({vol_sig.strength:.2f})" if vol_sig else "—",
                    "Dual Mom": dm_status,
                    "Consensus": f"{cons.direction} ({cons.strength:.2f})" if cons else "—",
                })

            sig_df = pd.DataFrame(rows)
            st.dataframe(sig_df, hide_index=True, use_container_width=True)
        else:
            st.info("Signals not yet computed.")

        # ── Signal Strength Heatmap ──────────────────────────────────
        st.markdown("### Signal Strength Heatmap")

        if signal_results:
            strategies = ["TSMOM", "VolTrend", "PairsArb"]
            symbols = config.ASSET_UNIVERSE
            heatmap_data = np.zeros((len(symbols), len(strategies)))

            for sig in signal_results.get("all_raw", []):
                if sig.symbol in symbols:
                    row_idx = symbols.index(sig.symbol)
                    # Map strategy name
                    strat_name = sig.strategy
                    if strat_name in strategies:
                        col_idx = strategies.index(strat_name)
                        val = sig.strength if sig.direction == "long" else -sig.strength
                        heatmap_data[row_idx][col_idx] = val

            fig_heat = go.Figure(data=go.Heatmap(
                z=heatmap_data,
                x=strategies,
                y=symbols,
                colorscale=[
                    [0.0, RED],
                    [0.5, "#161B22"],
                    [1.0, GREEN],
                ],
                zmid=0,
                zmin=-1,
                zmax=1,
                text=np.round(heatmap_data, 2),
                texttemplate="%{text}",
                textfont=dict(size=12, color="white"),
                hovertemplate="<b>%{y}</b> — %{x}<br>Strength: %{z:.3f}<extra></extra>",
                colorbar=dict(
                    title="Signal",
                    tickvals=[-1, -0.5, 0, 0.5, 1],
                    ticktext=["Short -1", "-0.5", "Neutral", "+0.5", "Long +1"],
                ),
            ))
            fig_heat.update_layout(
                **PLOTLY_LAYOUT,
                height=max(350, len(symbols) * 38),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig_heat, use_container_width=True)

        # ── Last Regime Detection Metadata ───────────────────────────
        st.markdown("### Regime Detection Metadata")
        try:
            from ta.trend import ADXIndicator
            from ta.volatility import AverageTrueRange

            meta_rows = []
            for sym in config.ASSET_UNIVERSE:
                if sym not in data:
                    continue
                df = data[sym]
                if len(df) < 60:
                    continue

                adx_ind = ADXIndicator(high=df["High"], low=df["Low"], close=df["Close"], window=14)
                atr_ind = AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"], window=14)

                adx = adx_ind.adx().iloc[-1]
                atr = atr_ind.average_true_range().iloc[-1]
                vol = df["Close"].pct_change().dropna().iloc[-60:].std() * np.sqrt(252)

                meta_rows.append({
                    "Symbol": sym,
                    "Regime": regimes_str.get(sym, "—"),
                    "ADX(14)": f"{adx:.1f}" if not pd.isna(adx) else "—",
                    "ATR(14)": f"${atr:.2f}" if not pd.isna(atr) else "—",
                    "Ann. Vol": f"{vol:.1%}" if not pd.isna(vol) else "—",
                })

            if meta_rows:
                st.dataframe(pd.DataFrame(meta_rows), hide_index=True, use_container_width=True)
        except Exception as e:
            st.caption(f"Could not load regime metadata: {e}")


# ── Tab 2: Pairs Arbitrage ───────────────────────────────────────────
with tab_pairs:
    st.markdown("### Cointegrated Pairs")

    data = _load_universe_data()
    if not data:
        st.warning("No data loaded.")
    else:
        try:
            from strategies.pairs_arb import PairsArbStrategy
            from core.regime_detector import RegimeDetector, Regime as R

            detector = RegimeDetector()
            import pandas as _pd
            _today = _pd.Timestamp.now()
            regimes = {}
            for _sym in data:
                regimes[_sym] = detector.detect({_sym: data[_sym]}, _today)
            pairs_strat = PairsArbStrategy()

            # Force recalculation
            ranging = [s for s, r in regimes.items() if r == R.RANGING and s in data]

            if len(ranging) < 2:
                st.info(f"Only {len(ranging)} assets in RANGING regime. Need ≥ 2 for pairs analysis.")
            else:
                pairs = pairs_strat._find_cointegrated_pairs(data, ranging)

                if not pairs:
                    st.info("No cointegrated pairs found at current p-value threshold.")
                else:
                    # Pairs table
                    pair_rows = []
                    for p in pairs:
                        pair_rows.append({
                            "Pair": f"{p['sym_a']} / {p['sym_b']}",
                            "p-value": f"{p['p_value']:.4f}",
                            "t-stat": f"{p['t_stat']:.2f}",
                            "Hedge Ratio": f"{p['hedge_ratio']:.4f}",
                        })

                    st.dataframe(pd.DataFrame(pair_rows), hide_index=True, use_container_width=True)

                    # Spread charts for top pairs
                    st.markdown("### Spread & Z-Score Charts")
                    for p in pairs[:3]:
                        sym_a, sym_b = p["sym_a"], p["sym_b"]
                        hr = p["hedge_ratio"]

                        if sym_a not in data or sym_b not in data:
                            continue

                        price_col = "Adj Close" if "Adj Close" in data[sym_a].columns else "Close"
                        pa = data[sym_a][price_col].dropna()
                        pb = data[sym_b][price_col].dropna()
                        common = pa.index.intersection(pb.index)
                        pa, pb = pa.loc[common], pb.loc[common]

                        spread = pa - hr * pb
                        z_mean = spread.rolling(60).mean()
                        z_std = spread.rolling(60).std()
                        zscore = (spread - z_mean) / z_std

                        st.markdown(f"#### {sym_a} / {sym_b}")
                        col_sp, col_z = st.columns(2)

                        with col_sp:
                            fig_sp = go.Figure()
                            fig_sp.add_trace(go.Scatter(
                                x=spread.index[-120:], y=spread.values[-120:],
                                mode="lines", line=dict(color=BLUE, width=1.5),
                                name="Spread",
                            ))
                            fig_sp.update_layout(**PLOTLY_LAYOUT, height=250, title="Spread")
                            st.plotly_chart(fig_sp, use_container_width=True)

                        with col_z:
                            z_vals = zscore.dropna().iloc[-120:]
                            fig_z = go.Figure()
                            fig_z.add_trace(go.Scatter(
                                x=z_vals.index, y=z_vals.values,
                                mode="lines", line=dict(color=YELLOW, width=1.5),
                                name="Z-Score",
                            ))
                            # Entry/exit lines
                            fig_z.add_hline(y=config.ZSCORE_ENTRY, line_dash="dash", line_color=RED, annotation_text="Entry +")
                            fig_z.add_hline(y=-config.ZSCORE_ENTRY, line_dash="dash", line_color=RED, annotation_text="Entry −")
                            fig_z.add_hline(y=config.ZSCORE_EXIT, line_dash="dot", line_color=GREEN, annotation_text="Exit +")
                            fig_z.add_hline(y=-config.ZSCORE_EXIT, line_dash="dot", line_color=GREEN, annotation_text="Exit −")
                            fig_z.add_hline(y=0, line_color="#21262D")
                            fig_z.update_layout(**PLOTLY_LAYOUT, height=250, title="Z-Score")
                            st.plotly_chart(fig_z, use_container_width=True)

        except Exception as e:
            st.error(f"Pairs analysis error: {e}")
