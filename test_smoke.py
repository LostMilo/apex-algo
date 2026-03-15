"""
test_smoke.py — End-to-end component instantiation smoke test.

Verifies every core component can be imported and instantiated without
errors. Does NOT execute trades or fetch live data.
"""

import sys
import os

# Ensure we're running from the project root
os.chdir(os.path.dirname(os.path.abspath(__file__)))

results = []
all_passed = True


def check(name: str, fn):
    global all_passed
    try:
        fn()
        results.append(f"  ✅  {name}")
    except Exception as e:
        results.append(f"  ❌  {name} — {type(e).__name__}: {e}")
        all_passed = False


# ── 1. Config ───────────────────────────────────────────────────────
def _config():
    import config
    assert hasattr(config, "ASSET_UNIVERSE"), "Missing ASSET_UNIVERSE"
    assert hasattr(config, "STARTING_CAPITAL"), "Missing STARTING_CAPITAL"
    assert hasattr(config, "MAX_DRAWDOWN_PCT"), "Missing MAX_DRAWDOWN_PCT"

check("1.  config.py (validate)", _config)


# ── 2. MemoryStore ──────────────────────────────────────────────────
def _memory_store():
    from learning.memory_store import MemoryStore
    ms = MemoryStore(":memory:")
    assert ms.get_trade_count() == 0

check("2.  MemoryStore", _memory_store)


# ── 3. DataAgent ────────────────────────────────────────────────────
def _data_agent():
    from data.data_agent import DataAgent
    agent = DataAgent()
    assert hasattr(agent, "get_ohlcv")
    assert hasattr(agent, "get_universe_data")

check("3.  DataAgent", _data_agent)


# ── 4a. TSMOMStrategy ──────────────────────────────────────────────
def _tsmom():
    from strategies.tsmom import TSMOMStrategy
    s = TSMOMStrategy()
    assert hasattr(s, "compute_signals")

check("4a. TSMOMStrategy", _tsmom)


# ── 4b. DualMomentumFilter ─────────────────────────────────────────
def _dual_mom():
    from strategies.dual_momentum import DualMomentumFilter
    s = DualMomentumFilter()
    assert hasattr(s, "compute_filter")

check("4b. DualMomentumFilter", _dual_mom)


# ── 4c. VolTrendStrategy ───────────────────────────────────────────
def _vol_trend():
    from strategies.vol_trend import VolTrendStrategy
    s = VolTrendStrategy()
    assert hasattr(s, "compute_signals")

check("4c. VolTrendStrategy", _vol_trend)


# ── 4d. PairsArbStrategy ──────────────────────────────────────────
def _pairs_arb():
    from strategies.pairs_arb import PairsArbStrategy
    s = PairsArbStrategy()
    assert hasattr(s, "generate_signals")

check("4d. PairsArbStrategy", _pairs_arb)


# ── 5. RegimeDetector ──────────────────────────────────────────────
def _regime():
    from core.regime_detector import RegimeDetector, Regime
    rd = RegimeDetector()
    assert hasattr(rd, "detect")
    assert Regime.TRENDING == "TRENDING"
    assert Regime.RANGING == "RANGING"
    assert Regime.RISK_OFF == "RISK_OFF"

check("5.  RegimeDetector", _regime)


# ── 6. ConsensusEngine ─────────────────────────────────────────────
def _consensus():
    import config
    from core.consensus_engine import ConsensusEngine
    ce = ConsensusEngine(config, {"consensus_weights": {}})
    assert hasattr(ce, "aggregate")

check("6.  ConsensusEngine", _consensus)


# ── 7. PositionSizer ──────────────────────────────────────────────
def _sizer():
    from risk.position_sizing import PositionSizer
    ps = PositionSizer()
    assert hasattr(ps, "compute")

check("7.  PositionSizer", _sizer)


# ── 8. ExitManager ────────────────────────────────────────────────
def _exits():
    from risk.exits import ExitManager
    em = ExitManager()
    assert hasattr(em, "check_all_exits")

check("8.  ExitManager", _exits)


# ── 9. ExperienceAgent ───────────────────────────────────────────
def _experience():
    import config
    from learning.memory_store import MemoryStore
    from core.consensus_engine import ConsensusEngine
    from learning.experience_agent import ExperienceAgent
    ms = MemoryStore(":memory:")
    ce = ConsensusEngine(config, {"consensus_weights": {}})
    ea = ExperienceAgent(config, ms, ce)
    assert hasattr(ea, "_recalculate_weights")

check("9.  ExperienceAgent", _experience)


# ── 10. AlpacaClient ──────────────────────────────────────────────
def _alpaca():
    from execution.alpaca_client import AlpacaClient
    # Paper mode — no real connection made until API calls
    client = AlpacaClient()
    assert hasattr(client, "execute_signal")
    assert hasattr(client, "get_account")
    assert hasattr(client, "get_positions")

check("10. AlpacaClient (paper)", _alpaca)


# ── 11. AlertManager ─────────────────────────────────────────────
def _alerts():
    import config
    from utils.alerting import AlertManager
    am = AlertManager(config)
    assert hasattr(am, "kill_switch_fired")
    assert hasattr(am, "trade_executed")

check("11. AlertManager", _alerts)


# ═══════════════════════════════════════════════════════════════════
# REPORT
# ═══════════════════════════════════════════════════════════════════
print()
print("═" * 60)
print("  END-TO-END SMOKE TEST")
print("═" * 60)
for line in results:
    print(line)
print()

if all_passed:
    print("  ✅ ALL 11 COMPONENTS INSTANTIATED SUCCESSFULLY")
    print()
    sys.exit(0)
else:
    failed = sum(1 for r in results if "❌" in r)
    print(f"  ❌ {failed} COMPONENT(S) FAILED — see above for details")
    print()
    sys.exit(1)
