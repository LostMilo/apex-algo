# Trading Algorithm — Setup Guide

## Prerequisites
- Python 3.11+
- ForexVPS or any Linux server with stable internet
- Alpaca account (free at [alpaca.markets](https://alpaca.markets) — use paper trading first)

## Installation

```bash
# 1. Copy project files to your VPS
scp -r Algorithm/ user@your-vps-ip:~/

# 2. Install dependencies
cd ~/Algorithm
pip install -r requirements.txt

# 3. Create your environment file
cp .env.template .env
```

Edit `.env` and fill in your Alpaca API keys (paper trading keys first):

```
ALPACA_API_KEY=your_paper_key_here
ALPACA_SECRET_KEY=your_paper_secret_here
```

## Getting Alpaca API Keys

1. Sign up at [alpaca.markets](https://alpaca.markets) (free)
2. Go to **Paper Trading → API Keys → Generate**
3. Copy **API Key ID** and **Secret Key** into `.env`

## Running the System

| Mode | Command |
|------|---------|
| Backtest | `python main.py --mode backtest` |
| Single cycle test | `python main.py --mode paper --run-once` |
| Paper trading | `python main.py --mode paper` |
| Live trading | `python main.py --mode live` |
| Dashboard | `streamlit run ui/dashboard.py --server.port 8501` |

Access the dashboard at: `http://YOUR_VPS_IP:8501`

## First Run Checklist

- [ ] `.env` file filled in with Alpaca paper trading keys
- [ ] `python main.py --mode backtest` runs without errors
- [ ] Backtest shows Sharpe > 0.5 (anything lower → review strategy params)
- [ ] Walk-forward shows robustness score > 0.4
- [ ] Paper trading starts and dashboard loads in browser
- [ ] Test alerts work (SMS + email via config page)
- [ ] Run for **90 days** of paper trading before switching to live

## Switching to Live Trading

> ⚠️ **WARNING:** Only do this after 90 days of successful paper trading validation.

1. Get **LIVE** Alpaca API keys (separate from paper keys)
2. Update `.env`:
   ```
   ALPACA_API_KEY=your_live_key
   ALPACA_SECRET_KEY=your_live_secret
   PAPER_TRADING=False
   USE_LIVE_EXECUTION=True
   ```
3. Run: `python main.py --mode live`

## Project Structure

```
Algorithm/
├── config.py              # All parameters (no hardcoded values)
├── main.py                # Entry point — backtest / paper / live
├── logger.py              # Structured logging (loguru)
├── data/
│   └── data_agent.py      # Single data source (Alpaca API)
├── strategies/
│   ├── tsmom.py           # Time Series Momentum
│   ├── dual_momentum.py   # Dual Momentum filter (gate)
│   ├── vol_trend.py       # Volatility Trend Following
│   └── pairs_arb.py       # Statistical Pairs Arbitrage
├── core/
│   ├── regime_detector.py # Market regime classification
│   └── consensus_engine.py# Signal aggregation + voting
├── risk/
│   ├── position_sizing.py # GARCH + Kelly Criterion sizing
│   └── exits.py           # Chandelier, hard stop, time stop, kill switch
├── backtest/
│   ├── engine.py          # Backtesting engine (anti-look-ahead)
│   ├── walk_forward.py    # Walk-forward optimization
│   └── metrics.py         # Performance metrics calculator
├── learning/
│   ├── memory_store.py    # SQLite knowledge database
│   └── experience_agent.py# Post-mortem analysis + weight tuning
├── execution/
│   └── alpaca_client.py   # Alpaca order execution (limit orders only)
├── utils/
│   └── alerting.py        # SMS + email alerts
└── ui/
    ├── dashboard.py        # Streamlit dashboard
    └── pages/              # Dashboard sub-pages
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `DataSourceError: USE_LIVE_DATA is False` | Set `USE_LIVE_DATA=True` in `.env` for paper/live modes |
| Dashboard won't load | Check port 8501 is open in firewall |
| No trades executing | Check kill switch status on dashboard; verify market hours |
| Alpaca API errors | Verify API keys in `.env`; check Alpaca status page |
