# XAUBot

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![MetaTrader5](https://img.shields.io/badge/Platform-MetaTrader5-blueviolet.svg)](https://www.metatrader5.com/)
[![Status](https://img.shields.io/badge/Status-Active%20Development-green.svg)]()

An ML/AI-based automated trading bot for **XAUUSD (Gold/USD)** on MetaTrader 5, combining breakout strategy with EMA trend filter, RSI momentum, and ATR volatility — optimized from 81 parameter combinations over 6 years of historical data (2020–2025).

---

## Overview

XAUBot is a signal-driven algorithmic trading system designed for XAUUSD scalping on the M5 timeframe. The bot integrates:

- **Range Breakout Detection** — dynamic upper/lower bands with configurable buffer
- **EMA Trend Filter** — 50-period EMA to confirm directional bias
- **RSI Momentum Signal** — entry gating via RSI thresholds (buy ≥ 45, sell ≤ 55)
- **ATR Volatility Filter** — avoids low-volatility periods (200–3500 pts range)
- **Trailing Stop Management** — dynamic SL trail activated at 80 pts profit
- **Session Filter** — London + New York sessions only (13:00–03:00 WIB)
- **Dollar-based Risk Management** — daily loss limit, daily profit target, and permanent halt circuit breaker

The strategy configuration was selected as **Rank #4 out of 81 tested combinations**, prioritizing lower drawdown (4.1% DD) over higher returns, making it suitable for small accounts starting at $100.

---

## Repository Structure

```
XAUBot/
├── data/                           # Historical OHLCV data (not tracked by git)
├── dump/                           # Dump/debug files (not tracked by git)
├── journal/
│   └── GoldBot_Journal.xlsx        # Trade journal log
├── logs/                           # Runtime trade logs (not tracked by git)
├── __pycache__/                    # Python cache (not tracked by git)
├── bot.py                          # Main bot — MT5 interface, signal loop, order execution
├── backtest.py                     # Backtesting engine (simple)
├── backtest_historical.py          # Extended historical backtesting
├── config.py                       # Centralized configuration & strategy parameters
├── debug_backtest.py               # Debugging utility for backtest runs
├── goldbot_state.json              # Persistent runtime state (daily PnL, halt status)
├── indicators.py                   # EMA, RSI, ATR, Range Breakout calculations
├── journal.py                      # Trade journal — Excel logging via openpyxl
├── optimize_historical.py          # Parameter optimizer — 81-combination grid search
├── requirements.txt                # Python dependencies
└── risk_manager.py                 # Dollar-based risk controls & circuit breaker
```

> **Note:** `data/`, `dump/`, and `logs/` are excluded from git via `.gitignore`.

---

## Strategy Overview

### Signal Logic

| Condition | Buy | Sell |
|---|---|---|
| Price vs Range | `close > upper_band` | `close < lower_band` |
| EMA Trend | `close > EMA(50)` | `close < EMA(50)` |
| RSI Filter | `RSI ≥ 45` | `RSI ≤ 55` |
| ATR Filter | `200 ≤ ATR ≤ 3500 pts` | `200 ≤ ATR ≤ 3500 pts` |
| Session | London / New York only | London / New York only |

### Optimized Parameters (Rank #4 / 81)

| Parameter | Value | Notes |
|---|---|---|
| Breakout Period | 15 candles | Lower DD vs period=10 |
| Breakout Buffer | 20 pts | More sensitive entry |
| EMA Period | 50 | Trend confirmation |
| ATR Min/Max | 200 / 3500 pts | Volatility gate |
| Stop Loss | 200 pts | Fixed SL |
| Trailing Activate | 80 pts | Trail starts at +$0.80 |
| Trail Distance | 80 pts | Dynamic SL distance |

### Backtest Results (2020–2025)

| Metric | Value |
|---|---|
| Win Rate | 72.1% |
| Profit Factor | 2.71 |
| Avg Trades/Day | 4.5 |
| Weekly Profit (est.) | +$20.15 |
| Max Drawdown | 4.1% |

---

## Risk Management

XAUBot uses a dollar-based risk system persistent across sessions via `goldbot_state.json`:

| Rule | Value |
|---|---|
| Max Daily Loss | $6.00 (3× SL) |
| Daily Profit Target | $15.00 (auto-stop) |
| Max Total Loss | $20.00 (permanent halt) |
| Max Open Positions | 1 |
| Max Spread | 150 pts |
| Commission Tracking | $5.00 / lot (MIFX) |
| Swap Avoidance | Close all at 23:00 WIB |

---

## Quick Start

### Prerequisites

- Python 3.10+
- MetaTrader 5 installed (Monex Trader / MIFX or any MT5 broker)
- Active MT5 demo or live account

### 1. Clone the repository

```bash
git clone https://github.com/faja27/Intermediate-project.git
cd Intermediate-project/XAUBot
```

### 2. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure credentials

Edit `config.py` and fill in your MT5 credentials:

```python
mt5_login: int = 12345678          # Your MT5 account number
mt5_password: str = "your_pass"    # Your MT5 password
mt5_server: str = "MIFX-Demo"      # Your broker server name
mt5_path: str = r"C:\Program Files\Monex Trader\terminal64.exe"
symbol: str = "XAUUSD.m"
magic_number: int = 20250101
```

### 5. Run the bot

```bash
python bot.py
```

---

## Backtesting

### Simple Backtest

```bash
python backtest.py
```

### Historical Backtest (6 years data)

```bash
python backtest_historical.py
```

### Parameter Optimization (81 combinations)

```bash
python optimize_historical.py
```

---

## Components

### `indicators.py`
Pure technical indicator calculations — no external TA library dependency:
- `Indicators.ema(prices, period)` — Exponential Moving Average
- `Indicators.rsi(prices, period)` — Relative Strength Index
- `Indicators.atr(high, low, close, period)` — Average True Range
- `Indicators.range_breakout(high, low, period, buffer)` — Dynamic breakout bands

### `risk_manager.py`
Dollar-based risk control with persistent state:
- Daily PnL tracking with auto-reset at midnight
- Circuit breaker on max total loss (permanent halt)
- Commission-inclusive PnL calculation ($5/lot)

### `journal.py`
Automated trade logging to Excel (`journal/GoldBot_Journal.xlsx`):
- Entry/exit price, lot size, PnL per trade
- Daily summary statistics

---

## Disclaimer

> ⚠️ **Trading involves significant risk of loss.**
> This bot is provided for educational and research purposes only.
> Always run on a **demo account for at least 2 weeks** before considering live deployment.
> Past backtest performance does not guarantee future results.

---

## License

This project is licensed under the [MIT License](LICENSE).
