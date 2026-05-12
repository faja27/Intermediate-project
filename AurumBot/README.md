# AurumBot — Automated Gold Trading Bot (XAUUSD)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![MetaTrader 5](https://img.shields.io/badge/Platform-MetaTrader%205-1A73E8.svg)](https://www.metatrader5.com/)
[![LightGBM](https://img.shields.io/badge/ML-LightGBM-9B59B6.svg)](https://lightgbm.readthedocs.io/)
[![Status](https://img.shields.io/badge/Status-Demo%20Testing-yellow.svg)]()

An automated trading bot for XAUUSD (Gold) using a hybrid approach — rule-based signals (MACD + RSI) combined with ML probability filtering (LightGBM), running on MetaTrader 5 via the Python API.

---

## Overview

AurumBot operates on M1 timeframe with M15 trend bias confirmation. Entry decisions combine classical technical analysis with two trained LightGBM classifiers (separate buy/sell models) built on 22 engineered features. TP/SL is adaptive using ATR multipliers for volatility-aware sizing.

---

## Repository Structure

```
AurumBot/
├── bot.py                  # Main trading loop (run after training)
├── train_model.py          # Feature engineering, ATR labeling, LightGBM training
├── config.py               # Central configuration (credentials, thresholds, paths)
├── collect_data.py         # Fetch historical M1/M15 data from MT5
├── convert_kaggle.py       # Convert Kaggle OHLCV data to bot format
├── convert_dukascopy.py    # Convert Dukascopy tick data to bot format
├── prepare_data.py         # Data preparation pipeline
├── read_hcc.py             # HCC data reader utility
├── debug_hcc.py            # Debug helper for HCC data
├── app.py                  # (Reserved) Dashboard / monitoring interface
├── models/
│   ├── lgb_xauusd_m1_buy.pkl   # Trained LightGBM buy model
│   └── lgb_xauusd_m1_sell.pkl  # Trained LightGBM sell model
├── data/
│   ├── raw_m1.csv          # M1 OHLCV data
│   └── raw_m15.csv         # M15 OHLCV data
├── logs/
│   └── trade.log           # Trade execution log
└── requirements.txt
```

---

## Strategy Overview

| Layer | Method |
|---|---|
| Trend filter | M15 linear regression slope bias (bullish / bearish / sideways) |
| Key level detection | Swing high/low (rolling max/min) |
| Supply & Demand zone | Consolidation width detection |
| Rule-based signal | MACD crossover + RSI momentum confirmation |
| ML confirmation | LightGBM classifier (22 features, separate buy/sell models) |
| TP/SL sizing | ATR-adaptive (TP=4×ATR, SL=1×ATR → 1:4 RR) |
| Risk control | Max floating loss circuit breaker (configurable) |

---

## ML Features (22 total)

`ret_1`, `ret_3`, `ret_5`, `body_ratio`, `upper_wick`, `lower_wick`, `atr_5`, `atr_14`, `volatility_5`, `rsi_slope`, `minute`, `volume_ratio`, `is_pin_bar`, `is_engulfing`, `is_inside_bar`, `near_key_level`, `in_sd_zone`, `m15_slope`, `m15_bias`, `consolidation_width`, `bos_bullish`, `bos_bearish`

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/faja27/Intermediate-project.git
cd Intermediate-project/AurumBot
```

### 2. Install dependencies

```bash
pip install MetaTrader5 pandas numpy lightgbm scikit-learn ta joblib
```

### 3. Configure credentials

Edit `config.py`:
```python
MT5_LOGIN    = "your_account_number"
MT5_PASSWORD = "your_password"
MT5_SERVER   = "your_broker_server"
```

### 4. Collect historical data

```bash
python collect_data.py
```

### 5. Train the models

```bash
python train_model.py
```

### 6. Run the bot

```bash
python bot.py
```

> Stop with `Ctrl+C`. The bot checks every 60 seconds.

---

## Configuration (config.py)

| Parameter | Default | Description |
|---|---|---|
| `SYMBOL` | `XAUUSD` | Trading instrument |
| `LOT` | `0.1` | Volume per trade |
| `TPSL_MODE` | `atr` | `atr` or `fixed` |
| `ATR_MULTIPLIER_TP` | `4.0` | TP = 4× ATR |
| `ATR_MULTIPLIER_SL` | `1.0` | SL = 1× ATR |
| `MAX_TRADE` | `2` | Max open positions |
| `THRESHOLD_BUY` | `0.55` | ML probability threshold (buy) |
| `THRESHOLD_SELL` | `0.55` | ML probability threshold (sell) |
| `MAX_FLOATING_LOSS` | `-50.0` | Emergency stop loss (USD) |
| `CHECK_INTERVAL` | `60` | Loop interval (seconds) |

---

## Requirements

- MetaTrader 5 installed and running
- Active MT5 account (demo recommended for testing)
- Python 3.9+
- XAUUSD symbol available on your broker

---

## Disclaimer

This bot is for **educational and research purposes only**. Trading financial instruments involves significant risk. Always test on a demo account before live deployment. Past backtesting performance does not guarantee future results.

---

## License

This project is licensed under the [MIT License](LICENSE).
