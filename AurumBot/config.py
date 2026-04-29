# =============================================================================
# config.py - Central configuration file for the trading bot
# All parameters are defined here. Edit this file to change bot behavior.
# =============================================================================

# --- MT5 Account Credentials ---
MT5_LOGIN    = ""       # Replace with your MT5 account number
MT5_PASSWORD = ""      # Replace with your MT5 password
MT5_SERVER   = ""      # Replace with your broker server (e.g. "FinexBisnisSolusi-Demo")

# --- Trading Symbol ---
SYMBOL = "XAUUSD"      # Gold vs USD (Finex format)

# --- Lot Size ---
LOT = 0.1              # Trade volume per position

# --- TP/SL Mode ---
# Two modes available:
# "fixed"  → use TP_POINTS and SL_POINTS directly
# "atr"    → use ATR multipliers (adaptive to market volatility)
TPSL_MODE = "atr"      # Recommended: "atr" for adaptive sizing

# --- Fixed TP/SL (used when TPSL_MODE = "fixed") ---
TP_POINTS = 39         # Take Profit in points
SL_POINTS = 13         # Stop Loss in points

# --- ATR-Based TP/SL (used when TPSL_MODE = "atr") ---
# TP = ATR_MULTIPLIER_TP * ATR_14
# SL = ATR_MULTIPLIER_SL * ATR_14
# RR ratio = TP_mult / SL_mult = 3.0 (1:3)
ATR_MULTIPLIER_TP = 4.0   # TP = 15x ATR (~13 points)
ATR_MULTIPLIER_SL = 1.0    # SL = 5x ATR (~4 points)
ATR_PERIOD        = 14     # ATR period for TP/SL calculation

RR_RATIO = 3.0            # Risk/Reward Ratio

# --- Position Management ---
MAX_TRADE    = 2       # Maximum number of open positions at once
MIN_DISTANCE = 20      # Minimum distance (points) between positions of same direction
MAGIC_NUMBER = 123456  # Unique identifier for this bot's orders

# --- Risk Control ---
MAX_FLOATING_LOSS = -50.0  # Maximum allowed floating loss in account currency (USD)

# --- Bot Loop ---
CHECK_INTERVAL = 60     # Seconds between each loop iteration

# --- ML Threshold Settings ---
# Phase 1: 0.60 - 0.65
# Phase 2: 0.70 - 0.75
# Phase 3: 0.80 - 0.85
THRESHOLD_BUY            = 0.55
THRESHOLD_SELL           = 0.55
SIDEWAYS_THRESHOLD_ADDON = 0.05

# --- M15 Trend Filter ---
M15_TREND_LOCK_THRESHOLD = 1.5
M15_BIAS_THRESHOLD       = 0.8
M15_SLOPE_LOOKBACK       = 20

# --- Key Level Detection ---
KEY_LEVEL_RADIUS = 8.0
SWING_LOOKBACK   = 20

# --- Supply & Demand Zone ---
SD_CONSOLIDATION_CANDLES = 5
SD_CONSOLIDATION_WIDTH   = 10.0

# --- Volume Filter ---
VOLUME_MA_PERIOD = 20

# --- Break of Structure (BoS) ---
BOS_LOOKBACK = 10

# --- ML Feature List ---
FEATURES = [
    'ret_1', 'ret_3', 'ret_5',
    'body_ratio', 'upper_wick', 'lower_wick',
    'atr_5', 'atr_14', 'volatility_5',
    'rsi_slope', 'minute',
    'volume_ratio',
    'is_pin_bar', 'is_engulfing', 'is_inside_bar',
    'near_key_level', 'in_sd_zone',
    'm15_slope', 'm15_bias',
    'consolidation_width',
    'bos_bullish', 'bos_bearish'
]

# --- File Paths ---
MODEL_BUY_PATH  = "models/lgb_xauusd_m1_buy.pkl"
MODEL_SELL_PATH = "models/lgb_xauusd_m1_sell.pkl"
DATA_M1_PATH    = "data/raw_m1.csv"
DATA_M15_PATH   = "data/raw_m15.csv"
LOG_FILE        = "logs/trade.log"

# --- Data Collection ---
DATA_LOOKBACK_DAYS = 2200
