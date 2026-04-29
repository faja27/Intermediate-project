# =============================================================================
# bot.py - Main Trading Bot (ATR-Adaptive TP/SL)
# Run this AFTER convert_kaggle.py and train_model.py
#
# Usage: python bot.py
# Stop : Ctrl+C
#
# CHANGELOG:
# - [FIX] Hapus TREND LOCK yang salah logika (memblokir entry saat trending)
# - [FIX] CHECK_INTERVAL diubah ke 60 detik (bukan 1 detik)
# - [FIX] Loop sekarang per menit, tidak boros resource
# - [INFO] Bias bearish/bullish sudah cukup handle arah entry via STEP 13/14
# - [FIX3] sell_signal: hapus syarat macd_0 > 0 (terlalu ketat saat downtrend)
# - [FIX3] buy_signal: hapus syarat macd_0 < 0 (terlalu ketat saat uptrend)
# - [FIX3] SELL L2: saat bearish, cukup near_support OR near_resistance
# - [FIX3] BUY L2: saat bullish, cukup near_support OR near_resistance
# =============================================================================

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import joblib
import ta
import time
import sys
import os
import logging
from datetime import datetime, timedelta

from config import (
    MT5_LOGIN, MT5_PASSWORD, MT5_SERVER,
    SYMBOL, LOT,
    TPSL_MODE, TP_POINTS, SL_POINTS,
    ATR_MULTIPLIER_TP, ATR_MULTIPLIER_SL, ATR_PERIOD,
    MAX_TRADE, MIN_DISTANCE, MAGIC_NUMBER,
    MAX_FLOATING_LOSS, CHECK_INTERVAL,
    THRESHOLD_BUY, THRESHOLD_SELL, SIDEWAYS_THRESHOLD_ADDON,
    M15_BIAS_THRESHOLD, M15_SLOPE_LOOKBACK,
    KEY_LEVEL_RADIUS, SWING_LOOKBACK,
    SD_CONSOLIDATION_CANDLES, SD_CONSOLIDATION_WIDTH,
    VOLUME_MA_PERIOD, BOS_LOOKBACK,
    FEATURES, MODEL_BUY_PATH, MODEL_SELL_PATH, LOG_FILE
)


# =============================================================================
# LOGGING SETUP
# =============================================================================

os.makedirs("logs",   exist_ok=True)
os.makedirs("models", exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger()

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
))
logger.addHandler(console_handler)


# =============================================================================
# MT5 CONNECTION
# =============================================================================

def connect_mt5(retry=3):
    """Initialize and connect to MT5 with retry logic."""
    for attempt in range(1, retry + 1):
        logger.info(f"Connecting to MT5 (attempt {attempt}/{retry})...")

        if not mt5.initialize():
            logger.error(f"MT5 init failed: {mt5.last_error()}")
            if attempt < retry:
                time.sleep(5)
            continue

        if MT5_LOGIN and MT5_PASSWORD and MT5_SERVER:
            authorized = mt5.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)
            if not authorized:
                logger.error(f"MT5 login failed: {mt5.last_error()}")
                mt5.shutdown()
                if attempt < retry:
                    time.sleep(5)
                continue

        if mt5.symbol_info(SYMBOL) is None:
            logger.error(f"Symbol {SYMBOL} not found.")
            mt5.shutdown()
            continue

        if not mt5.symbol_info(SYMBOL).visible:
            mt5.symbol_select(SYMBOL, True)

        logger.info("MT5 connected successfully.")
        return True

    logger.critical("Failed to connect to MT5. Exiting.")
    sys.exit(1)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_bars(symbol, timeframe, n):
    """Fetch latest n candles from MT5."""
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df[['open', 'high', 'low', 'close', 'tick_volume']]


def get_current_atr(df_m1, period=14):
    """
    Calculate current ATR from M1 data.
    - min_periods=7 agar NaN di tengah window tidak merusak hasil
    - Fallback ke bar sebelumnya kalau bar terakhir masih NaN
    """
    rng = (df_m1['high'] - df_m1['low']).replace(0, np.nan)
    atr = rng.rolling(period, min_periods=max(1, period // 2)).mean()

    for i in range(1, 6):
        val = float(atr.iloc[-i])
        if not np.isnan(val) and val > 0:
            return val

    return None


def get_m15_bias(df_m15):
    """Calculate M15 trend bias using linear regression slope."""
    prices = df_m15['close'].values[-M15_SLOPE_LOOKBACK:]
    if len(prices) < M15_SLOPE_LOOKBACK:
        return 'sideways', 0.0

    x = np.arange(len(prices))
    slope, _ = np.polyfit(x, prices, 1)
    slope = round(float(slope), 4)

    if slope > M15_BIAS_THRESHOLD:
        bias = 'bullish'
    elif slope < -M15_BIAS_THRESHOLD:
        bias = 'bearish'
    else:
        bias = 'sideways'

    return bias, slope


def detect_swing_levels(df, lookback):
    """Detect support and resistance using swing high/low."""
    support    = df['close'].rolling(lookback).min().iloc[-1]
    resistance = df['close'].rolling(lookback).max().iloc[-1]
    return support, resistance


def detect_sd_zone(df, n_candles, max_width):
    """Detect if price is in a Supply/Demand consolidation zone."""
    recent      = df.iloc[-n_candles:]
    price_range = recent['close'].max() - recent['close'].min()
    return price_range < max_width


def compute_candle_patterns(df):
    """Detect Pin Bar, Engulfing, Inside Bar from last 2 candles."""
    if len(df) < 2:
        return False, False, False

    curr = df.iloc[-1]
    prev = df.iloc[-2]
    rng  = curr['high'] - curr['low']

    if rng == 0:
        return False, False, False

    body       = abs(curr['close'] - curr['open'])
    upper_wick = (curr['high'] - max(curr['open'], curr['close'])) / rng
    lower_wick = (min(curr['open'], curr['close']) - curr['low']) / rng
    body_ratio = body / rng

    is_pin_bar    = (lower_wick > 2 * body_ratio) or (upper_wick > 2 * body_ratio)
    prev_body     = abs(prev['close'] - prev['open'])
    is_engulfing  = (body > prev_body) and \
                    ((curr['close'] > curr['open']) != (prev['close'] > prev['open']))
    is_inside_bar = (curr['high'] < prev['high']) and (curr['low'] > prev['low'])

    return bool(is_pin_bar), bool(is_engulfing), bool(is_inside_bar)


def compute_features(df_m1, df_m15):
    """Compute all 22 ML features from M1 and M15 data."""
    try:
        df = df_m1.copy()

        df['ret_1'] = df['close'].pct_change(1)
        df['ret_3'] = df['close'].pct_change(3)
        df['ret_5'] = df['close'].pct_change(5)

        rng = (df['high'] - df['low']).replace(0, np.nan)
        df['body_ratio']  = (df['close'] - df['open']).abs() / rng
        df['upper_wick']  = (df['high'] - df[['open', 'close']].max(axis=1)) / rng
        df['lower_wick']  = (df[['open', 'close']].min(axis=1) - df['low']) / rng
        df['atr_5']        = rng.rolling(5).mean()
        df['atr_14']       = rng.rolling(14).mean()
        df['volatility_5'] = df['ret_1'].rolling(5).std()

        rsi_ind         = ta.momentum.RSIIndicator(df['close'], window=14)
        df['rsi']       = rsi_ind.rsi()
        df['rsi_slope'] = df['rsi'].diff()
        df['minute']    = df.index.minute

        df['volume_ratio'] = df['tick_volume'] / \
                             df['tick_volume'].rolling(VOLUME_MA_PERIOD).mean()

        prev_body = (df['close'].shift(1) - df['open'].shift(1)).abs()
        curr_body = (df['close'] - df['open']).abs()
        prev_dir  = (df['close'].shift(1) > df['open'].shift(1)).astype(int)
        curr_dir  = (df['close'] > df['open']).astype(int)

        df['is_pin_bar']    = ((df['lower_wick'] > 2 * df['body_ratio']) |
                               (df['upper_wick'] > 2 * df['body_ratio'])).astype(int)
        df['is_engulfing']  = ((curr_body > prev_body) &
                               (curr_dir != prev_dir)).astype(int)
        df['is_inside_bar'] = ((df['high'] < df['high'].shift(1)) &
                               (df['low']  > df['low'].shift(1))).astype(int)

        swing_high = df['close'].rolling(SWING_LOOKBACK).max()
        swing_low  = df['close'].rolling(SWING_LOOKBACK).min()
        df['near_key_level'] = pd.concat(
            [(swing_high - df['close']).abs(),
             (swing_low  - df['close']).abs()], axis=1
        ).min(axis=1)

        rolling_max = df['close'].rolling(SD_CONSOLIDATION_CANDLES).max()
        rolling_min = df['close'].rolling(SD_CONSOLIDATION_CANDLES).min()
        df['consolidation_width'] = rolling_max - rolling_min
        df['in_sd_zone'] = (df['consolidation_width'] < SD_CONSOLIDATION_WIDTH).astype(int)

        bias, slope     = get_m15_bias(df_m15)
        df['m15_slope'] = slope
        df['m15_bias']  = 1 if bias == 'bullish' else (-1 if bias == 'bearish' else 0)

        prev_high = df['close'].shift(1).rolling(BOS_LOOKBACK).max()
        prev_low  = df['close'].shift(1).rolling(BOS_LOOKBACK).min()
        df['bos_bullish'] = (df['close'] > prev_high).astype(int)
        df['bos_bearish'] = (df['close'] < prev_low).astype(int)

        df.dropna(inplace=True)

        if len(df) == 0:
            return None

        return df.iloc[-1]

    except Exception as e:
        logger.error(f"Feature computation error: {e}")
        return None


def calculate_tpsl(df_m1):
    """Calculate TP and SL based on TPSL_MODE config."""
    if TPSL_MODE == "atr":
        atr = get_current_atr(df_m1, ATR_PERIOD)
        if atr and atr > 0:
            return ATR_MULTIPLIER_TP * atr, ATR_MULTIPLIER_SL * atr
        else:
            logger.warning(f"ATR calculation failed, using fixed TP/SL: {TP_POINTS}/{SL_POINTS}pts")
            return TP_POINTS, SL_POINTS
    else:
        return TP_POINTS, SL_POINTS


def open_position(direction, lot, tp_points, sl_points):
    """Open a new BUY or SELL position."""
    symbol_info = mt5.symbol_info(SYMBOL)
    if symbol_info is None:
        logger.error(f"Symbol info unavailable for {SYMBOL}")
        return None

    point = symbol_info.point
    tick  = mt5.symbol_info_tick(SYMBOL)
    if tick is None:
        logger.error("Tick data unavailable")
        return None

    if direction == 'buy':
        price      = tick.ask
        order_type = mt5.ORDER_TYPE_BUY
        tp         = price + tp_points * point
        sl         = price - sl_points * point
    else:
        price      = tick.bid
        order_type = mt5.ORDER_TYPE_SELL
        tp         = price - tp_points * point
        sl         = price + sl_points * point

    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       SYMBOL,
        "volume":       lot,
        "type":         order_type,
        "price":        price,
        "tp":           tp,
        "sl":           sl,
        "magic":        MAGIC_NUMBER,
        "comment":      f"AurumBot {direction.upper()}",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)

    if result is None:
        logger.error(f"order_send returned None. Error: {mt5.last_error()}")
        return None

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error(f"Order failed. retcode={result.retcode} | {result.comment}")
        return None

    return result


def modify_tp_sl(position, new_tp, new_sl):
    """Modify TP and SL of an existing position."""
    request = {
        "action":   mt5.TRADE_ACTION_SLTP,
        "position": position.ticket,
        "symbol":   position.symbol,
        "tp":       new_tp,
        "sl":       new_sl,
        "magic":    position.magic,
    }
    return mt5.order_send(request)


def close_all_positions(reason="MANUAL"):
    """Close all open positions."""
    positions = mt5.positions_get(symbol=SYMBOL)
    if not positions:
        return

    logger.info(f"Closing all positions. Reason: {reason}")

    for pos in positions:
        tick = mt5.symbol_info_tick(pos.symbol)
        if tick is None:
            continue

        price      = tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask
        order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY \
                     else mt5.ORDER_TYPE_BUY

        request = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "position":     pos.ticket,
            "symbol":       pos.symbol,
            "volume":       pos.volume,
            "type":         order_type,
            "price":        price,
            "magic":        pos.magic,
            "comment":      f"CLOSE_{reason}",
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"Position {pos.ticket} closed. Profit: {pos.profit:.2f}")
        else:
            logger.error(f"Failed to close {pos.ticket}.")


def recalculate_averaged_tpsl(direction, tp_points, sl_points):
    """Recalculate averaged TP/SL for all positions of same direction."""
    positions = mt5.positions_get(symbol=SYMBOL)
    if not positions:
        return

    symbol_info = mt5.symbol_info(SYMBOL)
    if symbol_info is None:
        return

    point    = symbol_info.point
    pos_type = mt5.POSITION_TYPE_BUY if direction == 'buy' else mt5.POSITION_TYPE_SELL
    filtered = [p for p in positions if p.type == pos_type and p.magic == MAGIC_NUMBER]

    if not filtered:
        return

    avg_price = sum(p.price_open for p in filtered) / len(filtered)

    if direction == 'buy':
        new_tp = avg_price + tp_points * point
        new_sl = avg_price - sl_points * point
    else:
        new_tp = avg_price - tp_points * point
        new_sl = avg_price + sl_points * point

    for pos in filtered:
        modify_tp_sl(pos, new_tp, new_sl)

    logger.info(
        f"Averaged TP/SL | {direction.upper()} | "
        f"Avg entry: {avg_price:.2f} | TP: {new_tp:.2f} | SL: {new_sl:.2f}"
    )


def get_position_counts():
    """Get counts and price levels of open positions."""
    positions    = mt5.positions_get(symbol=SYMBOL)
    buy_count    = 0
    sell_count   = 0
    buy_lowest   = 0.0
    sell_highest = 0.0

    if positions:
        for pos in positions:
            if pos.magic != MAGIC_NUMBER:
                continue
            if pos.type == mt5.POSITION_TYPE_BUY:
                buy_count += 1
                if buy_lowest == 0.0 or pos.price_open < buy_lowest:
                    buy_lowest = pos.price_open
            elif pos.type == mt5.POSITION_TYPE_SELL:
                sell_count += 1
                if sell_highest == 0.0 or pos.price_open > sell_highest:
                    sell_highest = pos.price_open

    return buy_count, sell_count, buy_lowest, sell_highest


def log_closed_trades(minutes_back=5):
    """Log recently closed trades."""
    time_to   = datetime.now()
    time_from = time_to - timedelta(minutes=minutes_back)
    deals     = mt5.history_deals_get(time_from, time_to)

    if not deals:
        return

    for deal in deals:
        if deal.entry != mt5.DEAL_ENTRY_OUT:
            continue

        direction = "BUY" if deal.type == mt5.ORDER_TYPE_BUY else "SELL"

        if deal.reason == mt5.DEAL_REASON_TP:
            reason = f"{direction} TP HIT"
        elif deal.reason == mt5.DEAL_REASON_SL:
            reason = f"{direction} SL HIT"
        else:
            reason = f"{direction} CLOSED"

        logger.info(f"{reason:<15} | Ticket: {deal.position_id} | Profit: {deal.profit:.2f}")


# =============================================================================
# MAIN LOOP
# =============================================================================

def main():
    print("=" * 60)
    print("  AURUMBOT - STARTING")
    print("=" * 60)

    connect_mt5()

    if not os.path.exists(MODEL_BUY_PATH) or not os.path.exists(MODEL_SELL_PATH):
        logger.critical(f"Model files not found. Run train_model.py first.")
        mt5.shutdown()
        sys.exit(1)

    model_buy  = joblib.load(MODEL_BUY_PATH)
    model_sell = joblib.load(MODEL_SELL_PATH)

    logger.info(f"Models loaded | Symbol: {SYMBOL} | "
                f"TPSL Mode: {TPSL_MODE} | "
                f"Threshold: {THRESHOLD_BUY}/{THRESHOLD_SELL}")
    logger.info(f"Filters: M15 bias + Key Level + S/D + MACD/RSI + ML (Volume filter disabled)")
    logger.info(f"Note: BoS filter DISABLED for demo phase")
    logger.info(f"Note: TREND LOCK removed — bias handled by entry conditions")

    if TPSL_MODE == "atr":
        logger.info(f"ATR TP/SL: TP={ATR_MULTIPLIER_TP}xATR | SL={ATR_MULTIPLIER_SL}xATR")
    else:
        logger.info(f"Fixed TP/SL: TP={TP_POINTS}pts | SL={SL_POINTS}pts")

    last_debug_minute = -1

    logger.info("Bot started. Entering main loop...")
    print("Bot is running. Press Ctrl+C to stop.\n")

    try:
        while True:

            # =============================================================
            # STEP 1 - RISK CONTROL
            # =============================================================
            try:
                account = mt5.account_info()
                if account is not None and account.profit <= MAX_FLOATING_LOSS:
                    logger.critical(
                        f"EMERGENCY STOP | Floating P&L: {account.profit:.2f}"
                    )
                    close_all_positions("EMERGENCY_STOP")
                    sys.exit(0)
            except Exception as e:
                logger.error(f"Risk control error: {e}")

            # =============================================================
            # STEP 2 - FETCH DATA
            # =============================================================
            try:
                df_m1  = get_bars(SYMBOL, mt5.TIMEFRAME_M1,  300)
                df_m15 = get_bars(SYMBOL, mt5.TIMEFRAME_M15,  50)

                if df_m1 is None or df_m15 is None:
                    logger.warning("Failed to fetch bars. Retrying...")
                    time.sleep(60)
                    continue
            except Exception as e:
                logger.error(f"Data fetch error: {e}")
                time.sleep(60)
                continue

            # =============================================================
            # STEP 3 - M15 BIAS
            # [FIX] TREND LOCK dihapus — logikanya terbalik (memblokir entry
            #        saat trending kuat). Arah entry sudah dikontrol di
            #        STEP 13 (bias != 'bearish') dan STEP 14 (bias != 'bullish')
            # =============================================================
            try:
                bias, slope = get_m15_bias(df_m15)
            except Exception as e:
                logger.error(f"Bias calculation error: {e}")
                time.sleep(60)
                continue

            # =============================================================
            # STEP 4 - M15 BIAS & THRESHOLD
            # Saat sideways, threshold dinaikkan (lebih selektif)
            # =============================================================
            if bias == 'sideways':
                threshold_buy  = THRESHOLD_BUY  + SIDEWAYS_THRESHOLD_ADDON
                threshold_sell = THRESHOLD_SELL + SIDEWAYS_THRESHOLD_ADDON
            else:
                threshold_buy  = THRESHOLD_BUY
                threshold_sell = THRESHOLD_SELL

            # =============================================================
            # STEP 5 - CALCULATE ATR-BASED TP/SL
            # =============================================================
            try:
                tp_pts, sl_pts = calculate_tpsl(df_m1)
            except Exception as e:
                logger.error(f"TP/SL calculation error: {e}")
                tp_pts, sl_pts = TP_POINTS, SL_POINTS

            # =============================================================
            # STEP 6 - FEATURE ENGINEERING
            # =============================================================
            try:
                last_bar = compute_features(df_m1, df_m15)
                if last_bar is None:
                    time.sleep(60)
                    continue

                p0 = last_bar['close']

                macd_indicator = ta.trend.MACD(
                    df_m1['close'], window_fast=12, window_slow=26, window_sign=9
                )
                macd_vals = macd_indicator.macd().values
                macd_0, macd_1 = macd_vals[-1], macd_vals[-2]

                rsi_vals = ta.momentum.RSIIndicator(
                    df_m1['close'], window=14
                ).rsi().values
                rsi_0, rsi_1 = rsi_vals[-1], rsi_vals[-2]

                bos_bullish = bool(last_bar.get('bos_bullish', 0))
                bos_bearish = bool(last_bar.get('bos_bearish', 0))

            except Exception as e:
                logger.error(f"Feature engineering error: {e}")
                time.sleep(60)
                continue

            # =============================================================
            # STEP 7 - KEY LEVEL & S/D CHECK
            # =============================================================
            try:
                support, resistance = detect_swing_levels(df_m1, SWING_LOOKBACK)
                in_sd_zone          = detect_sd_zone(
                    df_m1, SD_CONSOLIDATION_CANDLES, SD_CONSOLIDATION_WIDTH
                )
                near_support    = abs(p0 - support)    < KEY_LEVEL_RADIUS
                near_resistance = abs(p0 - resistance) < KEY_LEVEL_RADIUS

            except Exception as e:
                logger.error(f"Key level error: {e}")
                time.sleep(60)
                continue

            # =============================================================
            # STEP 8 - CANDLESTICK & VOLUME
            # =============================================================
            try:
                is_pin_bar, is_engulfing, is_inside_bar = compute_candle_patterns(df_m1)
                candle_confirm = is_pin_bar or is_engulfing or is_inside_bar
                volume_ratio   = last_bar.get('volume_ratio', 0)
                volume_confirm = volume_ratio > 1.0

            except Exception as e:
                logger.error(f"Pattern error: {e}")
                time.sleep(60)
                continue

            # =============================================================
            # STEP 9 - RULE-BASED SIGNALS (MACD + RSI)
            # =============================================================
            buy_signal  = (macd_1 < macd_0) and (rsi_1 < rsi_0) and (rsi_0 < 70)
            sell_signal = (macd_1 > macd_0) and (rsi_1 > rsi_0) and (rsi_0 > 30)

            # =============================================================
            # STEP 10 - ML CONFIRMATION
            # =============================================================
            try:
                X         = last_bar[FEATURES].to_frame().T
                prob_buy  = model_buy.predict_proba(X)[0, 1]
                prob_sell = model_sell.predict_proba(X)[0, 1]
            except Exception as e:
                logger.error(f"ML prediction error: {e}")
                time.sleep(60)
                continue

            # =============================================================
            # STEP 11 - POSITION COUNTS
            # =============================================================
            buy_count, sell_count, buy_lowest, sell_highest = get_position_counts()

            # =============================================================
            # STEP 12 - STATUS LOG (every minute)
            # =============================================================
            current_minute = datetime.now().minute
            if current_minute != last_debug_minute:
                last_debug_minute = current_minute
                logger.info(
                    f"STATUS | price: {p0:.2f} | bias: {bias:<8} | "
                    f"slope: {slope:.4f} | TP: {tp_pts:.1f}pts | SL: {sl_pts:.1f}pts | "
                    f"prob_buy: {prob_buy:.3f} | prob_sell: {prob_sell:.3f} | "
                    f"bos_bull: {bos_bullish} | bos_bear: {bos_bearish} | "
                    f"near_sup: {near_support} | near_res: {near_resistance} | "
                    f"vol_ok: {volume_confirm} | buy_pos: {buy_count} | sell_pos: {sell_count}"
                )

            # =============================================================
            # STEP 13 - BUY CONDITIONS
            # L1: bias tidak bearish (boleh bullish atau sideways)
            # L2: near_support (normal) | saat bullish: near_support OR near_resistance
            # L3: S&D zone ATAU candle pattern
            # L4: MACD + RSI signal (tanpa syarat posisi MACD)
            # L5: ML probability
            # L6: Max positions
            # =============================================================
            if (
                bias != 'bearish'               and
                (near_support or (bias == 'bullish' and near_resistance)) and  # L2: key level (relax saat bullish)
                (in_sd_zone or candle_confirm)  and
                buy_signal                      and
                prob_buy >= threshold_buy       and
                buy_count < MAX_TRADE           and
                (buy_count == 0 or (buy_lowest - p0) > MIN_DISTANCE)
            ):
                logger.info(
                    f"BUY SIGNAL | price: {p0:.2f} | prob: {prob_buy:.3f} | "
                    f"TP: {tp_pts:.1f}pts | SL: {sl_pts:.1f}pts | bias: {bias} | "
                    f"near_sup: {near_support} | vol: {volume_ratio:.2f}"
                )
                result = open_position('buy', LOT, tp_pts, sl_pts)
                if result is not None:
                    logger.info(f"BUY OPENED | ticket: {result.order} | price: {result.price:.2f}")
                    recalculate_averaged_tpsl('buy', tp_pts, sl_pts)
                else:
                    logger.warning("BUY order failed.")

            # =============================================================
            # STEP 14 - SELL CONDITIONS
            # L1: bias tidak bullish (boleh bearish atau sideways)
            # L2: near_resistance (normal) | saat bearish: near_resistance OR near_support
            # L3: S&D zone ATAU candle pattern
            # L4: MACD + RSI signal (tanpa syarat posisi MACD)
            # L5: ML probability
            # L6: Max positions
            # =============================================================
            if (
                bias != 'bullish'               and
                (near_resistance or (bias == 'bearish' and near_support)) and  # L2: key level (relax saat bearish)
                (in_sd_zone or candle_confirm)  and
                sell_signal                     and
                prob_sell >= threshold_sell     and
                sell_count < MAX_TRADE          and
                (sell_count == 0 or (p0 - sell_highest) > MIN_DISTANCE)
            ):
                logger.info(
                    f"SELL SIGNAL | price: {p0:.2f} | prob: {prob_sell:.3f} | "
                    f"TP: {tp_pts:.1f}pts | SL: {sl_pts:.1f}pts | bias: {bias} | "
                    f"near_res: {near_resistance} | vol: {volume_ratio:.2f}"
                )
                result = open_position('sell', LOT, tp_pts, sl_pts)
                if result is not None:
                    logger.info(f"SELL OPENED | ticket: {result.order} | price: {result.price:.2f}")
                    recalculate_averaged_tpsl('sell', tp_pts, sl_pts)
                else:
                    logger.warning("SELL order failed.")

            # =============================================================
            # STEP 15 - LOG CLOSED TRADES
            # =============================================================
            log_closed_trades(minutes_back=5)

            # =============================================================
            # STEP 16 - WAIT 60 DETIK
            # [FIX] Naik dari CHECK_INTERVAL (1 detik) ke 60 detik
            #       supaya loop per menit, tidak boros resource
            # =============================================================
            time.sleep(60)

    except KeyboardInterrupt:
        print("\nBot stopped by user (Ctrl+C).")
        logger.info("Bot stopped by user.")

    except Exception as e:
        logger.critical(f"Unexpected error: {e}", exc_info=True)

    finally:
        mt5.shutdown()
        logger.info("MT5 closed. Bot exited.")
        print("Bot exited.")


if __name__ == "__main__":
    main()
