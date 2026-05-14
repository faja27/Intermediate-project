"""
GoldBot Debug Backtest
Menampilkan berapa sinyal yang lolos/blocked per filter.
Jalankan ini untuk diagnosis kenapa tidak ada trades.
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from config import CONFIG
from indicators import Indicators


def debug_run():
    # Connect MT5
    init = (mt5.initialize(path=CONFIG.mt5_path)
            if CONFIG.mt5_path else mt5.initialize())
    if not init:
        print(f"MT5 init failed: {mt5.last_error()}")
        return

    if CONFIG.mt5_login and CONFIG.mt5_password:
        mt5.login(CONFIG.mt5_login,
                  password=CONFIG.mt5_password,
                  server=CONFIG.mt5_server)

    sym = mt5.symbol_info(CONFIG.symbol)
    if sym is None:
        print(f"Symbol {CONFIG.symbol} not found")
        mt5.shutdown()
        return
    if not sym.visible:
        mt5.symbol_select(CONFIG.symbol, True)

    end   = datetime.now()
    start = end - timedelta(days=90)
    rates = mt5.copy_rates_range(CONFIG.symbol, CONFIG.timeframe, start, end)
    mt5.shutdown()

    if rates is None:
        print("No data")
        return

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    print(f"Data: {len(df)} candles | {df['time'].iloc[0].date()} to {df['time'].iloc[-1].date()}")

    # Compute indicators
    close = df['close']
    high  = df['high']
    low   = df['low']
    point = 0.01

    ema_t  = Indicators.ema(close, CONFIG.ema_trend)
    rsi    = Indicators.rsi(close, CONFIG.rsi_period)
    atr    = Indicators.atr(high, low, close, CONFIG.atr_period)
    ub, rh, rl, lb = Indicators.range_breakout(
        high, low, CONFIG.breakout_period, CONFIG.breakout_buffer
    )

    df['ema_t'] = ema_t
    df['rsi']   = rsi
    df['atr']   = atr
    df['ub']    = ub
    df['lb']    = lb
    df['rh']    = rh
    df['rl']    = rl
    df = df.dropna().reset_index(drop=True)

    print(f"\nCandles setelah dropna: {len(df)}")
    print(f"\n=== SAMPLE DATA (5 candles terakhir) ===")
    sample = df.tail(5)[['time','close','ema_t','rsi','atr','ub','lb','rh','rl']]
    sample['atr_pts'] = sample['atr'] / point
    sample['range_pts'] = (sample['rh'] - sample['rl']) / point
    pd.set_option('display.float_format', '{:.2f}'.format)
    pd.set_option('display.max_columns', 15)
    pd.set_option('display.width', 200)
    print(sample.to_string(index=False))

    # Count filter passes
    total = len(df) - 1
    c_session = c_atr = c_above_ub = c_below_lb = 0
    c_buy_trend = c_sell_trend = 0
    c_buy_rsi = c_sell_rsi = 0
    c_buy_full = c_sell_full = 0

    atr_values    = []
    range_values  = []

    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        row  = df.iloc[i]

        # Session
        hour = (row['time'].hour + 7) % 24
        in_session = hour >= CONFIG.trade_hour_start or hour < CONFIG.trade_hour_end
        if in_session:
            c_session += 1

        # ATR
        atr_pts = prev['atr'] / point
        atr_values.append(atr_pts)
        range_pts = (prev['rh'] - prev['rl']) / point
        range_values.append(range_pts)

        atr_ok = CONFIG.atr_min_points <= atr_pts <= CONFIG.atr_max_points
        if in_session and atr_ok:
            c_atr += 1

        # Breakout
        above_ub = prev['close'] > prev['ub']
        below_lb = prev['close'] < prev['lb']
        if in_session and atr_ok and above_ub:
            c_above_ub += 1
        if in_session and atr_ok and below_lb:
            c_below_lb += 1

        # Trend filter
        if in_session and atr_ok and above_ub and prev['close'] > prev['ema_t']:
            c_buy_trend += 1
        if in_session and atr_ok and below_lb and prev['close'] < prev['ema_t']:
            c_sell_trend += 1

        # RSI filter
        if in_session and atr_ok and above_ub and prev['close'] > prev['ema_t'] and prev['rsi'] > CONFIG.rsi_min_buy:
            c_buy_rsi += 1
            c_buy_full += 1
        if in_session and atr_ok and below_lb and prev['close'] < prev['ema_t'] and prev['rsi'] < CONFIG.rsi_max_sell:
            c_sell_rsi += 1
            c_sell_full += 1

    print(f"\n=== STATISTIK INDIKATOR ===")
    print(f"ATR (points) - Min: {min(atr_values):.1f} | Max: {max(atr_values):.1f} | Avg: {sum(atr_values)/len(atr_values):.1f} | Median: {sorted(atr_values)[len(atr_values)//2]:.1f}")
    print(f"Range (points)- Min: {min(range_values):.1f} | Max: {max(range_values):.1f} | Avg: {sum(range_values)/len(range_values):.1f}")

    print(f"\n=== FILTER FUNNEL (dari {total} candles) ===")
    print(f"1. Session filter OK    : {c_session:>6} ({c_session/total*100:.1f}%)")
    print(f"2. ATR filter OK        : {c_atr:>6} ({c_atr/total*100:.1f}%)  [min={CONFIG.atr_min_points}, max={CONFIG.atr_max_points}]")
    print(f"3a. Breakout UP (BUY)   : {c_above_ub:>6} ({c_above_ub/total*100:.1f}%)")
    print(f"3b. Breakout DN (SELL)  : {c_below_lb:>6} ({c_below_lb/total*100:.1f}%)")
    print(f"4a. Trend OK (BUY)      : {c_buy_trend:>6} ({c_buy_trend/total*100:.1f}%)  [close > EMA{CONFIG.ema_trend}]")
    print(f"4b. Trend OK (SELL)     : {c_sell_trend:>6} ({c_sell_trend/total*100:.1f}%)  [close < EMA{CONFIG.ema_trend}]")
    print(f"5a. RSI OK (BUY)        : {c_buy_rsi:>6} ({c_buy_rsi/total*100:.1f}%)  [RSI > {CONFIG.rsi_min_buy}]")
    print(f"5b. RSI OK (SELL)       : {c_sell_rsi:>6} ({c_sell_rsi/total*100:.1f}%)  [RSI < {CONFIG.rsi_max_sell}]")
    print(f"─────────────────────────────────────────")
    print(f"TOTAL BUY signals       : {c_buy_full:>6}")
    print(f"TOTAL SELL signals      : {c_sell_full:>6}")
    print(f"TOTAL ALL signals       : {c_buy_full + c_sell_full:>6}")

    print(f"\n=== DIAGNOSIS ===")
    if c_session == 0:
        print("MASALAH: Session filter memblokir semua candle")
        print(f"  trade_hour_start={CONFIG.trade_hour_start}, trade_hour_end={CONFIG.trade_hour_end}")
    elif c_atr == 0:
        avg_atr = sum(atr_values)/len(atr_values)
        print(f"MASALAH: ATR filter terlalu ketat")
        print(f"  Config: min={CONFIG.atr_min_points}, max={CONFIG.atr_max_points}")
        print(f"  Actual ATR avg: {avg_atr:.1f} points")
        print(f"  FIX: ubah atr_min_points ke {avg_atr*0.3:.0f} dan atr_max_points ke {avg_atr*5:.0f}")
    elif c_above_ub + c_below_lb == 0:
        avg_range = sum(range_values)/len(range_values)
        print(f"MASALAH: Breakout tidak pernah terjadi")
        print(f"  breakout_period={CONFIG.breakout_period}, buffer={CONFIG.breakout_buffer} pts")
        print(f"  Avg range size: {avg_range:.1f} points")
        print(f"  FIX: turunkan breakout_period ke 10 atau hapus buffer")
    elif c_buy_trend + c_sell_trend == 0:
        print("MASALAH: EMA trend filter memblokir semua breakout")
        print(f"  ema_trend={CONFIG.ema_trend}")
        print("  FIX: turunkan ema_trend ke 20 atau 34")
    elif c_buy_full + c_sell_full == 0:
        print("MASALAH: RSI filter memblokir semua sinyal")
        print(f"  rsi_min_buy={CONFIG.rsi_min_buy}, rsi_max_sell={CONFIG.rsi_max_sell}")
        print("  FIX: ubah ke rsi_min_buy=40, rsi_max_sell=60")
    else:
        print(f"Filter OK! Ada {c_buy_full + c_sell_full} sinyal.")
        print("Mungkin daily limits terlalu agresif, atau issue lain di backtest logic.")

    print(f"\n=== CONFIG SAAT INI ===")
    print(f"breakout_period : {CONFIG.breakout_period}")
    print(f"breakout_buffer : {CONFIG.breakout_buffer} pts")
    print(f"ema_trend       : {CONFIG.ema_trend}")
    print(f"atr_min_points  : {CONFIG.atr_min_points}")
    print(f"atr_max_points  : {CONFIG.atr_max_points}")
    print(f"rsi_min_buy     : {CONFIG.rsi_min_buy}")
    print(f"rsi_max_sell    : {CONFIG.rsi_max_sell}")


if __name__ == "__main__":
    debug_run()
