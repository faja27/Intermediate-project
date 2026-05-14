"""
GoldBot Frequency Optimizer
Test berbagai kombinasi parameter untuk naikkan frekuensi + profit
menggunakan data historis 6 tahun (2020-2025)

Jalankan: python optimize_historical.py
"""

import pandas as pd
import numpy as np
import os, sys
from datetime import datetime
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CONFIG
from indicators import Indicators

CSV_PATH        = "data/XAU_M5.csv"
INITIAL_BALANCE = 100.0

# ============================================================================
# PARAMETER GRID - yang akan ditest
# ============================================================================
PARAM_GRID = {
    'breakout_period': [10, 15, 20],        # range pendek = lebih banyak signal
    'breakout_buffer': [20.0, 35.0, 50.0],  # buffer kecil = entry lebih mudah
    'daily_target':    [5.0, 10.0, 15.0],   # target lebih tinggi = bot lebih lama
    'trail_activate':  [80.0, 100.0, 120.0],# trail lebih awal = lebih banyak exit profit
}

# Minimum criteria
MIN_TRADES_DAY = 3.0    # minimal 3 trades/hari
MIN_WIN_RATE   = 0.50   # minimal 50% win rate
MIN_PF         = 1.50   # minimal profit factor 1.5
MAX_DD         = 25.0   # max drawdown 25%
MIN_PROFIT_YRS = 5      # minimal 5 dari 6 tahun profit


# ============================================================================
# LOAD DATA (sekali saja)
# ============================================================================

def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH, sep=None, engine='python')
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={'date': 'time'})
    df['time']  = pd.to_datetime(df['time'], format='%m/%d/%Y %H:%M',
                                 infer_datetime_format=True)
    for col in ['open','high','low','close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna().sort_values('time').reset_index(drop=True)
    return df


# ============================================================================
# FAST BACKTEST (streamlined untuk speed)
# ============================================================================

def fast_backtest(df_clean: pd.DataFrame, params: dict) -> dict:
    bp      = params['breakout_period']
    bb      = params['breakout_buffer']
    dt      = params['daily_target']
    ta      = params['trail_activate']
    td      = CONFIG.trail_distance_points
    sl_pts  = CONFIG.sl_points
    point   = 0.01
    lot     = CONFIG.lot_size
    comm    = CONFIG.commission_per_lot * lot

    close = df_clean['close']
    high  = df_clean['high']
    low   = df_clean['low']

    ema_t  = Indicators.ema(close, CONFIG.ema_trend)
    rsi    = Indicators.rsi(close, CONFIG.rsi_period)
    atr    = Indicators.atr(high, low, close, CONFIG.atr_period)
    ub, rh, rl, lb = Indicators.range_breakout(high, low, bp, bb)

    df = df_clean.copy()
    df['ema_t'] = ema_t
    df['rsi']   = rsi
    df['atr']   = atr
    df['ub']    = ub
    df['lb']    = lb
    df = df.dropna().reset_index(drop=True)

    balance    = INITIAL_BALANCE
    wins = losses = 0
    gp = gl = 0.0
    open_trade = None
    peak       = balance
    max_dd     = 0.0
    daily_pnl  = {}
    daily_tr   = {}
    yearly_pnl = {}

    for i in range(1, len(df)):
        row  = df.iloc[i]
        prev = df.iloc[i - 1]
        date = row['time'].strftime('%Y-%m-%d')
        year = row['time'].strftime('%Y')

        # Trailing update - pakai prev candle high/low, efektif di candle berikutnya
        if open_trade is not None:
            ep, sl, tp, direction, trail_on = open_trade
            if direction == "BUY":
                if (prev['high'] - ep) / point >= ta:
                    new_sl = prev['high'] - td * point
                    if new_sl > sl:
                        open_trade = (ep, new_sl, tp, direction, True)
            else:
                if (ep - prev['low']) / point >= ta:
                    new_sl = prev['low'] + td * point
                    if new_sl < sl:
                        open_trade = (ep, new_sl, tp, direction, True)

        # Exit - gunakan sl yang sudah di-update dari trailing
        if open_trade is not None:
            ep, sl, tp, direction, trail_on = open_trade
            ex = None
            if direction == "BUY":
                if row['low'] <= sl:     ex = sl
                elif row['high'] >= tp:  ex = tp   # tp=99999 = tidak pernah kena
            else:
                if row['high'] >= sl:    ex = sl
                elif row['low'] <= tp:   ex = tp   # tp=-99999 = tidak pernah kena

            hour_wib = (row['time'].hour + 7) % 24
            if ex is None and hour_wib == CONFIG.close_hour_wib:
                ex = row['close']

            if ex is not None:
                diff = (ex - ep if direction == "BUY" else ep - ex)
                pnl  = diff / point * 0.01 * (lot / 0.01) - comm
                balance += pnl
                peak    = max(peak, balance)
                max_dd  = max(max_dd, (peak - balance) / peak * 100)
                daily_pnl[date]  = daily_pnl.get(date, 0) + pnl
                daily_tr[date]   = daily_tr.get(date, 0) + 1
                yearly_pnl[year] = yearly_pnl.get(year, 0) + pnl
                if pnl > 0: wins += 1; gp += pnl
                else:        losses += 1; gl += abs(pnl)
                open_trade = None
                continue

        if open_trade is not None:
            continue

        # Daily limits
        if daily_pnl.get(date, 0) <= -CONFIG.max_daily_loss: continue
        if daily_pnl.get(date, 0) >= dt:                      continue
        if daily_tr.get(date, 0)  >= CONFIG.max_daily_trades: continue

        # Session
        hour_wib = (row['time'].hour + 7) % 24
        if not (hour_wib >= CONFIG.trade_hour_start or hour_wib < CONFIG.trade_hour_end):
            continue

        # ATR
        atr_pts = prev['atr'] / point
        if not (CONFIG.atr_min_points <= atr_pts <= CONFIG.atr_max_points):
            continue

        # Signal
        signal = None
        if (prev['close'] > prev['ub'] and
                prev['close'] > prev['ema_t'] and
                prev['rsi'] > CONFIG.rsi_min_buy):
            signal = "BUY"
        elif (prev['close'] < prev['lb'] and
                  prev['close'] < prev['ema_t'] and
                  prev['rsi'] < CONFIG.rsi_max_sell):
            signal = "SELL"

        if signal is None:
            continue

        entry = row['open']
        sl    = (entry - sl_pts * point if signal == "BUY"
                 else entry + sl_pts * point)
        # TP disabled: BUY pakai +99999, SELL pakai -99999
        # (mencegah SELL selalu exit karena low < 99999)
        tp_val = 99999.0 if signal == "BUY" else -99999.0
        open_trade = (entry, sl, tp_val, signal, False)

    total  = wins + losses
    if total == 0:
        return {'total_trades': 0, 'weekly_pnl': 0}

    days_t  = len(daily_pnl) or 1
    pf      = gp / (gl or 1e-9)
    wr      = wins / total
    tot_pnl = gp - gl
    avg_day = tot_pnl / days_t
    profit_years = sum(1 for v in yearly_pnl.values() if v > 0)

    return {
        'total_trades':  total,
        'trades_per_day':total / days_t,
        'win_rate':      wr,
        'profit_factor': pf,
        'total_pnl':     tot_pnl,
        'max_drawdown':  max_dd,
        'avg_daily':     avg_day,
        'weekly_pnl':    avg_day * 5,
        'profit_years':  profit_years,
        'yearly_pnl':    yearly_pnl,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("  GOLDBOT FREQUENCY OPTIMIZER - 6 TAHUN DATA")
    print("=" * 70)

    print("\nLoading 6-year dataset...")
    df_raw = load_data()
    print(f"  {len(df_raw):,} candles loaded")

    # Pre-compute base indicators (tidak berubah antar kombinasi)
    print("Pre-computing base indicators...")
    close = df_raw['close']
    high  = df_raw['high']
    low   = df_raw['low']
    df_raw['ema_t'] = Indicators.ema(close, CONFIG.ema_trend)
    df_raw['rsi']   = Indicators.rsi(close, CONFIG.rsi_period)
    df_raw['atr']   = Indicators.atr(high, low, close, CONFIG.atr_period)
    print("  Done.")

    keys   = list(PARAM_GRID.keys())
    combos = list(product(*[PARAM_GRID[k] for k in keys]))
    total  = len(combos)
    print(f"\nTesting {total} combinations...\n")

    results = []
    for idx, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        res    = fast_backtest(df_raw, params)
        res['params'] = params

        valid = (
            res['trades_per_day'] >= MIN_TRADES_DAY and
            res.get('win_rate', 0) >= MIN_WIN_RATE and
            res.get('profit_factor', 0) >= MIN_PF and
            res.get('max_drawdown', 99) <= MAX_DD and
            res.get('profit_years', 0) >= MIN_PROFIT_YRS
        )

        status = "OK" if valid else "below"
        print(
            f"[{idx:>2}/{total}] "
            f"bp={params['breakout_period']:>2} "
            f"buf={params['breakout_buffer']:>4.0f} "
            f"dt={params['daily_target']:>4.0f} "
            f"ta={params['trail_activate']:>4.0f} | "
            f"trades={res['trades_per_day']:>4.1f}/day "
            f"wr={res.get('win_rate',0):>5.1%} "
            f"pf={res.get('profit_factor',0):>5.2f} "
            f"dd={res.get('max_drawdown',0):>5.1f}% "
            f"wk=${res.get('weekly_pnl',0):>+6.2f} "
            f"yr={res.get('profit_years',0)}/6 | {status}"
        )
        results.append(res)

    # Filter valid
    valid_results = [r for r in results if
        r['trades_per_day'] >= MIN_TRADES_DAY and
        r.get('win_rate', 0) >= MIN_WIN_RATE and
        r.get('profit_factor', 0) >= MIN_PF and
        r.get('max_drawdown', 99) <= MAX_DD and
        r.get('profit_years', 0) >= MIN_PROFIT_YRS]

    print(f"\n{'='*70}")
    print(f"  {len(valid_results)} valid combinations from {total}")
    print(f"{'='*70}")

    if not valid_results:
        print("\nTidak ada yang lolos SEMUA kriteria.")
        print("Top 5 berdasarkan weekly P&L:")
        candidates = sorted(
            [r for r in results if r['total_trades'] > 0],
            key=lambda x: x.get('weekly_pnl', 0), reverse=True
        )
        for r in candidates[:5]:
            p = r['params']
            print(
                f"  bp={p['breakout_period']} buf={p['breakout_buffer']:.0f} "
                f"dt={p['daily_target']:.0f} ta={p['trail_activate']:.0f} | "
                f"{r['trades_per_day']:.1f}/day "
                f"wr={r.get('win_rate',0):.1%} "
                f"pf={r.get('profit_factor',0):.2f} "
                f"wk=${r.get('weekly_pnl',0):+.2f} "
                f"yr={r.get('profit_years',0)}/6"
            )
        return

    # Sort by weekly P&L
    valid_results.sort(key=lambda x: x['weekly_pnl'], reverse=True)

    print(f"\n{'Rank':<5} {'Trades/day':<12} {'WR':<8} {'PF':<7} {'Weekly':<10} {'DD':<8} {'Yrs':<6} Params")
    print("-" * 80)
    for i, r in enumerate(valid_results[:8], 1):
        p = r['params']
        print(
            f"#{i:<4} {r['trades_per_day']:<12.1f} "
            f"{r['win_rate']:.1%}   "
            f"{r['profit_factor']:.2f}   "
            f"${r['weekly_pnl']:>+7.2f}   "
            f"{r['max_drawdown']:.1f}%   "
            f"{r['profit_years']}/6  "
            f"bp={p['breakout_period']} buf={p['breakout_buffer']:.0f} "
            f"dt={p['daily_target']:.0f} ta={p['trail_activate']:.0f}"
        )

    best = valid_results[0]
    bp   = best['params']
    print(f"\n{'='*70}")
    print(f"BEST: bp={bp['breakout_period']} | buf={bp['breakout_buffer']:.0f} | "
          f"dt={bp['daily_target']:.0f} | ta={bp['trail_activate']:.0f}")
    print(f"      {best['trades_per_day']:.1f} trades/day | "
          f"WR={best['win_rate']:.1%} | PF={best['profit_factor']:.2f} | "
          f"Weekly=${best['weekly_pnl']:+.2f} | DD={best['max_drawdown']:.1f}%")
    print(f"\nUpdate config.py dengan parameter ini:")
    print(f"  breakout_period      = {bp['breakout_period']}")
    print(f"  breakout_buffer      = {bp['breakout_buffer']:.1f}")
    print(f"  daily_profit_target  = {bp['daily_target']:.1f}")
    print(f"  trail_activate_points= {bp['trail_activate']:.1f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
