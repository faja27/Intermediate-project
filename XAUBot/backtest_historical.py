"""
GoldBot Historical Backtest - CSV Dataset 2020-2025
Jalankan: python backtest_historical.py

Dataset: GoldBot/data/XAU_M5.csv
Kolom  : Date, Open, High, Low, Close, Volume
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# Import dari folder GoldBot
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CONFIG
from indicators import Indicators


# ============================================================================
# KONFIGURASI
# ============================================================================

CSV_PATH       = "data/XAU_M5.csv"
INITIAL_BALANCE= 100.0

# Periode yang akan ditest (bisa ubah untuk analisa per tahun)
# None = semua data
DATE_FROM = None   # contoh: "2020-01-01"
DATE_TO   = None   # contoh: "2025-12-31"


# ============================================================================
# LOAD & CLEAN DATA
# ============================================================================

def load_csv(path: str) -> pd.DataFrame:
    print(f"Loading {path}...")

    if not os.path.exists(path):
        print(f"ERROR: File tidak ditemukan: {path}")
        print(f"Pastikan file ada di: {os.path.abspath(path)}")
        sys.exit(1)

    # Baca file
    df = pd.read_csv(path, sep=None, engine='python')

    print(f"  Raw shape    : {df.shape}")
    print(f"  Columns      : {list(df.columns)}")
    print(f"  Sample row   : {df.iloc[0].to_dict()}")

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Rename ke format standar
    rename_map = {}
    for col in df.columns:
        if col in ('date', 'time', 'datetime', 'timestamp'):
            rename_map[col] = 'time'
        elif col in ('open', 'o'):
            rename_map[col] = 'open'
        elif col in ('high', 'h'):
            rename_map[col] = 'high'
        elif col in ('low', 'l'):
            rename_map[col] = 'low'
        elif col in ('close', 'c', 'price'):
            rename_map[col] = 'close'
        elif col in ('volume', 'vol', 'v', 'tick_volume', 'tickvol'):
            rename_map[col] = 'volume'

    df = df.rename(columns=rename_map)

    # Pastikan kolom wajib ada
    required = ['time', 'open', 'high', 'low', 'close']
    missing  = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: Kolom tidak ditemukan: {missing}")
        print(f"Kolom yang ada: {list(df.columns)}")
        sys.exit(1)

    if 'volume' not in df.columns:
        df['volume'] = 0

    # Parse datetime - coba berbagai format
    print("  Parsing datetime...")
    time_col = df['time'].astype(str).str.strip()

    parsed = None
    formats_to_try = [
        '%Y.%m.%d %H:%M',
        '%Y.%m.%d %H:%M:%S',
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d %H:%M',
        '%m/%d/%Y %H:%M',
        '%d/%m/%Y %H:%M',
        '%Y%m%d %H%M%S',
        '%Y%m%d%H%M%S',
    ]

    for fmt in formats_to_try:
        try:
            parsed = pd.to_datetime(time_col, format=fmt)
            print(f"  Datetime format: {fmt}")
            break
        except Exception:
            continue

    if parsed is None:
        # Fallback: pandas auto-detect
        try:
            parsed = pd.to_datetime(time_col, infer_datetime_format=True)
            print("  Datetime format: auto-detected")
        except Exception as e:
            print(f"ERROR: Tidak bisa parse datetime: {e}")
            print(f"Sample values: {time_col[:3].tolist()}")
            sys.exit(1)

    df['time'] = parsed

    # Sort dan clean
    df = df.sort_values('time').reset_index(drop=True)
    df = df.dropna(subset=['open','high','low','close'])

    # Convert price columns to float
    for col in ['open','high','low','close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['open','high','low','close'])

    # Filter tanggal jika ditentukan
    if DATE_FROM:
        df = df[df['time'] >= DATE_FROM]
    if DATE_TO:
        df = df[df['time'] <= DATE_TO]

    df = df.reset_index(drop=True)

    print(f"  Final rows   : {len(df):,}")
    print(f"  Date range   : {df['time'].iloc[0]} to {df['time'].iloc[-1]}")
    print(f"  Price range  : {df['close'].min():.2f} - {df['close'].max():.2f}")

    return df


# ============================================================================
# BACKTEST ENGINE
# ============================================================================

def run_backtest(df: pd.DataFrame, initial_balance: float = 100.0) -> dict:
    close = df['close']
    high  = df['high']
    low   = df['low']
    point = 0.01
    lot   = CONFIG.lot_size
    comm  = CONFIG.commission_per_lot * lot  # $0.05

    # Compute indicators
    print("Computing indicators...")
    ema_t  = Indicators.ema(close, CONFIG.ema_trend)
    rsi    = Indicators.rsi(close, CONFIG.rsi_period)
    atr    = Indicators.atr(high, low, close, CONFIG.atr_period)
    ub, rh, rl, lb = Indicators.range_breakout(
        high, low, CONFIG.breakout_period, CONFIG.breakout_buffer
    )

    df = df.copy()
    df['ema_t'] = ema_t
    df['rsi']   = rsi
    df['atr']   = atr
    df['ub']    = ub
    df['lb']    = lb
    df['rh']    = rh
    df['rl']    = rl
    df = df.dropna().reset_index(drop=True)
    print(f"  Candles after dropna: {len(df):,}")

    # Tracking
    balance    = initial_balance
    curve      = [balance]
    curve_dates= [df['time'].iloc[0]]
    wins = losses = 0
    gp = gl = 0.0
    open_trade = None     # (ep, sl, tp, direction, trail_active)
    peak       = balance
    max_dd     = 0.0
    daily_pnl  = {}
    daily_tr   = {}
    yearly_pnl = {}
    trades_log = []

    print("Running backtest...")
    total_rows = len(df)
    report_at  = set(range(0, total_rows, total_rows // 10))

    for i in range(1, total_rows):
        if i in report_at:
            pct = i / total_rows * 100
            print(f"  {pct:.0f}%... balance=${balance:.2f}")

        row  = df.iloc[i]
        prev = df.iloc[i - 1]
        date = row['time'].strftime('%Y-%m-%d')
        year = row['time'].strftime('%Y')

        # --- TRAILING STOP UPDATE ---
        if open_trade is not None and CONFIG.use_trailing:
            ep, sl, tp, direction, trail_active = open_trade
            if direction == "BUY":
                profit_pts = (row['high'] - ep) / point
                if profit_pts >= CONFIG.trail_activate_points:
                    new_sl = row['high'] - CONFIG.trail_distance_points * point
                    if new_sl > sl:
                        open_trade = (ep, new_sl, tp, direction, True)
            else:
                profit_pts = (ep - row['low']) / point
                if profit_pts >= CONFIG.trail_activate_points:
                    new_sl = row['low'] + CONFIG.trail_distance_points * point
                    if new_sl < sl:
                        open_trade = (ep, new_sl, tp, direction, True)

        # --- EXIT CHECK ---
        if open_trade is not None:
            ep, sl, tp, direction, trail_active = open_trade
            ex_price  = None
            ex_reason = None

            if direction == "BUY":
                if row['low'] <= sl:
                    ex_price  = sl
                    ex_reason = "TRAIL" if trail_active else "SL"
                elif row['high'] >= tp:
                    ex_price, ex_reason = tp, "TP"
            else:
                if row['high'] >= sl:
                    ex_price  = sl
                    ex_reason = "TRAIL" if trail_active else "SL"
                elif row['low'] <= tp:
                    ex_price, ex_reason = tp, "TP"

            # Swap close
            hour_utc = row['time'].hour
            hour_wib = (hour_utc + 7) % 24
            if ex_price is None and hour_wib == CONFIG.close_hour_wib:
                ex_price, ex_reason = row['close'], "SWAP"

            if ex_price is not None:
                price_diff = (ex_price - ep if direction == "BUY"
                              else ep - ex_price)
                pnl_gross  = price_diff / point * 0.01 * (lot / 0.01)
                pnl_net    = pnl_gross - comm

                balance += pnl_net
                curve.append(balance)
                curve_dates.append(row['time'])
                peak   = max(peak, balance)
                max_dd = max(max_dd, (peak - balance) / peak * 100)

                daily_pnl[date] = daily_pnl.get(date, 0) + pnl_net
                daily_tr[date]  = daily_tr.get(date, 0) + 1
                yearly_pnl[year]= yearly_pnl.get(year, 0) + pnl_net

                if pnl_net > 0: wins += 1; gp += pnl_net
                else:            losses += 1; gl += abs(pnl_net)

                trades_log.append({
                    'date': date, 'year': year,
                    'dir': direction, 'pnl': pnl_net,
                    'exit': ex_reason,
                    'entry': ep, 'exit_price': ex_price
                })
                open_trade = None
                continue

        # --- ENTRY CHECK ---
        if open_trade is not None:
            continue

        # Daily limits
        if daily_pnl.get(date, 0) <= -CONFIG.max_daily_loss:        continue
        if daily_pnl.get(date, 0) >= CONFIG.daily_profit_target:     continue
        if daily_tr.get(date, 0)  >= CONFIG.max_daily_trades:        continue

        # Session filter
        hour_wib = (row['time'].hour + 7) % 24
        in_session = (hour_wib >= CONFIG.trade_hour_start or
                      hour_wib < CONFIG.trade_hour_end)
        if not in_session:
            continue

        # ATR filter
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
        sl    = (entry - CONFIG.sl_points * point if signal == "BUY"
                 else entry + CONFIG.sl_points * point)
        tp    = (entry + CONFIG.tp_points * point if signal == "BUY"
                 else entry - CONFIG.tp_points * point)
        open_trade = (entry, sl, tp, signal, False)

    # ========================================================================
    # METRICS
    # ========================================================================
    total  = wins + losses
    if total == 0:
        return {'total_trades': 0}

    days_t  = len(daily_pnl) or 1
    years_t = len(yearly_pnl) or 1
    pf      = gp / (gl or 1e-9)
    wr      = wins / total
    tot_pnl = gp - gl

    returns = np.diff(curve) / np.array(curve[:-1])
    sharpe  = (returns.mean() / returns.std() * np.sqrt(252 * 288)
               if len(returns) > 1 and returns.std() > 0 else 0)

    # Per-year breakdown
    year_breakdown = {}
    for t in trades_log:
        y = t['year']
        if y not in year_breakdown:
            year_breakdown[y] = {'pnl': 0, 'trades': 0, 'wins': 0}
        year_breakdown[y]['pnl']    += t['pnl']
        year_breakdown[y]['trades'] += 1
        if t['pnl'] > 0:
            year_breakdown[y]['wins'] += 1

    return {
        'total_trades':    total,
        'wins':            wins,
        'losses':          losses,
        'win_rate':        wr,
        'profit_factor':   pf,
        'total_pnl':       tot_pnl,
        'total_return_pct':tot_pnl / initial_balance * 100,
        'final_balance':   balance,
        'avg_win':         gp / wins if wins else 0,
        'avg_loss':        -gl / losses if losses else 0,
        'max_drawdown_pct':max_dd,
        'sharpe':          sharpe,
        'days_traded':     days_t,
        'avg_daily_pnl':   tot_pnl / days_t,
        'win_days':        sum(1 for v in daily_pnl.values() if v > 0),
        'trades_per_day':  total / days_t,
        'commission_total':comm * total,
        'exit_tp':         sum(1 for t in trades_log if t['exit'] == 'TP'),
        'exit_sl':         sum(1 for t in trades_log if t['exit'] == 'SL'),
        'exit_trail':      sum(1 for t in trades_log if t['exit'] == 'TRAIL'),
        'exit_swap':       sum(1 for t in trades_log if t['exit'] == 'SWAP'),
        'year_breakdown':  year_breakdown,
        'curve':           curve,
        'curve_dates':     curve_dates,
    }


# ============================================================================
# REPORT
# ============================================================================

def print_report(r: dict):
    print("\n" + "=" * 65)
    print("     GOLDBOT HISTORICAL BACKTEST - XAUUSD.m 2020-2025")
    print("=" * 65)

    if r.get('total_trades', 0) == 0:
        print("No trades executed.")
        return

    years = sorted(r['year_breakdown'].keys())
    print(f"Period         : {years[0]} - {years[-1]}")
    print(f"Total Trades   : {r['total_trades']:,} ({r['trades_per_day']:.1f}/hari)")
    print(f"Win Rate       : {r['win_rate']:.1%}")
    print(f"Profit Factor  : {r['profit_factor']:.2f}")
    print(f"Total P&L      : ${r['total_pnl']:+,.2f}")
    print(f"Commission     : ${r['commission_total']:.2f}")
    print(f"Total Return   : {r['total_return_pct']:+.1f}%")
    print(f"Final Balance  : ${r['final_balance']:,.2f}")
    print(f"Avg Win        : ${r['avg_win']:.2f}")
    print(f"Avg Loss       : ${r['avg_loss']:.2f}")
    print(f"Max Drawdown   : {r['max_drawdown_pct']:.2f}%")
    print(f"Sharpe Ratio   : {r['sharpe']:.2f}")
    print(f"Days Traded    : {r['days_traded']:,}")
    print(f"Win Days       : {r['win_days']}/{r['days_traded']}")
    print(f"Avg Daily P&L  : ${r['avg_daily_pnl']:+.2f}")
    print(f"Exit TP/SL/Trail/Swap: {r['exit_tp']}/{r['exit_sl']}/{r['exit_trail']}/{r['exit_swap']}")

    # Weekly/monthly projection
    weekly  = r['avg_daily_pnl'] * 5
    monthly = weekly * 4
    print(f"\nProyeksi/minggu: ${weekly:+.2f}")
    print(f"Proyeksi/bulan : ${monthly:+.2f}")

    # Per-year breakdown
    print(f"\n{'─'*65}")
    print(f"{'Tahun':<8} {'Trades':>7} {'WR':>7} {'PF':>7} {'P&L':>10} {'Status'}")
    print(f"{'─'*65}")

    for year in years:
        yb = r['year_breakdown'][year]
        t  = yb['trades']
        w  = yb['wins']
        p  = yb['pnl']
        wr = w/t if t > 0 else 0
        status = "PROFIT" if p > 0 else "LOSS"
        pf_est = (w * r['avg_win']) / ((t-w) * abs(r['avg_loss']) + 0.001)
        print(f"{year:<8} {t:>7,} {wr:>7.1%} {pf_est:>7.2f} {p:>+10.2f}  {status}")

    print(f"{'─'*65}")

    # Assessment
    print(f"\n{'='*65}")
    ok_wr  = r['win_rate'] >= 0.48
    ok_pf  = r['profit_factor'] >= 1.3
    ok_dd  = r['max_drawdown_pct'] <= 25
    ok_yrs = sum(1 for yb in r['year_breakdown'].values() if yb['pnl'] > 0)
    total_yrs = len(r['year_breakdown'])

    print(f"Profitable years : {ok_yrs}/{total_yrs}")

    if ok_wr and ok_pf and ok_dd and ok_yrs >= total_yrs * 0.7:
        print("STATUS: ROBUST - strategi bekerja di berbagai kondisi market")
        print("        LAYAK untuk demo dan pertimbangkan live")
    elif ok_pf and ok_yrs >= total_yrs * 0.5:
        print("STATUS: MODERATE - bekerja di sebagian kondisi")
        print("        Perhatikan tahun mana yang loss, hindari kondisi itu")
    else:
        print("STATUS: TIDAK ROBUST - terlalu bergantung pada kondisi tertentu")
        print("        Jangan live sebelum diperbaiki")

    print(f"\nKEY INSIGHT:")
    best_year  = max(r['year_breakdown'].items(), key=lambda x: x[1]['pnl'])
    worst_year = min(r['year_breakdown'].items(), key=lambda x: x[1]['pnl'])
    print(f"  Best year  : {best_year[0]} (+${best_year[1]['pnl']:.2f})")
    print(f"  Worst year : {worst_year[0]} (${worst_year[1]['pnl']:+.2f})")
    print("=" * 65)
    print("\nREMINDER: Historical backtest tidak menjamin hasil future.")
    print("Demo 2 minggu tetap WAJIB sebelum live.")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  GOLDBOT HISTORICAL BACKTEST - 2020-2025")
    print("=" * 65)

    df = load_csv(CSV_PATH)
    print()

    result = run_backtest(df, INITIAL_BALANCE)
    print_report(result)
