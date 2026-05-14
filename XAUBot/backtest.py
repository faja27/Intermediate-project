"""
GoldBot Backtester - 90 hari data M5 XAUUSD.m
Include komisi $5/lot dalam kalkulasi P&L
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional

from config import CONFIG
from indicators import Indicators


class GoldBacktester:

    def __init__(self, config):
        self.config   = config
        self.point    = 0.01          # XAUUSD.m digits=2
        self.comm_lot = config.commission_per_lot * config.lot_size  # $0.05 per trade

    def fetch_data(self, days: int = 90) -> Optional[pd.DataFrame]:
        init = (mt5.initialize(path=self.config.mt5_path)
                if self.config.mt5_path else mt5.initialize())
        if not init:
            print(f"MT5 init failed: {mt5.last_error()}")
            return None

        if self.config.mt5_login and self.config.mt5_password:
            mt5.login(self.config.mt5_login,
                      password=self.config.mt5_password,
                      server=self.config.mt5_server)

        sym = mt5.symbol_info(self.config.symbol)
        if sym is None:
            print(f"Symbol {self.config.symbol} not found")
            mt5.shutdown()
            return None
        if not sym.visible:
            mt5.symbol_select(self.config.symbol, True)

        end   = datetime.now()
        start = end - timedelta(days=days)
        rates = mt5.copy_rates_range(
            self.config.symbol, self.config.timeframe, start, end
        )
        mt5.shutdown()

        if rates is None or len(rates) == 0:
            print("No data returned")
            return None

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        print(
            f"Data: {len(df)} candles M5 | "
            f"{df['time'].iloc[0].date()} to {df['time'].iloc[-1].date()}"
        )
        return df

    def run(self, df: pd.DataFrame, initial_balance: float = 100.0) -> dict:
        close = df['close']
        high  = df['high']
        low   = df['low']

        ema_t  = Indicators.ema(close, self.config.ema_trend)
        rsi    = Indicators.rsi(close, self.config.rsi_period)
        atr    = Indicators.atr(high, low, close, self.config.atr_period)
        ub, rh, rl, lb = Indicators.range_breakout(
            high, low, self.config.breakout_period, self.config.breakout_buffer
        )

        df = df.copy()
        df['ema_t']  = ema_t
        df['rsi']    = rsi
        df['atr']    = atr
        df['ub']     = ub
        df['lb']     = lb
        df['rh']     = rh
        df['rl']     = rl
        df = df.dropna().reset_index(drop=True)

        balance    = initial_balance
        curve      = [balance]
        wins = losses = 0
        gp = gl = 0.0
        open_trade = None
        peak       = balance
        max_dd     = 0.0
        daily_pnl  = {}
        daily_tr   = {}
        trades_log = []

        for i in range(1, len(df)):
            row  = df.iloc[i]
            prev = df.iloc[i - 1]
            date = row['time'].strftime('%Y-%m-%d')

            # --- TRAILING STOP ---
            trail_active = False
            if open_trade is not None and self.config.use_trailing:
                ep, sl, tp, direction, trail_was_active = open_trade
                if direction == "BUY":
                    profit_pts = (row['high'] - ep) / self.point
                    if profit_pts >= self.config.trail_activate_points:
                        new_sl = row['high'] - self.config.trail_distance_points * self.point
                        if new_sl > sl:
                            open_trade = (ep, new_sl, tp, direction, True)
                            sl = new_sl
                            trail_active = True
                        elif trail_was_active:
                            trail_active = True
                else:
                    profit_pts = (ep - row['low']) / self.point
                    if profit_pts >= self.config.trail_activate_points:
                        new_sl = row['low'] + self.config.trail_distance_points * self.point
                        if new_sl < sl:
                            open_trade = (ep, new_sl, tp, direction, True)
                            sl = new_sl
                            trail_active = True
                        elif trail_was_active:
                            trail_active = True

            # --- EXIT ---
            if open_trade is not None:
                ep, sl, tp, direction, trail_was_active = open_trade
                ex_price = None
                ex_reason = None

                if direction == "BUY":
                    if row['low'] <= sl:
                        ex_price = sl
                        ex_reason = "TRAIL" if trail_was_active else "SL"
                    elif row['high'] >= tp:
                        ex_price, ex_reason = tp, "TP"
                else:
                    if row['high'] >= sl:
                        ex_price = sl
                        ex_reason = "TRAIL" if trail_was_active else "SL"
                    elif row['low'] <= tp:
                        ex_price, ex_reason = tp, "TP"

                # Swap close
                hour = (row['time'].hour + 7) % 24
                if ex_price is None and hour == self.config.close_hour_wib:
                    ex_price, ex_reason = row['close'], "SWAP"

                if ex_price:
                    price_diff = (ex_price - ep if direction == "BUY"
                                  else ep - ex_price)
                    # 1 point = $0.01 per 0.01 lot
                    pnl_gross = price_diff / self.point * 0.01 * (self.config.lot_size / 0.01)
                    pnl_net   = pnl_gross - self.comm_lot

                    balance += pnl_net
                    curve.append(balance)
                    peak   = max(peak, balance)
                    max_dd = max(max_dd, (peak - balance) / peak * 100)

                    daily_pnl[date] = daily_pnl.get(date, 0) + pnl_net
                    daily_tr[date]  = daily_tr.get(date, 0) + 1

                    if pnl_net > 0:
                        wins += 1; gp += pnl_net
                    else:
                        losses += 1; gl += abs(pnl_net)

                    trades_log.append({
                        'date': date, 'dir': direction,
                        'pnl': pnl_net, 'exit': ex_reason
                    })
                    open_trade = None
                    continue

            # --- ENTRY ---
            if open_trade is not None:
                continue

            # Daily limits
            if daily_pnl.get(date, 0) <= -self.config.max_daily_loss:
                continue
            if daily_pnl.get(date, 0) >= self.config.daily_profit_target:
                continue
            if daily_tr.get(date, 0) >= self.config.max_daily_trades:
                continue

            # Session filter
            hour = (row['time'].hour + 7) % 24
            in_session = (hour >= self.config.trade_hour_start or
                          hour < self.config.trade_hour_end)
            if not in_session:
                continue

            # ATR filter
            atr_pts = prev['atr'] / self.point
            if (atr_pts < self.config.atr_min_points or
                    atr_pts > self.config.atr_max_points):
                continue

            # Signal
            signal = None
            if (prev['close'] > prev['ub'] and
                    prev['close'] > prev['ema_t'] and
                    prev['rsi'] > self.config.rsi_min_buy):
                signal = "BUY"
            elif (prev['close'] < prev['lb'] and
                      prev['close'] < prev['ema_t'] and
                      prev['rsi'] < self.config.rsi_max_sell):
                signal = "SELL"

            if signal is None:
                continue

            entry = row['open']
            sl    = (entry - self.config.sl_points * self.point if signal == "BUY"
                     else entry + self.config.sl_points * self.point)
            tp    = (entry + self.config.tp_points * self.point if signal == "BUY"
                     else entry - self.config.tp_points * self.point)
            open_trade = (entry, sl, tp, signal, False)

        # === METRICS ===
        total = wins + losses
        if total == 0:
            return {'total_trades': 0, 'note': 'No trades - check parameters'}

        days_t   = len(daily_pnl) or 1
        pf       = gp / (gl or 1e-9)
        wr       = wins / total
        tot_pnl  = gp - gl
        avg_day  = tot_pnl / days_t

        returns = np.diff(curve) / np.array(curve[:-1])
        sharpe  = (returns.mean() / returns.std() * np.sqrt(252 * 288)
                   if len(returns) > 1 and returns.std() > 0 else 0)

        return {
            'total_trades':     total,
            'wins':             wins,
            'losses':           losses,
            'win_rate':         wr,
            'profit_factor':    pf,
            'total_pnl':        tot_pnl,
            'total_return_pct': tot_pnl / initial_balance * 100,
            'final_balance':    balance,
            'avg_win':          gp / wins if wins else 0,
            'avg_loss':         -gl / losses if losses else 0,
            'max_drawdown_pct': max_dd,
            'sharpe':           sharpe,
            'days_traded':      days_t,
            'avg_daily_pnl':    avg_day,
            'win_days':         sum(1 for v in daily_pnl.values() if v > 0),
            'trades_per_day':   total / days_t,
            'exit_tp':          sum(1 for t in trades_log if t['exit'] == 'TP'),
            'exit_sl':          sum(1 for t in trades_log if t['exit'] == 'SL'),
            'exit_trail':       sum(1 for t in trades_log if t['exit'] == 'TRAIL'),
            'exit_swap':        sum(1 for t in trades_log if t['exit'] == 'SWAP'),
            'commission_total': self.comm_lot * total,
        }


def print_report(r: dict, days: int):
    print("\n" + "=" * 62)
    print("         GOLDBOT BACKTEST RESULTS - XAUUSD.m")
    print("=" * 62)

    if r.get('total_trades', 0) == 0:
        print(f"No trades. {r.get('note', '')}")
        return

    print(f"Period           : {days} hari")
    print(f"Total Trades     : {r['total_trades']} ({r['trades_per_day']:.1f}/hari)")
    print(f"Win Rate         : {r['win_rate']:.1%}")
    print(f"Profit Factor    : {r['profit_factor']:.2f}")
    print(f"Total P&L (net)  : ${r['total_pnl']:+.2f}")
    print(f"Commission paid  : ${r['commission_total']:.2f}")
    print(f"Total Return     : {r['total_return_pct']:+.2f}%")
    print(f"Final Balance    : ${r['final_balance']:.2f}")
    print(f"Avg Win (net)    : ${r['avg_win']:.2f}")
    print(f"Avg Loss (net)   : ${r['avg_loss']:.2f}")
    print(f"Max Drawdown     : {r['max_drawdown_pct']:.2f}%")
    print(f"Sharpe Ratio     : {r['sharpe']:.2f}")
    print(f"Days Traded      : {r['days_traded']}")
    print(f"Win Days         : {r['win_days']}/{r['days_traded']}")
    print(f"Avg Daily P&L    : ${r['avg_daily_pnl']:+.2f}")
    print(f"Exit TP/SL/Trail : {r['exit_tp']}/{r['exit_sl']}/{r['exit_trail']}")

    weekly  = r['avg_daily_pnl'] * 5
    monthly = weekly * 4
    print(f"\nProyeksi/minggu  : ${weekly:+.2f}")
    print(f"Proyeksi/bulan   : ${monthly:+.2f}")

    print("\n" + "-" * 62)
    ok_wr  = r['win_rate'] >= 0.48
    ok_pf  = r['profit_factor'] >= 1.15
    ok_dd  = r['max_drawdown_pct'] <= 20
    ok_tr  = r['trades_per_day'] >= 1.0

    if ok_wr and ok_pf and ok_dd and ok_tr:
        print("STATUS : LAYAK ditest di DEMO")
    elif r['profit_factor'] >= 1.0:
        print("STATUS : BORDERLINE - perlu tuning lebih")
        if not ok_wr:  print(f"         Win rate {r['win_rate']:.1%} < 48% target")
        if not ok_tr:  print(f"         Trades/hari {r['trades_per_day']:.1f} < 1.0 target")
    else:
        print("STATUS : TIDAK LAYAK - jangan deploy")

    print("=" * 62)
    print("\nREMINDER: Demo dulu 2 minggu. Jangan skip ke real.")


if __name__ == "__main__":
    bt   = GoldBacktester(CONFIG)
    DAYS = 90

    print(f"Fetching {DAYS} hari data M5 XAUUSD.m...")
    df = bt.fetch_data(days=DAYS)
    if df is not None:
        print("Running backtest...")
        result = bt.run(df, initial_balance=100.0)
        print_report(result, DAYS)
