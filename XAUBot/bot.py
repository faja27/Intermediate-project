"""
================================================================================
GOLDBOT - Breakout Scalping Bot untuk XAUUSD.m
Strategy  : Range Breakout + EMA Trend + RSI Momentum + ATR Filter
Platform  : MetaTrader 5 | Broker: MIFX (Monex Trader)
Modal     : $100 | Lot: 0.01 | Target: $1.5-3/minggu
================================================================================
DISCLAIMER: Trading berisiko. Demo dulu minimal 2 minggu. Tidak ada jaminan profit.
================================================================================
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import logging
import signal
import os
from datetime import datetime

from config import CONFIG
from indicators import Indicators
from risk_manager import RiskManager
from journal import TradingJournal


# ============================================================================
# LOGGER
# ============================================================================

def setup_logger(log_file: str) -> logging.Logger:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger("GoldBot")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger


# ============================================================================
# MT5 INTERFACE
# ============================================================================

class MT5Interface:

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def connect(self) -> bool:
        init = (mt5.initialize(path=self.config.mt5_path)
                if self.config.mt5_path else mt5.initialize())
        if not init:
            self.logger.error(f"MT5 init failed: {mt5.last_error()}")
            return False

        if self.config.mt5_login and self.config.mt5_password:
            ok = mt5.login(
                self.config.mt5_login,
                password=self.config.mt5_password,
                server=self.config.mt5_server
            )
            if not ok:
                self.logger.error(f"Login failed: {mt5.last_error()}")
                mt5.shutdown()
                return False

        sym = mt5.symbol_info(self.config.symbol)
        if sym is None:
            self.logger.error(f"Symbol {self.config.symbol} not found")
            mt5.shutdown()
            return False
        if not sym.visible:
            mt5.symbol_select(self.config.symbol, True)

        acc = mt5.account_info()
        self.logger.info(
            f"Connected | {acc.login} @ {acc.server} | "
            f"Balance: ${acc.balance:.2f} | Equity: ${acc.equity:.2f}"
        )
        return True

    def disconnect(self):
        mt5.shutdown()
        self.logger.info("Disconnected")

    def get_rates(self, count: int = 150) -> pd.DataFrame:
        rates = mt5.copy_rates_from_pos(
            self.config.symbol, self.config.timeframe, 0, count
        )
        if rates is None or len(rates) == 0:
            return None
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df

    def get_balance(self) -> float:
        acc = mt5.account_info()
        return acc.balance if acc else 0.0

    def get_equity(self) -> float:
        acc = mt5.account_info()
        return acc.equity if acc else 0.0

    def get_spread_points(self) -> float:
        tick = mt5.symbol_info_tick(self.config.symbol)
        info = mt5.symbol_info(self.config.symbol)
        if not tick or not info:
            return 9999.0
        return round((tick.ask - tick.bid) / info.point)

    def get_positions(self) -> list:
        pos = mt5.positions_get(symbol=self.config.symbol)
        if pos is None:
            return []
        return [p for p in pos if p.magic == self.config.magic_number]

    def place_order(self, direction: str, sl: float, tp: float) -> bool:
        tick = mt5.symbol_info_tick(self.config.symbol)
        info = mt5.symbol_info(self.config.symbol)
        if not tick or not info:
            return False

        mt5_type = mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL
        price    = tick.ask if direction == "BUY" else tick.bid
        sl       = round(sl, info.digits)
        tp       = round(tp, info.digits)

        # MIFX Gold menggunakan FOK
        request = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       self.config.symbol,
            "volume":       self.config.lot_size,
            "type":         mt5_type,
            "price":        price,
            "sl":           sl,
            "tp":           tp,
            "deviation":    30,
            "magic":        self.config.magic_number,
            "comment":      "GoldBot",
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }

        result = mt5.order_send(request)

        # Fallback ke mode lain jika FOK gagal
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            for mode in [mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_RETURN]:
                request["type_filling"] = mode
                result = mt5.order_send(request)
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    break

        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            self.logger.info(
                f"ORDER {direction} | Price: {result.price:.2f} | "
                f"SL: {sl:.2f} | TP: {tp:.2f} | Ticket: {result.order}"
            )
            return True

        self.logger.error(
            f"Order FAILED | retcode={result.retcode if result else 'None'} | "
            f"{result.comment if result else ''}"
        )
        return False

    def modify_sl(self, ticket: int, new_sl: float) -> bool:
        """Update trailing stop loss."""
        info = mt5.symbol_info(self.config.symbol)
        request = {
            "action":   mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl":       round(new_sl, info.digits),
        }
        result = mt5.order_send(request)
        return result and result.retcode == mt5.TRADE_RETCODE_DONE

    def close_position(self, pos) -> float:
        tick = mt5.symbol_info_tick(pos.symbol)
        if not tick:
            return 0.0

        mt5_type = (mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY
                    else mt5.ORDER_TYPE_BUY)
        price = tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask

        request = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       pos.symbol,
            "volume":       pos.volume,
            "type":         mt5_type,
            "position":     pos.ticket,
            "price":        price,
            "deviation":    30,
            "magic":        self.config.magic_number,
            "comment":      "GoldBot_Close",
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }

        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            self.logger.info(
                f"CLOSED #{pos.ticket} | PnL: ${pos.profit:+.2f}"
            )
            return pos.profit
        return 0.0

    def close_all_positions(self, reason: str = "MANUAL"):
        """Close semua posisi - dipanggil sebelum swap time."""
        positions = self.get_positions()
        if not positions:
            return
        self.logger.info(f"Closing all positions | Reason: {reason}")
        for pos in positions:
            self.close_position(pos)


# ============================================================================
# MAIN GOLDBOT
# ============================================================================

class GoldBot:

    def __init__(self, config):
        self.config  = config
        self.logger  = setup_logger(config.log_file)
        self.mt5     = MT5Interface(config, self.logger)
        self.risk    = RiskManager(config, self.logger)
        self.running = True
        self._closed_tickets = set()
        self._last_order_time = None  # track kapan terakhir order dikirim
        self._open_times = {}         # ticket -> (datetime_open, price_open, direction, sl, tp)
        self.journal = TradingJournal(config.journal_file)

        # Load ticket yang sudah ada di jurnal supaya tidak duplikat saat restart
        try:
            import pandas as pd
            xl = pd.read_excel(config.journal_file, sheet_name=None)
            for sheet_df in xl.values():
                if 'Catatan' in sheet_df.columns:
                    for val in sheet_df['Catatan'].dropna():
                        ticket_str = str(val).replace('#', '').strip()
                        if ticket_str.isdigit():
                            self._closed_tickets.add(int(ticket_str))
        except Exception:
            pass  # file belum ada atau kosong, tidak masalah

        signal.signal(signal.SIGINT,  self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

    def _shutdown(self, signum, frame):
        self.logger.info("Shutdown signal. Closing positions...")
        self.mt5.close_all_positions("SHUTDOWN")
        # Tambah separator ringkasan hari ini ke jurnal
        try:
            summary = self.risk.get_daily_summary()
            self.journal.add_daily_separator(
                daily_pnl   = self.risk.daily_pnl,
                trade_count = self.risk.daily_trades,
            )
        except Exception:
            pass
        self.running = False

    def _is_trading_session(self) -> bool:
        """
        Gold aktif: London (14:00-23:00 WIB) + NY (19:00-03:00 WIB)
        Hindari Asian session (sepi, breakout sering false)
        Logic: jam >= 13 ATAU jam < 3
        """
        hour = datetime.now().hour
        return hour >= self.config.trade_hour_start or hour < self.config.trade_hour_end

    def _should_close_for_swap(self) -> bool:
        """Tutup posisi sebelum swap charge (23:00 WIB)."""
        if not self.config.close_before_swap:
            return False
        hour   = datetime.now().hour
        minute = datetime.now().minute
        return hour == self.config.close_hour_wib and minute >= 50

    def analyze(self, df: pd.DataFrame) -> dict:
        close = df['close']
        high  = df['high']
        low   = df['low']

        ema_trend              = Indicators.ema(close, self.config.ema_trend)
        rsi                    = Indicators.rsi(close, self.config.rsi_period)
        atr                    = Indicators.atr(high, low, close, self.config.atr_period)
        upper_band, range_high, range_low, lower_band = Indicators.range_breakout(
            high, low, self.config.breakout_period, self.config.breakout_buffer
        )

        # Pakai candle yang sudah close (-2), bukan forming (-1)
        i = -2
        return {
            'time':        df['time'].iloc[i],
            'close':       close.iloc[i],
            'high':        high.iloc[i],
            'low':         low.iloc[i],
            'ema_trend':   ema_trend.iloc[i],
            'rsi':         rsi.iloc[i],
            'atr':         atr.iloc[i],           # dalam price units
            'atr_points':  atr.iloc[i] / 0.01,    # convert ke points (digits=2)
            'upper_band':  upper_band.iloc[i],
            'lower_band':  lower_band.iloc[i],
            'range_high':  range_high.iloc[i],
            'range_low':   range_low.iloc[i],
        }

    def get_signal(self, ind: dict) -> str:
        # ATR filter
        if (ind['atr_points'] < self.config.atr_min_points or
                ind['atr_points'] > self.config.atr_max_points):
            return None

        # BUY: breakout atas range + uptrend + momentum bullish
        if (ind['close'] > ind['upper_band'] and
                ind['close'] > ind['ema_trend'] and
                ind['rsi'] > self.config.rsi_min_buy):
            return "BUY"

        # SELL: breakout bawah range + downtrend + momentum bearish
        if (ind['close'] < ind['lower_band'] and
                ind['close'] < ind['ema_trend'] and
                ind['rsi'] < self.config.rsi_max_sell):
            return "SELL"

        return None

    def execute_trade(self, signal: str, ind: dict) -> bool:
        tick = mt5.symbol_info_tick(self.config.symbol)
        if not tick:
            return False

        point = 0.01  # XAUUSD.m digits=2, 1 point = 0.01

        if signal == "BUY":
            price = tick.ask
            sl    = price - (self.config.sl_points * point)
            tp    = 99999.0   # TP disabled - exit via trailing stop
        else:
            price = tick.bid
            sl    = price + (self.config.sl_points * point)
            tp    = 0.01      # TP disabled untuk SELL - exit via trailing stop

        # Log estimasi
        range_size = (ind['range_high'] - ind['range_low']) / point
        self.logger.info(
            f"SIGNAL {signal} | Price: {price:.2f} | "
            f"Range: {ind['range_low']:.2f}-{ind['range_high']:.2f} ({range_size:.0f}pts) | "
            f"RSI: {ind['rsi']:.1f} | ATR: {ind['atr_points']:.0f}pts | "
            f"SL: {self.config.sl_points:.0f}pts | TP: {self.config.tp_points:.0f}pts"
        )
        # Set cooldown DULU sebelum kirim order (cegah race condition)
        self._last_order_time = datetime.now()
        open_time = datetime.now()
        success = self.mt5.place_order(signal, sl, tp)
        if not success:
            self._last_order_time = None  # reset kalau order gagal
        else:
            # Simpan info buka posisi untuk jurnal (pakai ticket dari posisi terbaru)
            time.sleep(0.5)  # beri MT5 waktu register
            positions = self.mt5.get_positions()
            if positions:
                newest = max(positions, key=lambda p: p.ticket)
                self._open_times[newest.ticket] = (open_time, price, signal, sl, tp)
        return success

    def manage_trailing_stop(self):
        """Update trailing stop untuk posisi terbuka."""
        if not self.config.use_trailing:
            return

        positions = self.mt5.get_positions()
        point = 0.01

        for pos in positions:
            tick = mt5.symbol_info_tick(pos.symbol)
            if not tick:
                continue

            profit_points = 0.0
            new_sl        = None

            if pos.type == mt5.POSITION_TYPE_BUY:
                profit_points = (tick.bid - pos.price_open) / point
                if profit_points >= self.config.trail_activate_points:
                    new_sl = tick.bid - (self.config.trail_distance_points * point)
                    if new_sl > pos.sl:    # hanya geser ke atas
                        self.mt5.modify_sl(pos.ticket, new_sl)
                        self.logger.info(
                            f"TRAIL BUY #{pos.ticket} | "
                            f"Profit: {profit_points:.0f}pts | New SL: {new_sl:.2f}"
                        )

            elif pos.type == mt5.POSITION_TYPE_SELL:
                profit_points = (pos.price_open - tick.ask) / point
                if profit_points >= self.config.trail_activate_points:
                    new_sl = tick.ask + (self.config.trail_distance_points * point)
                    if new_sl < pos.sl:    # hanya geser ke bawah
                        self.mt5.modify_sl(pos.ticket, new_sl)
                        self.logger.info(
                            f"TRAIL SELL #{pos.ticket} | "
                            f"Profit: {profit_points:.0f}pts | New SL: {new_sl:.2f}"
                        )

    def check_closed_trades(self):
        """Catat trade yang baru close ke risk manager."""
        from datetime import timedelta
        # Query 24 jam kebelakang, lalu filter manual by tanggal lokal WIB.
        # Cara paling robust - tidak bergantung pada timezone MT5 broker.
        since = datetime.now() - timedelta(hours=24)
        deals = mt5.history_deals_get(since, datetime.now())
        if not deals:
            return

        today_date = datetime.now().date()  # tanggal hari ini WIB

        for deal in deals:
            if deal.magic != self.config.magic_number:
                continue
            if deal.entry != mt5.DEAL_ENTRY_OUT:
                continue
            if deal.ticket in self._closed_tickets:
                continue

            # Hanya proses trade yang close HARI INI (lokal WIB)
            deal_date = datetime.fromtimestamp(deal.time).date()
            if deal_date != today_date:
                continue

            self._closed_tickets.add(deal.ticket)
            self.risk.record_trade(deal.profit)

            reason = ("TP"     if deal.reason == mt5.DEAL_REASON_TP  else
                      "SL"     if deal.reason == mt5.DEAL_REASON_SL  else
                      "TRAIL"  if deal.reason == mt5.DEAL_REASON_SO  else
                      "MANUAL")
            self.logger.info(
                f"CLOSED | {reason} | Profit: ${deal.profit:+.2f} | "
                f"{self.risk.get_daily_summary()}"
            )

            # Catat ke jurnal Excel
            try:
                close_time = datetime.fromtimestamp(deal.time)
                open_info  = self._open_times.pop(deal.position_id, None)
                open_time  = open_info[0] if open_info else close_time
                price_open = open_info[1] if open_info else deal.price
                direction  = open_info[2] if open_info else ("BUY" if deal.type == mt5.DEAL_TYPE_SELL else "SELL")
                sl_val     = open_info[3] if open_info else 0.0
                tp_val     = open_info[4] if open_info else 0.0
                self.journal.log_trade(
                    ticket      = deal.position_id,
                    symbol      = deal.symbol,
                    direction   = direction,
                    lot         = deal.volume,
                    price_open  = price_open,
                    price_close = deal.price,
                    sl          = sl_val,
                    tp          = tp_val,
                    profit      = deal.profit,
                    exit_reason = reason,
                    open_time   = open_time,
                    close_time  = close_time,
                    commission  = abs(deal.commission) if deal.commission else 0.05,
                )
            except Exception as je:
                self.logger.warning(f"Journal write failed: {je}")

    def cycle(self):
        balance = self.mt5.get_balance()
        self.risk.initialize(balance)

        # Cek trade yang baru close
        self.check_closed_trades()

        # Swap protection - close sebelum 23:50 WIB
        if self._should_close_for_swap():
            self.mt5.close_all_positions("SWAP_PROTECTION")
            return

        # Trailing stop management
        self.manage_trailing_stop()

        # Circuit breakers
        can_trade, reason = self.risk.check_can_trade(balance)
        if not can_trade:
            if "target reached" in reason or "limit" in reason.lower():
                self.logger.info(f"No trade: {reason}")
            else:
                self.logger.warning(f"No trade: {reason}")
            return

        # Session filter
        if not self._is_trading_session():
            return

        # Spread filter
        spread = self.mt5.get_spread_points()
        if spread > self.config.max_spread_points:
            self.logger.debug(f"Spread {spread:.0f}pts > {self.config.max_spread_points}, skip")
            return

        # Max 1 posisi - cek posisi DAN cooldown 60 detik setelah order
        if len(self.mt5.get_positions()) >= self.config.max_open_positions:
            return

        # Cooldown 60 detik setelah order terakhir
        # (MT5 butuh beberapa detik register posisi, perpanjang dari 30 → 60)
        if self._last_order_time is not None:
            elapsed = (datetime.now() - self._last_order_time).total_seconds()
            if elapsed < 60:
                return

        # Fetch & analisa
        df = self.mt5.get_rates(count=150)
        if df is None or len(df) < self.config.breakout_period + 10:
            self.logger.warning("Insufficient data")
            return

        ind = self.analyze(df)

        # NaN check
        if any(pd.isna(v) for k, v in ind.items() if k != 'time'):
            return

        signal = self.get_signal(ind)
        if signal:
            self.execute_trade(signal, ind)

    def run(self):
        self.logger.info("=" * 65)
        self.logger.info("GOLDBOT STARTING - XAUUSD.m Breakout Scalping")
        self.logger.info(f"Lot: {self.config.lot_size} | SL: {self.config.sl_points}pts | TP: {self.config.tp_points}pts")
        self.logger.info(f"Daily limit: -${self.config.max_daily_loss} | Target: +${self.config.daily_profit_target}")
        self.logger.info("=" * 65)

        if not self.mt5.connect():
            self.logger.error("MT5 connection failed. Exiting.")
            return

        try:
            while self.running:
                try:
                    self.cycle()
                except Exception as e:
                    self.logger.exception(f"Cycle error: {e}")

                for _ in range(self.config.check_interval_seconds):
                    if not self.running:
                        break
                    time.sleep(1)
        finally:
            self.mt5.disconnect()
            self.logger.info("GoldBot stopped.")


if __name__ == "__main__":
    bot = GoldBot(CONFIG)
    bot.run()
