"""
GoldBot Config - FINAL OPTIMIZED
Berdasarkan optimizer 81 kombinasi x 6 tahun data (2020-2025)
Rank #4: bp=15 | buf=20 | dt=15 | ta=80
  4.5 trades/day | WR=72.1% | PF=2.71 | Weekly=$+20.15 | DD=4.1%

Dipilih Rank #4 bukan #1 karena DD 4.1% vs 5.1% (lebih aman untuk $100)
"""

from dataclasses import dataclass
import MetaTrader5 as mt5


@dataclass
class GoldConfig:
    # ========================================================================
    # MT5 CONNECTION - MIFX
    # ========================================================================
    mt5_login: int = ""
    mt5_password: str = ""           # isi password demo MIFX
    mt5_server: str = ""
    mt5_path: str = r"C:\Program Files\Monex Trader\terminal64.exe"

    # ========================================================================
    # TRADING PARAMETERS
    # ========================================================================
    symbol: str = ""
    timeframe: int = mt5.TIMEFRAME_M5
    magic_number: int = ""

    # ========================================================================
    # STRATEGY - OPTIMIZED (Rank #4 dari 81 kombinasi)
    # ========================================================================
    breakout_period: int = 15        # optimal: 15 (DD lebih rendah vs bp=10)
    breakout_buffer: float = 20.0    # 20pts buffer = lebih sensitif
    ema_trend: int = 50
    rsi_period: int = 14
    rsi_min_buy: float = 45.0
    rsi_max_sell: float = 55.0
    atr_period: int = 14
    atr_min_points: float = 200.0
    atr_max_points: float = 3500.0

    # Trailing primary (no fixed TP)
    sl_points: float = 200.0
    tp_points: float = 99999.0       # BUY: disabled
                                     # SELL: bot.py pakai 0.01
    use_trailing: bool = True
    trail_activate_points: float = 80.0   # OPTIMIZED: 120 -> 80
    trail_distance_points: float = 80.0

    # ========================================================================
    # SESSION FILTER - London + NY
    # ========================================================================
    trade_hour_start: int = 13       # 13:00 WIB
    trade_hour_end: int = 3          # 03:00 WIB

    # ========================================================================
    # RISK MANAGEMENT
    # ========================================================================
    lot_size: float = 0.01
    commission_per_lot: float = 5.0

    max_daily_loss: float = 6.00     # 3x SL = circuit breaker TETAP AKTIF
    daily_profit_target: float = 15.0   # Auto-stop saat profit +$15/hari
    max_total_loss: float = 20.00    # 20% modal = halt permanen
    max_daily_trades: int = 999      # NONAKTIF
    max_open_positions: int = 1      # MAX 1 posisi sekaligus
    max_spread_points: float = 150.0

    close_before_swap: bool = True   # close jam 23:00 WIB (hindari swap -4.82%)
    close_hour_wib: int = 23

    # ========================================================================
    # OPERATIONAL
    # ========================================================================
    check_interval_seconds: int = 10
    log_file: str = "logs/goldbot.log"
    state_file: str = "goldbot_state.json"
    journal_file: str = "journal/GoldBot_Journal.xlsx"


CONFIG = GoldConfig()
