# =============================================================================
# collect_data.py - Historical Data Collection from MetaTrader 5
# Run this script ONCE before training the model.
#
# Usage: python collect_data.py
# =============================================================================

import MetaTrader5 as mt5
import pandas as pd
import os
from config import (
    MT5_LOGIN, MT5_PASSWORD, MT5_SERVER,
    SYMBOL, DATA_M1_PATH, DATA_M15_PATH,
    DATA_LOOKBACK_DAYS
)


# =============================================================================
# CONNECTION
# =============================================================================

def connect_mt5():
    """Initialize and connect to MetaTrader 5."""
    print("Connecting to MetaTrader 5...")

    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialization failed. Error: {mt5.last_error()}")

    if MT5_LOGIN and MT5_PASSWORD and MT5_SERVER:
        authorized = mt5.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)
        if not authorized:
            mt5.shutdown()
            raise RuntimeError(f"MT5 login failed. Error: {mt5.last_error()}")
        print(f"Logged in to account #{MT5_LOGIN} on server {MT5_SERVER}")

    symbol_info = mt5.symbol_info(SYMBOL)
    if symbol_info is None:
        mt5.shutdown()
        raise RuntimeError(f"Symbol {SYMBOL} not found in Market Watch.")

    if not symbol_info.visible:
        mt5.symbol_select(SYMBOL, True)

    print(f"MT5 connected. Symbol: {SYMBOL}")
    return True


# =============================================================================
# DATA COLLECTION
# =============================================================================

def collect_bars_safe(timeframe, timeframe_name, days=365):
    """
    Collect ALL available historical bars from broker.
    Uses largest possible batch size for MT5 (max 99999 bars per request).

    Args:
        timeframe:      MT5 timeframe constant
        timeframe_name: Human-readable name (e.g. "M1")
        days:           Max days to keep (will keep all if broker has less)

    Returns:
        pd.DataFrame with OHLCV data
    """
    print(f"\nCollecting {timeframe_name} data...")

    # MT5 hard limit is 99999 bars per request
    # M1 52 days  = ~62,400 bars -> need large batch
    # M15 52 days = ~4,160 bars  -> smaller batch fine
    if timeframe == mt5.TIMEFRAME_M1:
        batch_sizes = [99000, 70000, 50000, 30000, 10000]
    else:
        batch_sizes = [20000, 10000, 5000, 3000, 1000]

    rates = None

    for batch in batch_sizes:
        print(f"  Trying batch size: {batch:,} bars...")
        result = mt5.copy_rates_from_pos(SYMBOL, timeframe, 0, batch)
        if result is not None and len(result) > 0:
            rates = result
            print(f"  Success with batch size: {batch:,}")
            break
        else:
            print(f"  Failed (error: {mt5.last_error()}), trying smaller batch...")

    if rates is None:
        raise RuntimeError(
            f"Failed to collect {timeframe_name} data with any batch size.\n"
            f"Last error: {mt5.last_error()}\n"
            f"Pastikan:\n"
            f"  1. MT5 terbuka dan sudah login\n"
            f"  2. Chart XAUUSD M1 dan M15 sudah dibuka di MT5\n"
            f"  3. Scroll chart ke kiri sejauh mungkin untuk load history\n"
            f"  4. Tunggu MT5 selesai load (loading bar di bawah hilang)"
        )

    # Convert to DataFrame
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = df[['time', 'open', 'high', 'low', 'close', 'tick_volume']]
    df.set_index('time', inplace=True)
    df.sort_index(inplace=True)

    print(f"  Total bars fetched  : {len(df):,}")
    print(f"  Full date range     : {df.index[0]} to {df.index[-1]}")

    # Filter by days — tapi kalau data lebih sedikit dari filter, ambil semua
    cutoff = df.index[-1] - pd.Timedelta(days=days)
    df_filtered = df[df.index >= cutoff]

    if len(df_filtered) < 500:
        print(f"  WARNING: Filter {days} hari terlalu ketat, mengambil semua data yang tersedia...")
        df_filtered = df.copy()

    print(f"  After {days}-day filter : {len(df_filtered):,} bars")
    print(f"  Final date range    : {df_filtered.index[0]} to {df_filtered.index[-1]}")

    # Info kelengkapan data
    if timeframe == mt5.TIMEFRAME_M1:
        actual_days = (df_filtered.index[-1] - df_filtered.index[0]).days
        bars_per_day = len(df_filtered) / max(actual_days, 1)
        print(f"  Rata-rata bars/hari : {bars_per_day:.0f} (normal: ~1.200-1.440)")

    return df_filtered


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main function to collect and save historical data."""
    print("=" * 60)
    print("  AURUMBOT - DATA COLLECTION")
    print("=" * 60)

    os.makedirs("data", exist_ok=True)

    try:
        connect_mt5()

        # Collect M1 data (ambil semaksimal mungkin)
        df_m1 = collect_bars_safe(
            mt5.TIMEFRAME_M1, "M1", days=DATA_LOOKBACK_DAYS
        )

        # Collect M15 data
        df_m15 = collect_bars_safe(
            mt5.TIMEFRAME_M15, "M15", days=DATA_LOOKBACK_DAYS
        )

        # Validasi minimal data
        if len(df_m1) < 5000:
            print("\n[!] WARNING: Data M1 sangat sedikit (<5.000 bars).")
            print("    Training mungkin tidak optimal.")
            print("    Coba scroll chart MT5 ke kiri lebih jauh lalu jalankan ulang.")

        if len(df_m15) < 500:
            print("\n[!] WARNING: Data M15 sangat sedikit (<500 bars).")

        # Save to CSV
        print(f"\nSaving data...")
        df_m1.to_csv(DATA_M1_PATH)
        df_m15.to_csv(DATA_M15_PATH)

        print(f"\n[OK] M1  saved : {DATA_M1_PATH}  ({len(df_m1):,} rows)")
        print(f"[OK] M15 saved : {DATA_M15_PATH} ({len(df_m15):,} rows)")

        print("\n" + "=" * 60)
        print("  Data collection COMPLETE!")
        print("  Next step: python train_model.py")
        print("=" * 60)

    except RuntimeError as e:
        print(f"\n[ERROR] {e}")

    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        mt5.shutdown()
        print("\nMT5 connection closed.")


if __name__ == "__main__":
    main()