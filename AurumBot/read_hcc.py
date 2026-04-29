# =============================================================================
# read_hcc.py - Baca file MT5 history cache (.hc) dan convert ke CSV
#
# Temuan format (dari reverse engineering file 2015.hcc):
#   - Bar size    : 60 bytes (MqlRates standard)
#   - Header size : VARIABLE per file (di-detect otomatis)
#   - Urutan data : bisa acak, akan di-sort ulang
#   - Bar format  : int64_ts + double*4(OHLC) + int64_vol + int32_spread + int64_realvol
#
# Sumber data:
#   M1  -> cache/M1.hc
#   M15 -> cache/M15.hc
#
# Cara pakai:
#   1. Pastikan MT5 sudah DITUTUP
#   2. Jalankan: python read_hcc.py
#   3. Output: data/raw_m1.csv dan data/raw_m15.csv
# =============================================================================

import struct
import pandas as pd
import os
from datetime import datetime

# =============================================================================
# KONFIGURASI PATH
# =============================================================================

CACHE_DIR = (
    r"C:\Users\fmoch\AppData\Roaming\MetaQuotes\Terminal"
    r"\C84535A6B43B3F94C032314C1C9A9F5B\bases"
    r"\FinexBisnisSolusi-Demo\history\XAUUSD\cache"
)

OUTPUT_M1  = "data/raw_m1.csv"
OUTPUT_M15 = "data/raw_m15.csv"

# Filter: ambil data mulai tahun ini
YEAR_FROM = 2016

# =============================================================================
# FORMAT BAR (CONFIRMED via reverse engineering)
# =============================================================================

BAR_FORMAT = '<qddddqiq'
BAR_SIZE   = struct.calcsize(BAR_FORMAT)  # = 60 bytes

TS_MIN    = int(datetime(2010, 1, 1).timestamp())
TS_MAX    = int(datetime(2040, 1, 1).timestamp())
PRICE_MIN = 200.0
PRICE_MAX = 15000.0


# =============================================================================
# AUTO-DETECT HEADER SIZE
# =============================================================================

def detect_header_size(raw):
    """
    Scan byte per byte untuk menemukan offset pertama bar valid.
    Header MT5 .hc bersifat variable size — tidak bisa diasumsi fixed.
    """
    print(f"  Detecting header size (scanning up to 50,000 bytes)...")

    for pos in range(0, min(len(raw) - BAR_SIZE * 2, 50000)):
        try:
            ts = struct.unpack_from('<q', raw, pos)[0]
            if not (TS_MIN <= ts <= TS_MAX):
                continue

            o = struct.unpack_from('<d', raw, pos + 8)[0]
            h = struct.unpack_from('<d', raw, pos + 16)[0]
            l = struct.unpack_from('<d', raw, pos + 24)[0]
            c = struct.unpack_from('<d', raw, pos + 32)[0]

            if not all(PRICE_MIN < x < PRICE_MAX for x in [o, h, l, c]):
                continue
            if h < max(o, c) or l > min(o, c):
                continue

            # Konfirmasi dengan cek bar ke-2
            pos2 = pos + BAR_SIZE
            if pos2 + BAR_SIZE <= len(raw):
                ts2 = struct.unpack_from('<q', raw, pos2)[0]
                o2  = struct.unpack_from('<d', raw, pos2 + 8)[0]
                h2  = struct.unpack_from('<d', raw, pos2 + 16)[0]
                if (TS_MIN <= ts2 <= TS_MAX and
                        PRICE_MIN < o2 < PRICE_MAX and
                        PRICE_MIN < h2 < PRICE_MAX):
                    return pos

        except Exception:
            continue

    return None


# =============================================================================
# BACA SATU FILE .hc
# =============================================================================

def read_mt5_cache_file(filepath):
    """
    Baca file MT5 cache (.hc) dan return DataFrame OHLCV.
    Auto-detect header, validasi semua bar.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File tidak ditemukan: {filepath}")

    file_size = os.path.getsize(filepath)
    print(f"  File : {os.path.basename(filepath)}")
    print(f"  Size : {file_size / 1024 / 1024:.1f} MB  ({file_size:,} bytes)")

    with open(filepath, 'rb') as f:
        raw = f.read()

    # Detect header
    header_size = detect_header_size(raw)
    if header_size is None:
        raise RuntimeError(
            f"Tidak bisa menemukan bar data di {os.path.basename(filepath)}.\n"
            f"Pastikan:\n"
            f"  1. MT5 sudah ditutup\n"
            f"  2. Chart XAUUSD pernah dibuka dan di-scroll\n"
            f"  3. File tidak corrupt"
        )
    print(f"  Header size : {header_size} bytes")
    print(f"  Est. bars   : {(file_size - header_size) // BAR_SIZE:,}")

    # Parse semua bar
    records = []
    skipped = 0
    pos     = header_size

    while pos + BAR_SIZE <= len(raw):
        try:
            ts, o, h, l, c, tv, spread, rv = struct.unpack_from(BAR_FORMAT, raw, pos)

            ts_ok    = TS_MIN <= ts <= TS_MAX
            price_ok = all(PRICE_MIN < x < PRICE_MAX for x in [o, h, l, c])
            ohlc_ok  = h >= max(o, c) and l <= min(o, c)

            if ts_ok and price_ok and ohlc_ok:
                records.append((ts, o, h, l, c, int(tv)))
            else:
                skipped += 1

        except Exception:
            skipped += 1

        pos += BAR_SIZE

    print(f"  Bars valid  : {len(records):,}")
    if skipped > 0:
        print(f"  Bars skipped: {skipped:,}")

    if len(records) == 0:
        raise RuntimeError(f"Tidak ada bar valid di {os.path.basename(filepath)}")

    # Convert ke DataFrame
    df = pd.DataFrame(
        records,
        columns=['time', 'open', 'high', 'low', 'close', 'tick_volume']
    )
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df.sort_index(inplace=True)

    # Hapus duplikat (keep last = bar lebih baru)
    before = len(df)
    df = df[~df.index.duplicated(keep='last')]
    if len(df) < before:
        print(f"  Duplikat dihapus: {before - len(df):,}")

    return df


# =============================================================================
# FILTER & REPORT
# =============================================================================

def filter_and_report(df, label, year_from, tf_minutes):
    """Filter data mulai year_from dan cetak statistik kualitas."""

    cutoff = pd.Timestamp(f"{year_from}-01-01")
    df     = df[df.index >= cutoff].copy()

    if len(df) == 0:
        print(f"  [!] Tidak ada data setelah {year_from}")
        return df

    actual_days  = (df.index[-1] - df.index[0]).days
    bars_per_day = len(df) / max(actual_days, 1)

    # Deteksi timeframe aktual dari median gap
    gaps = df.index.to_series().diff().dropna()
    median_gap_min = gaps.median().total_seconds() / 60
    tf_detected = f"~{median_gap_min:.0f} menit"

    print(f"  After filter (>= {year_from})")
    print(f"    Bars      : {len(df):,}")
    print(f"    Range     : {df.index[0]} to {df.index[-1]}")
    print(f"    Bars/hari : {bars_per_day:.0f}")
    print(f"    TF median : {tf_detected}")

    expected_per_day = (23 * 60) / tf_minutes
    if bars_per_day < expected_per_day * 0.4:
        print(f"  [!] Bars/hari jauh lebih rendah dari ekspektasi ({expected_per_day:.0f})")
        print(f"      Kemungkinan ada gap data atau timeframe berbeda")
    else:
        print(f"    Kualitas  : OK")

    return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 55)
    print("  AURUMBOT - READ MT5 CACHE FILES (.hc)")
    print("=" * 55)
    print(f"  Cache dir : ...\\XAUUSD\\cache")
    print(f"  Filter    : >= {YEAR_FROM}")
    print()

    if not os.path.exists(CACHE_DIR):
        print(f"[ERROR] Folder cache tidak ditemukan:")
        print(f"  {CACHE_DIR}")
        print()
        print("Pastikan path CACHE_DIR di script sudah benar.")
        return

    os.makedirs("data", exist_ok=True)

    df_m1  = None
    df_m15 = None

    # -------------------------------------------------------------------------
    # M1
    # -------------------------------------------------------------------------
    print(f"{'='*40}")
    print("  Processing M1 (cache/M1.hc)")
    print(f"{'='*40}")
    try:
        df_m1 = read_mt5_cache_file(os.path.join(CACHE_DIR, "M1.hc"))
        df_m1 = filter_and_report(df_m1, "M1", YEAR_FROM, tf_minutes=1)

        if len(df_m1) > 0:
            df_m1.to_csv(OUTPUT_M1)
            print(f"\n  [OK] Saved: {OUTPUT_M1}  ({len(df_m1):,} rows)")
        else:
            print("  [SKIP] Tidak ada data untuk disimpan")

    except Exception as e:
        print(f"\n  [ERROR] {e}")
        import traceback
        traceback.print_exc()
        return

    # -------------------------------------------------------------------------
    # M15
    # -------------------------------------------------------------------------
    print(f"\n{'='*40}")
    print("  Processing M15 (cache/M15.hc)")
    print(f"{'='*40}")
    try:
        df_m15 = read_mt5_cache_file(os.path.join(CACHE_DIR, "M15.hc"))
        df_m15 = filter_and_report(df_m15, "M15", YEAR_FROM, tf_minutes=15)

        if len(df_m15) > 0:
            df_m15.to_csv(OUTPUT_M15)
            print(f"\n  [OK] Saved: {OUTPUT_M15}  ({len(df_m15):,} rows)")
        else:
            print("  [SKIP] Tidak ada data untuk disimpan")

    except Exception as e:
        print(f"\n  [ERROR] {e}")
        import traceback
        traceback.print_exc()
        return

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    m1_count  = len(df_m1)  if df_m1  is not None else 0
    m15_count = len(df_m15) if df_m15 is not None else 0

    print(f"\n{'='*55}")
    print(f"  SELESAI!")
    print(f"  M1  : {m1_count:,} bars -> {OUTPUT_M1}")
    print(f"  M15 : {m15_count:,} bars -> {OUTPUT_M15}")
    print(f"\n  Next step: python train_model.py")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
