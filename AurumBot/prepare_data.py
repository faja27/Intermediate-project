# =============================================================================
# prepare_data.py - Konversi Kaggle dan merge dengan data Finex
#
# Hasil akhir:
#   data/raw_m1.csv  = Kaggle FILTER_FROM - 2026-01-07 + Finex 2026-01-08+
#   data/raw_m15.csv = Kaggle FILTER_FROM - 2025-06-13 + Finex 2025-06-13+
#
# Cara ganti filter:
#   Ubah FILTER_FROM di bawah (2023-01-01, 2022-01-01, dst)
#   JANGAN ubah FINEX_M1_START dan FINEX_M15_START
#
# Cara pakai:
#   python prepare_data.py
# =============================================================================

import pandas as pd
import os

# =============================================================================
# KONFIGURASI
# =============================================================================

INPUT_M1  = "data/XAU_1m_data.csv"
INPUT_M15 = "data/XAU_15m_data.csv"
FINEX_M1  = "data/raw_m1.csv"
FINEX_M15 = "data/raw_m15.csv"
OUTPUT_M1  = "data/raw_m1.csv"
OUTPUT_M15 = "data/raw_m15.csv"

# ← UBAH INI untuk ganti rentang data training
# 2023-01-01 = ~700k bars M1 (harga 1800-5000, lebih relevan)
# 2022-01-01 = ~1.1jt bars M1
# 2020-01-01 = ~2jt bars M1 (paling banyak tapi banyak data harga lama)
FILTER_FROM = "2023-01-01"

# ← JANGAN UBAH INI — hardcoded tanggal bar pertama Finex asli
# Diperlukan agar cutoff tidak ikut berubah kalau raw file sudah merged
FINEX_M1_START  = "2026-01-08 03:40:00"
FINEX_M15_START = "2025-06-13 21:15:00"

MAX_ROWS_M1  = 2_000_000
MAX_ROWS_M15 = 200_000
CHUNK_SIZE   = 100_000


# =============================================================================
# STEP 1: LOAD FINEX DATA
# =============================================================================

def load_finex(filepath, label, start_date):
    """
    Load Finex data dan filter hanya dari start_date ke atas.
    start_date di-hardcode agar tidak terpengaruh kalau file sudah merged.
    """
    print(f"  Loading Finex {label}: {filepath}")
    if not os.path.exists(filepath):
        print(f"  [!] File tidak ditemukan: {filepath}")
        return None

    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    df.index.name = 'time'

    # Filter hanya bar Finex asli (bukan bagian Kaggle yang sudah merged)
    df = df[df.index >= start_date]

    if len(df) == 0:
        print(f"  [!] Tidak ada data Finex {label} setelah {start_date}")
        return None

    print(f"  Finex {label}: {len(df):,} bars | {df.index[0]} to {df.index[-1]}")
    return df


# =============================================================================
# STEP 2: KONVERSI KAGGLE (chunked)
# =============================================================================

def convert_kaggle(input_path, timeframe_name, max_rows, cutoff_before):
    """
    Baca Kaggle CSV dengan chunked reading.
    Ambil data >= FILTER_FROM dan < cutoff_before (hindari overlap Finex).
    """
    print(f"\n  Converting Kaggle {timeframe_name}...")
    print(f"  Input  : {input_path}")
    print(f"  Filter : >= {FILTER_FROM}")
    print(f"  Cutoff : < {cutoff_before} (hindari overlap Finex)")

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"File Kaggle tidak ditemukan: {input_path}\n"
            f"Pastikan file ada di folder data/"
        )

    chunks_kept = []
    total_read  = 0
    total_kept  = 0

    reader = pd.read_csv(
        input_path,
        sep=';',
        header=0,
        names=['time', 'open', 'high', 'low', 'close', 'tick_volume'],
        engine='python',
        on_bad_lines='skip',
        chunksize=CHUNK_SIZE
    )

    for i, chunk in enumerate(reader):
        total_read += len(chunk)

        # Parse datetime
        chunk['time'] = pd.to_datetime(
            chunk['time'], format='%Y.%m.%d %H:%M', errors='coerce'
        )
        chunk = chunk.dropna(subset=['time'])
        chunk.set_index('time', inplace=True)

        # Filter >= FILTER_FROM
        chunk = chunk[chunk.index >= FILTER_FROM]

        # Potong < cutoff_before
        if len(chunk) > 0:
            chunk = chunk[chunk.index < cutoff_before]

        if len(chunk) > 0:
            for col in ['open', 'high', 'low', 'close', 'tick_volume']:
                chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
            chunk.dropna(inplace=True)
            chunks_kept.append(chunk)
            total_kept += len(chunk)

        if (i + 1) % 20 == 0:
            print(f"  Progress: {total_read:,} rows read | {total_kept:,} rows kept...")

        if total_kept >= max_rows:
            print(f"  Reached {max_rows:,} rows limit — stopping.")
            break

    if not chunks_kept:
        raise ValueError(
            f"Tidak ada data Kaggle {timeframe_name} antara {FILTER_FROM} "
            f"dan {cutoff_before}.\n"
            f"Cek file: {input_path}"
        )

    df = pd.concat(chunks_kept)
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep='last')]

    print(f"  Kaggle {timeframe_name}: {len(df):,} bars | "
          f"{df.index[0]} to {df.index[-1]}")
    return df


# =============================================================================
# STEP 3: MERGE + SAVE
# =============================================================================

def merge_and_save(df_kaggle, df_finex, output_path, label):
    """Gabungkan Kaggle dan Finex, hapus duplikat, sort, simpan."""

    if df_finex is not None and len(df_finex) > 0:
        print(f"\n  Merging {label}...")
        df_merged = pd.concat([df_kaggle, df_finex])
        df_merged.sort_index(inplace=True)
        df_merged = df_merged[~df_merged.index.duplicated(keep='last')]
        df_merged = df_merged[['open', 'high', 'low', 'close', 'tick_volume']]

        print(f"  Kaggle : {len(df_kaggle):,} bars")
        print(f"  Finex  : {len(df_finex):,} bars")
        print(f"  Merged : {len(df_merged):,} bars")
        print(f"  Range  : {df_merged.index[0]} to {df_merged.index[-1]}")

        # Cek gap besar
        gaps     = df_merged.index.to_series().diff().dropna()
        big_gaps = gaps[gaps > pd.Timedelta(days=5)]
        if len(big_gaps) > 0:
            print(f"  Gap > 5 hari: {len(big_gaps)} kali (wajar untuk weekend/holiday)")
            for idx, gap in big_gaps.head(3).items():
                print(f"    {idx}: gap {gap.days} hari")

        df_merged.to_csv(output_path)
        print(f"  [OK] Saved: {output_path} ({len(df_merged):,} rows)")
        return df_merged

    else:
        print(f"\n  Finex {label} tidak tersedia, simpan Kaggle saja...")
        df_kaggle = df_kaggle[['open', 'high', 'low', 'close', 'tick_volume']]
        df_kaggle.to_csv(output_path)
        print(f"  [OK] Saved: {output_path} ({len(df_kaggle):,} rows)")
        return df_kaggle


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("  AURUMBOT - PREPARE DATA")
    print("=" * 60)
    print(f"  Kaggle filter  : >= {FILTER_FROM}")
    print(f"  Finex M1 dari  : {FINEX_M1_START}")
    print(f"  Finex M15 dari : {FINEX_M15_START}")
    print(f"  Max M1 rows    : {MAX_ROWS_M1:,}")
    print(f"  Max M15 rows   : {MAX_ROWS_M15:,}")

    os.makedirs("data", exist_ok=True)

    # -------------------------------------------------------------------------
    # STEP 1: Load Finex (filter by hardcoded start date)
    # -------------------------------------------------------------------------
    print(f"\n{'='*40}")
    print("  STEP 1: Load Finex data")
    print(f"{'='*40}")

    df_finex_m1  = load_finex(FINEX_M1,  "M1",  FINEX_M1_START)
    df_finex_m15 = load_finex(FINEX_M15, "M15", FINEX_M15_START)

    print(f"\n  Kaggle M1  akan dipotong sebelum : {FINEX_M1_START}")
    print(f"  Kaggle M15 akan dipotong sebelum : {FINEX_M15_START}")

    # -------------------------------------------------------------------------
    # STEP 2: Convert Kaggle
    # -------------------------------------------------------------------------
    print(f"\n{'='*40}")
    print("  STEP 2: Convert Kaggle CSV")
    print(f"{'='*40}")

    try:
        df_kaggle_m1 = convert_kaggle(
            INPUT_M1, "M1",
            max_rows=MAX_ROWS_M1,
            cutoff_before=FINEX_M1_START
        )
    except Exception as e:
        print(f"  [ERROR] Kaggle M1: {e}")
        return

    try:
        df_kaggle_m15 = convert_kaggle(
            INPUT_M15, "M15",
            max_rows=MAX_ROWS_M15,
            cutoff_before=FINEX_M15_START
        )
    except Exception as e:
        print(f"  [ERROR] Kaggle M15: {e}")
        return

    # -------------------------------------------------------------------------
    # STEP 3: Merge & Save
    # -------------------------------------------------------------------------
    print(f"\n{'='*40}")
    print("  STEP 3: Merge & Save")
    print(f"{'='*40}")

    df_m1_final  = merge_and_save(df_kaggle_m1,  df_finex_m1,  OUTPUT_M1,  "M1")
    df_m15_final = merge_and_save(df_kaggle_m15, df_finex_m15, OUTPUT_M15, "M15")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  SELESAI!")
    print(f"  M1  : {len(df_m1_final):,} bars")
    print(f"        {df_m1_final.index[0]} to {df_m1_final.index[-1]}")
    print(f"  M15 : {len(df_m15_final):,} bars")
    print(f"        {df_m15_final.index[0]} to {df_m15_final.index[-1]}")
    print(f"\n  Next step: python train_model.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
