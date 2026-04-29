# =============================================================================
# convert_kaggle.py - Convert Kaggle XAUUSD CSV to AurumBot format
# Run this ONCE before train_model.py
#
# Kaggle format (semicolon-separated):
# Date;Open;High;Low;Close;Volume
# 2004.06.11 07:18;384;384.1;384;384;3
#
# Usage: python convert_kaggle.py
# =============================================================================

import pandas as pd
import os


# =============================================================================
# SETTINGS
# =============================================================================

INPUT_M1   = "data/XAU_1m_data.csv"
INPUT_M15  = "data/XAU_15m_data.csv"
OUTPUT_M1  = "data/raw_m1.csv"
OUTPUT_M15 = "data/raw_m15.csv"

# Filter: only keep data from this date onwards
# 2024-01-01 = ~500,000 M1 rows (1 year) — fast and sufficient
FILTER_FROM = "2024-01-01"

# Chunk size for reading large files — avoids RAM overflow
CHUNK_SIZE = 100_000


# =============================================================================
# CONVERSION
# =============================================================================

def convert_kaggle(input_path, output_path, timeframe_name):
    """
    Convert Kaggle XAUUSD CSV to AurumBot format using chunked reading.
    Chunked reading prevents RAM overflow on large files (300MB+).

    Kaggle format:
    - Separator  : semicolon (;)
    - Date format: YYYY.MM.DD HH:MM
    - Columns    : Date, Open, High, Low, Close, Volume

    AurumBot format:
    - Separator  : comma (,)
    - Index      : time (datetime)
    - Columns    : open, high, low, close, tick_volume
    """
    print(f"\nConverting {timeframe_name} data...")
    print(f"  Input      : {input_path}")
    print(f"  Output     : {output_path}")
    print(f"  Filter from: {FILTER_FROM}")

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"File not found: {input_path}\n"
            f"Make sure the Kaggle CSV file is in the data/ folder."
        )

    chunks_kept = []
    total_read  = 0
    total_kept  = 0

    # Read in chunks to avoid RAM overflow
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

        # Filter to FILTER_FROM onwards
        chunk = chunk[chunk.index >= FILTER_FROM]

        if len(chunk) > 0:
            # Ensure numeric
            for col in ['open', 'high', 'low', 'close', 'tick_volume']:
                chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
            chunk.dropna(inplace=True)
            chunks_kept.append(chunk)
            total_kept += len(chunk)

        # Progress update every 10 chunks
        if (i + 1) % 10 == 0:
            print(f"  Progress: {total_read:,} rows read | {total_kept:,} rows kept...")

        # Stop early if we have enough data
        if total_kept >= 600_000:
            print(f"  Reached 600,000 rows limit — stopping early to save time.")
            break

    if not chunks_kept:
        raise ValueError(
            f"No data found after {FILTER_FROM}. "
            f"Try changing FILTER_FROM to an earlier date."
        )

    # Combine all chunks
    print(f"  Combining chunks...")
    df = pd.concat(chunks_kept)
    df.sort_index(inplace=True)

    # Save
    df.to_csv(output_path)

    print(f"  Rows converted : {len(df):,}")
    print(f"  Date range     : {df.index[0]} to {df.index[-1]}")
    print(f"  Saved to       : {output_path} ✅")

    return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("  AURUMBOT - KAGGLE DATA CONVERTER")
    print("=" * 60)
    print(f"  Filtering from : {FILTER_FROM}")
    print(f"  Max M1 rows    : 600,000 (~1 year)")

    try:
        df_m1  = convert_kaggle(INPUT_M1,  OUTPUT_M1,  "M1")
        df_m15 = convert_kaggle(INPUT_M15, OUTPUT_M15, "M15")

        print("\n" + "=" * 60)
        print("  CONVERSION COMPLETE!")
        print("=" * 60)
        print(f"\n  M1  : {len(df_m1):,} rows")
        print(f"        {df_m1.index[0]} to {df_m1.index[-1]}")
        print(f"\n  M15 : {len(df_m15):,} rows")
        print(f"        {df_m15.index[0]} to {df_m15.index[-1]}")
        print(f"\nNext step: python train_model.py")
        print("=" * 60)

    except FileNotFoundError as e:
        print(f"\nERROR: {e}")

    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
