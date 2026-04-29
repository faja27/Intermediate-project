# =============================================================================
# convert_dukascopy.py - Convert Dukascopy CSV to AurumBot format
# Run this ONCE before train_model.py
#
# Usage: python convert_dukascopy.py
# =============================================================================

import pandas as pd
import os


# =============================================================================
# SETTINGS
# =============================================================================

INPUT_M1   = "data/XAUUSD_M1.csv"
INPUT_M15  = "data/XAUUSD_M15.csv"
OUTPUT_M1  = "data/raw_m1.csv"
OUTPUT_M15 = "data/raw_m15.csv"


# =============================================================================
# CONVERSION
# =============================================================================

def convert_dukascopy(input_path, output_path, timeframe_name):
    """
    Convert Dukascopy CSV to AurumBot format.

    Dukascopy quirk: header has 6 names but data rows have 7 fields (tab-separated).
    Header : Time  Open  High  Low  Close  Volume
    Data   : Time  Open  High  Low  Close  1  Volume

    We force 7 column names and drop the extra column.
    """
    print(f"\nConverting {timeframe_name} data...")
    print(f"  Input  : {input_path}")
    print(f"  Output : {output_path}")

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"File not found: {input_path}\n"
            f"Make sure XAUUSD_{timeframe_name}.csv is in the data/ folder."
        )

    # Force read with 7 column names, skip the original header row
    # This bypasses the header/data mismatch issue
    df = pd.read_csv(
        input_path,
        sep='\t',
        skiprows=1,          # skip original header
        header=None,         # no header in data
        names=['time', 'open', 'high', 'low', 'close', '_extra', 'tick_volume'],
        engine='python',     # more flexible parser
        on_bad_lines='skip'  # skip any malformed lines
    )

    print(f"  Raw rows read  : {len(df):,}")

    # Drop extra column
    df = df.drop(columns=['_extra'], errors='ignore')

    # Parse datetime
    df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    df = df.dropna(subset=['time'])
    df.set_index('time', inplace=True)
    df.sort_index(inplace=True)

    # Ensure numeric columns
    for col in ['open', 'high', 'low', 'close', 'tick_volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop any remaining NaN
    before = len(df)
    df.dropna(inplace=True)
    after = len(df)
    if before != after:
        print(f"  Dropped {before - after} invalid rows")

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
    print("  AURUMBOT - DUKASCOPY DATA CONVERTER")
    print("=" * 60)

    try:
        df_m1  = convert_dukascopy(INPUT_M1,  OUTPUT_M1,  "M1")
        df_m15 = convert_dukascopy(INPUT_M15, OUTPUT_M15, "M15")

        print("\n" + "=" * 60)
        print("  CONVERSION COMPLETE!")
        print("=" * 60)
        print(f"\n  M1  : {len(df_m1):,} rows")
        print(f"        {df_m1.index[0]} to {df_m1.index[-1]}")
        print(f"\n  M15 : {len(df_m15):,} rows")
        print(f"        {df_m15.index[0]} to {df_m15.index[-1]}")
        print(f"\nNext step: python train_model.py")

    except FileNotFoundError as e:
        print(f"\nERROR: {e}")

    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
