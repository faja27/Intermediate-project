# =============================================================================
# debug_hcc.py - Investigasi format binary file .hcc MT5
#
# Cara pakai: python debug_hcc.py
# =============================================================================

import struct
import os

HCC_FILE = (
    r"C:\Users\fmoch\AppData\Roaming\MetaQuotes\Terminal"
    r"\C84535A6B43B3F94C032314C1C9A9F5B\bases"
    r"\FinexBisnisSolusi-Demo\history\XAUUSD\2020.hcc"
)

BAR_FORMATS = {
    # format_string : (bar_size, label)
    '<qddddqiq' : (60, 'MT5 MqlRates 60-byte'),
    '<IddddI'   : (44, 'MT4 style 44-byte'),
    '<iddddii'  : (40, 'compact 40-byte'),
}

def try_read_bars(raw, offset, fmt, bar_size, n=5):
    """Coba baca n bar dari offset tertentu, return list hasil."""
    results = []
    pos = offset
    for i in range(n):
        if pos + bar_size > len(raw):
            break
        try:
            bar = struct.unpack_from(fmt, raw, pos)
            results.append(bar)
        except Exception as e:
            results.append(None)
        pos += bar_size
    return results

def looks_like_xauusd(bar, fmt):
    """Cek apakah bar terlihat seperti data XAUUSD yang valid."""
    try:
        if fmt == '<qddddqiq':
            ts, o, h, l, c = bar[0], bar[1], bar[2], bar[3], bar[4]
        elif fmt == '<Iddddii':
            ts, o, h, l, c = bar[0], bar[1], bar[2], bar[3], bar[4]
        else:
            return False, "unknown fmt"

        # Cek timestamp (2010–2030)
        ts_ok = 1262304000 < ts < 1893456000

        # Cek harga (XAUUSD range semua era: 200–15000)
        price_ok = all(200 < x < 15000 for x in [o, h, l, c])

        # Cek logika OHLC
        ohlc_ok = h >= max(o, c) and l <= min(o, c) and h >= l

        from datetime import datetime
        ts_str = datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d') if ts_ok else f"BAD_TS({ts})"

        return ts_ok and price_ok and ohlc_ok, f"ts={ts_str} O={o:.2f} H={h:.2f} L={l:.2f} C={c:.2f}"
    except:
        return False, "parse error"

def main():
    print("=" * 60)
    print("  DEBUG - Investigasi Format .hcc MT5")
    print("=" * 60)

    if not os.path.exists(HCC_FILE):
        print(f"[ERROR] File tidak ditemukan: {HCC_FILE}")
        return

    with open(HCC_FILE, 'rb') as f:
        raw = f.read()

    print(f"\nFile   : {os.path.basename(HCC_FILE)}")
    print(f"Size   : {len(raw):,} bytes ({len(raw)/1024:.1f} KB)")

    # -------------------------------------------------------------------------
    # 1. Print hex dump dari 256 byte pertama (untuk lihat header)
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  HEX DUMP - 256 bytes pertama (header)")
    print(f"{'='*60}")
    for i in range(0, min(256, len(raw)), 16):
        hex_part = ' '.join(f'{b:02x}' for b in raw[i:i+16])
        asc_part = ''.join(chr(b) if 32 <= b < 127 else '.' for b in raw[i:i+16])
        print(f"  {i:04x}: {hex_part:<48}  {asc_part}")

    # -------------------------------------------------------------------------
    # 2. Coba berbagai offset dan format
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  SCANNING - Cari offset + format yang valid")
    print(f"{'='*60}")

    offsets_to_try = [0, 4, 8, 12, 16, 24, 32, 48, 64, 72, 80, 96, 112, 128, 148, 160, 192, 256]

    best_offset = None
    best_fmt    = None
    best_label  = None
    best_score  = 0

    for offset in offsets_to_try:
        for fmt, (bar_size, label) in BAR_FORMATS.items():
            bars = try_read_bars(raw, offset, fmt, bar_size, n=10)
            score = 0
            details = []
            for bar in bars:
                if bar is None:
                    break
                ok, desc = looks_like_xauusd(bar, fmt)
                if ok:
                    score += 1
                    details.append(desc)

            if score >= 3:
                print(f"\n  [HIT] offset={offset}, fmt={label}, score={score}/10")
                for d in details[:3]:
                    print(f"        {d}")

                if score > best_score:
                    best_score  = score
                    best_offset = offset
                    best_fmt    = fmt
                    best_label  = label

    # -------------------------------------------------------------------------
    # 3. Kalau tidak ketemu, coba brute force setiap byte
    # -------------------------------------------------------------------------
    if best_offset is None:
        print(f"\n  Tidak ketemu dengan offset standar, brute force...")
        for offset in range(0, min(512, len(raw))):
            for fmt, (bar_size, label) in BAR_FORMATS.items():
                bars = try_read_bars(raw, offset, fmt, bar_size, n=5)
                score = 0
                for bar in bars:
                    if bar is None:
                        break
                    ok, _ = looks_like_xauusd(bar, fmt)
                    if ok:
                        score += 1
                if score >= 3:
                    print(f"  [HIT] offset={offset}, fmt={label}, score={score}/5")
                    if score > best_score:
                        best_score  = score
                        best_offset = offset
                        best_fmt    = fmt
                        best_label  = label

    # -------------------------------------------------------------------------
    # 4. Hasil
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    if best_offset is not None:
        print(f"  HASIL: offset={best_offset}, format={best_label}")
        print(f"  Score : {best_score}/10 bars valid")

        # Estimasi total bars
        fmt_size = BAR_FORMATS[best_fmt][0]
        total_bars = (len(raw) - best_offset) // fmt_size
        print(f"  Estimasi total bars di file ini: {total_bars:,}")
        print(f"\n  => Salin nilai ini ke chat!")
        print(f"     HEADER_SIZE = {best_offset}")
        print(f"     BAR_FORMAT  = '{best_fmt}'")
        print(f"     BAR_SIZE    = {fmt_size}")
    else:
        print(f"  TIDAK KETEMU format yang cocok!")
        print(f"  => Share output ini ke chat untuk analisis lebih lanjut.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
