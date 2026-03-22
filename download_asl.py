"""
download_asl.py
Downloads Google ASL Signs parquet files for target signs.

Usage:
    python download_asl.py
"""

import pandas as pd
import subprocess
import os
import sys
import zipfile
from pathlib import Path
from collections import defaultdict

# ── Config ────────────────────────────────────────
ASL_DIR      = Path("D:/signvoice/data/raw/asl-signs")
TARGET_SIGNS = [
    'drink', 'who', 'cow', 'bird', 'brown',
    'cat', 'kiss', 'go', 'think', 'man'
]
MAX_PER_SIGN = 80
# ─────────────────────────────────────────────────


def extract_csv():
    zip_path = ASL_DIR / "train.csv.zip"
    csv_path = ASL_DIR / "train.csv"
    if csv_path.exists():
        print("  train.csv already extracted.")
        return
    if zip_path.exists():
        print("  Extracting train.csv...")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(ASL_DIR)
        print("  Extracted.")
    else:
        print(f"ERROR: train.csv not found at {ASL_DIR}")
        sys.exit(1)


def download_file(path: str, out_dir: Path) -> bool:
    result = subprocess.run(
        [
            'kaggle', 'competitions', 'download',
            '-c', 'asl-signs',
            '-f', path,
            '-p', str(out_dir),
            '--quiet',
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return result.returncode == 0


def verify_parquet(parquet_path: Path) -> bool:
    """Check if parquet file is valid and has correct structure."""
    try:
        df = pd.read_parquet(parquet_path)
        return len(df) > 0
    except Exception:
        return False


def main():
    print("=" * 55)
    print("  SignVoice — Download Google ASL Signs")
    print("=" * 55)
    print(f"\n  Target signs : {TARGET_SIGNS}")
    print(f"  Max per sign : {MAX_PER_SIGN}")

    # Step 1 — Extract CSV
    print("\n[1/3] Checking train.csv...")
    extract_csv()

    # Step 2 — Load CSV
    df = pd.read_csv(ASL_DIR / "train.csv")
    print(f"  Total samples in CSV : {len(df)}")

    subset = df[df['sign'].isin(TARGET_SIGNS)]
    print(f"  Files for target signs : {len(subset)}")
    print()
    print("  Available per sign:")
    for sign in TARGET_SIGNS:
        count = len(subset[subset['sign'] == sign])
        print(f"    {sign:15s} : {count}")

    # Step 3 — Download
    print(f"\n[2/3] Downloading parquet files...")
    print(f"  This takes 20-40 minutes\n")

    sign_downloaded = defaultdict(int)
    sign_failed     = defaultdict(int)
    sign_skipped    = defaultdict(int)

    for sign in TARGET_SIGNS:
        sign_files = subset[subset['sign'] == sign]
        print(f"  Downloading '{sign}' "
              f"({len(sign_files)} available, "
              f"max {MAX_PER_SIGN})...")

        for _, row in sign_files.iterrows():
            if sign_downloaded[sign] >= MAX_PER_SIGN:
                sign_skipped[sign] += 1
                continue

            path     = row['path']
            out_dir  = ASL_DIR / Path(path).parent
            out_dir.mkdir(parents=True, exist_ok=True)

            out_file = ASL_DIR / path
            if out_file.exists() and out_file.stat().st_size > 100:
                sign_downloaded[sign] += 1
                continue

            success = download_file(path, out_dir)
            if success:
                sign_downloaded[sign] += 1
            else:
                sign_failed[sign] += 1

        print(f"    Done={sign_downloaded[sign]} | "
              f"Failed={sign_failed[sign]} | "
              f"Skipped={sign_skipped[sign]}")

    # Step 4 — Verify
    print(f"\n[3/3] Verifying downloads...")
    all_parquet = list(
        (ASL_DIR / "train_landmark_files").rglob("*.parquet")
    )
    print(f"  Total parquet files : {len(all_parquet)}")

    if all_parquet:
        sample = all_parquet[0]
        df_sample = pd.read_parquet(sample)
        print(f"\n  Sample file: {sample.name}")
        print(f"  Columns : {df_sample.columns.tolist()}")
        print(f"  Rows    : {len(df_sample)}")
        if 'type' in df_sample.columns:
            print(f"  Types   : {df_sample['type'].unique().tolist()}")
        print(f"\n  First 3 rows:")
        print(df_sample.head(3).to_string())

    # Summary
    total_downloaded = sum(sign_downloaded.values())
    total_failed     = sum(sign_failed.values())

    print("\n" + "=" * 55)
    print("  Download Summary")
    print("=" * 55)
    print(f"  Total downloaded : {total_downloaded}")
    print(f"  Total failed     : {total_failed}")
    print()
    print("  Per sign:")
    for sign in TARGET_SIGNS:
        ok     = sign_downloaded[sign]
        fail   = sign_failed[sign]
        status = "✓" if ok >= 20 else "⚠"
        print(f"    {status} {sign:15s} : {ok:>3} ok | {fail:>3} failed")

    print()
    if total_downloaded < 50:
        print("  WARNING: Too few files.")
        print("  Accept rules at:")
        print("  https://www.kaggle.com/competitions/asl-signs/rules")
    else:
        print("  Next command:")
        print("  python scripts/preprocess_asl.py")
    print("=" * 55)


if __name__ == "__main__":
    main()