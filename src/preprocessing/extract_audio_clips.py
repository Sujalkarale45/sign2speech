"""
extract_audio_clips.py
Extracts audio from How2Sign video clips using ffmpeg.
Handles filenames starting with - (dash) correctly.

CSV columns: VIDEO_ID, VIDEO_NAME, SENTENCE_ID,
             SENTENCE_NAME, START, END, SENTENCE

SENTENCE_NAME = -d5dN54tH2E_0-1-rgb_front
Video file    = -d5dN54tH2E_0-1-rgb_front.mp4

Usage:
    python extract_audio_clips.py

Output:
    data/processed/audio_clips/<safe_name>.wav
    data/processed/audio_clips/name_mapping.json
"""

import subprocess
import re
import os
import json
import pandas as pd
from pathlib import Path

# ── Config ────────────────────────────────────────
METADATA_DIR = Path("data/raw/metadata")
VIDEO_DIRS   = [
    Path("data/raw/videos/test"),
    Path("data/raw/videos/val"),
    Path("data/raw/videos/train"),
]
OUT_DIR = Path("data/processed/audio_clips")
SR      = 22050
OUT_DIR.mkdir(parents=True, exist_ok=True)
# ─────────────────────────────────────────────────


def safe_name(sentence_name: str) -> str:
    """
    Convert SENTENCE_NAME to a safe filename.
    -d5dN54tH2E_0-1-rgb_front
    → d5dN54tH2E_0_1_rgb_front
    Removes leading dash, replaces remaining dashes with underscores.
    """
    name = sentence_name.strip()
    name = name.lstrip('-')
    name = name.replace('-', '_')
    return name


def build_video_index() -> tuple[dict, dict]:
    """
    Scan all video dirs and build lookups.
    Returns:
        index      : stem → Path  (exact match)
        base_index : base → [Path] (suffix-stripped match)
    """
    index      = {}
    base_index = {}

    for d in VIDEO_DIRS:
        if not d.exists():
            continue
        for mp4 in d.glob("*.mp4"):
            stem = mp4.stem
            index[stem] = mp4
            base = re.sub(r'-\d+-rgb_front$', '', stem)
            if base not in base_index:
                base_index[base] = []
            base_index[base].append(mp4)

    print(f"  Indexed {len(index)} video files")
    return index, base_index


def find_video(sentence_name: str,
               index: dict,
               base_index: dict) -> Path | None:
    """Find video for a sentence name using multiple strategies."""

    # Strategy 1 — exact stem match
    if sentence_name in index:
        return index[sentence_name]

    # Strategy 2 — base name (strip suffix number)
    base = re.sub(r'-\d+-rgb_front$', '', sentence_name)
    if base in base_index:
        return base_index[base][0]

    # Strategy 3 — try different suffix numbers
    for n in range(1, 15):
        alt = re.sub(r'-\d+-rgb_front$',
                     f'-{n}-rgb_front', sentence_name)
        if alt in index:
            return index[alt]

    # Strategy 4 — partial match on first 20 chars
    prefix = sentence_name[:20]
    for stem, path in index.items():
        if prefix in stem:
            return path

    return None


def extract_audio_safe(video_path: Path,
                       out_path: Path) -> tuple[bool, str]:
    """
    Extract audio to a temp file first (avoids - prefix issue),
    then rename to final path.
    """
    # Use temp file without leading dash
    tmp_path = out_path.parent / f"_tmp_{out_path.name}"

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path.resolve()),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar",     str(SR),
        "-ac",     "1",
        str(tmp_path.resolve()),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if result.returncode == 0 and tmp_path.exists():
        tmp_path.rename(out_path)
        return True, ""

    # Cleanup temp if exists
    if tmp_path.exists():
        tmp_path.unlink()

    return False, result.stderr


def process_split(csv_path: Path,
                  index: dict,
                  base_index: dict,
                  name_mapping: dict) -> int:
    """Process one CSV split and extract audio."""
    df = pd.read_csv(csv_path, sep="\t")

    print(f"\n  [{csv_path.name}]")
    print(f"  Clips    : {len(df)}")
    print(f"  Columns  : {df.columns.tolist()}")

    sent_col = "SENTENCE_NAME"
    if sent_col not in df.columns:
        print(f"  ERROR: SENTENCE_NAME not found")
        return 0

    ok   = 0
    skip = 0
    err  = 0

    for _, row in df.iterrows():
        sent_name = str(row[sent_col]).strip()
        sname     = safe_name(sent_name)
        out_file  = OUT_DIR / f"{sname}.wav"

        # Store mapping: safe_name → original SENTENCE_NAME
        name_mapping[sname] = {
            "sentence_name": sent_name,
            "sentence"     : str(row.get("SENTENCE", "")),
            "start"        : float(row.get("START", 0)),
            "end"          : float(row.get("END", 0)),
            "video_id"     : str(row.get("VIDEO_ID", "")),
        }

        # Skip if already extracted
        if out_file.exists() and out_file.stat().st_size > 500:
            skip += 1
            continue

        # Find video
        video_path = find_video(sent_name, index, base_index)
        if video_path is None:
            skip += 1
            if skip <= 3:
                print(f"  [SKIP] not found: {sent_name}")
            continue

        # Extract
        success, stderr = extract_audio_safe(video_path, out_file)

        if success:
            ok += 1
            if ok % 100 == 0:
                print(f"  [OK]   {ok} extracted...")
        else:
            err += 1
            if err <= 3:
                lines = [l for l in stderr.splitlines()
                         if l.strip()]
                last  = lines[-1] if lines else "unknown"
                print(f"  [ERR]  {sent_name}: {last}")

    print(f"  → OK={ok} | skipped={skip} | errors={err}")
    return ok


def main():
    print("=" * 55)
    print("  How2Sign — Extract Audio Clips")
    print("=" * 55)

    # Check ffmpeg
    r = subprocess.run(["ffmpeg", "-version"],
                       capture_output=True)
    if r.returncode != 0:
        print("ERROR: ffmpeg not found.")
        return

    # Show video dirs
    print("\n  Video directories:")
    for d in VIDEO_DIRS:
        n = len(list(d.glob("*.mp4"))) if d.exists() else 0
        print(f"    {d} : {n} files {'✓' if n > 0 else '✗'}")

    # Find CSVs
    csvs = sorted(METADATA_DIR.glob("*.csv"))
    if not csvs:
        print(f"\nERROR: No CSV files in {METADATA_DIR}")
        return

    print(f"\n  CSV files: {[c.name for c in csvs]}")

    # Build index
    print(f"\n  Building video index...")
    index, base_index = build_video_index()

    # Process all CSVs
    name_mapping = {}
    total_ok     = 0

    for csv in csvs:
        ok        = process_split(csv, index,
                                  base_index, name_mapping)
        total_ok += ok

    # Save name mapping — needed for preprocessing
    mapping_path = OUT_DIR / "name_mapping.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(name_mapping, f, indent=2,
                  ensure_ascii=False)
    print(f"\n  Mapping saved → {mapping_path}")
    print(f"  Mapping entries : {len(name_mapping)}")

    # Summary
    total_wav = len(list(OUT_DIR.glob("*.wav")))
    print("\n" + "=" * 55)
    print("  Extraction Complete!")
    print("=" * 55)
    print(f"  Total .wav files : {total_wav}")
    print(f"  Output folder    : {OUT_DIR}")

    if total_wav > 0:
        wavs = list(OUT_DIR.glob("*.wav"))[:5]
        print(f"\n  Sample files:")
        for w in wavs:
            size = w.stat().st_size / 1024
            print(f"    {w.name} ({size:.1f} KB)")

        print(f"\n  Next step:")
        print(f"  python scripts/preprocess_how2sign.py")
    else:
        print(f"\n  WARNING: No audio extracted.")
        print(f"  Check video dirs contain .mp4 files.")

    print("=" * 55)


if __name__ == "__main__":
    main()