"""
scripts/preprocess_asl.py
Preprocesses Google ASL Signs dataset + WLASL combined.
Reads pre-extracted MediaPipe parquet files from Google ASL.
Combines with WLASL videos for better generalization.

Usage:
    python scripts/preprocess_asl.py
"""

import json
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing.normalizer import KeypointNormalizer
from src.preprocessing.mel_utils import gloss_to_mel

# ── Config ────────────────────────────────────────
ASL_DIR      = Path("data/raw/asl-signs")
WLASL_DIR    = Path("data/raw/wlasl-processed")
OUTPUT_DIR   = Path("data/processed")
MAX_PER_SIGN = 80
TARGET_SIGNS = [
    'drink', 'who', 'cow', 'bird', 'brown',
    'cat', 'kiss', 'go', 'think', 'man'
]

# MediaPipe landmark indices
FACE_IDX = [61, 291, 199, 1, 33, 263]
POSE_IDX = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 0, 1, 2]
# ─────────────────────────────────────────────────


def parquet_to_keypoints(parquet_path: str) -> np.ndarray:
    """
    Convert Google ASL parquet file to (T, 183) keypoint array.
    Google ASL parquet columns: frame, type, landmark_index, x, y, z
    Output: (T, 183) = left_hand(63) + right_hand(63) + pose(45) + face(12)
    """
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        raise ValueError(f"Cannot read parquet: {e}")

    # Check required columns
    required = {'frame', 'type', 'landmark_index', 'x', 'y', 'z'}
    if not required.issubset(df.columns):
        # Try alternate column format
        df.columns = [c.split('.')[-1] for c in df.columns]

    frames = sorted(df['frame'].unique())
    if len(frames) == 0:
        raise ValueError("No frames found")

    keypoints = []

    for frame_idx in frames:
        frame_df = df[df['frame'] == frame_idx]

        def get_hand(lm_type: str) -> np.ndarray:
            """Get 21 hand landmarks → (63,)"""
            subset = frame_df[frame_df['type'] == lm_type]
            pts    = []
            for idx in range(21):
                row = subset[subset['landmark_index'] == idx]
                if len(row) == 0:
                    pts.extend([0.0, 0.0, 0.0])
                else:
                    pts.extend([
                        float(row['x'].iloc[0]),
                        float(row['y'].iloc[0]),
                        float(row['z'].iloc[0]),
                    ])
            return np.array(pts, dtype=np.float32)

        def get_pose(indices: list) -> np.ndarray:
            """Get selected pose landmarks → (45,)"""
            subset = frame_df[frame_df['type'] == 'pose']
            pts    = []
            for idx in indices:
                row = subset[subset['landmark_index'] == idx]
                if len(row) == 0:
                    pts.extend([0.0, 0.0, 0.0])
                else:
                    pts.extend([
                        float(row['x'].iloc[0]),
                        float(row['y'].iloc[0]),
                        float(row['z'].iloc[0]),
                    ])
            return np.array(pts, dtype=np.float32)

        def get_face(indices: list) -> np.ndarray:
            """Get selected face landmarks x,y only → (12,)"""
            subset = frame_df[frame_df['type'] == 'face']
            pts    = []
            for idx in indices:
                row = subset[subset['landmark_index'] == idx]
                if len(row) == 0:
                    pts.extend([0.0, 0.0])
                else:
                    pts.extend([
                        float(row['x'].iloc[0]),
                        float(row['y'].iloc[0]),
                    ])
            return np.array(pts, dtype=np.float32)

        left_hand  = get_hand('left_hand')    # 63
        right_hand = get_hand('right_hand')   # 63
        pose       = get_pose(POSE_IDX)       # 45
        face       = get_face(FACE_IDX)       # 12

        frame_vec = np.concatenate([left_hand, right_hand, pose, face])
        assert frame_vec.shape[0] == 183, \
            f"Expected 183 features, got {frame_vec.shape[0]}"
        keypoints.append(frame_vec)

    return np.stack(keypoints, axis=0)  # (T, 183)


def process_google_asl(df: pd.DataFrame) -> tuple:
    """Process Google ASL parquet files for target signs."""
    print(f"\n[1/5] Processing Google ASL parquet files...")

    asl_subset  = df[df['sign'].isin(TARGET_SIGNS)]
    sign_count  = defaultdict(int)
    manifest    = []
    kp_list     = []
    failed      = 0
    skipped     = 0

    total = len(asl_subset)
    print(f"  Files available : {total}")

    for i, (_, row) in enumerate(asl_subset.iterrows()):
        sign = row['sign']
        if sign_count[sign] >= MAX_PER_SIGN:
            skipped += 1
            continue

        parquet_path = ASL_DIR / row['path']
        if not parquet_path.exists():
            failed += 1
            continue

        try:
            kp      = parquet_to_keypoints(str(parquet_path))
            vid_id  = f"asl_{sign}_{sign_count[sign]:04d}"
            kp_path = OUTPUT_DIR / "keypoints" / f"{vid_id}.npy"
            np.save(kp_path, kp)
            kp_list.append(kp)

            manifest.append({
                "video_id"     : vid_id,
                "gloss"        : sign,
                "keypoint_file": str(kp_path),
                "kp_frames"    : int(kp.shape[0]),
                "source"       : "google_asl",
            })
            sign_count[sign] += 1

            if len(manifest) % 100 == 0:
                print(f"  [{len(manifest)}] processed | "
                      f"failed={failed}")

        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"  SKIP {row['path']}: {e}")

    print(f"  Google ASL done : {len(manifest)} samples | "
          f"failed={failed} | skipped={skipped}")
    print(f"  Per sign        :")
    for sign in TARGET_SIGNS:
        print(f"    {sign:15s} : {sign_count[sign]}")

    return manifest, kp_list


def process_wlasl() -> tuple:
    """Process WLASL videos for target signs."""
    print(f"\n[2/5] Processing WLASL videos...")

    json_path = WLASL_DIR / "WLASL_v0.3.json"
    video_dir = WLASL_DIR / "videos"

    if not json_path.exists() or not video_dir.exists():
        print("  WLASL not found — skipping")
        return [], []

    from src.preprocessing.extractor import KeypointExtractor
    extractor = KeypointExtractor()

    with open(json_path) as f:
        data = json.load(f)

    # Build position-based match
    all_videos     = sorted(video_dir.glob("*.mp4"))
    json_instances = []
    for entry in data:
        for inst in entry["instances"]:
            json_instances.append({
                "gloss"   : entry["gloss"].lower(),
                "video_id": str(inst["video_id"]),
            })

    # Filter to target signs
    wlasl_samples = []
    for video, inst in zip(all_videos, json_instances):
        if inst["gloss"] in TARGET_SIGNS:
            wlasl_samples.append({
                "video_path": str(video),
                "video_id"  : video.stem,
                "gloss"     : inst["gloss"],
            })

    print(f"  WLASL matched : {len(wlasl_samples)} videos")

    manifest = []
    kp_list  = []
    failed   = 0

    for i, s in enumerate(wlasl_samples):
        try:
            kp      = extractor.process_video(s["video_path"])
            vid_id  = f"wlasl_{s['video_id']}"
            kp_path = OUTPUT_DIR / "keypoints" / f"{vid_id}.npy"
            np.save(kp_path, kp)
            kp_list.append(kp)

            manifest.append({
                "video_id"     : vid_id,
                "gloss"        : s["gloss"],
                "keypoint_file": str(kp_path),
                "kp_frames"    : int(kp.shape[0]),
                "source"       : "wlasl",
            })

            if i % 20 == 0:
                print(f"  [{i:>4}/{len(wlasl_samples)}] "
                      f"{s['gloss']:15s} | frames={kp.shape[0]}")

        except Exception as e:
            failed += 1
            if failed <= 3:
                print(f"  SKIP {s['video_id']}: {e}")

    print(f"  WLASL done : {len(manifest)} samples | failed={failed}")
    return manifest, kp_list


def synthesize_mels(samples: list) -> list:
    """Synthesize gTTS mel targets for all samples."""
    print(f"\n[3/5] Synthesizing mel targets ({len(samples)} samples)...")
    mel_cache = {}
    final     = []

    for s in samples:
        gloss = s['gloss']
        try:
            if gloss not in mel_cache:
                print(f"  Synthesizing: '{gloss}'")
                mel_cache[gloss] = gloss_to_mel(gloss)

            mel      = mel_cache[gloss].copy()
            mel_path = OUTPUT_DIR / "mels" / f"{s['video_id']}.npy"
            np.save(mel_path, mel)
            s["mel_file"]   = str(mel_path)
            s["mel_frames"] = int(mel.shape[1])
            final.append(s)

        except Exception as e:
            print(f"  SKIP mel '{gloss}': {e}")

    print(f"  Mel done : {len(final)} samples")
    return final


def split_by_gloss(samples: list):
    """Split ensuring every gloss appears in train/val/test."""
    groups = defaultdict(list)
    for s in samples:
        groups[s['gloss']].append(s)

    train, val, test = [], [], []
    for gloss, items in groups.items():
        random.shuffle(items)
        n     = len(items)
        t_end = max(1, int(0.8 * n))
        v_end = max(t_end + 1, int(0.9 * n))
        train.extend(items[:t_end])
        val.extend(items[t_end:v_end])
        test.extend(items[v_end:] if v_end < n else items[-1:])

    return train, val, test


def main():
    print("=" * 55)
    print("  SignVoice — Google ASL + WLASL Preprocessing")
    print("=" * 55)
    print(f"\n  Target signs : {TARGET_SIGNS}")

    # Validate
    csv_path = ASL_DIR / "train.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found.")
        sys.exit(1)

    # Create output dirs
    (OUTPUT_DIR / "keypoints").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "mels").mkdir(parents=True, exist_ok=True)

    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"\n  CSV loaded : {len(df)} total samples")

    # Verify target signs exist
    available = set(df['sign'].unique())
    missing   = set(TARGET_SIGNS) - available
    if missing:
        print(f"  WARNING: Signs not in CSV: {missing}")

    # Process both datasets
    asl_manifest, asl_kp   = process_google_asl(df)
    wlasl_manifest, wlasl_kp = process_wlasl()

    # Combine
    all_samples = asl_manifest + wlasl_manifest
    all_kp      = asl_kp + wlasl_kp

    if len(all_samples) == 0:
        print("ERROR: No samples processed.")
        sys.exit(1)

    print(f"\n  Combined : {len(all_samples)} total samples")
    counts = Counter(s['gloss'] for s in all_samples)
    print(f"  Per sign :")
    for sign in sorted(counts.keys()):
        src_asl  = sum(1 for s in all_samples
                      if s['gloss'] == sign and s['source'] == 'google_asl')
        src_wlasl = sum(1 for s in all_samples
                       if s['gloss'] == sign and s['source'] == 'wlasl')
        print(f"    {sign:15s} : {counts[sign]:>4} "
              f"(ASL={src_asl} WLASL={src_wlasl})")

    # Synthesize mels
    final = synthesize_mels(all_samples)

    # Fit normalizer
    print(f"\n[4/5] Fitting normalizer on {len(all_kp)} sequences...")
    normalizer = KeypointNormalizer(
        str(OUTPUT_DIR / "keypoint_stats.npz")
    )
    normalizer.fit(all_kp)

    # Split
    print(f"\n[5/5] Creating splits...")
    train, val, test = split_by_gloss(final)

    for split, sdata in [("train", train),
                          ("val",   val),
                          ("test",  test)]:
        path = OUTPUT_DIR / f"{split}_manifest.json"
        with open(path, "w") as f:
            json.dump(sdata, f, indent=2)
        glosses = sorted(set(s["gloss"] for s in sdata))
        sources = Counter(s["source"] for s in sdata)
        print(f"  {split:>5} : {len(sdata):>4} samples | "
              f"glosses={len(glosses)} | {dict(sources)}")

    # Summary
    glosses_used = sorted(set(s["gloss"] for s in final))
    print("\n" + "=" * 55)
    print("  Preprocessing Complete!")
    print("=" * 55)
    print(f"  Total   : {len(final)}")
    print(f"  Train   : {len(train)}")
    print(f"  Val     : {len(val)}")
    print(f"  Test    : {len(test)}")
    print(f"  Signs   : {glosses_used}")
    print(f"\n  Next commands:")
    print(f"  python scripts/test_inference.py")
    print(f"  python scripts/realtime_demo.py")
    print("=" * 55)


if __name__ == "__main__":
    main()