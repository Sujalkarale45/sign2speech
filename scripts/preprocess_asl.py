"""
scripts/preprocess_asl.py
Robust + Fast preprocessing for Google ASL + optional WLASL

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
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing.normalizer import KeypointNormalizer
from src.preprocessing.mel_utils import gloss_to_mel

# ── Config ────────────────────────────────────────
ASL_DIR      = Path("data/raw/asl-signs")
OUTPUT_DIR   = Path("data/processed")
MAX_PER_SIGN = 400

TARGET_SIGNS = [
    'drink', 'who', 'cow', 'bird', 'brown',
    'cat', 'kiss', 'go', 'think', 'man'
]

FACE_IDX = [61, 291, 199, 1, 33, 263]
POSE_IDX = [11,12,13,14,15,16,23,24,25,26,27,28,0,1,2]

EXPECTED_FEATURES = 183
# ─────────────────────────────────────────────────


# ── FAST PARQUET PARSER ──────────────────────────
def parquet_to_keypoints(path: Path) -> np.ndarray:

    df = pd.read_parquet(path)

    required = {'frame', 'type', 'landmark_index', 'x', 'y', 'z'}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns in {path.name}")

    df = df.sort_values("frame")
    grouped = df.groupby("frame")

    keypoints = []

    for _, frame_df in grouped:

        def extract(lm_type, indices, dims):
            subset = frame_df[frame_df['type'] == lm_type]

            lookup = {
                int(row['landmark_index']): row
                for _, row in subset.iterrows()
            }

            pts = []
            for idx in indices:
                if idx in lookup:
                    r = lookup[idx]
                    pts.extend([r['x'], r['y']])
                    if dims == 3:
                        pts.append(r['z'])
                else:
                    pts.extend([0.0] * dims)

            return pts

        left  = extract('left_hand', range(21), 3)
        right = extract('right_hand', range(21), 3)
        pose  = extract('pose', POSE_IDX, 3)
        face  = extract('face', FACE_IDX, 2)

        vec = np.array(left + right + pose + face, dtype=np.float32)

        if vec.shape[0] != EXPECTED_FEATURES:
            raise ValueError(f"{path.name}: wrong feature size")

        vec = np.nan_to_num(vec)
        keypoints.append(vec)

    if len(keypoints) < 8:
        raise ValueError(f"{path.name}: too few frames")

    return np.stack(keypoints)


# ── GOOGLE ASL PROCESSING ─────────────────────────
def process_google_asl(df):

    print("\n[1/4] Processing Google ASL...")

    df['sign'] = df['sign'].str.lower().str.strip()
    subset = df[df['sign'].isin(TARGET_SIGNS)]

    counts = defaultdict(int)
    manifest = []
    kp_list = []

    for _, row in tqdm(subset.iterrows(), total=len(subset)):

        sign = row['sign']

        if counts[sign] >= MAX_PER_SIGN:
            continue

        path = ASL_DIR / row['path']
        if not path.exists():
            continue

        try:
            kp = parquet_to_keypoints(path)

            # ✅ length control
            if kp.shape[0] > 120:
                kp = kp[:120]

            if kp.shape[0] < 8:
                continue

            vid_id = f"{sign}_{counts[sign]:04d}"
            kp_path = OUTPUT_DIR / "keypoints" / f"{vid_id}.npy"

            np.save(kp_path, kp)

            manifest.append({
                "video_id": vid_id,
                "gloss": sign,
                "keypoint_file": str(kp_path),
                "frames": int(kp.shape[0])
            })

            kp_list.append(kp)
            counts[sign] += 1

        except Exception as e:
            continue

    print("\nSamples per sign:")
    for s in TARGET_SIGNS:
        print(f"{s:10}: {counts[s]}")

    return manifest, kp_list


# ── MEL GENERATION ───────────────────────────────
def synthesize_mels(samples):

    print("\n[2/4] Generating mels...")

    cache = {}

    for s in tqdm(samples):
        g = s['gloss']

        if g not in cache:
            cache[g] = gloss_to_mel(g)

        mel = cache[g]

        mel_path = OUTPUT_DIR / "mels" / f"{s['video_id']}.npy"
        np.save(mel_path, mel)

        s["mel_file"] = str(mel_path)

    return samples


# ── SPLIT ───────────────────────────────────────
def split_data(samples):

    groups = defaultdict(list)

    for s in samples:
        groups[s['gloss']].append(s)

    train, val, test = [], [], []

    for g, items in groups.items():
        random.shuffle(items)
        n = len(items)

        train += items[:int(0.8*n)]
        val   += items[int(0.8*n):int(0.9*n)]
        test  += items[int(0.9*n):]

    return train, val, test


# ── MAIN ────────────────────────────────────────
def main():

    print("="*60)
    print("ASL Preprocessing (FAST & STABLE)")
    print("="*60)

    csv_path = ASL_DIR / "train.csv"

    if not csv_path.exists():
        print("Dataset missing")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "keypoints").mkdir(exist_ok=True)
    (OUTPUT_DIR / "mels").mkdir(exist_ok=True)

    df = pd.read_csv(csv_path)

    manifest, kp_list = process_google_asl(df)

    if not manifest:
        print("No data processed")
        return

    manifest = synthesize_mels(manifest)

    print("\n[3/4] Fitting normalizer...")
    normalizer = KeypointNormalizer(str(OUTPUT_DIR / "keypoint_stats.npz"))
    normalizer.fit(kp_list)

    print("\n[4/4] Splitting dataset...")
    train, val, test = split_data(manifest)

    for name, data in zip(["train","val","test"], [train,val,test]):
        with open(OUTPUT_DIR / f"{name}_manifest.json","w") as f:
            json.dump(data, f, indent=2)

        print(f"{name}: {len(data)} samples")

    print("\n✅ DONE")
    print(f"Total samples: {len(manifest)}")


if __name__ == "__main__":
    main()