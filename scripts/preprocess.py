"""
scripts/preprocess.py
Fixed for Kaggle WLASL-processed dataset where video files
are renamed sequentially (00335.mp4) but JSON has original IDs.
Each video gets its own unique mel target with small variation.
"""

import json
import os
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing.extractor import KeypointExtractor
from src.preprocessing.normalizer import KeypointNormalizer
from src.preprocessing.mel_utils import gloss_to_mel

# ── Config ────────────────────────────────────────
VIDEO_DIR     = Path("data/raw/wlasl-processed/videos")
JSON_PATH     = Path("data/raw/wlasl-processed/WLASL_v0.3.json")
OUTPUT_DIR    = Path("data/processed")
MAX_GLOSSES   = 20
MAX_PER_GLOSS = 20
# ─────────────────────────────────────────────────


def build_index_by_order() -> list[dict]:
    """
    Match video files to glosses by sequential position.
    Kaggle WLASL renames videos sequentially but JSON has original IDs.
    """
    with open(JSON_PATH) as f:
        data = json.load(f)

    all_videos = sorted(VIDEO_DIR.glob("*.mp4"))
    print(f"  Total videos on disk : {len(all_videos)}")

    json_instances = []
    for entry in data:
        gloss = entry["gloss"]
        for inst in entry["instances"]:
            json_instances.append({
                "gloss"      : gloss,
                "original_id": str(inst["video_id"]),
                "split"      : inst.get("split", "train"),
            })

    print(f"  Total JSON instances : {len(json_instances)}")
    print(f"  Matching by position : {min(len(all_videos), len(json_instances))} pairs")

    matched = []
    for video, inst in zip(all_videos, json_instances):
        matched.append({
            "video_path" : str(video),
            "video_id"   : video.stem,
            "gloss"      : inst["gloss"],
            "original_id": inst["original_id"],
            "split"      : inst["split"],
        })

    gloss_counts = Counter(m["gloss"] for m in matched)
    top_glosses  = set(g for g, _ in gloss_counts.most_common(MAX_GLOSSES))

    print(f"\n  Top {MAX_GLOSSES} glosses:")
    for g, c in gloss_counts.most_common(MAX_GLOSSES):
        print(f"    {g:20s} : {c} videos")

    filtered   = []
    gloss_seen = defaultdict(int)

    for m in matched:
        if m["gloss"] not in top_glosses:
            continue
        if gloss_seen[m["gloss"]] >= MAX_PER_GLOSS:
            continue
        filtered.append(m)
        gloss_seen[m["gloss"]] += 1

    print(f"\n  After filtering : {len(filtered)} samples")
    print(f"  Glosses used    : {sorted(set(m['gloss'] for m in filtered))}")

    return filtered


def main():
    print("=" * 55)
    print("  SignVoice — Preprocessing Pipeline")
    print("=" * 55)

    if not VIDEO_DIR.exists():
        print(f"\nERROR: {VIDEO_DIR} not found.")
        sys.exit(1)

    total_mp4 = len(list(VIDEO_DIR.glob("*.mp4")))
    print(f"\n  Videos on disk : {total_mp4}")

    if total_mp4 == 0:
        print("ERROR: No .mp4 files found.")
        sys.exit(1)

    (OUTPUT_DIR / "keypoints").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "mels").mkdir(parents=True, exist_ok=True)

    # ── Step 1: Build index ──────────────────────
    print(f"\n[1/4] Building video-gloss index...")
    samples = build_index_by_order()

    if len(samples) == 0:
        print("\nERROR: No samples found.")
        sys.exit(1)

    # ── Step 2: Extract keypoints ────────────────
    print(f"\n[2/4] Extracting keypoints ({len(samples)} videos)...")
    print(f"  Estimated time : ~{max(1, len(samples) * 3 // 60)} mins")

    extractor = KeypointExtractor()
    all_kp    = []
    manifest  = []
    failed    = 0

    for i, s in enumerate(samples):
        try:
            kp      = extractor.process_video(s["video_path"])
            kp_path = OUTPUT_DIR / "keypoints" / f"{s['video_id']}.npy"
            np.save(kp_path, kp)
            all_kp.append(kp)
            s["keypoint_file"] = str(kp_path)
            s["kp_frames"]     = int(kp.shape[0])
            manifest.append(s)

            if i % 25 == 0:
                print(f"  [{i:>4}/{len(samples)}] {s['gloss']:20s} | frames={kp.shape[0]}")

        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"  SKIP {s['video_id']} ({s['gloss']}): {e}")

    print(f"\n  Extracted : {len(manifest)} | Failed : {failed}")

    # ── Step 3: Synthesize mel targets ───────────
    print(f"\n[3/4] Synthesizing mel targets (gTTS)...")
    mel_cache  = {}
    final      = []
    mel_failed = 0

    for s in manifest:
        gloss = s["gloss"]
        try:
            # Synthesize once per gloss, cache it
            if gloss not in mel_cache:
                print(f"  Synthesizing : '{gloss}'")
                mel_cache[gloss] = gloss_to_mel(gloss)

            # Give each video its own unique mel with small noise
            # This prevents model from collapsing to one output per gloss
            base_mel = mel_cache[gloss]
            noise    = np.random.normal(0, 0.05, base_mel.shape).astype(np.float32)
            mel      = base_mel + noise

            mel_path = OUTPUT_DIR / "mels" / f"{s['video_id']}.npy"
            np.save(mel_path, mel)
            s["mel_file"]   = str(mel_path)
            s["mel_frames"] = int(mel.shape[1])
            final.append(s)

        except Exception as e:
            mel_failed += 1
            print(f"  SKIP mel '{gloss}': {e}")

    print(f"  Done : {len(final)} | Failed : {mel_failed}")

    # ── Step 4: Fit normalizer ───────────────────
    print(f"\n[4/4] Fitting normalizer on {len(all_kp)} sequences...")
    normalizer = KeypointNormalizer(str(OUTPUT_DIR / "keypoint_stats.npz"))
    normalizer.fit(all_kp)

    # ── Split 80/10/10 ───────────────────────────
    n     = len(final)
    train = final[:int(0.8 * n)]
    val   = final[int(0.8 * n):int(0.9 * n)]
    test  = final[int(0.9 * n):]

    for split, sdata in [("train", train), ("val", val), ("test", test)]:
        path = OUTPUT_DIR / f"{split}_manifest.json"
        with open(path, "w") as f:
            json.dump(sdata, f, indent=2)
        print(f"  {split:>5} : {len(sdata):>4} samples")

    # ── Summary ──────────────────────────────────
    glosses_used = sorted(set(s["gloss"] for s in final))
    print("\n" + "=" * 55)
    print("  Preprocessing Complete!")
    print("=" * 55)
    print(f"  Total samples  : {n}")
    print(f"  Train          : {len(train)}")
    print(f"  Val            : {len(val)}")
    print(f"  Test           : {len(test)}")
    print(f"  Glosses        : {len(glosses_used)}")
    print(f"  Signs learned  : {glosses_used}")
    print(f"\n  Next:")
    print(f"  python scripts/train.py --config configs/lightweight.yaml")
    print("=" * 55)


if __name__ == "__main__":
    main()