"""
scripts/preprocess.py
Fixed split — every gloss appears in train/val/test.
"""

import json
import os
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing.extractor import KeypointExtractor
from src.preprocessing.normalizer import KeypointNormalizer
from src.preprocessing.mel_utils import gloss_to_mel

VIDEO_DIR     = Path("data/raw/wlasl-processed/videos")
JSON_PATH     = Path("data/raw/wlasl-processed/WLASL_v0.3.json")
OUTPUT_DIR    = Path("data/processed")
MAX_GLOSSES   = 10
MAX_PER_GLOSS = 30


def build_index_by_order():
    with open(JSON_PATH) as f:
        data = json.load(f)

    all_videos     = sorted(VIDEO_DIR.glob("*.mp4"))
    json_instances = []
    for entry in data:
        for inst in entry["instances"]:
            json_instances.append({
                "gloss"      : entry["gloss"],
                "original_id": str(inst["video_id"]),
            })

    print(f"  Videos on disk   : {len(all_videos)}")
    print(f"  JSON instances   : {len(json_instances)}")

    matched = []
    for video, inst in zip(all_videos, json_instances):
        matched.append({
            "video_path" : str(video),
            "video_id"   : video.stem,
            "gloss"      : inst["gloss"],
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

    print(f"\n  After filtering  : {len(filtered)} samples")
    print(f"  Glosses          : {sorted(set(m['gloss'] for m in filtered))}")
    return filtered


def split_by_gloss(samples):
    """
    Split dataset ensuring every gloss appears in train/val/test.
    80/10/10 split done per gloss separately.
    """
    gloss_groups = defaultdict(list)
    for s in samples:
        gloss_groups[s["gloss"]].append(s)

    train, val, test = [], [], []
    for gloss, items in gloss_groups.items():
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
    print("  SignVoice — Preprocessing Pipeline")
    print("=" * 55)

    if not VIDEO_DIR.exists():
        print(f"ERROR: {VIDEO_DIR} not found.")
        sys.exit(1)

    total_mp4 = len(list(VIDEO_DIR.glob("*.mp4")))
    print(f"\n  Videos on disk : {total_mp4}")

    (OUTPUT_DIR / "keypoints").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "mels").mkdir(parents=True, exist_ok=True)

    print(f"\n[1/4] Building index...")
    samples = build_index_by_order()

    print(f"\n[2/4] Extracting keypoints ({len(samples)} videos)...")
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
            if i % 20 == 0:
                print(f"  [{i:>4}/{len(samples)}] {s['gloss']:20s} | frames={kp.shape[0]}")
        except Exception as e:
            failed += 1
            if failed <= 3:
                print(f"  SKIP {s['video_id']}: {e}")

    print(f"\n  Extracted : {len(manifest)} | Failed : {failed}")

    print(f"\n[3/4] Synthesizing mel targets...")
    mel_cache = {}
    final     = []

    for s in manifest:
        gloss = s["gloss"]
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

    print(f"\n[4/4] Fitting normalizer...")
    normalizer = KeypointNormalizer(str(OUTPUT_DIR / "keypoint_stats.npz"))
    normalizer.fit(all_kp)

    # Split per gloss so every gloss is in all splits
    train, val, test = split_by_gloss(final)

    for split, sdata in [("train", train), ("val", val), ("test", test)]:
        with open(OUTPUT_DIR / f"{split}_manifest.json", "w") as f:
            json.dump(sdata, f, indent=2)
        glosses = sorted(set(s["gloss"] for s in sdata))
        print(f"  {split:>5} : {len(sdata):>4} samples | glosses={glosses}")

    glosses_used = sorted(set(s["gloss"] for s in final))
    print("\n" + "=" * 55)
    print("  Preprocessing Complete!")
    print("=" * 55)
    print(f"  Total   : {len(final)}")
    print(f"  Train   : {len(train)}")
    print(f"  Val     : {len(val)}")
    print(f"  Test    : {len(test)}")
    print(f"  Glosses : {glosses_used}")
    print(f"\n  Next: python scripts/test_inference.py")
    print("=" * 55)


if __name__ == "__main__":
    main()