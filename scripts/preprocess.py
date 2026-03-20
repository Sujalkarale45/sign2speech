"""
scripts/preprocess.py
Fixed version — matches WLASL video filenames directly without zero-padding.
"""

import json
import os
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing.extractor import KeypointExtractor
from src.preprocessing.normalizer import KeypointNormalizer
from src.preprocessing.mel_utils import gloss_to_mel

# ── Config ────────────────────────────────────────
VIDEO_DIR     = Path("data/raw/wlasl-processed/videos")
JSON_PATH     = Path("data/raw/wlasl-processed/WLASL_v0.3.json")
OUTPUT_DIR    = Path("data/processed")
MAX_GLOSSES   = 1000    # keep plenty of glosses; should allow 380+ if files exist
MAX_PER_GLOSS = 999     # very high limit to include all available files per gloss (380+ total)
# ─────────────────────────────────────────────────


def build_video_index():
    """
    Match video files to glosses.
    Tries both raw video_id and zero-padded (zfill 5) formats.
    """
    # Build lookup from JSON: both formats → gloss
    id_to_gloss = {}
    with open(JSON_PATH) as f:
        data = json.load(f)

    for entry in data[:MAX_GLOSSES]:
        gloss = entry["gloss"]
        for inst in entry["instances"]:
            vid_id = str(inst["video_id"])
            # store both raw and zero-padded versions
            id_to_gloss[vid_id]            = gloss   # e.g. "69241"
            id_to_gloss[vid_id.zfill(5)]   = gloss   # e.g. "69241" (same here)
            id_to_gloss[vid_id.zfill(6)]   = gloss   # fallback

    # Scan disk and match
    samples      = []
    unmatched    = []
    gloss_counts = {}

    for mp4 in sorted(VIDEO_DIR.glob("*.mp4")):
        stem  = mp4.stem                     # e.g. "69241"
        gloss = id_to_gloss.get(stem)        # direct match

        # If no direct match, try stripping leading zeros
        if gloss is None:
            gloss = id_to_gloss.get(stem.lstrip("0") or "0")

        if gloss is None:
            # Fallback: assign unknown gloss and still include the sample
            gloss = "unknown"
            unmatched.append(mp4.name)

        count = gloss_counts.get(gloss, 0)
        if count >= MAX_PER_GLOSS:
            continue

        samples.append({
            "video_path" : str(mp4),
            "gloss"      : gloss,
            "video_id"   : stem,
        })
        gloss_counts[gloss] = count + 1

    print(f"  Matched   : {len(samples)} videos")
    print(f"  Unmatched : {len(unmatched)} (belong to glosses outside top {MAX_GLOSSES})")
    print(f"  Glosses   : {sorted(gloss_counts.keys())}")
    print(f"  Per gloss : {gloss_counts}")
    return samples


def main():
    print("=" * 55)
    print("  SignVoice — Preprocessing Pipeline")
    print("=" * 55)

    if not VIDEO_DIR.exists():
        print(f"\nERROR: {VIDEO_DIR} not found.")
        sys.exit(1)

    total_mp4 = len(list(VIDEO_DIR.glob("*.mp4")))
    print(f"\n  Total .mp4 on disk : {total_mp4}")

    (OUTPUT_DIR / "keypoints").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "mels").mkdir(parents=True, exist_ok=True)

    # ── Step 1: Build index ──────────────────────
    print(f"\n[1/4] Building video index...")
    samples = build_video_index()

    if len(samples) == 0:
        print("\nERROR: Still 0 matches. Run the debug command below:")
        print('python -c "from pathlib import Path; '
              'print([p.stem for p in '
              'sorted(Path(\'data/raw/wlasl-processed/videos\').glob(\'*.mp4\'))[:5]])"')
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

            if i % 20 == 0:
                print(f"  [{i:>4}/{len(samples)}] {s['gloss']:15s} | frames={kp.shape[0]}")

        except Exception as e:
            failed += 1
            if failed <= 3:
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
            if gloss not in mel_cache:
                print(f"  Synthesizing : '{gloss}'")
                mel_cache[gloss] = gloss_to_mel(gloss)

            mel      = mel_cache[gloss]
            mel_path = OUTPUT_DIR / "mels" / f"{s['video_id']}.npy"
            np.save(mel_path, mel)
            s["mel_file"]   = str(mel_path)
            s["mel_frames"] = int(mel.shape[1])
            final.append(s)

        except Exception as e:
            mel_failed += 1
            print(f"  SKIP mel '{gloss}': {e}")

    print(f"  Mel done : {len(final)} | Failed : {mel_failed}")

    # ── Step 4: Fit normalizer ───────────────────
    print(f"\n[4/4] Fitting normalizer...")
    normalizer = KeypointNormalizer(str(OUTPUT_DIR / "keypoint_stats.npz"))
    normalizer.fit(all_kp)

    # ── Split 80/10/10 ───────────────────────────
    n     = len(final)
    train = final[:int(0.8 * n)]
    val   = final[int(0.8 * n):int(0.9 * n)]
    test  = final[int(0.9 * n):]

    for split, data in [("train", train), ("val", val), ("test", test)]:
        path = OUTPUT_DIR / f"{split}_manifest.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  {split:>5} : {len(data):>4} samples")

    # ── Summary ──────────────────────────────────
    glosses_used = sorted(set(s["gloss"] for s in final))
    print("\n" + "=" * 55)
    print("  Preprocessing Complete!")
    print("=" * 55)
    print(f"  Total samples : {n}")
    print(f"  Train         : {len(train)}")
    print(f"  Val           : {len(val)}")
    print(f"  Test          : {len(test)}")
    print(f"  Glosses       : {glosses_used}")
    print(f"\n  Next:")
    print(f"  python scripts/train.py --config configs/lightweight.yaml")
    print("=" * 55)


if __name__ == "__main__":
    main()