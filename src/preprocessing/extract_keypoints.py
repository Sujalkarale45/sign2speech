"""
extract_keypoints.py
Runs MediaPipe Holistic on every sentence-level video clip and saves
a (T, 225) numpy array per clip.

  225 = pose(33×3) + left_hand(21×3) + right_hand(21×3)

Expected input layout:
  data/raw/videos/test/<SENTENCE_NAME>.mp4
  data/raw/videos/val/<SENTENCE_NAME>.mp4
  data/raw/metadata/how2sign_*.csv

Output:
  data/processed/keypoints/<SENTENCE_NAME>.npy
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
VIDEO_DIRS   = [Path("data/raw/videos/test"), Path("data/raw/videos/val")]
METADATA_DIR = Path("data/raw/metadata")
OUT_DIR      = Path("data/processed/keypoints")
OUT_DIR.mkdir(parents=True, exist_ok=True)
# ─────────────────────────────────────────────────────────────────────────────

mp_holistic = mp.solutions.holistic


def find_video(sent_name: str) -> Path | None:
    """Search all VIDEO_DIRS for the clip."""
    for d in VIDEO_DIRS:
        p = d / f"{sent_name}.mp4"
        if p.exists():
            return p
    return None


def extract_keypoints(video_path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    seq = []

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = holistic.process(rgb)

            pose = (
                np.array([[lm.x, lm.y, lm.z]
                           for lm in result.pose_landmarks.landmark]).flatten()
                if result.pose_landmarks else np.zeros(33 * 3)
            )
            lh = (
                np.array([[lm.x, lm.y, lm.z]
                           for lm in result.left_hand_landmarks.landmark]).flatten()
                if result.left_hand_landmarks else np.zeros(21 * 3)
            )
            rh = (
                np.array([[lm.x, lm.y, lm.z]
                           for lm in result.right_hand_landmarks.landmark]).flatten()
                if result.right_hand_landmarks else np.zeros(21 * 3)
            )

            seq.append(np.concatenate([pose, lh, rh]))   # 225-dim

    cap.release()
    return np.array(seq)    # (T, 225)


if __name__ == "__main__":
    # Collect all unique SENTENCE_NAMEs from every metadata CSV
    all_names: set[str] = set()
    for csv in METADATA_DIR.glob("how2sign_*.csv"):
        df = pd.read_csv(csv, sep="\t")
        all_names.update(df["SENTENCE_NAME"].astype(str).tolist())

    print(f"Total sentence clips to process: {len(all_names)}")
    ok = skip = err = 0

    for sent_name in sorted(all_names):
        out = OUT_DIR / f"{sent_name}.npy"
        if out.exists():
            skip += 1
            continue

        vp = find_video(sent_name)
        if vp is None:
            print(f"  [SKIP] video not found: {sent_name}")
            skip += 1
            continue

        try:
            kp = extract_keypoints(vp)
            if kp.shape[0] == 0:
                print(f"  [WARN] empty keypoints: {sent_name}")
                skip += 1
                continue
            np.save(str(out), kp)
            print(f"  [OK]   {sent_name}  shape={kp.shape}")
            ok += 1
        except Exception as exc:
            print(f"  [ERR]  {sent_name}: {exc}")
            err += 1

    print(f"\nDone → {ok} saved, {skip} skipped, {err} errors")
    print(f"Keypoints saved to: {OUT_DIR}")
