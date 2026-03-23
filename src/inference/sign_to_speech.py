"""
sign_to_speech.py
Full inference pipeline:
  Sign video → MediaPipe keypoints → MelPredictor → HiFi-GAN → .wav

Usage:
  python src/inference/sign_to_speech.py --video path/to/clip.mp4 --output out.wav
"""

import argparse
import sys
import cv2
import mediapipe as mp
import numpy as np
import torch
import soundfile as sf
from pathlib import Path

# ── Add project root to path so src.* imports work ────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.mel_predictor import MelPredictor

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_CKPT    = "outputs/mel_predictor_best.pt"
HIFIGAN_DIR     = "outputs/hifigan"     # clone https://github.com/jik876/hifi-gan here
SR              = 22050
MAX_KP_LEN      = 300
MAX_MEL_LEN     = 800
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
# ─────────────────────────────────────────────────────────────────────────────

mp_holistic = mp.solutions.holistic


# ── 1. Extract keypoints ──────────────────────────────────────────────────────

def video_to_keypoints(video_path: str) -> np.ndarray:
    """Returns (T, 225) float32 array."""
    cap = cv2.VideoCapture(video_path)
    seq = []

    with mp_holistic.Holistic(min_detection_confidence=0.5,
                               min_tracking_confidence=0.5) as holistic:
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
            seq.append(np.concatenate([pose, lh, rh]))

    cap.release()
    return np.array(seq, dtype=np.float32)   # (T, 225)


def pad_keypoints(kp: np.ndarray, max_len: int) -> np.ndarray:
    T = kp.shape[0]
    if T >= max_len:
        return kp[:max_len]
    return np.vstack([kp, np.zeros((max_len - T, kp.shape[1]), dtype=np.float32)])


# ── 2. Keypoints → Mel ────────────────────────────────────────────────────────

def keypoints_to_mel(kp: np.ndarray, ckpt_path: str) -> np.ndarray:
    """Returns (80, T) mel spectrogram."""
    model = MelPredictor(max_mel_len=MAX_MEL_LEN).to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()

    kp_pad = pad_keypoints(kp, MAX_KP_LEN)                          # (300, 225)
    tensor = torch.from_numpy(kp_pad).unsqueeze(0).to(DEVICE)       # (1, 300, 225)

    with torch.no_grad():
        mel = model(tensor)                                          # (1, 80, 800)

    return mel.squeeze(0).cpu().numpy()                              # (80, 800)


# ── 3. Mel → Waveform via HiFi-GAN ───────────────────────────────────────────

def mel_to_wav_hifigan(mel: np.ndarray, hifigan_dir: str) -> np.ndarray | None:
    """
    Converts mel → waveform using a local HiFi-GAN checkpoint.

    Setup:
      git clone https://github.com/jik876/hifi-gan outputs/hifigan
      # Download checkpoint LJ_FT_T2_V1 into outputs/hifigan/

    Returns float32 waveform array, or None if HiFi-GAN is not set up.
    """
    import importlib, json

    hifigan_path = Path(hifigan_dir)
    if not hifigan_path.exists():
        print(f"[WARN] HiFi-GAN not found at {hifigan_dir}. "
              f"Returning mel only.\n"
              f"  → git clone https://github.com/jik876/hifi-gan {hifigan_dir}")
        return None

    sys.path.insert(0, str(hifigan_path))
    from models import Generator   # HiFi-GAN internal import
    from env import AttrDict

    config_file = next(hifigan_path.glob("**/*.json"), None)
    ckpt_file   = next(hifigan_path.glob("**/*.pt"),   None)

    if config_file is None or ckpt_file is None:
        print("[WARN] HiFi-GAN config or checkpoint not found.")
        return None

    with open(config_file) as f:
        h = AttrDict(json.load(f))

    generator = Generator(h).to(DEVICE)
    state     = torch.load(ckpt_file, map_location=DEVICE)
    generator.load_state_dict(state["generator"])
    generator.eval()
    generator.remove_weight_norm()

    mel_tensor = torch.from_numpy(mel).unsqueeze(0).to(DEVICE)   # (1, 80, T)
    with torch.no_grad():
        wav = generator(mel_tensor).squeeze().cpu().numpy()

    return wav


# ── Main pipeline ─────────────────────────────────────────────────────────────

def sign_to_speech(video_path: str,
                   ckpt_path:  str = DEFAULT_CKPT,
                   output_wav: str = "output.wav",
                   hifigan_dir: str = HIFIGAN_DIR) -> str:
    print(f"\n[1/3] Extracting keypoints from: {video_path}")
    kp = video_to_keypoints(video_path)
    print(f"      Keypoints shape: {kp.shape}")

    print(f"[2/3] Predicting mel spectrogram  (checkpoint: {ckpt_path})")
    mel = keypoints_to_mel(kp, ckpt_path)
    print(f"      Mel shape: {mel.shape}")

    print(f"[3/3] Converting mel → waveform  (HiFi-GAN: {hifigan_dir})")
    wav = mel_to_wav_hifigan(mel, hifigan_dir)

    if wav is not None:
        sf.write(output_wav, wav, SR)
        print(f"\n✓ Audio saved → {output_wav}")
        return output_wav
    else:
        mel_out = output_wav.replace(".wav", "_mel.npy")
        np.save(mel_out, mel)
        print(f"\n✓ Mel saved → {mel_out}  (plug into HiFi-GAN to get audio)")
        return mel_out


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",   required=True, help="Path to input .mp4")
    parser.add_argument("--ckpt",    default=DEFAULT_CKPT)
    parser.add_argument("--output",  default="output.wav")
    parser.add_argument("--hifigan", default=HIFIGAN_DIR)
    args = parser.parse_args()

    sign_to_speech(args.video, args.ckpt, args.output, args.hifigan)
