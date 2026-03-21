"""
scripts/realtime_demo.py
Real-time ASL to Speech demo using webcam.
Classification-based: keypoints → predicted gloss → gTTS speech.
Works reliably with small datasets.

Usage:
    python scripts/realtime_demo.py

Controls:
    SPACE  → manually trigger prediction
    R      → reset buffer
    Q      → quit
"""

import os
import sys
import time
import json
import threading
import subprocess
import tempfile
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml
import soundfile as sf
import sounddevice as sd
import mediapipe as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing.normalizer import KeypointNormalizer

# ── Config ───────────────────────────────────────
WINDOW_NAME    = "SignVoice — ASL to Speech Demo"
MIN_FRAMES     = 15
MAX_FRAMES     = 90
FPS_TARGET     = 15
SAMPLE_RATE    = 22050
STILL_THRESH   = 10
# ─────────────────────────────────────────────────

POSE_INDICES         = [11,12,13,14,15,16,23,24,25,26,27,28,0,1,2]
FACE_EMOTION_INDICES = [33,263,61,291,199,1]


# ── Classifier model ──────────────────────────────

class SignClassifier(nn.Module):
    """
    Lightweight transformer classifier.
    Input:  (B, T, 183) keypoints
    Output: (B, n_classes) logits
    """
    def __init__(self, input_dim=183, d_model=128,
                 n_heads=4, n_layers=2, n_classes=5):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        layer     = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=256, dropout=0.1,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.classifier  = nn.Linear(d_model, n_classes)
        self.dropout     = nn.Dropout(0.3)

    def forward(self, x, mask=None):
        x = self.proj(x)
        x = self.transformer(x, src_key_padding_mask=mask)
        x = x.mean(dim=1)
        return self.classifier(self.dropout(x))


# ── Keypoint extraction ───────────────────────────

def extract_frame_keypoints(results) -> np.ndarray:
    """Extract 183-dim keypoint vector from MediaPipe results."""

    def hand(lms):
        if lms is None:
            return np.zeros(63, dtype=np.float32)
        return np.array(
            [[l.x, l.y, l.z] for l in lms.landmark],
            dtype=np.float32
        ).flatten()

    def pose(lms):
        if lms is None:
            return np.zeros(45, dtype=np.float32)
        pts = []
        for idx in POSE_INDICES:
            l = lms.landmark[idx]
            pts.extend([l.x, l.y, l.z])
        return np.array(pts, dtype=np.float32)

    def face(lms):
        if lms is None:
            return np.zeros(12, dtype=np.float32)
        pts = []
        for idx in FACE_EMOTION_INDICES:
            l = lms.landmark[idx]
            pts.extend([l.x, l.y])
        return np.array(pts, dtype=np.float32)

    return np.concatenate([
        hand(results.left_hand_landmarks),
        hand(results.right_hand_landmarks),
        pose(results.pose_landmarks),
        face(results.face_landmarks),
    ])


# ── Audio ─────────────────────────────────────────

def speak_gloss(gloss: str) -> np.ndarray:
    """Convert gloss word to audio using gTTS. Returns wav array."""
    from gtts import gTTS

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        tmp_mp3 = f.name
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_wav = f.name

    try:
        tts = gTTS(text=gloss.lower(), lang="en", slow=False)
        tts.save(tmp_mp3)
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_mp3,
             "-ar", str(SAMPLE_RATE), tmp_wav],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        wav, _ = sf.read(tmp_wav)
        return wav.astype(np.float32)
    finally:
        for p in (tmp_mp3, tmp_wav):
            if os.path.exists(p):
                os.unlink(p)


def play_audio(wav: np.ndarray):
    """Play audio through speakers in background thread."""
    def _play():
        sd.play(wav, SAMPLE_RATE)
        sd.wait()
    threading.Thread(target=_play, daemon=True).start()


# ── Sign detector ─────────────────────────────────

class SignDetector:
    """Detects when signing stops based on hand stillness."""

    def __init__(self):
        self.prev_pos    = None
        self.still_count = 0

    def update(self, keypoints: np.ndarray) -> bool:
        """Returns True when sign is complete (hands stopped)."""
        curr_pos = keypoints[:4]

        if self.prev_pos is None:
            self.prev_pos = curr_pos
            return False

        motion         = np.mean(np.abs(curr_pos - self.prev_pos))
        self.prev_pos  = curr_pos

        if motion < 0.005:
            self.still_count += 1
        else:
            self.still_count = 0

        return self.still_count == STILL_THRESH

    def reset(self):
        self.prev_pos    = None
        self.still_count = 0


# ── UI drawing ────────────────────────────────────

def draw_ui(frame, status, buffer_len,
            last_prediction, fps, hands_detected):
    """Draw overlay UI on webcam frame."""
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h-140), (w, h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

    color = (0, 255, 100) if hands_detected else (100, 100, 100)

    cv2.putText(frame, f"Status: {status}",
                (15, h-110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(frame, f"Buffer: {buffer_len} / {MAX_FRAMES} frames",
                (15, h-82), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.putText(frame, f"Last: {last_prediction}",
                (15, h-52), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 100), 2)
    cv2.putText(frame, f"FPS:{fps:.0f}  SPACE=predict  R=reset  Q=quit",
                (15, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    # Buffer progress bar
    bar_w = int((buffer_len / MAX_FRAMES) * (w - 30))
    cv2.rectangle(frame, (15, h-95), (w-15, h-87), (40, 40, 40), -1)
    bar_color = (0, 200, 100) if buffer_len < MAX_FRAMES * 0.8 else (0, 140, 255)
    cv2.rectangle(frame, (15, h-95), (15 + bar_w, h-87), bar_color, -1)

    # Hand indicator dot
    dot_color = (0, 255, 100) if hands_detected else (60, 60, 60)
    cv2.circle(frame, (w-25, 25), 14, dot_color, -1)
    cv2.putText(frame, "hands", (w-70, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)

    return frame


# ── Main ──────────────────────────────────────────

def main():
    print("=" * 55)
    print("  SignVoice — Real-Time ASL to Speech Demo")
    print("=" * 55)

    # Load config
    with open("configs/lightweight.yaml") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device : {device}")

    # Load normalizer
    normalizer = KeypointNormalizer(config["data"]["stats_path"])
    normalizer.load()
    print(f"  Normalizer : loaded")

    # Load classifier checkpoint
    ckpt_path = "checkpoints/classifier.pt"
    if not os.path.exists(ckpt_path):
        print(f"\nERROR: {ckpt_path} not found.")
        print("Run first: python scripts/test_inference.py")
        print("That trains and saves the classifier.")
        sys.exit(1)

    ckpt    = torch.load(ckpt_path, map_location=device)
    glosses = ckpt["glosses"]
    print(f"  Glosses    : {glosses}")

    model = SignClassifier(n_classes=len(glosses)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"  Classifier : loaded ({len(glosses)} classes)")

    # Pre-generate audio for each gloss (faster at runtime)
    print(f"\n  Pre-generating audio for {len(glosses)} glosses...")
    gloss_audio = {}
    for g in glosses:
        try:
            gloss_audio[g] = speak_gloss(g)
            print(f"    '{g}' : OK")
        except Exception as e:
            print(f"    '{g}' : FAILED — {e}")

    # MediaPipe
    mp_holistic = mp.solutions.holistic
    mp_drawing  = mp.solutions.drawing_utils
    holistic    = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    detector = SignDetector()

    # Webcam
    print(f"\n  Opening webcam...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)

    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        sys.exit(1)

    print(f"  Webcam     : opened")
    print(f"\n  Signs this model knows: {glosses}")
    print(f"\n  READY — Start signing!")
    print(f"  Controls: SPACE=predict now  R=reset  Q=quit")
    print("=" * 55)

    # State
    kp_buffer       = []
    last_prediction = "waiting..."
    status          = "waiting for hands..."
    fps             = 0.0
    prev_time       = time.time()
    is_predicting   = False

    os.makedirs("outputs", exist_ok=True)

    def run_prediction(buf_copy):
        """Classify sign and play audio."""
        nonlocal last_prediction, is_predicting, kp_buffer

        try:
            kp   = normalizer.normalize(buf_copy)
            kp_t = torch.from_numpy(kp).unsqueeze(0).to(device)

            with torch.no_grad():
                logits   = model(kp_t)
                probs    = torch.softmax(logits, dim=1)
                pred_idx = logits.argmax(dim=1).item()
                conf     = probs[0, pred_idx].item()

            gloss = glosses[pred_idx]

            # Play pre-generated audio
            if gloss in gloss_audio:
                play_audio(gloss_audio[gloss])

            last_prediction = f"{gloss}  ({conf:.0%})"
            print(f"  Predicted: '{gloss}'  confidence={conf:.0%}  frames={len(buf_copy)}")

            # Save audio
            out_path = f"outputs/live_{int(time.time())}_{gloss}.wav"
            if gloss in gloss_audio:
                sf.write(out_path, gloss_audio[gloss], SAMPLE_RATE)

        except Exception as e:
            last_prediction = f"error: {str(e)[:25]}"
            print(f"  Prediction error: {e}")
        finally:
            kp_buffer     = []
            detector.reset()
            is_predicting = False

    # ── Main loop ─────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        curr_time = time.time()
        fps       = 0.9 * fps + 0.1 * (1.0 / max(curr_time - prev_time, 1e-6))
        prev_time = curr_time

        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        hands_detected = (
            results.left_hand_landmarks  is not None or
            results.right_hand_landmarks is not None
        )

        kp = extract_frame_keypoints(results)

        if hands_detected:
            kp_buffer.append(kp)
            if len(kp_buffer) > MAX_FRAMES:
                kp_buffer.pop(0)
            status = f"signing... ({len(kp_buffer)} frames)"

            # Auto trigger when hands stop moving
            if (detector.update(kp) and
                    len(kp_buffer) >= MIN_FRAMES and
                    not is_predicting):
                status        = "predicting..."
                is_predicting = True
                buf_copy      = np.array(kp_buffer.copy())
                threading.Thread(
                    target=run_prediction,
                    args=(buf_copy,),
                    daemon=True
                ).start()
        else:
            status = "waiting for hands..."
            detector.reset()

        # Draw hand landmarks
        mp_drawing.draw_landmarks(
            frame,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 200, 100), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(0, 150, 80),  thickness=2),
        )
        mp_drawing.draw_landmarks(
            frame,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(100, 180, 255), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(80, 140, 200),  thickness=2),
        )
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(200, 200, 200), thickness=1, circle_radius=2),
            mp_drawing.DrawingSpec(color=(150, 150, 150), thickness=1),
        )

        frame = draw_ui(
            frame, status, len(kp_buffer),
            last_prediction, fps, hands_detected
        )

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("\nQuitting...")
            break

        elif key == ord('r'):
            kp_buffer       = []
            detector.reset()
            last_prediction = "reset"
            print("  Buffer reset")

        elif (key == ord(' ') and
              len(kp_buffer) >= MIN_FRAMES and
              not is_predicting):
            status        = "predicting..."
            is_predicting = True
            buf_copy      = np.array(kp_buffer.copy())
            threading.Thread(
                target=run_prediction,
                args=(buf_copy,),
                daemon=True
            ).start()

    cap.release()
    cv2.destroyAllWindows()
    holistic.close()
    print("Demo closed.")


if __name__ == "__main__":
    main()