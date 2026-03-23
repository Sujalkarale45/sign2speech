"""
scripts/realtime_demo.py
Real-time ASL to Speech demo using webcam.
Auto-detects model architecture from checkpoint.

Controls:
    SPACE  → manually trigger prediction
    R      → reset buffer
    Q      → quit
"""

import os
import sys
import time
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

# ── Config ────────────────────────────────────────
WINDOW_NAME  = "SignVoice — ASL to Speech Demo"
MIN_FRAMES   = 15
MAX_FRAMES   = 90
FPS_TARGET   = 15
SAMPLE_RATE  = 22050
STILL_THRESH = 10

POSE_INDICES         = [11,12,13,14,15,16,23,24,25,26,27,28,0,1,2]
FACE_EMOTION_INDICES = [33,263,61,291,199,1]


# ── Classifier — auto config from checkpoint ──────

class SignClassifier(nn.Module):
    def __init__(self, input_dim=183, d_model=64,
                 n_heads=2, n_layers=1, n_classes=10):
        super().__init__()
        # Ensure n_heads divides d_model
        while d_model % n_heads != 0 and n_heads > 1:
            n_heads -= 1

        self.proj = nn.Linear(input_dim, d_model)
        self.bn   = nn.BatchNorm1d(d_model)
        layer     = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=0.5,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=n_layers
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(d_model // 2, n_classes),
        )

    def forward(self, x, mask=None):
        B, T, F = x.shape
        x = self.proj(x)
        x = self.bn(x.reshape(B * T, -1)).reshape(B, T, -1)
        x = self.transformer(x, src_key_padding_mask=mask)
        x = x.mean(dim=1)
        return self.classifier(x)


def load_classifier(ckpt_path: str, device: str):
    """
    Load classifier and auto-detect architecture from checkpoint.
    Never fails due to size mismatch.
    """
    ckpt    = torch.load(ckpt_path, map_location=device,
                         weights_only=False)
    glosses = ckpt["glosses"]
    state   = ckpt["model"]

    # Auto-detect config from saved weights
    d_model   = int(state["proj.bias"].shape[0])
    n_layers  = sum(1 for k in state
                    if "self_attn.in_proj_bias" in k)
    n_classes = int(state[
        [k for k in state if k.endswith("classifier.4.bias")
         or k.endswith("classifier.4.weight")][0]
    ].shape[0])

    # Detect n_heads from attention weight shape
    attn_w  = state["transformer.layers.0.self_attn.in_proj_weight"]
    n_heads = 2
    for h in [4, 2, 1]:
        if d_model % h == 0:
            n_heads = h
            break

    print(f"  Model config : d_model={d_model} "
          f"n_layers={n_layers} "
          f"n_heads={n_heads} "
          f"n_classes={n_classes}")

    model = SignClassifier(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        n_classes=n_classes,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, glosses


# ── Keypoint extraction ───────────────────────────

def extract_frame_keypoints(results) -> np.ndarray:
    def hand(lms):
        if lms is None:
            return np.zeros(63, dtype=np.float32)
        pts = []
        for lm in lms.landmark:
            pts.extend([lm.x, lm.y, lm.z])
        return np.array(pts, dtype=np.float32)

    def pose(lms):
        if lms is None:
            return np.zeros(45, dtype=np.float32)
        pts = []
        for idx in POSE_INDICES:
            lm = lms.landmark[idx]
            pts.extend([lm.x, lm.y, lm.z])
        return np.array(pts, dtype=np.float32)

    def face(lms):
        if lms is None:
            return np.zeros(12, dtype=np.float32)
        pts = []
        for idx in FACE_EMOTION_INDICES:
            lm = lms.landmark[idx]
            pts.extend([lm.x, lm.y])
        return np.array(pts, dtype=np.float32)

    kp = np.concatenate([
        hand(results.left_hand_landmarks),
        hand(results.right_hand_landmarks),
        pose(results.pose_landmarks),
        face(results.face_landmarks),
    ])
    # Replace NaN with 0
    return np.nan_to_num(kp, nan=0.0, posinf=0.0, neginf=0.0)


# ── Audio ─────────────────────────────────────────

def speak_system(gloss: str):
    """Speak using Windows system TTS."""
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty('rate', 140)
        engine.setProperty('volume', 1.0)
        voices = engine.getProperty('voices')
        if voices:
            engine.setProperty('voice', voices[0].id)
        engine.say(gloss)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        print(f"  TTS error: {e}")
        speak_gtts(gloss)


def speak_gtts(gloss: str):
    """Fallback: gTTS speech."""
    from gtts import gTTS
    tmp_mp3 = tempfile.mktemp(suffix=".mp3")
    tmp_wav = tempfile.mktemp(suffix=".wav")
    try:
        gTTS(text=gloss.lower(), lang="en", slow=False).save(tmp_mp3)
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_mp3,
             "-ar", str(SAMPLE_RATE),
             "-ac", "1",
             "-af", "volume=2.0",
             tmp_wav],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        wav, sr = sf.read(tmp_wav)
        sd.play(wav.astype(np.float32), samplerate=sr)
        sd.wait()
    finally:
        for p in [tmp_mp3, tmp_wav]:
            try:
                os.unlink(p)
            except Exception:
                pass


def play_gloss(gloss: str):
    """Play speech in background thread."""
    threading.Thread(
        target=speak_system,
        args=(gloss,),
        daemon=True
    ).start()


# ── Sign detector ─────────────────────────────────

class SignDetector:
    def __init__(self):
        self.prev_pos    = None
        self.still_count = 0

    def update(self, keypoints: np.ndarray) -> bool:
        curr_pos = keypoints[:4].copy()
        if self.prev_pos is None:
            self.prev_pos = curr_pos
            return False
        motion        = float(np.mean(np.abs(curr_pos - self.prev_pos)))
        self.prev_pos = curr_pos
        if motion < 0.005:
            self.still_count += 1
        else:
            self.still_count = 0
        return self.still_count == STILL_THRESH

    def reset(self):
        self.prev_pos    = None
        self.still_count = 0


# ── UI ────────────────────────────────────────────

def draw_ui(frame, status, buffer_len,
            last_pred, fps, hands_detected, glosses):
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h-160), (w, h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

    color = (0, 255, 100) if hands_detected else (100, 100, 100)

    cv2.putText(frame, f"Status: {status}",
                (15, h-130), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2)

    cv2.putText(frame, f"Buffer: {buffer_len}/{MAX_FRAMES}",
                (15, h-100), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (200, 200, 200), 1)

    # Last prediction — big yellow text
    cv2.putText(frame, f"{last_pred}",
                (15, h-58), cv2.FONT_HERSHEY_SIMPLEX,
                1.4, (0, 255, 255), 3)

    # Signs known
    signs_text = "Signs: " + " | ".join(glosses)
    cv2.putText(frame, signs_text,
                (15, h-25), cv2.FONT_HERSHEY_SIMPLEX,
                0.38, (180, 180, 180), 1)

    cv2.putText(frame, "SPACE=predict  R=reset  Q=quit",
                (w-270, h-8), cv2.FONT_HERSHEY_SIMPLEX,
                0.38, (120, 120, 120), 1)

    # Progress bar
    bar_w = int((min(buffer_len, MAX_FRAMES) / MAX_FRAMES) * (w - 30))
    cv2.rectangle(frame, (15, h-113), (w-15, h-105),
                  (40, 40, 40), -1)
    bar_c = (0, 200, 100) if buffer_len < MAX_FRAMES * 0.8 \
            else (0, 140, 255)
    if bar_w > 0:
        cv2.rectangle(frame, (15, h-113),
                      (15 + bar_w, h-105), bar_c, -1)

    # Hand dot
    dot_c = (0, 255, 100) if hands_detected else (60, 60, 60)
    cv2.circle(frame, (w-25, 25), 14, dot_c, -1)
    cv2.putText(frame, "hands", (w-72, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38, (160, 160, 160), 1)

    # FPS
    cv2.putText(frame, f"FPS:{fps:.0f}",
                (15, h-8), cv2.FONT_HERSHEY_SIMPLEX,
                0.38, (100, 100, 100), 1)

    return frame


# ── Main ──────────────────────────────────────────

def main():
    print("=" * 55)
    print("  SignVoice — Real-Time ASL to Speech Demo")
    print("=" * 55)

    # Install pyttsx3
    try:
        import pyttsx3
    except ImportError:
        print("  Installing pyttsx3...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "pyttsx3", "-q"],
            check=True
        )

    # Load config
    with open("configs/lightweight.yaml") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device     : {device}")

    # Load normalizer
    normalizer = KeypointNormalizer(config["data"]["stats_path"])
    normalizer.load()
    print(f"  Normalizer : loaded")

    # Load classifier — auto detects architecture
    ckpt_path = "checkpoints/classifier.pt"
    if not os.path.exists(ckpt_path):
        print(f"\nERROR: {ckpt_path} not found.")
        print("Run: python scripts/test_inference.py")
        sys.exit(1)

    model, glosses = load_classifier(ckpt_path, device)
    print(f"  Glosses    : {glosses}")
    print(f"  Classes    : {len(glosses)}")

    # Test audio
    print(f"\n  Testing audio...")
    play_gloss("sign voice ready")
    time.sleep(2)

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
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)

    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        sys.exit(1)

    print(f"  Webcam     : opened")
    print(f"\n  Signs known: {glosses}")
    print(f"  READY — Start signing!")
    print(f"  SPACE=predict  R=reset  Q=quit")
    print("=" * 55)

    kp_buffer     = []
    last_pred     = "waiting..."
    status        = "waiting for hands..."
    fps           = 0.0
    prev_time     = time.time()
    is_predicting = False

    os.makedirs("outputs", exist_ok=True)

    def run_prediction(buf_copy):
        nonlocal last_pred, is_predicting, kp_buffer
        try:
            kp   = normalizer.normalize(buf_copy)
            kp   = np.nan_to_num(kp, nan=0.0,
                                  posinf=0.0, neginf=0.0)
            kp_t = torch.from_numpy(kp).unsqueeze(0).to(device)

            with torch.no_grad():
                logits   = model(kp_t)
                probs    = torch.softmax(logits, dim=1)
                pred_idx = int(logits.argmax(dim=1).item())
                conf     = float(probs[0, pred_idx].item())

            gloss = glosses[pred_idx]
            print(f"  → '{gloss}'  conf={conf:.0%}  "
                  f"frames={len(buf_copy)}")

            play_gloss(gloss)
            last_pred = f"{gloss}  ({conf:.0%})"

            # Save audio
            try:
                from gtts import gTTS
                tmp_mp3 = tempfile.mktemp(suffix=".mp3")
                out_wav = (f"outputs/live_"
                           f"{int(time.time())}_{gloss}.wav")
                gTTS(text=gloss.lower(),
                     lang="en", slow=False).save(tmp_mp3)
                subprocess.run(
                    ["ffmpeg", "-y", "-i", tmp_mp3,
                     "-ar", str(SAMPLE_RATE),
                     "-ac", "1", out_wav],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                os.unlink(tmp_mp3)
            except Exception:
                pass

        except Exception as e:
            last_pred = f"error: {str(e)[:20]}"
            print(f"  Error: {e}")
        finally:
            kp_buffer     = []
            detector.reset()
            is_predicting = False

    # ── Main webcam loop ──────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        now       = time.time()
        fps       = 0.9 * fps + 0.1 / max(now - prev_time, 1e-6)
        prev_time = now

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

            if (detector.update(kp) and
                    len(kp_buffer) >= MIN_FRAMES and
                    not is_predicting):
                status        = "predicting..."
                is_predicting = True
                buf_copy      = np.array(kp_buffer.copy())
                threading.Thread(
                    target=run_prediction,
                    args=(buf_copy,),
                    daemon=True,
                ).start()
        else:
            status = "waiting for hands..."
            detector.reset()

        # Draw landmarks
        mp_drawing.draw_landmarks(
            frame,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(
                color=(0, 200, 100),
                thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(
                color=(0, 150, 80), thickness=2),
        )
        mp_drawing.draw_landmarks(
            frame,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(
                color=(100, 180, 255),
                thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(
                color=(80, 140, 200), thickness=2),
        )
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(
                color=(200, 200, 200),
                thickness=1, circle_radius=2),
            mp_drawing.DrawingSpec(
                color=(150, 150, 150), thickness=1),
        )

        frame = draw_ui(
            frame, status, len(kp_buffer),
            last_pred, fps, hands_detected, glosses
        )

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("\nQuitting...")
            break

        elif key == ord('r'):
            kp_buffer  = []
            detector.reset()
            last_pred  = "reset"
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
                daemon=True,
            ).start()

    cap.release()
    cv2.destroyAllWindows()
    holistic.close()
    print("Demo closed.")


if __name__ == "__main__":
    main()