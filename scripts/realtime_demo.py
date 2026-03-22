"""
scripts/realtime_demo.py
Real-time ASL to Speech — 10 sign demo.
Uses pyttsx3 system voice for clear audio.

Controls:
    SPACE → predict
    R     → reset
    Q     → quit
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

WINDOW_NAME   = "SignVoice — ASL to Speech Demo"
MIN_FRAMES    = 15
MAX_FRAMES    = 90
SAMPLE_RATE   = 22050
STILL_THRESH  = 10

POSE_INDICES         = [11,12,13,14,15,16,23,24,25,26,27,28,0,1,2]
FACE_EMOTION_INDICES = [33,263,61,291,199,1]


class SignClassifier(nn.Module):
    def __init__(self, input_dim=183, d_model=128,
                 n_heads=4, n_layers=2, n_classes=10):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.bn   = nn.BatchNorm1d(d_model)
        layer     = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=256, dropout=0.3,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=n_layers
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_classes),
        )

    def forward(self, x, mask=None):
        B, T, F = x.shape
        x = self.proj(x)
        x = self.bn(x.reshape(B * T, -1)).reshape(B, T, -1)
        x = self.transformer(x, src_key_padding_mask=mask)
        x = x.mean(dim=1)
        return self.classifier(x)


def extract_frame_keypoints(results):
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

    return np.concatenate([
        hand(results.left_hand_landmarks),
        hand(results.right_hand_landmarks),
        pose(results.pose_landmarks),
        face(results.face_landmarks),
    ])


def speak(gloss: str):
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


def play_async(gloss: str):
    """Play speech in background thread."""
    threading.Thread(
        target=speak, args=(gloss,), daemon=True
    ).start()


class SignDetector:
    def __init__(self):
        self.prev_pos    = None
        self.still_count = 0

    def update(self, keypoints):
        curr = keypoints[:4].copy()
        if self.prev_pos is None:
            self.prev_pos = curr
            return False
        motion        = float(np.mean(np.abs(curr - self.prev_pos)))
        self.prev_pos = curr
        if motion < 0.005:
            self.still_count += 1
        else:
            self.still_count = 0
        return self.still_count == STILL_THRESH

    def reset(self):
        self.prev_pos    = None
        self.still_count = 0


def draw_ui(frame, status, buf_len, last_pred,
            fps, hands_detected, glosses):
    h, w = frame.shape[:2]

    ov = frame.copy()
    cv2.rectangle(ov, (0, h-160), (w, h), (0, 0, 0), -1)
    frame = cv2.addWeighted(ov, 0.7, frame, 0.3, 0)

    color = (0, 255, 100) if hands_detected else (100, 100, 100)
    cv2.putText(frame, f"Status: {status}",
                (15, h-130), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2)
    cv2.putText(frame, f"Buffer: {buf_len}/{MAX_FRAMES}",
                (15, h-100), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (180, 180, 180), 1)

    # Big prediction text
    cv2.putText(frame, last_pred,
                (15, h-58), cv2.FONT_HERSHEY_SIMPLEX,
                1.4, (0, 255, 255), 3)

    # Signs list
    signs_str = "  ".join(glosses)
    cv2.putText(frame, signs_str[:60],
                (15, h-28), cv2.FONT_HERSHEY_SIMPLEX,
                0.38, (150, 150, 150), 1)

    cv2.putText(frame, "SPACE=predict  R=reset  Q=quit",
                (15, h-10), cv2.FONT_HERSHEY_SIMPLEX,
                0.38, (120, 120, 120), 1)

    # Progress bar
    bar_w = int((min(buf_len, MAX_FRAMES) / MAX_FRAMES) * (w-30))
    cv2.rectangle(frame, (15, h-112), (w-15, h-104),
                  (40, 40, 40), -1)
    if bar_w > 0:
        bar_c = (0, 200, 100) if buf_len < MAX_FRAMES*0.8 \
                else (0, 140, 255)
        cv2.rectangle(frame, (15, h-112),
                      (15+bar_w, h-104), bar_c, -1)

    # Hand dot
    dot_c = (0, 255, 100) if hands_detected else (60, 60, 60)
    cv2.circle(frame, (w-25, 25), 14, dot_c, -1)

    cv2.putText(frame, f"FPS:{fps:.0f}",
                (w-65, h-10), cv2.FONT_HERSHEY_SIMPLEX,
                0.38, (120, 120, 120), 1)

    return frame


def main():
    print("=" * 55)
    print("  SignVoice — 10-Sign ASL to Speech Demo")
    print("=" * 55)

    # Install pyttsx3
    try:
        import pyttsx3
    except ImportError:
        subprocess.run(
            [sys.executable, "-m", "pip",
             "install", "pyttsx3", "-q"],
            check=True
        )

    with open("configs/lightweight.yaml") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device : {device}")

    normalizer = KeypointNormalizer(config["data"]["stats_path"])
    normalizer.load()

    ckpt_path = "checkpoints/classifier.pt"
    if not os.path.exists(ckpt_path):
        print(f"\nERROR: {ckpt_path} not found.")
        print("Run: python scripts/test_inference.py")
        sys.exit(1)

    ckpt    = torch.load(ckpt_path, map_location=device,
                         weights_only=False)
    glosses = ckpt["glosses"]
    print(f"  Signs known : {glosses}")

    model = SignClassifier(n_classes=len(glosses)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"  Model loaded: {len(glosses)} classes")

    # Test audio
    print("\n  Testing audio...")
    play_async("sign voice ready")
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

    print("\n  Opening webcam...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)

    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        sys.exit(1)

    print("  READY!")
    print(f"  Signs: {glosses}")
    print("  SPACE=predict  R=reset  Q=quit")
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
            kp_t = torch.from_numpy(kp).unsqueeze(0).to(device)

            with torch.no_grad():
                logits   = model(kp_t)
                probs    = torch.softmax(logits, dim=1)
                pred_idx = int(logits.argmax(dim=1).item())
                conf     = float(probs[0, pred_idx].item())

            gloss = glosses[pred_idx]
            print(f"  → '{gloss}'  conf={conf:.0%}  "
                  f"frames={len(buf_copy)}")

            play_async(gloss)
            last_pred = f"{gloss}  ({conf:.0%})"

            # Save audio
            out = f"outputs/live_{int(time.time())}_{gloss}.wav"
            try:
                from gtts import gTTS
                tmp = tempfile.mktemp(suffix=".mp3")
                tts = gTTS(text=gloss, lang="en", slow=False)
                tts.save(tmp)
                subprocess.run(
                    ["ffmpeg", "-y", "-i", tmp,
                     "-ar", "22050", out],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                os.unlink(tmp)
            except Exception:
                pass

        except Exception as e:
            last_pred = f"error: {str(e)[:20]}"
            print(f"  Error: {e}")
        finally:
            kp_buffer     = []
            detector.reset()
            is_predicting = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        now       = time.time()
        fps       = 0.9*fps + 0.1/max(now-prev_time, 1e-6)
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

        mp_drawing.draw_landmarks(
            frame,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(
                color=(0,200,100), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(
                color=(0,150,80), thickness=2),
        )
        mp_drawing.draw_landmarks(
            frame,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(
                color=(100,180,255), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(
                color=(80,140,200), thickness=2),
        )
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(
                color=(200,200,200), thickness=1, circle_radius=2),
            mp_drawing.DrawingSpec(
                color=(150,150,150), thickness=1),
        )

        frame = draw_ui(
            frame, status, len(kp_buffer),
            last_pred, fps, hands_detected, glosses
        )
        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            kp_buffer  = []
            detector.reset()
            last_pred  = "reset"
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