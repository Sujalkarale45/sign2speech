"""
gradio_app.py
SignVoice — ASL to Speech via Gradio with webcam
Run: python gradio_app.py
"""

import gradio as gr
import torch
import torch.nn as nn
import numpy as np
import yaml
import tempfile
import subprocess
import os
import sys
import cv2
import threading
from pathlib import Path

sys.path.insert(0, '.')
from src.preprocessing.normalizer import KeypointNormalizer


# ── Model ─────────────────────────────────────────

class SignClassifier(nn.Module):
    def __init__(self, input_dim=183, d_model=128,
                 n_heads=4, n_layers=2, n_classes=10):
        super().__init__()
        while d_model % n_heads != 0 and n_heads > 1:
            n_heads -= 1
        self.proj = nn.Linear(input_dim, d_model)
        self.bn   = nn.BatchNorm1d(d_model)
        layer     = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=0.3, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=n_layers
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
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


# ── Load model ────────────────────────────────────

print("Loading model...")
device = "cuda" if torch.cuda.is_available() else "cpu"

with open("configs/lightweight.yaml") as f:
    config = yaml.safe_load(f)

normalizer = KeypointNormalizer(config["data"]["stats_path"])
normalizer.load()

ckpt    = torch.load("checkpoints/classifier.pt",
                     map_location=device, weights_only=False)
glosses = ckpt["glosses"]
state   = ckpt["model"]
d_model   = int(state["proj.bias"].shape[0])
n_layers  = sum(1 for k in state
                if "self_attn.in_proj_bias" in k)
n_classes = int(state[
    [k for k in state
     if "classifier.4" in k and "bias" in k][0]
].shape[0])

model = SignClassifier(
    d_model=d_model,
    n_layers=n_layers,
    n_classes=n_classes,
).to(device)
model.load_state_dict(state)
model.eval()
print(f"Model loaded | Signs: {glosses} | Device: {device}")


# ── MediaPipe ─────────────────────────────────────

import mediapipe as mp
mp_holistic = mp.solutions.holistic
mp_drawing  = mp.solutions.drawing_utils
holistic    = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

POSE_IDX = [11,12,13,14,15,16,23,24,25,26,27,28,0,1,2]
FACE_IDX = [33,263,61,291,199,1]


# ── Keypoint extraction ───────────────────────────

def extract_keypoints(frame):
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(rgb)

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
        for idx in POSE_IDX:
            lm = lms.landmark[idx]
            pts.extend([lm.x, lm.y, lm.z])
        return np.array(pts, dtype=np.float32)

    def face(lms):
        if lms is None:
            return np.zeros(12, dtype=np.float32)
        pts = []
        for idx in FACE_IDX:
            lm = lms.landmark[idx]
            pts.extend([lm.x, lm.y])
        return np.array(pts, dtype=np.float32)

    kp = np.concatenate([
        hand(results.left_hand_landmarks),
        hand(results.right_hand_landmarks),
        pose(results.pose_landmarks),
        face(results.face_landmarks),
    ])
    return np.nan_to_num(kp, nan=0.0), results


# ── Audio ─────────────────────────────────────────

def gloss_to_audio(gloss: str) -> str:
    from gtts import gTTS
    tmp_mp3 = tempfile.mktemp(suffix=".mp3")
    out_wav = tempfile.mktemp(suffix=".wav")
    try:
        gTTS(text=gloss.lower(), lang="en",
             slow=False).save(tmp_mp3)
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_mp3,
             "-ar", "22050", "-ac", "1", out_wav],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        os.unlink(tmp_mp3)
        return out_wav
    except Exception as e:
        print(f"Audio error: {e}")
        return None


# ── Global state ──────────────────────────────────

kp_buffer     = []
last_pred     = "waiting..."
last_conf     = 0.0
still_count   = 0
prev_pos      = None
is_predicting = False


def reset_buffer():
    global kp_buffer, still_count, prev_pos
    global last_pred, last_conf, is_predicting
    kp_buffer     = []
    still_count   = 0
    prev_pos      = None
    last_pred     = "waiting..."
    last_conf     = 0.0
    is_predicting = False
    return "Buffer reset ✓"


def run_prediction():
    global kp_buffer, last_pred, last_conf, is_predicting
    if len(kp_buffer) < 15:
        return
    is_predicting = True
    try:
        kp   = np.stack(kp_buffer.copy(), axis=0)
        kp   = normalizer.normalize(kp)
        kp   = np.nan_to_num(kp, nan=0.0)
        kp_t = torch.from_numpy(kp).unsqueeze(0).to(device)

        with torch.no_grad():
            logits   = model(kp_t)
            probs    = torch.softmax(logits, dim=1)
            pred_idx = logits.argmax(dim=1).item()
            conf     = float(probs[0, pred_idx].item())

        last_pred = glosses[pred_idx]
        last_conf = conf
        print(f"  → '{last_pred}' conf={conf:.0%}")

        # Speak
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('rate', 140)
            engine.setProperty('volume', 1.0)
            engine.say(last_pred)
            engine.runAndWait()
            engine.stop()
        except Exception:
            pass

    except Exception as e:
        print(f"Prediction error: {e}")
    finally:
        kp_buffer     = []
        is_predicting = False


# ── Frame processing ──────────────────────────────

def process_frame(frame):
    """Process each webcam frame."""
    global kp_buffer, still_count, prev_pos, is_predicting

    if frame is None:
        return None, "No frame", "—"

    # Convert RGB (Gradio) to BGR (OpenCV)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    kp, results = extract_keypoints(frame_bgr)

    hands_detected = (
        results.left_hand_landmarks  is not None or
        results.right_hand_landmarks is not None
    )

    if hands_detected:
        kp_buffer.append(kp)
        if len(kp_buffer) > 90:
            kp_buffer.pop(0)

        curr_pos = kp[:4]
        if prev_pos is not None:
            motion = float(np.mean(np.abs(curr_pos - prev_pos)))
            if motion < 0.005:
                still_count += 1
            else:
                still_count = 0
        prev_pos = curr_pos.copy()

        if (still_count == 10 and
                len(kp_buffer) >= 15 and
                not is_predicting):
            threading.Thread(
                target=run_prediction, daemon=True
            ).start()
            still_count = 0
    else:
        still_count = 0

    # Draw landmarks
    annotated = frame_bgr.copy()

    mp_drawing.draw_landmarks(
        annotated,
        results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(
            color=(0, 200, 100), thickness=2, circle_radius=3),
        mp_drawing.DrawingSpec(
            color=(0, 150, 80), thickness=2),
    )
    mp_drawing.draw_landmarks(
        annotated,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(
            color=(100, 180, 255), thickness=2, circle_radius=3),
        mp_drawing.DrawingSpec(
            color=(80, 140, 200), thickness=2),
    )
    mp_drawing.draw_landmarks(
        annotated,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(
            color=(200, 200, 200), thickness=1, circle_radius=2),
        mp_drawing.DrawingSpec(
            color=(150, 150, 150), thickness=1),
    )

    # Status text
    status = "Signing..." if hands_detected else "Show hands"
    color  = (0, 255, 100) if hands_detected else (80, 80, 80)
    cv2.putText(annotated, status,
                (15, 35), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, color, 2)

    buf = len(kp_buffer)
    cv2.putText(annotated, f"Buffer: {buf}/90",
                (15, 65), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (200, 200, 200), 1)

    # Show prediction on frame
    if last_pred not in ["waiting...", "—"]:
        h = annotated.shape[0]
        cv2.putText(annotated,
                    f"{last_pred.upper()} ({last_conf:.0%})",
                    (15, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 255, 255), 3)

    # Progress bar
    h, w = annotated.shape[:2]
    bar_w = int((buf / 90) * (w - 30))
    cv2.rectangle(annotated, (15, h-48),
                  (w-15, h-40), (40, 40, 40), -1)
    if bar_w > 0:
        cv2.rectangle(annotated, (15, h-48),
                      (15+bar_w, h-40), (0, 200, 100), -1)

    # Convert back to RGB for Gradio
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    pred_text = (f"## 🤟 {last_pred.upper()}"
                 if last_pred != "waiting..."
                 else "## Waiting...")
    conf_text = f"Confidence: **{last_conf:.0%}**"

    return annotated_rgb, pred_text, conf_text


def manual_predict(frame):
    """Manual predict button handler."""
    global is_predicting
    if len(kp_buffer) >= 15 and not is_predicting:
        run_prediction()

    audio = None
    if last_pred not in ["waiting...", "—"]:
        audio = gloss_to_audio(last_pred)

    pred_text = (f"## 🤟 {last_pred.upper()}"
                 if last_pred != "waiting..."
                 else "## Waiting...")
    conf_text = f"Confidence: **{last_conf:.0%}**"

    return pred_text, conf_text, audio


# ── Gradio UI ─────────────────────────────────────

with gr.Blocks(title="SignVoice — ASL to Speech") as demo:

    gr.Markdown("""
    # 🤟 SignVoice — ASL to Speech
    **Real-time sign language to speech — no text at any stage**
    """)

    with gr.Row():

        # Left — webcam input + annotated output
        with gr.Column(scale=2):
            webcam_input = gr.Image(
                sources=["webcam"],
                streaming=True,
                label="📹 Webcam — Start signing",
                type="numpy",
            )
            annotated_frame = gr.Image(
                label="🖼️ Detected Landmarks",
                type="numpy",
            )
            with gr.Row():
                predict_btn = gr.Button(
                    "🔍 Predict Now",
                    variant="primary",
                    size="lg",
                )
                reset_btn = gr.Button(
                    "🔄 Reset Buffer",
                    variant="secondary",
                    size="lg",
                )
            status_box = gr.Textbox(
                value="Ready — show hands to start",
                label="Status",
                interactive=False,
            )

        # Right — results
        with gr.Column(scale=1):
            pred_out = gr.Markdown("## Waiting...")
            conf_out = gr.Markdown("Confidence: —")

            audio_out = gr.Audio(
                label="🔊 Speech Output",
                autoplay=True,
                type="filepath",
            )

            gr.Markdown("---")
            gr.Markdown("## 🤟 Signs I Know")
            gr.Markdown(
                "\n".join([f"• **{g}**" for g in glosses])
            )

            gr.Markdown("---")
            gr.Markdown("""
            ## 📖 How to use
            1. Allow camera access above
            2. Perform an ASL sign clearly
            3. Hold still — auto predicts!
            4. Or press **Predict Now**

            **Signs:** """ +
            ", ".join(glosses)
            )

    # Stream handler
    webcam_input.stream(
        fn=process_frame,
        inputs=webcam_input,
        outputs=[annotated_frame, pred_out, conf_out],
        stream_every=0.1,
    )

    # Predict button
    predict_btn.click(
        fn=manual_predict,
        inputs=webcam_input,
        outputs=[pred_out, conf_out, audio_out],
    )

    # Reset button
    reset_btn.click(
        fn=reset_buffer,
        inputs=None,
        outputs=status_box,
    )

    gr.Markdown("""
    ---
    Built by **Sujal Karale** | B.Tech CSE (AI/ML) — RCOEM Nagpur |
    [GitHub](https://github.com/Sujalkarale45/sign2speech) |
    [LinkedIn](https://linkedin.com/in/sujal-karale/)
    """)


# ── Launch ────────────────────────────────────────

if __name__ == "__main__":
    demo.launch(
        share=True,
        server_port=7860,
        show_error=True,
        inbrowser=True,
        theme=gr.themes.Soft(),
    )