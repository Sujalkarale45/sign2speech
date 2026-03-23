"""
app.py
SignVoice web app using Streamlit.
Run: streamlit run app.py
"""

import streamlit as st
import cv2
import torch
import torch.nn as nn
import numpy as np
import tempfile
import subprocess
import os
import sys
import yaml
import soundfile as sf
from pathlib import Path

sys.path.insert(0, '.')
from src.preprocessing.normalizer import KeypointNormalizer

# ── Page config ───────────────────────────────────
st.set_page_config(
    page_title="SignVoice — ASL to Speech",
    page_icon="🤟",
    layout="wide",
)

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
            dim_feedforward=d_model * 2, dropout=0.3,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.classifier  = nn.Sequential(
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
        if mask is not None:
            valid = (~mask).float().unsqueeze(-1)
            x = (x * valid).sum(1) / valid.sum(1).clamp(min=1e-8)
        else:
            x = x.mean(dim=1)
        return self.classifier(x)


@st.cache_resource
def load_model():
    """Load model and normalizer once."""
    import mediapipe as mp

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
    n_layers  = sum(1 for k in state if "self_attn.in_proj_bias" in k)
    n_classes = int(state[[k for k in state
                            if "classifier.4" in k and "bias" in k][0]].shape[0])

    model = SignClassifier(
        d_model=d_model,
        n_layers=n_layers,
        n_classes=n_classes
    ).to(device)
    model.load_state_dict(state)
    model.eval()

    holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    return model, normalizer, glosses, device, holistic


def extract_keypoints(frame, holistic):
    """Extract MediaPipe keypoints from frame."""
    POSE_IDX = [11,12,13,14,15,16,23,24,25,26,27,28,0,1,2]
    FACE_IDX = [33,263,61,291,199,1]

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


def predict_sign(frames, model, normalizer, glosses, device):
    """Predict sign from collected frames."""
    kp_array = np.stack(frames, axis=0)
    kp       = normalizer.normalize(kp_array)
    kp       = np.nan_to_num(kp, nan=0.0)
    kp_t     = torch.from_numpy(kp).unsqueeze(0).to(device)

    with torch.no_grad():
        logits   = model(kp_t)
        probs    = torch.softmax(logits, dim=1)
        pred_idx = logits.argmax(dim=1).item()
        conf     = probs[0, pred_idx].item()

    return glosses[pred_idx], conf


def text_to_audio(gloss: str) -> bytes:
    """Convert gloss to audio bytes."""
    from gtts import gTTS
    tmp_mp3 = tempfile.mktemp(suffix=".mp3")
    tmp_wav = tempfile.mktemp(suffix=".wav")
    try:
        gTTS(text=gloss.lower(), lang="en", slow=False).save(tmp_mp3)
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_mp3,
             "-ar", "22050", "-ac", "1", tmp_wav],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        with open(tmp_wav, "rb") as f:
            return f.read()
    finally:
        for p in [tmp_mp3, tmp_wav]:
            try: os.unlink(p)
            except: pass


# ── UI ────────────────────────────────────────────

def main():
    # Header
    st.title("🤟 SignVoice — ASL to Speech")
    st.markdown(
        "**Direct sign language video to speech — no text at any stage.**"
    )
    st.divider()

    # Load model
    with st.spinner("Loading model..."):
        model, normalizer, glosses, device, holistic = load_model()

    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        **SignVoice** converts ASL signs directly
        to spoken audio using a Temporal Transformer.

        **Pipeline:**
```
        Webcam → MediaPipe → Transformer
        → Classifier → TTS → Speech
```

        **No text at any stage.**
        """)

        st.header("🤟 Signs Known")
        for g in glosses:
            st.write(f"• {g}")

        st.header("⚙️ Settings")
        min_frames  = st.slider("Min frames to collect", 10, 40, 20)
        conf_thresh = st.slider("Confidence threshold", 0.3, 0.9, 0.5)

    # Main columns
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📹 Webcam Feed")
        run_demo = st.checkbox("Start Webcam", value=False)
        frame_placeholder = st.empty()

    with col2:
        st.subheader("🔊 Prediction")
        pred_placeholder  = st.empty()
        conf_placeholder  = st.empty()
        audio_placeholder = st.empty()
        history_placeholder = st.empty()

        st.subheader("📊 Statistics")
        stats_placeholder = st.empty()

    # State
    if "kp_buffer"    not in st.session_state:
        st.session_state.kp_buffer    = []
    if "predictions"  not in st.session_state:
        st.session_state.predictions  = []
    if "total_preds"  not in st.session_state:
        st.session_state.total_preds  = 0
    if "prev_pos"     not in st.session_state:
        st.session_state.prev_pos     = None
    if "still_count"  not in st.session_state:
        st.session_state.still_count  = 0

    if run_demo:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        import mediapipe as mp
        mp_drawing = mp.solutions.drawing_utils
        mp_holistic = mp.solutions.holistic

        stop_button = st.button("⏹ Stop Webcam")

        while not stop_button:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            kp, results = extract_keypoints(frame, holistic)

            hands_detected = (
                results.left_hand_landmarks  is not None or
                results.right_hand_landmarks is not None
            )

            if hands_detected:
                st.session_state.kp_buffer.append(kp)
                if len(st.session_state.kp_buffer) > 90:
                    st.session_state.kp_buffer.pop(0)

                # Motion detection
                curr_pos = kp[:4]
                if st.session_state.prev_pos is not None:
                    motion = float(np.mean(np.abs(
                        curr_pos - st.session_state.prev_pos
                    )))
                    if motion < 0.005:
                        st.session_state.still_count += 1
                    else:
                        st.session_state.still_count = 0
                st.session_state.prev_pos = curr_pos

                # Auto predict when hands stop
                if (st.session_state.still_count == 10 and
                        len(st.session_state.kp_buffer) >= min_frames):

                    try:
                        gloss, conf = predict_sign(
                            st.session_state.kp_buffer,
                            model, normalizer, glosses, device
                        )

                        if conf >= conf_thresh:
                            st.session_state.predictions.append({
                                "gloss": gloss,
                                "conf" : conf,
                            })
                            st.session_state.total_preds += 1

                            # Show prediction
                            pred_placeholder.markdown(
                                f"### 🤟 **{gloss.upper()}**"
                            )
                            conf_placeholder.progress(
                                conf,
                                text=f"Confidence: {conf:.0%}"
                            )

                            # Generate and play audio
                            audio_bytes = text_to_audio(gloss)
                            audio_placeholder.audio(
                                audio_bytes,
                                format="audio/wav",
                                autoplay=True,
                            )

                    except Exception as e:
                        st.error(f"Prediction error: {e}")

                    # Reset buffer
                    st.session_state.kp_buffer   = []
                    st.session_state.still_count = 0

            else:
                st.session_state.still_count = 0
                st.session_state.prev_pos    = None

            # Draw landmarks on frame
            mp_drawing.draw_landmarks(
                frame,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(
                    color=(0, 200, 100), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(
                    color=(0, 150, 80), thickness=2),
            )
            mp_drawing.draw_landmarks(
                frame,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(
                    color=(100, 180, 255), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(
                    color=(80, 140, 200), thickness=2),
            )

            # Add status overlay
            status = "Signing..." if hands_detected else "Waiting for hands..."
            cv2.putText(frame, status,
                        (15, 35), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 100) if hands_detected
                        else (100, 100, 100), 2)

            buf_len = len(st.session_state.kp_buffer)
            cv2.putText(frame, f"Buffer: {buf_len}/90",
                        (15, 65), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (200, 200, 200), 1)

            # Show frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(
                frame_rgb, channels="RGB", use_column_width=True
            )

            # Show prediction history
            if st.session_state.predictions:
                history = "\n".join([
                    f"• **{p['gloss']}** ({p['conf']:.0%})"
                    for p in reversed(
                        st.session_state.predictions[-5:]
                    )
                ])
                history_placeholder.markdown(
                    f"**Recent predictions:**\n{history}"
                )

            # Stats
            stats_placeholder.markdown(f"""
            | Metric | Value |
            |--------|-------|
            | Total predictions | {st.session_state.total_preds} |
            | Buffer size | {buf_len} / 90 |
            | Hands detected | {'Yes ✓' if hands_detected else 'No'} |
            | Device | {device.upper()} |
            """)

        cap.release()

    else:
        # Show instructions when webcam is off
        frame_placeholder.info(
            "👆 Check 'Start Webcam' to begin the demo"
        )
        pred_placeholder.markdown("### Waiting for signs...")

        # Show sample info
        st.subheader("📋 How to use")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown("""
            **Step 1**
            Check 'Start Webcam'
            and allow camera access
            """)
        with col_b:
            st.markdown("""
            **Step 2**
            Perform an ASL sign
            clearly in front of camera
            """)
        with col_c:
            st.markdown("""
            **Step 3**
            Hold still — system
            speaks the word automatically
            """)

        st.subheader("🏗️ Architecture")
        st.code("""
Sign Video
    ↓
MediaPipe (183 landmarks/frame)
    ↓
Temporal Transformer Encoder
    ↓
Sign Classifier (10 classes)
    ↓
Text-to-Speech (pyttsx3/gTTS)
    ↓
Speech Output 🔊
        """)


if __name__ == "__main__":
    main()