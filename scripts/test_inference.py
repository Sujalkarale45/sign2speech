"""
scripts/test_inference.py
Classification-based inference:
  keypoints → predicted gloss → gTTS speech
This works reliably with small datasets (40-200 samples).
"""

import json
import os
import sys
import time
import subprocess
import tempfile
import torch
import torch.nn as nn
import numpy as np
import yaml
import soundfile as sf
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing.normalizer import KeypointNormalizer


# ── Simple classifier model ───────────────────────
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
        x   = self.proj(x)
        x   = self.transformer(x, src_key_padding_mask=mask)
        x   = x.mean(dim=1)       # global average pool
        return self.classifier(self.dropout(x))


def gloss_to_audio(gloss: str, out_path: str, sr: int = 22050):
    """Convert gloss word to speech using gTTS."""
    from gtts import gTTS
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        tmp_mp3 = f.name
    try:
        tts = gTTS(text=gloss.lower(), lang="en", slow=False)
        tts.save(tmp_mp3)
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_mp3,
             "-ar", str(sr), out_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    finally:
        if os.path.exists(tmp_mp3):
            os.unlink(tmp_mp3)


def train_classifier(
    train_manifest: list,
    val_manifest: list,
    glosses: list,
    normalizer: KeypointNormalizer,
    device: str,
    epochs: int = 100,
) -> SignClassifier:
    """Train a simple sign classifier."""
    from torch.nn.utils.rnn import pad_sequence

    gloss_to_idx = {g: i for i, g in enumerate(glosses)}
    n_classes    = len(glosses)

    model     = SignClassifier(n_classes=n_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    def make_batch(samples):
        kps     = []
        labels  = []
        lengths = []
        for s in samples:
            kp  = normalizer.normalize(np.load(s["keypoint_file"]))
            kps.append(torch.from_numpy(kp))
            labels.append(gloss_to_idx[s["gloss"]])
            lengths.append(kp.shape[0])

        padded = pad_sequence(kps, batch_first=True, padding_value=0.0)
        labels = torch.tensor(labels)
        max_len = padded.size(1)
        mask   = torch.arange(max_len).unsqueeze(0) >= torch.tensor(lengths).unsqueeze(1)
        return padded.to(device), labels.to(device), mask.to(device)

    print(f"\n  Training classifier ({n_classes} classes, {epochs} epochs)...")
    best_val_acc = 0.0
    best_state   = None

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        import random
        random.shuffle(train_manifest)

        # Mini batches of 8
        total_loss = 0.0
        for i in range(0, len(train_manifest), 8):
            batch = train_manifest[i:i+8]
            if not batch:
                continue
            kp, lbl, mask = make_batch(batch)
            logits        = model(kp, mask)
            loss          = criterion(logits, lbl)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Validation
        if epoch % 20 == 0 or epoch == epochs:
            model.eval()
            correct = 0
            with torch.no_grad():
                kp, lbl, mask = make_batch(val_manifest)
                logits        = model(kp, mask)
                preds         = logits.argmax(dim=1)
                correct       = (preds == lbl).sum().item()
            val_acc = correct / len(val_manifest) * 100
            print(f"  Epoch {epoch:>3} | loss={total_loss/max(1,len(train_manifest)//8):.4f} | val_acc={val_acc:.1f}%")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state   = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    print(f"\n  Best val accuracy: {best_val_acc:.1f}%")
    return model


def main():
    print("=" * 55)
    print("  SignVoice — Classification-Based Inference")
    print("=" * 55)

    with open("configs/lightweight.yaml") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device : {device}")

    # Load normalizer
    normalizer = KeypointNormalizer(config["data"]["stats_path"])
    normalizer.load()

    # Load manifests
    with open("data/processed/train_manifest.json") as f:
        train_data = json.load(f)
    with open("data/processed/val_manifest.json") as f:
        val_data = json.load(f)
    with open("data/processed/test_manifest.json") as f:
        test_data = json.load(f)

    glosses = sorted(set(s["gloss"] for s in train_data))
    print(f"  Glosses : {glosses}")
    print(f"  Train   : {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

    # Train classifier
    model = train_classifier(
        train_data, val_data, glosses, normalizer, device, epochs=150
    )

    # Save classifier
    torch.save({
        "model"  : model.state_dict(),
        "glosses": glosses,
    }, "checkpoints/classifier.pt")
    print(f"\n  Classifier saved → checkpoints/classifier.pt")

    # Run inference on test set
    print(f"\n  Running inference on {len(test_data)} test samples...")
    Path("outputs").mkdir(exist_ok=True)

    model.eval()
    from torch.nn.utils.rnn import pad_sequence

    correct = 0
    results = []

    for i, sample in enumerate(test_data):
        gloss_true = sample["gloss"]

        kp     = normalizer.normalize(np.load(sample["keypoint_file"]))
        kp_t   = torch.from_numpy(kp).unsqueeze(0).to(device)

        with torch.no_grad():
            logits    = model(kp_t)
            probs     = torch.softmax(logits, dim=1)
            pred_idx  = logits.argmax(dim=1).item()
            pred_prob = probs[0, pred_idx].item()

        gloss_pred = glosses[pred_idx]
        is_correct = gloss_pred == gloss_true

        if is_correct:
            correct += 1

        # Generate audio for prediction
        out_path = f"outputs/test_{i+1}_{gloss_pred}.wav"
        gloss_to_audio(gloss_pred, out_path)

        print(f"  [{i+1:>2}/{len(test_data)}] "
              f"true={gloss_true:12s} "
              f"pred={gloss_pred:12s} "
              f"conf={pred_prob:.2f} "
              f"{'✓' if is_correct else '✗'}")

        results.append({
            "true"     : gloss_true,
            "predicted": gloss_pred,
            "correct"  : is_correct,
            "audio"    : out_path,
            "confidence": pred_prob,
        })

    # Summary
    accuracy = correct / len(test_data) * 100
    print("\n" + "=" * 55)
    print("  Inference Complete!")
    print("=" * 55)
    print(f"  Test accuracy : {accuracy:.1f}%  ({correct}/{len(test_data)})")
    print(f"  Audio files   : outputs/ folder")
    print()
    print("  Per-gloss accuracy:")
    for g in glosses:
        g_samples = [r for r in results if r["true"] == g]
        g_correct = sum(1 for r in g_samples if r["correct"])
        if g_samples:
            print(f"    {g:15s} : {g_correct}/{len(g_samples)} = {g_correct/len(g_samples)*100:.0f}%")
    print()
    print("  Audio guide:")
    print("    Correct predictions  → audio says the right word")
    print("    Wrong predictions    → audio says wrong word")
    print("    Goal: >60% accuracy for a good demo")
    print("=" * 55)


if __name__ == "__main__":
    main()