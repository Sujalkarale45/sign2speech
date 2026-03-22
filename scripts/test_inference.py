"""
scripts/test_inference.py
Classification-based inference for 10-sign model.
Trains on Google ASL + WLASL combined dataset.

Usage:
    python scripts/test_inference.py
"""

import json
import os
import sys
import subprocess
import tempfile
import random
import torch
import torch.nn as nn
import numpy as np
import yaml
import soundfile as sf
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocessing.normalizer import KeypointNormalizer


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


def augment(kp: np.ndarray) -> np.ndarray:
    kp = kp + np.random.normal(0, 0.01, kp.shape).astype(np.float32)
    kp = kp * np.random.uniform(0.9, 1.1)
    if kp.shape[0] > 5:
        n_drop   = random.randint(0, max(1, kp.shape[0] // 10))
        drop_idx = random.sample(range(kp.shape[0]), n_drop)
        kp[drop_idx] = 0.0
    return kp


def gloss_to_audio(gloss: str, out_path: str, sr: int = 22050):
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


def make_batch(samples, normalizer, gloss_to_idx,
               device, do_augment=False):
    kps, labels, lengths = [], [], []
    for s in samples:
        if s["gloss"] not in gloss_to_idx:
            continue
        kp = normalizer.normalize(np.load(s["keypoint_file"]))
        if do_augment:
            kp = augment(kp)
        kps.append(torch.from_numpy(kp))
        labels.append(gloss_to_idx[s["gloss"]])
        lengths.append(kp.shape[0])

    if not kps:
        return None, None, None

    padded  = pad_sequence(kps, batch_first=True, padding_value=0.0)
    labels  = torch.tensor(labels)
    max_len = padded.size(1)
    mask    = (
        torch.arange(max_len).unsqueeze(0) >=
        torch.tensor(lengths).unsqueeze(1)
    )
    return padded.to(device), labels.to(device), mask.to(device)


def train_classifier(train_data, val_data, glosses,
                     normalizer, device, epochs=200):
    gloss_to_idx = {g: i for i, g in enumerate(glosses)}
    n_classes    = len(glosses)

    model     = SignClassifier(n_classes=n_classes).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=1e-3
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    best_acc   = 0.0
    best_state = None
    patience   = 0

    print(f"\n  Training ({n_classes} classes, {epochs} epochs)...")
    print(f"  Train={len(train_data)} Val={len(val_data)}")

    for epoch in range(1, epochs + 1):
        model.train()
        random.shuffle(train_data)
        total_loss = 0.0
        n_batches  = 0

        for i in range(0, len(train_data), 16):
            batch = train_data[i:i+16]
            if not batch:
                continue
            kp, lbl, mask = make_batch(
                batch, normalizer, gloss_to_idx,
                device, do_augment=True
            )
            if kp is None:
                continue
            logits = model(kp, mask)
            loss   = criterion(logits, lbl)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        scheduler.step()

        if epoch % 20 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                kp, lbl, mask = make_batch(
                    val_data, normalizer,
                    gloss_to_idx, device,
                    do_augment=False
                )
                if kp is None:
                    continue
                logits  = model(kp, mask)
                preds   = logits.argmax(dim=1)
                val_acc = (preds == lbl).float().mean().item() * 100

            avg_loss = total_loss / max(1, n_batches)
            print(f"  Epoch {epoch:>3} | "
                  f"loss={avg_loss:.4f} | "
                  f"val_acc={val_acc:.1f}%")

            if val_acc > best_acc:
                best_acc   = val_acc
                best_state = {k: v.clone()
                              for k, v in model.state_dict().items()}
                patience   = 0
            else:
                patience  += 1
                if patience >= 5:
                    print(f"  Early stopping at epoch {epoch}")
                    break

    if best_state:
        model.load_state_dict(best_state)
    print(f"\n  Best val accuracy: {best_acc:.1f}%")
    return model


def main():
    print("=" * 55)
    print("  SignVoice — 10-Sign Classification")
    print("=" * 55)

    with open("configs/lightweight.yaml") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device : {device}")

    normalizer = KeypointNormalizer(config["data"]["stats_path"])
    normalizer.load()

    with open("data/processed/train_manifest.json") as f:
        train_data = json.load(f)
    with open("data/processed/val_manifest.json") as f:
        val_data = json.load(f)
    with open("data/processed/test_manifest.json") as f:
        test_data = json.load(f)

    glosses = sorted(set(s["gloss"] for s in train_data))
    print(f"  Glosses : {glosses}")
    print(f"  Train={len(train_data)} "
          f"Val={len(val_data)} "
          f"Test={len(test_data)}")

    model = train_classifier(
        train_data, val_data, glosses,
        normalizer, device, epochs=200
    )

    Path("checkpoints").mkdir(exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "glosses": glosses},
        "checkpoints/classifier.pt"
    )
    print(f"\n  Saved → checkpoints/classifier.pt")

    # Test
    print(f"\n  Testing on {len(test_data)} samples...")
    Path("outputs").mkdir(exist_ok=True)
    model.eval()
    correct    = 0
    results    = []
    g_to_idx   = {g: i for i, g in enumerate(glosses)}

    for i, sample in enumerate(test_data):
        gloss_true = sample["gloss"]
        if gloss_true not in g_to_idx:
            continue

        kp   = normalizer.normalize(
            np.load(sample["keypoint_file"])
        )
        kp_t = torch.from_numpy(kp).unsqueeze(0).to(device)

        with torch.no_grad():
            logits    = model(kp_t)
            probs     = torch.softmax(logits, dim=1)
            pred_idx  = logits.argmax(dim=1).item()
            pred_prob = probs[0, pred_idx].item()

        gloss_pred = glosses[pred_idx]
        is_correct = gloss_pred == gloss_true
        if is_correct:
            correct += 1

        out_path = f"outputs/test_{i+1}_{gloss_pred}.wav"
        gloss_to_audio(gloss_pred, out_path)

        mark = "✓" if is_correct else "✗"
        print(f"  [{i+1:>3}/{len(test_data)}] "
              f"true={gloss_true:12s} "
              f"pred={gloss_pred:12s} "
              f"conf={pred_prob:.2f} {mark}")

        results.append({
            "true"      : gloss_true,
            "predicted" : gloss_pred,
            "correct"   : is_correct,
            "confidence": pred_prob,
        })

    if results:
        accuracy = correct / len(results) * 100
        print("\n" + "=" * 55)
        print("  Results")
        print("=" * 55)
        print(f"  Accuracy : {accuracy:.1f}%  "
              f"({correct}/{len(results)})")
        print(f"\n  Per-gloss:")
        for g in glosses:
            g_res = [r for r in results if r["true"] == g]
            if g_res:
                g_acc = sum(1 for r in g_res if r["correct"])
                bar   = "█" * g_acc + "░" * (len(g_res) - g_acc)
                print(f"    {g:12s} : {bar} "
                      f"{g_acc}/{len(g_res)} = "
                      f"{g_acc/len(g_res)*100:.0f}%")

        print()
        if accuracy >= 70:
            print("  EXCELLENT — ready for demo!")
        elif accuracy >= 50:
            print("  GOOD — acceptable for demo")
        else:
            print("  LOW — need more data")
        print("=" * 55)


if __name__ == "__main__":
    main()