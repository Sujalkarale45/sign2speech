"""
scripts/test_inference.py
Classification-based inference:
  keypoints → predicted gloss → gTTS speech
Saves checkpoint compatible with realtime_demo.py
"""

import json
import os
import sys
import random
import tempfile
import subprocess
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import yaml
import soundfile as sf
from torch.nn.utils.rnn import pad_sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.preprocessing.normalizer import KeypointNormalizer


# ── Model ─────────────────────────────────────────

class SignClassifier(nn.Module):
    """
    Transformer classifier for ASL sign recognition.
    Architecture auto-saved in checkpoint for compatibility.
    """
    def __init__(self, input_dim=183, d_model=128,
                 n_heads=4, n_layers=2, n_classes=10):
        super().__init__()

        # Ensure n_heads divides d_model
        while d_model % n_heads != 0 and n_heads > 1:
            n_heads -= 1

        self.proj = nn.Linear(input_dim, d_model)
        self.bn   = nn.BatchNorm1d(d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=0.3,
            batch_first=True,
            norm_first=True,
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
        if mask is not None:
            valid = (~mask).float().unsqueeze(-1)
            x = (x * valid).sum(1) / valid.sum(1).clamp(min=1e-8)
        else:
            x = x.mean(dim=1)
        return self.classifier(x)


# ── Utilities ─────────────────────────────────────

def augment(kp: np.ndarray) -> np.ndarray:
    """Data augmentation for keypoints."""
    if random.random() < 0.6:
        kp = kp + np.random.normal(0, 0.008, kp.shape).astype(np.float32)
    if random.random() < 0.3:
        kp = kp * random.uniform(0.94, 1.06)
    if kp.shape[0] > 5 and random.random() < 0.3:
        n    = random.randint(1, max(1, kp.shape[0] // 10))
        idxs = random.sample(range(kp.shape[0]), n)
        kp[idxs] = 0.0
    return np.clip(kp, -10, 10)


def gloss_to_audio(gloss: str, out_path: str, sr: int = 22050):
    """Convert gloss to speech wav file."""
    from gtts import gTTS
    tmp_mp3 = tempfile.mktemp(suffix=".mp3")
    try:
        gTTS(text=gloss.lower(), lang="en", slow=False).save(tmp_mp3)
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_mp3,
             "-ar", str(sr), "-ac", "1", out_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        print(f"  Audio error for '{gloss}': {e}")
        # Fallback silence
        sf.write(out_path,
                 np.zeros(sr // 2, dtype=np.float32), sr)
    finally:
        try:
            os.unlink(tmp_mp3)
        except Exception:
            pass


def make_batch(samples, normalizer, gloss_to_idx,
               device, do_augment=False):
    """Create padded batch from samples."""
    kps, labels, lengths = [], [], []

    for s in samples:
        if s["gloss"] not in gloss_to_idx:
            continue
        try:
            kp = normalizer.normalize(
                np.load(s["keypoint_file"])
            )
            kp = np.nan_to_num(kp, nan=0.0,
                               posinf=0.0, neginf=0.0)
            if do_augment:
                kp = augment(kp)
            kps.append(torch.from_numpy(kp).float())
            labels.append(gloss_to_idx[s["gloss"]])
            lengths.append(kp.shape[0])
        except Exception:
            continue

    if not kps:
        return None, None, None

    padded  = pad_sequence(kps, batch_first=True,
                           padding_value=0.0)
    labels  = torch.tensor(labels, dtype=torch.long,
                           device=device)
    max_len = padded.size(1)
    mask    = (
        torch.arange(max_len, device=device).unsqueeze(0) >=
        torch.tensor(lengths, device=device).unsqueeze(1)
    )
    return padded.to(device), labels, mask


# ── Training ──────────────────────────────────────

def train_classifier(train_data, val_data, glosses,
                     normalizer, device, epochs=200):
    gloss_to_idx = {g: i for i, g in enumerate(glosses)}
    n_classes    = len(glosses)

    model     = SignClassifier(n_classes=n_classes).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=5e-4, weight_decay=1e-2
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    best_acc   = 0.0
    best_state = None
    patience   = 0
    max_pat    = 40

    print(f"\n  Training ({n_classes} classes, "
          f"max {epochs} epochs)...")
    print(f"  Train={len(train_data)} Val={len(val_data)}")

    for epoch in range(1, epochs + 1):
        model.train()
        random.shuffle(train_data)
        total_loss = 0.0
        n_batches  = 0

        for i in range(0, len(train_data), 8):
            batch = train_data[i:i+8]
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
                model.parameters(), 0.5
            )
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        scheduler.step()

        # Validate every 10 epochs
        if epoch % 10 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                kp, lbl, mask = make_batch(
                    val_data, normalizer,
                    gloss_to_idx, device,
                    do_augment=False
                )
                if kp is None:
                    continue
                preds   = model(kp, mask).argmax(1)
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
                if patience >= max_pat:
                    print(f"  Early stop at epoch {epoch}")
                    break

    if best_state:
        model.load_state_dict(best_state)
    print(f"\n  Best val accuracy: {best_acc:.1f}%")
    return model


# ── Main ──────────────────────────────────────────

def main():
    print("=" * 55)
    print("  SignVoice — Classification Inference")
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

    # Train
    model = train_classifier(
        train_data, val_data, glosses,
        normalizer, device, epochs=200
    )

    # Save checkpoint — compatible with realtime_demo.py
    Path("checkpoints").mkdir(exist_ok=True)
    torch.save(
        {
            "model"   : model.state_dict(),
            "glosses" : glosses,
            "d_model" : 128,
            "n_heads" : 4,
            "n_layers": 2,
        },
        "checkpoints/classifier.pt"
    )
    print(f"\n  Saved → checkpoints/classifier.pt")

    # Test
    print(f"\n  Testing on {len(test_data)} samples...")
    Path("outputs").mkdir(exist_ok=True)
    model.eval()
    correct      = 0
    results      = []
    gloss_to_idx = {g: i for i, g in enumerate(glosses)}

    for i, sample in enumerate(test_data):
        gloss_true = sample["gloss"]
        if gloss_true not in gloss_to_idx:
            continue

        try:
            kp   = normalizer.normalize(
                np.load(sample["keypoint_file"])
            )
            kp   = np.nan_to_num(kp, nan=0.0,
                                  posinf=0.0, neginf=0.0)
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
            print(f"  [{i+1:>2}/{len(test_data)}] "
                  f"true={gloss_true:12s} "
                  f"pred={gloss_pred:12s} "
                  f"conf={pred_prob:.2f} {mark}")

            results.append({
                "true"      : gloss_true,
                "predicted" : gloss_pred,
                "correct"   : is_correct,
                "confidence": pred_prob,
            })

        except Exception as e:
            print(f"  ERROR sample {i}: {e}")

    # Summary
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
                print(f"    {g:15s} : {bar} "
                      f"{g_acc}/{len(g_res)} = "
                      f"{g_acc/len(g_res)*100:.0f}%")
        print()
        if accuracy >= 60:
            print("  GOOD — ready for demo!")
        elif accuracy >= 40:
            print("  OKAY — acceptable for demo")
        else:
            print("  LOW — need more data")
        print(f"\n  Best signs to demo:")
        best = [g for g in glosses
                if any(r["correct"] and r["true"] == g
                       for r in results)]
        print(f"  {best}")
        print("=" * 55)


if __name__ == "__main__":
    main()