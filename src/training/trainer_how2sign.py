"""
trainer.py
Trains the MelPredictor model on How2Sign keypoint→mel pairs.

Usage:
  python src/training/trainer.py
  python src/training/trainer.py --epochs 100 --batch_size 16 --lr 3e-4
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import time

from src.dataset.how2sign_dataset import How2SignDataset
from src.models.mel_predictor import MelPredictor

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
# ─────────────────────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--batch_size", type=int,   default=8)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--val_split",  type=float, default=0.1,
                   help="Fraction of val set used for validation during training")
    p.add_argument("--save_every", type=int,   default=5,
                   help="Save checkpoint every N epochs")
    return p.parse_args()


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for kp, mel in loader:
        kp, mel  = kp.to(DEVICE), mel.to(DEVICE)
        pred_mel = model(kp)
        loss     = criterion(pred_mel, mel)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    for kp, mel in loader:
        kp, mel  = kp.to(DEVICE), mel.to(DEVICE)
        pred_mel = model(kp)
        total_loss += criterion(pred_mel, mel).item()
    return total_loss / len(loader)


def main():
    args = get_args()
    print(f"Device : {DEVICE}")
    print(f"Epochs : {args.epochs}  |  Batch : {args.batch_size}  |  LR : {args.lr}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    val_full = How2SignDataset(split="val")

    # Use a small portion of val set for online validation
    val_size   = max(1, int(len(val_full) * args.val_split))
    train_size = len(val_full) - val_size
    train_ds, val_ds = random_split(val_full, [train_size, val_size],
                                    generator=torch.Generator().manual_seed(42))

    # Also pull in test split if available
    try:
        test_ds   = How2SignDataset(split="test")
        train_ds  = torch.utils.data.ConcatDataset([train_ds, test_ds])
        print(f"Combined training samples: {len(train_ds)}")
    except Exception:
        print(f"Training samples: {len(train_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0, pin_memory=False)

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = MelPredictor().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=args.epochs)
    criterion = nn.L1Loss()

    best_val  = float("inf")

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        t0        = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss   = evaluate(model, val_loader, criterion)
        scheduler.step()
        elapsed    = time.time() - t0

        print(f"Epoch {epoch:03d}/{args.epochs}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  "
              f"time={elapsed:.1f}s")

        # Save best checkpoint
        if val_loss < best_val:
            best_val = val_loss
            ckpt = OUTPUTS_DIR / "mel_predictor_best.pt"
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "val_loss": val_loss}, str(ckpt))
            print(f"  ✓ Best model saved → {ckpt}")

        # Periodic checkpoint
        if epoch % args.save_every == 0:
            ckpt = OUTPUTS_DIR / f"mel_predictor_epoch{epoch:03d}.pt"
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "val_loss": val_loss}, str(ckpt))

    print(f"\nTraining complete. Best val loss: {best_val:.4f}")
    print(f"Checkpoints saved to: {OUTPUTS_DIR}")


if __name__ == "__main__":
    main()
