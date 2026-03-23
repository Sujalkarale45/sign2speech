"""
scripts/train.py
Training entrypoint for SignVoice (Sign Language → Speech)

Usage examples:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/lightweight.yaml --resume checkpoints/last.pt
    python scripts/train.py --config configs/default.yaml --device cuda:1
"""

import argparse
import os
import random
import sys
from pathlib import Path
import yaml
import torch
import numpy as np

# Make sure parent directory is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from torch.utils.data import DataLoader

# Assuming these are now correctly located
from src.dataset.dataset import SignVoiceDataset, signvoice_collate_fn
from src.models.signvoice import SignVoiceModel
from src.preprocessing.normalizer import KeypointNormalizer
from src.training.trainer import Trainer


def set_seed(seed: int = 42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # set True later if desired


def get_device(device_str: str | None) -> torch.device:
    """Select best available device."""
    if device_str and device_str.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(device_str)
        print("Requested CUDA but not available → falling back to cpu")
    elif device_str == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description="Train SignVoice model")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from (.pt or .pth)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device override: cuda, cuda:0, mps, cpu")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--no-benchmark", action="store_true",
                        help="Disable cudnn.benchmark (more reproducible but slower)")

    args = parser.parse_args()

    # ── Reproducibility ────────────────────────────────────────
    set_seed(args.seed)
    if not args.no_benchmark and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # ── Load config ────────────────────────────────────────────
    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    with config_path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print(f"Loaded config: {config_path.name}")
    print(f"  Experiment name: {config.get('experiment', {}).get('name', 'unnamed')}")

    # ── Device ─────────────────────────────────────────────────
    device = get_device(args.device or config["training"].get("device"))
    print(f"Using device: {device}")

    # ── Paths & checks ─────────────────────────────────────────
    processed_dir = Path(config["data"]["processed_dir"])
    stats_path = Path(config["data"]["stats_path"])

    if not stats_path.is_file():
        print(f"Error: Keypoint stats file missing: {stats_path}")
        print("Run preprocessing first: python scripts/preprocess_asl.py")
        sys.exit(1)

    for split in ["train", "val"]:
        manifest = processed_dir / f"{split}_manifest.json"
        if not manifest.is_file():
            print(f"Error: {split}_manifest.json not found in {processed_dir}")
            sys.exit(1)

    # Create checkpoint dir
    ckpt_dir = Path(config["training"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ───────────────────────────────────────────────────
    normalizer = KeypointNormalizer(str(stats_path))
    try:
        normalizer.load()
    except Exception as e:
        print(f"Failed to load normalizer stats: {e}")
        sys.exit(1)

    train_ds = SignVoiceDataset(
        manifest_path=processed_dir / "train_manifest.json",
        normalizer=normalizer,
        augment=config["training"].get("augment", True),
    )

    val_ds = SignVoiceDataset(
        manifest_path=processed_dir / "val_manifest.json",
        normalizer=normalizer,
        augment=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"].get("num_workers", 2),
        pin_memory=torch.cuda.is_available(),
        collate_fn=signvoice_collate_fn,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"].get("num_workers", 2),
        pin_memory=torch.cuda.is_available(),
        collate_fn=signvoice_collate_fn,
    )

    print(f"Train samples: {len(train_ds):,d} | batches: {len(train_loader):,d}")
    print(f"  Val samples: {len(val_ds):,d}   | batches: {len(val_loader):,d}")

    # ── Model & Trainer ────────────────────────────────────────
    model = SignVoiceModel(config["model"]).to(device)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        checkpoint_dir=ckpt_dir,
    )

    # Resume if requested
    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
        print(f"Resumed from epoch {start_epoch}")

    # ── Start training ─────────────────────────────────────────
    try:
        trainer.fit(start_epoch=start_epoch)
    except KeyboardInterrupt:
        print("\nInterrupted — saving checkpoint...")
        trainer.save_checkpoint(is_best=False, epoch=trainer.current_epoch)
        print("Exiting gracefully.")
    except Exception as e:
        print(f"Training crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()