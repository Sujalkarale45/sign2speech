"""
scripts/train.py
CLI entrypoint for training SignVoice.
Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/lightweight.yaml
"""
import argparse, os, sys, yaml, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from torch.utils.data import DataLoader
from src.models.signvoice import SignVoiceModel
from src.dataset.dataset import SignVoiceDataset
from src.dataset.collate import collate_fn
from src.preprocessing.normalizer import KeypointNormalizer
from src.training.trainer import Trainer


def main(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = config["training"]["device"] if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    normalizer = KeypointNormalizer(config["data"]["stats_path"])
    normalizer.load()

    processed = config["data"]["processed_dir"]
    train_ds  = SignVoiceDataset(f"{processed}train_manifest.json", normalizer)
    val_ds    = SignVoiceDataset(f"{processed}val_manifest.json",   normalizer)

    train_loader = DataLoader(train_ds, batch_size=config["training"]["batch_size"],
                              shuffle=True, collate_fn=collate_fn, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=config["training"]["batch_size"],
                              shuffle=False, collate_fn=collate_fn, num_workers=2)

    model   = SignVoiceModel(config)
    trainer = Trainer(model, train_loader, val_loader, config, device)
    trainer.fit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    main(parser.parse_args())