"""
src/dataset/dataset.py
PyTorch Dataset + collate function for SignVoice (keypoints → mel spectrogram)
"""

import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict, Any

from src.preprocessing.normalizer import KeypointNormalizer


class SignVoiceDataset(Dataset):
    """
    Loads normalized keypoints + target mel spectrograms from preprocessed .npy files.

    Manifest format (JSON list of dicts):
        [
          {
            "video_id": str,
            "gloss": str,
            "keypoint_file": str or Path,
            "mel_file": str or Path,
            "kp_frames": int,
            "mel_frames": int,
            "source": str (optional)
          },
          ...
        ]
    """

    def __init__(
        self,
        manifest_path: str | Path,
        normalizer: KeypointNormalizer,
        augment: bool = False,
        noise_std: float = 0.015,       # small keypoint noise
        preload: bool = False,          # load all data into RAM (only for small datasets!)
    ):
        """
        Args:
            manifest_path: Path to train/val/test_manifest.json
            normalizer:    Fitted KeypointNormalizer instance
            augment:       Whether to apply random keypoint noise during training
            noise_std:     Std dev of Gaussian noise added to keypoints (if augment=True)
            preload:       If True, loads all arrays into memory at init (faster but RAM-heavy)
        """
        manifest_path = Path(manifest_path)
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        with manifest_path.open(encoding="utf-8") as f:
            self.samples: List[Dict[str, Any]] = json.load(f)

        self.normalizer = normalizer
        self.augment = augment
        self.noise_std = noise_std
        self.preload = preload

        self.keypoints = []
        self.mels = []

        if preload:
            print(f"[Dataset] Preloading {len(self.samples)} samples into memory...")
            for s in self.samples:
                try:
                    kp = np.load(s["keypoint_file"], allow_pickle=False).astype(np.float32)
                    ml = np.load(s["mel_file"], allow_pickle=False).astype(np.float32)
                    self.keypoints.append(kp)
                    self.mels.append(ml)
                except Exception as e:
                    print(f"Failed to preload {s.get('video_id', '?')}: {e}")
                    self.keypoints.append(None)
                    self.mels.append(None)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        """
        Returns:
            keypoints: (T_kp, 183) normalized float32 tensor
            mel:       (80, T_mel) log-mel float32 tensor
            kp_len:    original keypoint sequence length (before padding)
            mel_len:   original mel sequence length (before padding)
        """
        if self.preload:
            kp = self.keypoints[idx]
            ml = self.mels[idx]
            if kp is None or ml is None:
                # Fallback on error — return zero tensors (or raise depending on policy)
                return (
                    torch.zeros((16, 183), dtype=torch.float32),
                    torch.zeros((80, 32), dtype=torch.float32),
                    16, 32
                )
        else:
            s = self.samples[idx]
            try:
                kp_path = Path(s["keypoint_file"])
                mel_path = Path(s["mel_file"])
                kp = np.load(kp_path, allow_pickle=False).astype(np.float32)   # (T, 183)
                ml = np.load(mel_path, allow_pickle=False).astype(np.float32)  # (80, T_mel)
            except Exception as e:
                print(f"Load error {s.get('video_id', '?')}: {e}")
                # Return dummy small tensors so training doesn't crash completely
                return (
                    torch.zeros((16, 183), dtype=torch.float32),
                    torch.zeros((80, 32), dtype=torch.float32),
                    16, 32
                )

        # Normalize keypoints
        kp = self.normalizer.normalize(kp)  # still numpy

        # Optional lightweight augmentation (only on train)
        if self.augment and random.random() < 0.65:
            kp += np.random.normal(0, self.noise_std, kp.shape).astype(np.float32)

        # Convert to torch tensors
        kp_tensor = torch.from_numpy(kp)           # (T_kp, 183)
        mel_tensor = torch.from_numpy(ml)          # (80, T_mel)

        return kp_tensor, mel_tensor, kp.shape[0], ml.shape[1]


def signvoice_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, int, int]]):
    """
    Collate function for DataLoader — pads sequences to max length in batch.
    Returns:
        kp_padded:    (B, max_T_kp, 183)
        mel_padded:   (B, 80, max_T_mel)
        kp_lengths:   (B,)
        mel_lengths:  (B,)
    """
    kps, mels, kp_lens, mel_lens = zip(*batch)

    # Find max lengths in this batch
    max_kp_len = max(kp_lens)
    max_mel_len = max(mel_lens)

    # Pad keypoints (right-pad with zeros)
    kp_padded = torch.zeros(
        (len(batch), max_kp_len, 183),
        dtype=torch.float32
    )
    for i, kp in enumerate(kps):
        kp_padded[i, :kp.shape[0], :] = kp

    # Pad mels (right-pad with small negative value or zero — Tacotron usually uses ~ -4 ~ -6)
    mel_padded = torch.full(
        (len(batch), 80, max_mel_len),
        fill_value=-6.0,   # common stop token region value for log-mel
        dtype=torch.float32
    )
    for i, mel in enumerate(mels):
        mel_padded[i, :, :mel.shape[1]] = mel

    kp_lengths = torch.tensor(kp_lens, dtype=torch.long)
    mel_lengths = torch.tensor(mel_lens, dtype=torch.long)

    return kp_padded, mel_padded, kp_lengths, mel_lengths


# ── Example usage ───────────────────────────────────────────────
if __name__ == "__main__":
    from src.preprocessing.normalizer import KeypointNormalizer

    norm = KeypointNormalizer("data/processed/keypoint_stats.npz")
    norm.load()  # assuming it has load method

    ds = SignVoiceDataset(
        manifest_path="data/processed/train_manifest.json",
        normalizer=norm,
        augment=True,
        preload=False
    )

    dl = DataLoader(
        ds,
        batch_size=8,
        shuffle=True,
        num_workers=2,
        collate_fn=signvoice_collate_fn,
        drop_last=True,
    )

    for batch in dl:
        kp, mel, kp_len, mel_len = batch
        print(f"Batch shapes: kp={kp.shape}, mel={mel.shape}, lens={kp_len}, {mel_len}")
        break