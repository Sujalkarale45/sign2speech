"""
dataset.py
PyTorch Dataset that loads preprocessed keypoint and mel .npy files
described by a JSON manifest.
"""
import json
import numpy as np
from torch.utils.data import Dataset
from src.preprocessing.normalizer import KeypointNormalizer


class SignVoiceDataset(Dataset):
    """
    Loads (keypoints, mel) pairs from disk.

    Manifest format (JSON list):
        [{"id": "...", "gloss": "...",
          "keypoint_file": "...", "mel_file": "..."}, ...]
    """

    def __init__(self, manifest_path: str, normalizer: KeypointNormalizer):
        """
        Args:
            manifest_path: Path to train/val/test manifest JSON.
            normalizer:    Fitted KeypointNormalizer instance.
        """
        with open(manifest_path) as f:
            self.samples = json.load(f)
        self.normalizer = normalizer

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            keypoints: (T, 183) normalized float32 array.
            mel:       (80, T_mel) log-mel float32 array.
        """
        s  = self.samples[idx]
        kp = np.load(s["keypoint_file"])   # (T, 183)
        ml = np.load(s["mel_file"])        # (80, T_mel)
        return self.normalizer.normalize(kp), ml