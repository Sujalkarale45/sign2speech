"""
how2sign_dataset.py
PyTorch Dataset that pairs keypoint arrays with mel spectrograms
using SENTENCE_NAME as the common key.

Returns:
  keypoints : FloatTensor  (max_kp_len, 225)
  mel       : FloatTensor  (80, max_mel_len)
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from pathlib import Path


class How2SignDataset(Dataset):
    CSV_MAP = {
        "test": "how2sign_test.csv",
        "val":  "how2sign_val.csv",
    }

    def __init__(
        self,
        split: str = "val",
        keypoints_dir: str = "data/processed/keypoints",
        mels_dir: str      = "data/processed/mels",
        metadata_dir: str  = "data/raw/metadata",
        max_kp_len: int    = 300,
        max_mel_len: int   = 800,
    ):
        assert split in self.CSV_MAP, f"split must be one of {list(self.CSV_MAP)}"

        self.kp_dir  = Path(keypoints_dir)
        self.mel_dir = Path(mels_dir)
        self.max_kp  = max_kp_len
        self.max_mel = max_mel_len

        csv = Path(metadata_dir) / self.CSV_MAP[split]
        df  = pd.read_csv(csv, sep="\t")

        self.samples = []
        missing = 0
        for _, row in df.iterrows():
            name = str(row["SENTENCE_NAME"])
            kp_path  = self.kp_dir  / f"{name}.npy"
            mel_path = self.mel_dir / f"{name}.npy"
            if kp_path.exists() and mel_path.exists():
                self.samples.append({
                    "name":     name,
                    "sentence": str(row["SENTENCE"]),
                })
            else:
                missing += 1

        print(f"[How2SignDataset | {split}]  "
              f"{len(self.samples)} pairs loaded, {missing} skipped (not preprocessed yet)")

    # ── helpers ───────────────────────────────────────────────────────────────

    def _pad_kp(self, arr: np.ndarray) -> np.ndarray:
        """Pad or trim keypoint array to (max_kp, 225)."""
        T = arr.shape[0]
        if T >= self.max_kp:
            return arr[:self.max_kp]
        pad = np.zeros((self.max_kp - T, arr.shape[1]), dtype=np.float32)
        return np.vstack([arr, pad])

    def _pad_mel(self, mel: np.ndarray) -> np.ndarray:
        """Pad or trim mel to (80, max_mel)."""
        _, T = mel.shape
        mel  = mel[:, :self.max_mel]
        if mel.shape[1] < self.max_mel:
            mel = np.pad(mel, ((0, 0), (0, self.max_mel - mel.shape[1])))
        return mel

    # ── Dataset API ───────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        name = self.samples[idx]["name"]

        kp  = np.load(self.kp_dir  / f"{name}.npy").astype(np.float32)  # (T, 225)
        mel = np.load(self.mel_dir / f"{name}.npy").astype(np.float32)  # (80, T)

        kp  = self._pad_kp(kp)   # (max_kp, 225)
        mel = self._pad_mel(mel)  # (80, max_mel)

        return torch.from_numpy(kp), torch.from_numpy(mel)

    def get_sentence(self, idx: int) -> str:
        return self.samples[idx]["sentence"]
