"""
collate.py
Custom collate function for batching variable-length keypoint sequences
and mel spectrograms with proper padding.
"""
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch: list[tuple]) -> tuple:
    """
    Pads keypoints and mels within a batch.

    Args:
        batch: List of (keypoints (T,183), mel (80,T_mel)) tuples.

    Returns:
        key_padded:  (B, T_max, 183)
        key_lengths: (B,)
        mel_padded:  (B, 80, T_mel_max)
        mel_lengths: (B,)
    """
    keypoints, mels = zip(*batch)

    key_lengths = torch.tensor([k.shape[0] for k in keypoints])
    mel_lengths = torch.tensor([m.shape[1] for m in mels])

    key_padded = pad_sequence(
        [torch.from_numpy(k) for k in keypoints],
        batch_first=True,
        padding_value=0.0,
    )  # (B, T_max, 183)

    mel_padded = pad_sequence(
        [torch.from_numpy(m.T) for m in mels],   # transpose → (T_mel, 80)
        batch_first=True,
        padding_value=-11.5,                       # log(1e-5) = silence
    ).permute(0, 2, 1)                             # → (B, 80, T_mel_max)

    return key_padded, key_lengths, mel_padded, mel_lengths