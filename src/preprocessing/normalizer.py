"""
normalizer.py
Fits per-feature z-score normalization statistics over training data
and applies them at runtime.
"""
import numpy as np


class KeypointNormalizer:
    """
    Computes and applies per-feature mean/std normalization.
    Statistics are fitted on training data and persisted to disk.
    """

    def __init__(self, stats_path: str = "data/keypoint_stats.npz"):
        self.stats_path = stats_path
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None  = None

    def fit(self, sequences: list[np.ndarray]) -> None:
        """
        Fit mean and std from a list of (T_i, 183) arrays.
        Saves stats to self.stats_path.
        """
        all_frames = np.concatenate(sequences, axis=0)
        self.mean  = all_frames.mean(axis=0)
        self.std   = all_frames.std(axis=0) + 1e-8
        np.savez(self.stats_path, mean=self.mean, std=self.std)
        print(f"[Normalizer] Fitted on {all_frames.shape[0]} frames → {self.stats_path}")

    def load(self) -> None:
        """Load previously fitted stats from disk."""
        data      = np.load(self.stats_path)
        self.mean = data["mean"]
        self.std  = data["std"]

    def normalize(self, seq: np.ndarray) -> np.ndarray:
        """
        Normalize a single (T, 183) sequence.

        Returns:
            (T, 183) float32 array with zero mean, unit variance per feature.
        """
        assert self.mean is not None, "Call fit() or load() before normalize()."
        return ((seq - self.mean) / self.std).astype(np.float32)