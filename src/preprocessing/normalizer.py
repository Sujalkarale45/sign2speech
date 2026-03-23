import numpy as np

class KeypointNormalizer:
    def __init__(self, stats_path: str = "data/keypoint_stats.npz"):
        self.stats_path = stats_path
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, sequences: list[np.ndarray]) -> None:
        """Fit mean and std from a list of (T_i, 183) arrays, ignoring NaN/Inf."""
        # Filter out frames with NaN/Inf
        cleaned_sequences = [seq[np.isfinite(seq).all(axis=1)] for seq in sequences]
        if not cleaned_sequences:
            raise ValueError("No valid keypoint frames to fit statistics.")

        all_frames = np.concatenate(cleaned_sequences, axis=0)
        self.mean = all_frames.mean(axis=0)
        self.std = all_frames.std(axis=0) + 1e-8
        np.savez(self.stats_path, mean=self.mean, std=self.std)
        print(f"[Normalizer] Fitted on {all_frames.shape[0]} frames → {self.stats_path}")

    def load(self) -> None:
        data = np.load(self.stats_path)
        self.mean = data["mean"]
        self.std = data["std"]

    def normalize(self, seq: np.ndarray) -> np.ndarray:
        """
        Normalize a single (T, 183) sequence, replacing NaN/Inf with zeros.
        """
        assert self.mean is not None, "Call fit() or load() before normalize()."

        # Replace NaN/Inf with 0 before normalizing
        seq_clean = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
        return ((seq_clean - self.mean) / self.std).astype(np.float32)