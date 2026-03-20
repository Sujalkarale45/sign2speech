"""
vocoder.py
Thin wrapper around a pretrained HiFi-GAN checkpoint.
Converts log-mel spectrograms → raw waveforms.
"""
import json
import torch


class HiFiGANWrapper:
    """
    Loads HiFi-GAN generator from a local checkpoint directory.
    Expects the standard HiFi-GAN repo structure:
      checkpoint_dir/
        generator_*.pth.tar   ← model weights
        config.json           ← architecture config
    """

    def __init__(self, checkpoint_dir: str, device: str = "cuda"):
        self.device = device
        self.generator = self._load(checkpoint_dir)

    def _load(self, checkpoint_dir: str):
        """Load generator weights. Returns model in eval mode."""
        raise NotImplementedError(
            "Clone https://github.com/jik876/hifi-gan and call load_checkpoint(). "
            "See notebooks/03_inference_demo.ipynb for integration example."
        )

    @torch.no_grad()
    def mel_to_wav(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: (1, 80, T) log-mel tensor on self.device.

        Returns:
            waveform: (1, N) float32 tensor, 22050 Hz.
        """
        return self.generator(mel).squeeze(1)