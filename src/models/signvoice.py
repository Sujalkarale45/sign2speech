"""
signvoice.py — UPDATED
"""
import torch
import torch.nn as nn
from .encoder import TemporalTransformerEncoder
from .decoder import MelDecoder
from .postnet import PostNet
from .emotion import EmotionEmbedding


class SignVoiceModel(nn.Module):
    """End-to-end Sign Language → Mel Spectrogram model."""

    def __init__(self, config: dict):
        super().__init__()
        m = config["model"]

        self.encoder = TemporalTransformerEncoder(
            input_dim=m["input_dim"],
            d_model=m["d_model"],
            n_heads=m["encoder_heads"],
            n_layers=m["encoder_layers"],
            ff_dim=m["encoder_ff_dim"],
            dropout=m["encoder_dropout"],
        )
        self.decoder = MelDecoder(
            n_mels=m["n_mels"],
            d_model=m["d_model"],
            n_heads=m["decoder_heads"],
            n_layers=m["decoder_layers"],
            ff_dim=m["decoder_ff_dim"],
            dropout=m["decoder_dropout"],
        )
        self.postnet = PostNet(
            n_mels=m["n_mels"],
            channels=m["postnet_channels"],
            kernel=m["postnet_kernel"],
            n_layers=m["postnet_layers"],
        )
        self.use_emotion = m.get("use_emotion", False)
        if self.use_emotion:
            self.emotion = EmotionEmbedding(emotion_dim=m["emotion_dim"])

    def make_padding_mask(
        self,
        lengths: torch.Tensor,
        max_len: int
    ) -> torch.Tensor:
        """Returns (B, max_len) bool mask. True = pad position."""
        return (
            torch.arange(max_len, device=lengths.device)
            .unsqueeze(0) >= lengths.unsqueeze(1)
        )

    def forward(
        self,
        keypoints: torch.Tensor,       # (B, T, 183)
        key_lengths: torch.Tensor,     # (B,)
        mel_input: torch.Tensor,       # (B, T_mel, 80)
        face_kp: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            mel_pre:   (B, T_mel, 80)
            mel_post:  (B, T_mel, 80)
            stop_pred: (B, T_mel, 1)
        """
        src_mask = self.make_padding_mask(key_lengths, keypoints.size(1))

        enc_out = self.encoder(
            keypoints,
            src_key_padding_mask=src_mask
        )

        mel_pre, stop_pred = self.decoder(
            encoder_out=enc_out,
            mel_input=mel_input,
            memory_key_padding_mask=src_mask,
        )

        # PostNet: (B, T, 80) → transpose → (B, 80, T) → postnet → back
        mel_post = self.postnet(
            mel_pre.transpose(1, 2)
        ).transpose(1, 2)

        return mel_pre, mel_post, stop_pred