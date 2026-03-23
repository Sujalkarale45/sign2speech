"""
mel_predictor.py
Transformer-based model that maps sign-language keypoint sequences
to mel spectrograms — no text at any stage.

Architecture:
  keypoints (B, T_kp, 225)
    → Linear projection → (B, T_kp, d_model)
    → Transformer Encoder
    → Adaptive average pool → (B, d_model)
    → Linear head → (B, n_mels × mel_frames)
    → Reshape → (B, n_mels, mel_frames)
"""

import torch
import torch.nn as nn


class MelPredictor(nn.Module):
    def __init__(
        self,
        input_dim:   int = 225,
        d_model:     int = 256,
        nhead:       int = 4,
        num_layers:  int = 4,
        n_mels:      int = 80,
        max_mel_len: int = 800,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.n_mels      = n_mels
        self.max_mel_len = max_mel_len

        # Project raw keypoints into model dimension
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
        )

        # Positional encoding (learned)
        self.pos_emb = nn.Embedding(1000, d_model)   # supports up to 1000 frames

        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Output head: global pool → predict full mel
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, n_mels * max_mel_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T_kp, 225)  — padded keypoint sequence
        Returns:
            mel: (B, 80, max_mel_len)
        """
        B, T, _ = x.shape
        pos      = torch.arange(T, device=x.device).unsqueeze(0)   # (1, T)
        x        = self.input_proj(x) + self.pos_emb(pos)           # (B, T, d_model)
        x        = self.transformer(x)                               # (B, T, d_model)
        x        = x.mean(dim=1)                                     # (B, d_model)
        x        = self.output_head(x)                               # (B, n_mels*mel_len)
        x        = x.view(B, self.n_mels, self.max_mel_len)         # (B, 80, 800)
        return x


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = MelPredictor()
    dummy = torch.randn(4, 300, 225)   # batch of 4, 300 frames
    out   = model(dummy)
    print(f"Input  : {dummy.shape}")
    print(f"Output : {out.shape}")    # expect (4, 80, 800)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params : {params:,}")
