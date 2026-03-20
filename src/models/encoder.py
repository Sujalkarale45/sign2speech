"""
encoder.py
Temporal Transformer Encoder: maps (B, T, 183) keypoint sequences
to (B, T, d_model) context representations.
"""
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding added to input embeddings."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model)"""
        return self.dropout(x + self.pe[:, : x.size(1)])


class TemporalTransformerEncoder(nn.Module):
    """
    Projects keypoints to d_model, adds positional encoding,
    then applies N transformer encoder layers.
    """

    def __init__(
        self,
        input_dim: int   = 183,
        d_model: int     = 256,
        n_heads: int     = 4,
        n_layers: int    = 4,
        ff_dim: int      = 1024,
        dropout: float   = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc    = PositionalEncoding(d_model, dropout)
        layer           = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=ff_dim, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm        = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:                      (B, T, 183)
            src_key_padding_mask:   (B, T) bool, True = ignore

        Returns:
            (B, T, d_model)
        """
        x = self.pos_enc(self.input_proj(x))
        return self.norm(self.transformer(x, src_key_padding_mask=src_key_padding_mask))