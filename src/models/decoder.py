"""
decoder.py
Autoregressive Mel Decoder with cross-attention over encoder hidden states.
Predicts mel frames one step at a time (teacher-forced during training).
"""
import torch
import torch.nn as nn


class MelDecoder(nn.Module):
    """
    Transformer decoder that attends to encoder output to generate mel frames.

    At each step, predicts:
      - next mel frame  (B, n_mels)
      - stop token      (B, 1)   — sigmoid > 0.5 = end of sequence
    """

    def __init__(
        self,
        n_mels: int    = 80,
        d_model: int   = 256,
        n_heads: int   = 4,
        n_layers: int  = 4,
        ff_dim: int    = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.mel_proj = nn.Linear(n_mels, d_model)
        layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=ff_dim, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.transformer  = nn.TransformerDecoder(layer, num_layers=n_layers)
        self.norm         = nn.LayerNorm(d_model)
        self.mel_out      = nn.Linear(d_model, n_mels)
        self.stop_out     = nn.Linear(d_model, 1)

    def forward(
        self,
        encoder_out: torch.Tensor,
        mel_input: torch.Tensor,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            encoder_out:              (B, T_src, d_model)
            mel_input:                (B, T_tgt, n_mels)  — teacher-forced targets
            memory_key_padding_mask:  (B, T_src) bool

        Returns:
            mel_out:   (B, T_tgt, n_mels)
            stop_out:  (B, T_tgt, 1)
        """
        tgt = self.mel_proj(mel_input)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt.size(1), device=tgt.device
        )
        out = self.norm(self.transformer(
            tgt, encoder_out,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        ))
        return self.mel_out(out), self.stop_out(out)