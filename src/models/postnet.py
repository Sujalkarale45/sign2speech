"""
postnet.py
5-layer 1D CNN PostNet that refines the decoder's raw mel output.
Mirrors Tacotron 2's postnet design.
"""
import torch
import torch.nn as nn


class PostNet(nn.Module):
    """
    Residual CNN that predicts a mel residual added to the decoder output.
    Input/output: (B, n_mels, T).
    """

    def __init__(
        self,
        n_mels: int    = 80,
        channels: int  = 512,
        kernel: int    = 5,
        n_layers: int  = 5,
        dropout: float = 0.5,
    ):
        super().__init__()
        padding = (kernel - 1) // 2
        layers  = []

        for i in range(n_layers):
            in_ch  = n_mels    if i == 0           else channels
            out_ch = n_mels    if i == n_layers - 1 else channels
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel, padding=padding),
                nn.BatchNorm1d(out_ch),
                nn.Tanh() if i < n_layers - 1 else nn.Identity(),
                nn.Dropout(dropout),
            ]

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, n_mels, T) — raw decoder mel output.

        Returns:
            (B, n_mels, T) — mel + postnet residual.
        """
        return x + self.net(x)