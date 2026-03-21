"""
postnet.py — UPDATED
5-layer CNN PostNet. Fixed layer construction bug.
"""
import torch
import torch.nn as nn


class PostNet(nn.Module):
    """
    Residual CNN that predicts a mel residual added to decoder output.
    Input/output shape: (B, n_mels, T).
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
            in_ch  = n_mels    if i == 0            else channels
            out_ch = n_mels    if i == n_layers - 1  else channels
            layers.append(nn.Conv1d(in_ch, out_ch, kernel, padding=padding))
            layers.append(nn.BatchNorm1d(out_ch))
            if i < n_layers - 1:
                layers.append(nn.Tanh())
            layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, n_mels, T)

        Returns:
            (B, n_mels, T) — x + postnet residual
        """
        return x + self.net(x)