"""
emotion.py
Optional emotion embedding branch.
Takes face keypoints (B, T, 12) and produces a (B, emotion_dim) embedding
that is added to the decoder's initial hidden state.
"""
import torch
import torch.nn as nn


class EmotionEmbedding(nn.Module):
    """
    Lightweight GRU that summarizes face keypoint dynamics into
    a fixed-size emotion vector, injected as decoder bias.
    """

    def __init__(self, face_dim: int = 12, emotion_dim: int = 32, hidden: int = 64):
        super().__init__()
        self.gru     = nn.GRU(face_dim, hidden, batch_first=True, bidirectional=True)
        self.proj    = nn.Linear(hidden * 2, emotion_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, face_kp: torch.Tensor) -> torch.Tensor:
        """
        Args:
            face_kp: (B, T, 12) — face landmark sequence.

        Returns:
            (B, emotion_dim) — emotion context vector.
        """
        _, h  = self.gru(face_kp)           # h: (2, B, hidden)
        h_cat = torch.cat([h[0], h[1]], -1) # (B, hidden*2)
        return self.dropout(self.proj(h_cat))