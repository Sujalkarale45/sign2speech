"""
losses.py
Combined loss for mel prediction: L1 + MSE on mel frames + BCE on stop token.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MelLoss(nn.Module):
    """
    Weighted combination of:
      - L1  loss on pre-postnet mel
      - MSE loss on pre-postnet mel
      - L1  loss on post-postnet mel  (postnet refinement supervision)
      - BCE loss on stop token
    """

    def __init__(self, l1_w=1.0, mse_w=1.0, stop_w=1.0, postnet_w=1.0):
        super().__init__()
        self.l1_w      = l1_w
        self.mse_w     = mse_w
        self.stop_w    = stop_w
        self.postnet_w = postnet_w

    def forward(
        self,
        mel_pre:   torch.Tensor,   # (B, T, 80)
        mel_post:  torch.Tensor,   # (B, T, 80)
        stop_pred: torch.Tensor,   # (B, T, 1)
        mel_target:torch.Tensor,   # (B, T, 80)
        stop_target:torch.Tensor,  # (B, T, 1)
    ) -> tuple[torch.Tensor, dict]:
        """
        Returns:
            total_loss: scalar tensor.
            breakdown:  dict with individual loss values for logging.
        """
        l1      = F.l1_loss(mel_pre,  mel_target)
        mse     = F.mse_loss(mel_pre, mel_target)
        post_l1 = F.l1_loss(mel_post, mel_target)
        stop    = F.binary_cross_entropy_with_logits(stop_pred, stop_target)

        total = self.l1_w*l1 + self.mse_w*mse + self.postnet_w*post_l1 + self.stop_w*stop

        return total, {"l1": l1.item(), "mse": mse.item(),
                       "postnet": post_l1.item(), "stop": stop.item()}