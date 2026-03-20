"""
scheduler.py
Warmup + cosine annealing LR scheduler (Transformer-style).
"""
import torch
from torch.optim.lr_scheduler import _LRScheduler
import math


class WarmupCosineScheduler(_LRScheduler):
    """
    Linear warmup for `warmup_steps`, then cosine decay to `min_lr`.
    """

    def __init__(self, optimizer, warmup_steps: int, total_steps: int,
                 min_lr: float = 1e-6, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.total_steps  = total_steps
        self.min_lr       = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        step = self.last_epoch + 1
        if step < self.warmup_steps:
            scale = step / max(1, self.warmup_steps)
        else:
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            scale    = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [max(self.min_lr, base_lr * scale) for base_lr in self.base_lrs]