"""
trainer.py — UPDATED
Fixed mel transpose and stop token target creation.
"""
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from .losses import MelLoss
from .scheduler import WarmupCosineScheduler


class Trainer:
    """Encapsulates training loop for SignVoiceModel."""

    def __init__(
        self,
        model,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: str = "cuda",
    ):
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.config       = config
        self.device       = device

        t = config["training"]
        self.epochs        = t["epochs"]
        self.grad_clip     = t["grad_clip"]
        self.save_interval = t["save_interval"]
        self.ckpt_dir      = t["checkpoint_dir"]

        lw = config["loss"]
        self.criterion = MelLoss(
            lw["l1_weight"], lw["mse_weight"],
            lw["stop_weight"], lw["postnet_weight"]
        )

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=t["learning_rate"],
            weight_decay=t["weight_decay"],
        )

        total_steps    = self.epochs * len(train_loader)
        self.scheduler = WarmupCosineScheduler(
            self.optimizer, t["warmup_steps"], total_steps
        )

        self.writer      = SummaryWriter()
        self.global_step = 0

    def _prepare_batch(self, batch):
        """
        Unpack batch and move to device.
        Returns keypoints, key_lens, mels (B, T_mel, 80), mel_lens.
        """
        keys, key_lens, mels, mel_lens = batch

        keys     = keys.to(self.device)                    # (B, T, 183)
        key_lens = key_lens.to(self.device)
        # mels from dataloader: (B, 80, T_mel) → need (B, T_mel, 80)
        mels     = mels.permute(0, 2, 1).to(self.device)  # (B, T_mel, 80)
        mel_lens = mel_lens.to(self.device)

        return keys, key_lens, mels, mel_lens

    def _make_stop_target(self, mel_lens, T_mel):
        """Build stop token target: 1 at last real frame, 0 elsewhere."""
        B          = mel_lens.size(0)
        stop_tgt   = torch.zeros(B, T_mel, 1, device=self.device)
        for b in range(B):
            idx = min(mel_lens[b].item() - 1, T_mel - 1)
            stop_tgt[b, idx] = 1.0
        return stop_tgt

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0

        for batch in tqdm(self.train_loader, desc=f"Train {epoch:03d}"):
            keys, key_lens, mels, mel_lens = self._prepare_batch(batch)

            # Teacher forcing: shift mel right by one frame
            mel_in  = torch.cat(
                [torch.zeros_like(mels[:, :1, :]), mels[:, :-1, :]], dim=1
            )
            stop_tgt = self._make_stop_target(mel_lens, mels.size(1))

            mel_pre, mel_post, stop_pred = self.model(keys, key_lens, mel_in)

            loss, breakdown = self.criterion(
                mel_pre, mel_post, stop_pred, mels, stop_tgt
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip
            )
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()

            for k, v in breakdown.items():
                self.writer.add_scalar(f"train/{k}", v, self.global_step)
            self.writer.add_scalar("train/loss", loss.item(), self.global_step)
            self.global_step += 1

        return total_loss / len(self.train_loader)

    def val_epoch(self, epoch: int) -> float:
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Val   {epoch:03d}"):
                keys, key_lens, mels, mel_lens = self._prepare_batch(batch)
                mel_in   = torch.cat(
                    [torch.zeros_like(mels[:, :1, :]), mels[:, :-1, :]], dim=1
                )
                stop_tgt = self._make_stop_target(mel_lens, mels.size(1))

                mel_pre, mel_post, stop_pred = self.model(keys, key_lens, mel_in)
                loss, _ = self.criterion(
                    mel_pre, mel_post, stop_pred, mels, stop_tgt
                )
                total_loss += loss.item()

        val_loss = total_loss / len(self.val_loader)
        self.writer.add_scalar("val/loss", val_loss, epoch)
        return val_loss

    def fit(self):
        """Full training loop with checkpointing."""
        os.makedirs(self.ckpt_dir, exist_ok=True)

        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss   = self.val_epoch(epoch)

            print(
                f"Epoch {epoch:03d} | "
                f"train={train_loss:.4f} | "
                f"val={val_loss:.4f} | "
                f"lr={self.scheduler.get_last_lr()[0]:.2e}"
            )

            if epoch % self.save_interval == 0:
                path = os.path.join(
                    self.ckpt_dir, f"ckpt_epoch{epoch:03d}.pt"
                )
                torch.save({
                    "epoch"     : epoch,
                    "model"     : self.model.state_dict(),
                    "optimizer" : self.optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss"  : val_loss,
                }, path)
                print(f"  Saved → {path}")