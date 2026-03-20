"""
trainer.py
Trainer class: manages train/val loops, checkpointing, and TensorBoard logging.
"""
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from .losses import MelLoss
from .scheduler import WarmupCosineScheduler


class Trainer:
    """Encapsulates training loop for SignVoiceModel."""

    def __init__(self, model, train_loader: DataLoader, val_loader: DataLoader,
                 config: dict, device: str = "cuda"):
        self.model       = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.config       = config
        self.device       = device

        t = config["training"]
        self.epochs       = t["epochs"]
        self.grad_clip    = t["grad_clip"]
        self.save_interval = t["save_interval"]
        self.ckpt_dir     = t["checkpoint_dir"]

        lw = config["loss"]
        self.criterion = MelLoss(lw["l1_weight"], lw["mse_weight"],
                                 lw["stop_weight"], lw["postnet_weight"])

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"]
        )
        total_steps = self.epochs * len(train_loader)
        self.scheduler = WarmupCosineScheduler(
            self.optimizer, t["warmup_steps"], total_steps
        )
        self.writer = SummaryWriter()
        self.global_step = 0

    def train_epoch(self, epoch: int) -> float:
        """Run one training epoch. Returns mean loss."""
        self.model.train()
        total_loss = 0.0

        for keys, key_lens, mels, mel_lens in tqdm(self.train_loader, desc=f"Train {epoch}"):
            keys    = keys.to(self.device)
            key_lens= key_lens.to(self.device)
            mels    = mels.permute(0, 2, 1).to(self.device)   # (B, T_mel, 80)

            # Teacher-forced: shift mels right by one frame
            mel_in  = torch.cat([torch.zeros_like(mels[:, :1]), mels[:, :-1]], dim=1)
            stop_tgt = torch.zeros(*mels.shape[:2], 1, device=self.device)
            for b, length in enumerate(mel_lens):
                if length - 1 < stop_tgt.size(1):
                    stop_tgt[b, length - 1] = 1.0

            mel_pre, mel_post, stop_pred = self.model(keys, key_lens, mel_in)
            loss, breakdown = self.criterion(mel_pre, mel_post, stop_pred, mels, stop_tgt)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            self.writer.add_scalar("train/loss", loss.item(), self.global_step)
            self.global_step += 1

        return total_loss / len(self.train_loader)

    def val_epoch(self, epoch: int) -> float:
        """Run validation. Returns mean loss."""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for keys, key_lens, mels, mel_lens in tqdm(self.val_loader, desc=f"Val {epoch}"):
                keys    = keys.to(self.device)
                key_lens= key_lens.to(self.device)
                mels    = mels.permute(0, 2, 1).to(self.device)
                mel_in  = torch.cat([torch.zeros_like(mels[:, :1]), mels[:, :-1]], dim=1)
                stop_tgt = torch.zeros(*mels.shape[:2], 1, device=self.device)
                mel_pre, mel_post, stop_pred = self.model(keys, key_lens, mel_in)
                loss, _ = self.criterion(mel_pre, mel_post, stop_pred, mels, stop_tgt)
                total_loss += loss.item()

        val_loss = total_loss / len(self.val_loader)
        self.writer.add_scalar("val/loss", val_loss, epoch)
        return val_loss

    def fit(self):
        """Full training loop."""
        import os
        os.makedirs(self.ckpt_dir, exist_ok=True)

        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss   = self.val_epoch(epoch)
            print(f"Epoch {epoch:03d} | train={train_loss:.4f} | val={val_loss:.4f}")

            if epoch % self.save_interval == 0:
                path = f"{self.ckpt_dir}/ckpt_epoch{epoch:03d}.pt"
                torch.save({"epoch": epoch, "model": self.model.state_dict(),
                            "optimizer": self.optimizer.state_dict()}, path)
                print(f"  Saved → {path}")