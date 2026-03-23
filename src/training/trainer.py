"""
src/training/trainer.py
Training loop + checkpointing for SignVoice model with AMP (mixed precision) support
"""

import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn.utils.clip_grad_norm_ as clip_grad_norm_

try:
    from torch.cuda.amp import autocast, GradScaler
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False
    autocast = lambda: (lambda f: f)  # no-op context
    GradScaler = lambda: None

from .losses import MelLoss
from .scheduler import WarmupCosineScheduler


class Trainer:
    def __init__(
        self,
        model,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: torch.device,
        checkpoint_dir: str = "checkpoints",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.checkpoint_dir = checkpoint_dir

        t = config["training"]

        self.epochs = t["epochs"]
        self.grad_clip = t.get("grad_clip", 1.0)
        self.save_interval = t.get("save_interval", 5)
        self.best_val_loss = float("inf")

        # Mixed Precision (AMP)
        self.use_amp = t.get("use_amp", True) and AMP_AVAILABLE and device.type == "cuda"
        if self.use_amp:
            self.scaler = GradScaler()
            print("Mixed Precision (AMP) ENABLED")
        else:
            self.scaler = None
            print("Mixed Precision (AMP) DISABLED")

        # Loss
        lw = config.get("loss", {})
        self.criterion = MelLoss(
            l1_weight=lw.get("l1_weight", 1.0),
            mse_weight=lw.get("mse_weight", 1.0),
            stop_weight=lw.get("stop_weight", 5.0),
            postnet_weight=lw.get("postnet_weight", 0.5),
        )

        # Optimizer & scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=t["learning_rate"],
            weight_decay=t.get("weight_decay", 1e-6),
            betas=(0.9, 0.98),
            eps=1e-9,
        )

        total_steps = self.epochs * len(train_loader)
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_steps=t.get("warmup_steps", 4000),
            total_steps=total_steps,
            min_lr_ratio=0.01,
        )

        # Logging
        self.writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, "logs"))
        self.global_step = 0
        self.current_epoch = 0

        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _prepare_batch(self, batch):
        kp_padded, mel_padded, kp_lengths, mel_lengths = batch

        kp_padded = kp_padded.to(self.device)          # (B, T_kp, 183)
        mel_padded = mel_padded.permute(0, 2, 1).to(self.device)  # (B, T_mel, 80)
        kp_lengths = kp_lengths.to(self.device)
        mel_lengths = mel_lengths.to(self.device)

        return kp_padded, kp_lengths, mel_padded, mel_lengths

    @staticmethod
    def _create_stop_target(mel_lengths, max_len: int, device):
        B = mel_lengths.size(0)
        stop = torch.zeros(B, max_len, 1, device=device)
        indices = torch.clamp(mel_lengths - 1, min=0, max=max_len - 1)
        stop[torch.arange(B, device=device), indices, 0] = 1.0
        return stop

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch:03d} [train]")
        for batch in pbar:
            keys, key_lens, mels, mel_lens = self._prepare_batch(batch)

            mel_input = torch.cat(
                [torch.zeros_like(mels[:, :1, :]), mels[:, :-1, :]], dim=1
            )

            stop_target = self._create_stop_target(mel_lens, mels.size(1), self.device)

            # ── Mixed Precision Forward ─────────────────────────────
            if self.use_amp:
                with autocast():
                    mel_before, mel_after, stop_pred = self.model(
                        keys, key_lens, mel_input
                    )
                    loss, loss_dict = self.criterion(
                        mel_before, mel_after, stop_pred,
                        mels, stop_target, mel_lens
                    )
            else:
                mel_before, mel_after, stop_pred = self.model(
                    keys, key_lens, mel_input
                )
                loss, loss_dict = self.criterion(
                    mel_before, mel_after, stop_pred,
                    mels, stop_target, mel_lens
                )

            # Backward + optimize
            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.grad_clip > 0:
                    clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            self.scheduler.step()

            total_loss += loss.item()
            self.global_step += 1

            self.writer.add_scalar("train/loss", loss.item(), self.global_step)
            for k, v in loss_dict.items():
                self.writer.add_scalar(f"train/{k}", v, self.global_step)

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / num_batches

    @torch.no_grad()
    def validate_epoch(self, epoch: int) -> float:
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)

        for batch in tqdm(self.val_loader, desc=f"Epoch {epoch:03d} [val]  "):
            keys, key_lens, mels, mel_lens = self._prepare_batch(batch)

            mel_input = torch.cat(
                [torch.zeros_like(mels[:, :1, :]), mels[:, :-1, :]], dim=1
            )
            stop_target = self._create_stop_target(mel_lens, mels.size(1), self.device)

            # ── Mixed Precision Inference (still useful for memory) ──
            if self.use_amp:
                with autocast():
                    mel_before, mel_after, stop_pred = self.model(
                        keys, key_lens, mel_input
                    )
                    loss, _ = self.criterion(
                        mel_before, mel_after, stop_pred,
                        mels, stop_target, mel_lens
                    )
            else:
                mel_before, mel_after, stop_pred = self.model(
                    keys, key_lens, mel_input
                )
                loss, _ = self.criterion(
                    mel_before, mel_after, stop_pred,
                    mels, stop_target, mel_lens
                )

            total_loss += loss.item()

        avg_loss = total_loss / num_batches
        self.writer.add_scalar("val/loss", avg_loss, epoch)
        return avg_loss

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }

        if is_best:
            path = os.path.join(self.checkpoint_dir, "best.pt")
        else:
            path = os.path.join(self.checkpoint_dir, f"epoch_{epoch:04d}.pt")

        torch.save(state, path)
        print(f"  Saved checkpoint → {path}")

    def load_checkpoint(self, path: str) -> int:
        if not os.path.isfile(path):
            print(f"Checkpoint not found: {path}")
            return 1

        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.scheduler.load_state_dict(state["scheduler_state_dict"])

        if self.scaler and "scaler_state_dict" in state and state["scaler_state_dict"]:
            self.scaler.load_state_dict(state["scaler_state_dict"])

        self.best_val_loss = state.get("best_val_loss", float("inf"))
        start_epoch = state["epoch"] + 1

        print(f"Loaded checkpoint from epoch {state['epoch']}, "
              f"best val loss: {self.best_val_loss:.4f}")
        return start_epoch

    def fit(self, start_epoch: int = 1):
        print(f"Starting training from epoch {start_epoch}")

        try:
            for epoch in range(start_epoch, self.epochs + 1):
                self.current_epoch = epoch

                train_loss = self.train_epoch(epoch)
                val_loss = self.validate_epoch(epoch)

                print(f"Epoch {epoch:03d}/{self.epochs} | "
                      f"train: {train_loss:.4f} | val: {val_loss:.4f} | "
                      f"lr: {self.scheduler.get_last_lr()[0]:.2e}")

                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    print(f"  → New best val loss: {val_loss:.4f}")

                if epoch % self.save_interval == 0 or is_best or epoch == self.epochs:
                    self.save_checkpoint(epoch, is_best=is_best)

        except KeyboardInterrupt:
            print("\nInterrupted — saving current state...")
            self.save_checkpoint(self.current_epoch, is_best=False)
            print("Exiting gracefully.")

        except Exception as e:
            print(f"Training failed: {e}")
            import traceback
            traceback.print_exc()

        finally:
            self.writer.close()
            print("Training finished.")