"""Training loop for autoregressive CyTOF sequence models.

Implements:
- Next-token prediction with teacher forcing
- AdamW optimiser
- Early stopping on validation loss
- Best-checkpoint saving
- Optional mixed precision (torch.amp)
- Per-epoch logging via :class:`~src.utils.logging.TrainingLogger`
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.models.base import BaseModel
from src.utils.logging import TrainingLogger

logger = logging.getLogger(__name__)


# ===========================================================================
# Dataset
# ===========================================================================

class CellSequenceDataset(Dataset):
    """PyTorch Dataset wrapping tokenised cell sequences.

    Args:
        sequences: List of integer token-id lists (each is one cell).
        pad_id: Padding token id.
        max_seq_len: Sequences longer than this are truncated.
    """

    def __init__(
        self,
        sequences: list[list[int]],
        pad_id: int = 0,
        max_seq_len: int = 128,
    ) -> None:
        self.sequences = [seq[:max_seq_len] for seq in sequences]
        self.pad_id = pad_id
        self.max_len = max(len(s) for s in self.sequences) if self.sequences else 1

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        seq = self.sequences[idx]
        return {
            "input_ids": torch.tensor(seq, dtype=torch.long),
            "attention_mask": torch.ones(len(seq), dtype=torch.bool),
        }

    @staticmethod
    def collate_fn(
        batch: list[dict[str, torch.Tensor]],
        pad_id: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Pad a batch to uniform length."""
        max_len = max(b["input_ids"].size(0) for b in batch)
        input_ids_list, mask_list = [], []
        for b in batch:
            curr_len = b["input_ids"].size(0)
            pad = max_len - curr_len
            input_ids_list.append(
                torch.cat([b["input_ids"], torch.full((pad,), pad_id, dtype=torch.long)])
            )
            mask_list.append(
                torch.cat([b["attention_mask"], torch.zeros(pad, dtype=torch.bool)])
            )
        return {
            "input_ids": torch.stack(input_ids_list),
            "attention_mask": torch.stack(mask_list),
        }


# ===========================================================================
# Trainer
# ===========================================================================

class Trainer:
    """Manages training and validation of a :class:`~src.models.base.BaseModel`.

    Args:
        model: The model to train.
        train_loader: DataLoader yielding ``{input_ids, attention_mask}`` dicts.
        val_loader: DataLoader for validation loss computation.
        output_dir: Directory for checkpoints and logs.
        lr: Learning rate.
        weight_decay: AdamW weight decay.
        max_epochs: Maximum training epochs.
        patience: Early stopping patience (epochs without val improvement).
        grad_clip: Gradient clipping norm (0 = disabled).
        mixed_precision: Enable ``torch.amp`` mixed precision on CUDA.
        device: Training device.
        pad_id: Padding token id (used to mask loss).
    """

    def __init__(
        self,
        model: BaseModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        output_dir: str | Path,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        max_epochs: int = 100,
        patience: int = 10,
        grad_clip: float = 1.0,
        mixed_precision: bool = False,
        device: str | torch.device = "cpu",
        pad_id: int = 0,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.grad_clip = grad_clip
        self.pad_id = pad_id

        self.device = torch.device(device) if isinstance(device, str) else device
        self.model.to(self.device)

        self.optimiser = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # Mixed precision: supported on MPS (bfloat16) and CUDA (float16).
        # MPS does not support GradScaler, so we disable it on MPS.
        self.use_amp = mixed_precision and self.device.type in ("mps", "cuda")
        self.scaler = (
            torch.cuda.amp.GradScaler()
            if (self.use_amp and self.device.type == "cuda")
            else None
        )

        self.metric_logger = TrainingLogger(self.output_dir)
        self._best_val_loss = math.inf
        self._epochs_no_improve = 0
        self._best_checkpoint = self.output_dir / "best_checkpoint.pt"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> dict[str, Any]:
        """Run the full training loop.

        Returns:
            Dictionary with final metrics::

                {
                    "best_val_loss": float,
                    "best_epoch": int,
                    "total_epochs": int,
                    "checkpoint": str,
                }
        """
        best_epoch = 0
        logger.info(
            "Starting training: max_epochs=%d  patience=%d  device=%s",
            self.max_epochs, self.patience, self.device,
        )

        for epoch in range(1, self.max_epochs + 1):
            train_loss = self._train_epoch()
            val_loss, val_ppl = self._val_epoch()
            current_lr = self.optimiser.param_groups[0]["lr"]

            self.metric_logger.log(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_perplexity=val_ppl,
                lr=current_lr,
            )

            if val_loss < self._best_val_loss:
                self._best_val_loss = val_loss
                best_epoch = epoch
                self._epochs_no_improve = 0
                self._save_checkpoint(epoch)
                logger.info(
                    "Epoch %d: val_loss improved to %.4f — checkpoint saved.",
                    epoch, val_loss,
                )
            else:
                self._epochs_no_improve += 1
                logger.info(
                    "Epoch %d: val_loss=%.4f (no improvement %d/%d).",
                    epoch, val_loss, self._epochs_no_improve, self.patience,
                )
                if self._epochs_no_improve >= self.patience:
                    logger.info("Early stopping triggered at epoch %d.", epoch)
                    break

        return {
            "best_val_loss": self._best_val_loss,
            "best_epoch": best_epoch,
            "total_epochs": epoch,
            "checkpoint": str(self._best_checkpoint),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        n_tokens = 0

        for batch in self.train_loader:
            input_ids = batch["input_ids"].to(self.device)
            attn_mask = batch["attention_mask"].to(self.device)

            # Autoregressive: predict token[t+1] from tokens[0:t+1]
            src = input_ids[:, :-1]
            tgt = input_ids[:, 1:]
            src_mask = attn_mask[:, :-1]
            tgt_mask = attn_mask[:, 1:]

            self.optimiser.zero_grad()

            if self.use_amp:
                # Use bfloat16 on MPS, float16 on CUDA
                amp_dtype = (
                    torch.bfloat16
                    if self.device.type == "mps"
                    else torch.float16
                )
                with torch.amp.autocast(device_type=self.device.type, dtype=amp_dtype):
                    logits = self.model(src, attention_mask=src_mask)
                    loss = self._compute_loss(logits, tgt, tgt_mask)
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    if self.grad_clip > 0:
                        self.scaler.unscale_(self.optimiser)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.scaler.step(self.optimiser)
                    self.scaler.update()
                else:
                    loss.backward()
                    if self.grad_clip > 0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimiser.step()
            else:
                logits = self.model(src, attention_mask=src_mask)
                loss = self._compute_loss(logits, tgt, tgt_mask)
                loss.backward()
                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimiser.step()

            n = tgt_mask.sum().item()
            total_loss += loss.item() * n
            n_tokens += n

        return total_loss / max(n_tokens, 1)

    @torch.no_grad()
    def _val_epoch(self) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        n_tokens = 0

        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            attn_mask = batch["attention_mask"].to(self.device)

            src = input_ids[:, :-1]
            tgt = input_ids[:, 1:]
            src_mask = attn_mask[:, :-1]
            tgt_mask = attn_mask[:, 1:]

            logits = self.model(src, attention_mask=src_mask)
            loss = self._compute_loss(logits, tgt, tgt_mask)

            n = tgt_mask.sum().item()
            total_loss += loss.item() * n
            n_tokens += n

        avg_loss = total_loss / max(n_tokens, 1)
        perplexity = math.exp(min(avg_loss, 100))  # cap to avoid overflow
        return avg_loss, perplexity

    def _compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-entropy loss, ignoring padding positions."""
        B, T, V = logits.shape
        return nn.functional.cross_entropy(
            logits.reshape(B * T, V),
            targets.reshape(B * T),
            ignore_index=self.pad_id,
            reduction="mean",
        )

    def _save_checkpoint(self, epoch: int) -> None:
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimiser_state_dict": self.optimiser.state_dict(),
            "val_loss": self._best_val_loss,
            "model_class": self.model.__class__.__name__,
            "vocab_size": self.model.vocab_size,
            "d_model": self.model.d_model,
            "max_seq_len": self.model.max_seq_len,
        }
        torch.save(state, self._best_checkpoint)
