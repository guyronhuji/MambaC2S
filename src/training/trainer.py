"""Training loop for CyTOF sequence and vector models.

Supports two modes:
  - ``mode="sequence"`` — autoregressive next-token prediction (Transformer/LSTM/GRU)
  - ``mode="vector"``   — supervised cross-entropy classification (MLP/DeepSets)
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Optional, Union

import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.models.base import BaseModel, VectorBaseModel
from src.utils.logging import TrainingLogger

logger = logging.getLogger(__name__)


# ===========================================================================
# Datasets
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


class CellVectorDataset(Dataset):
    """PyTorch Dataset for raw marker-vector supervised classification.

    Args:
        markers: Float array of shape ``(n_cells, n_markers)``.
        labels: Integer label array of shape ``(n_cells,)``.
    """

    def __init__(self, markers: np.ndarray, labels: np.ndarray) -> None:
        self.markers = torch.tensor(markers, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"markers": self.markers[idx], "labels": self.labels[idx]}


class CellUnlabeledDataset(Dataset):
    """PyTorch Dataset for unsupervised autoencoder training (no labels).

    Args:
        markers: Float array of shape ``(n_cells, n_markers)``.
    """

    def __init__(self, markers: np.ndarray) -> None:
        self.markers = torch.tensor(markers, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.markers)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"markers": self.markers[idx]}


# ===========================================================================
# Trainer
# ===========================================================================

class Trainer:
    """Manages training and validation of sequence or vector models.

    Args:
        model: The model to train (BaseModel or VectorBaseModel).
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        output_dir: Directory for checkpoints and logs.
        lr: Learning rate.
        weight_decay: AdamW weight decay.
        max_epochs: Maximum training epochs.
        patience: Early stopping patience (epochs without val improvement).
        grad_clip: Gradient clipping norm (0 = disabled).
        mixed_precision: Enable ``torch.amp`` mixed precision.
        device: Training device.
        pad_id: Padding token id (sequence mode only).
        mode: ``"sequence"`` for autoregressive LM; ``"vector"`` for supervised
            classifier; ``"reconstruction"`` for unsupervised autoencoder (MSE).
    """

    def __init__(
        self,
        model: Union[BaseModel, VectorBaseModel],
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
        mode: str = "sequence",
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
        self.mode = mode

        self.device = torch.device(device) if isinstance(device, str) else device
        self.model.to(self.device)

        self.optimiser = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # Mixed precision: supported on MPS (bfloat16) and CUDA (float16).
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
            "Starting training: mode=%s  max_epochs=%d  patience=%d  device=%s",
            self.mode, self.max_epochs, self.patience, self.device,
        )

        epoch_bar = tqdm(
            range(1, self.max_epochs + 1),
            desc="Epochs",
            unit="epoch",
            dynamic_ncols=True,
            file=sys.stderr,
        )
        for epoch in epoch_bar:
            if self.mode == "sequence":
                train_loss = self._train_epoch()
                val_loss, val_metric = self._val_epoch()
                metric_label, metric_fmt = "ppl", f"{val_metric:.1f}"
            elif self.mode == "reconstruction":
                train_loss = self._train_epoch_reconstruction()
                val_loss, val_metric = self._val_epoch_reconstruction()
                metric_label, metric_fmt = "mse", f"{val_metric:.4f}"
            else:
                train_loss = self._train_epoch_vector()
                val_loss, val_metric = self._val_epoch_vector()
                metric_label, metric_fmt = "acc", f"{val_metric:.4f}"

            current_lr = self.optimiser.param_groups[0]["lr"]
            self.metric_logger.log(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_perplexity=val_metric,
                lr=current_lr,
            )

            improved = val_loss < self._best_val_loss
            if improved:
                self._best_val_loss = val_loss
                best_epoch = epoch
                self._epochs_no_improve = 0
                self._save_checkpoint(epoch)

            epoch_bar.set_postfix(
                {
                    "train": f"{train_loss:.4f}",
                    "val": f"{val_loss:.4f}",
                    metric_label: metric_fmt,
                    "best": f"e{best_epoch}",
                    "": "*" if improved else "",
                },
                refresh=False,
            )

            if not improved:
                self._epochs_no_improve += 1
                if self._epochs_no_improve >= self.patience:
                    tqdm.write(f"Early stopping at epoch {epoch}.")
                    break

        return {
            "best_val_loss": self._best_val_loss,
            "best_epoch": best_epoch,
            "total_epochs": epoch,
            "checkpoint": str(self._best_checkpoint),
        }

    # ------------------------------------------------------------------
    # Sequence mode (autoregressive LM)
    # ------------------------------------------------------------------

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = torch.tensor(0.0, device=self.device)
        n_tokens = torch.tensor(0, device=self.device)

        for batch in self.train_loader:
            input_ids = batch["input_ids"].to(self.device)
            attn_mask = batch["attention_mask"].to(self.device)

            src = input_ids[:, :-1]
            tgt = input_ids[:, 1:]
            src_mask = attn_mask[:, :-1]
            tgt_mask = attn_mask[:, 1:]

            self.optimiser.zero_grad()

            if self.use_amp:
                amp_dtype = torch.bfloat16 if self.device.type == "mps" else torch.float16
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

            n = tgt_mask.sum()
            total_loss = total_loss + loss.detach() * n
            n_tokens = n_tokens + n

        return (total_loss / n_tokens.clamp(min=1)).item()

    @torch.no_grad()
    def _val_epoch(self) -> tuple[float, float]:
        self.model.eval()
        total_loss = torch.tensor(0.0, device=self.device)
        n_tokens = torch.tensor(0, device=self.device)

        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            attn_mask = batch["attention_mask"].to(self.device)

            src = input_ids[:, :-1]
            tgt = input_ids[:, 1:]
            src_mask = attn_mask[:, :-1]
            tgt_mask = attn_mask[:, 1:]

            logits = self.model(src, attention_mask=src_mask)
            loss = self._compute_loss(logits, tgt, tgt_mask)

            n = tgt_mask.sum()
            total_loss = total_loss + loss.detach() * n
            n_tokens = n_tokens + n

        avg_loss = (total_loss / n_tokens.clamp(min=1)).item()
        perplexity = math.exp(min(avg_loss, 100))
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

    # ------------------------------------------------------------------
    # Vector mode (supervised classifier)
    # ------------------------------------------------------------------

    def _train_epoch_vector(self) -> float:
        self.model.train()
        total_loss = torch.tensor(0.0, device=self.device)
        n_cells = torch.tensor(0, device=self.device)

        for batch in self.train_loader:
            markers = batch["markers"].to(self.device)
            labels = batch["labels"].to(self.device)

            self.optimiser.zero_grad()
            logits = self.model(markers)  # (B, n_classes)
            loss = nn.functional.cross_entropy(logits, labels)
            loss.backward()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimiser.step()

            n = torch.tensor(markers.size(0), device=self.device)
            total_loss = total_loss + loss.detach() * n
            n_cells = n_cells + n

        return (total_loss / n_cells.clamp(min=1)).item()

    @torch.no_grad()
    def _val_epoch_vector(self) -> tuple[float, float]:
        """Returns (val_loss, val_accuracy)."""
        self.model.eval()
        total_loss = torch.tensor(0.0, device=self.device)
        n_correct = torch.tensor(0, device=self.device)
        n_cells = torch.tensor(0, device=self.device)

        for batch in self.val_loader:
            markers = batch["markers"].to(self.device)
            labels = batch["labels"].to(self.device)

            logits = self.model(markers)
            loss = nn.functional.cross_entropy(logits, labels)

            n = torch.tensor(markers.size(0), device=self.device)
            total_loss = total_loss + loss.detach() * n
            n_cells = n_cells + n
            n_correct = n_correct + (logits.argmax(dim=-1) == labels).sum()

        avg_loss = (total_loss / n_cells.clamp(min=1)).item()
        accuracy = (n_correct / n_cells.clamp(min=1)).item()
        return avg_loss, accuracy

    # ------------------------------------------------------------------
    # Reconstruction mode (unsupervised autoencoder)
    # ------------------------------------------------------------------

    def _train_epoch_reconstruction(self) -> float:
        self.model.train()
        total_loss = torch.tensor(0.0, device=self.device)
        n_cells = torch.tensor(0, device=self.device)

        for batch in self.train_loader:
            markers = batch["markers"].to(self.device)

            self.optimiser.zero_grad()
            recon, _z = self.model(markers)
            loss = nn.functional.mse_loss(recon, markers)
            loss.backward()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimiser.step()

            n = torch.tensor(markers.size(0), device=self.device)
            total_loss = total_loss + loss.detach() * n
            n_cells = n_cells + n

        return (total_loss / n_cells.clamp(min=1)).item()

    @torch.no_grad()
    def _val_epoch_reconstruction(self) -> tuple[float, float]:
        """Returns (val_loss, val_mse) — both are the same MSE value."""
        self.model.eval()
        total_loss = torch.tensor(0.0, device=self.device)
        n_cells = torch.tensor(0, device=self.device)

        for batch in self.val_loader:
            markers = batch["markers"].to(self.device)
            recon, _z = self.model(markers)
            loss = nn.functional.mse_loss(recon, markers)

            n = torch.tensor(markers.size(0), device=self.device)
            total_loss = total_loss + loss.detach() * n
            n_cells = n_cells + n

        avg_loss = (total_loss / n_cells.clamp(min=1)).item()
        return avg_loss, avg_loss

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int) -> None:
        state: dict[str, Any] = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimiser_state_dict": self.optimiser.state_dict(),
            "val_loss": self._best_val_loss,
            "model_class": self.model.__class__.__name__,
            "d_model": self.model.d_model,
            "mode": self.mode,
        }
        if hasattr(self.model, "vocab_size"):
            state["vocab_size"] = self.model.vocab_size
            state["max_seq_len"] = self.model.max_seq_len
        if hasattr(self.model, "n_markers"):
            state["n_markers"] = self.model.n_markers
            state["n_classes"] = self.model.n_classes
        torch.save(state, self._best_checkpoint)
