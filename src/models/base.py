"""Abstract base classes for all CyTOF models.

Sequence models (Transformer, LSTM, GRU):
  - BaseModel: forward() → next-token logits, encode() → pooled embedding

Vector models (MLP, DeepSets):
  - VectorBaseModel: forward() → class logits, encode() → d_model embedding
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, Optional

import torch
import torch.nn as nn


class BaseModel(ABC, nn.Module):
    """Abstract base for autoregressive CyTOF sequence models.

    Subclasses must implement :meth:`forward` and :meth:`get_hidden_states`.
    :meth:`encode` and :meth:`generate` have default implementations that
    rely on those two methods.
    """

    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

    @abstractmethod
    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute next-token logits for every position.

        Args:
            tokens: Integer token ids, shape ``(batch, seq_len)``.
            attention_mask: Boolean mask of shape ``(batch, seq_len)``.
                ``True`` = valid token, ``False`` = padding.
                If None, all tokens are treated as valid.

        Returns:
            Logits of shape ``(batch, seq_len, vocab_size)``.
        """
        ...

    @abstractmethod
    def get_hidden_states(
        self,
        tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return internal hidden states (no LM head projection).

        Args:
            tokens: Integer token ids, shape ``(batch, seq_len)``.
            attention_mask: Boolean mask of shape ``(batch, seq_len)``.

        Returns:
            Hidden states of shape ``(batch, seq_len, d_model)``.
        """
        ...

    def encode(
        self,
        tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pooling: Literal["mean", "last"] = "mean",
    ) -> torch.Tensor:
        """Extract a fixed-size cell embedding via pooling.

        Args:
            tokens: Integer token ids, shape ``(batch, seq_len)``.
            attention_mask: Boolean mask; ``True`` = real token.
            pooling: ``"mean"`` averages over non-padding positions;
                ``"last"`` uses the last non-padding hidden state.

        Returns:
            Embeddings of shape ``(batch, d_model)``.
        """
        hidden = self.get_hidden_states(tokens, attention_mask)  # (B, T, D)

        if pooling == "mean":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()  # (B, T, 1)
                summed = (hidden * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1.0)
                return summed / counts
            else:
                return hidden.mean(dim=1)

        elif pooling == "last":
            if attention_mask is not None:
                # Last valid position for each sequence
                lengths = attention_mask.long().sum(dim=1) - 1  # (B,)
                lengths = lengths.clamp(min=0)
                batch_size = tokens.size(0)
                idx = lengths.unsqueeze(1).unsqueeze(2).expand(batch_size, 1, self.d_model)
                return hidden.gather(1, idx).squeeze(1)
            else:
                return hidden[:, -1, :]

        else:
            raise ValueError(f"Unknown pooling: {pooling!r}. Choose 'mean' or 'last'.")

    @torch.no_grad()
    def generate(
        self,
        prefix: torch.Tensor,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        do_sample: bool = False,
        eos_id: int = 3,
        pad_id: int = 0,
    ) -> torch.Tensor:
        """Autoregressively generate tokens following a prefix.

        Args:
            prefix: Integer token ids, shape ``(batch, prefix_len)``.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Softmax temperature; lower = more greedy.
            do_sample: If True, sample from the distribution; otherwise greedy.
            eos_id: Stop generation when this token id is produced.
            pad_id: ID used for padding in the output.

        Returns:
            Generated sequences including prefix, shape
            ``(batch, prefix_len + n_generated)``.
        """
        self.eval()
        current = prefix.clone()
        finished = torch.zeros(prefix.size(0), dtype=torch.bool, device=prefix.device)

        for _ in range(max_new_tokens):
            if finished.all():
                break

            logits = self.forward(current)  # (B, T, V)
            next_logits = logits[:, -1, :] / temperature  # (B, V)

            if do_sample:
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)  # (B, 1)

            # Replace finished sequences with PAD
            next_token[finished] = pad_id
            current = torch.cat([current, next_token], dim=1)
            finished = finished | (next_token.squeeze(1) == eos_id)

        return current

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        params = self.count_parameters()
        return (
            f"{self.__class__.__name__}("
            f"vocab_size={self.vocab_size}, "
            f"d_model={self.d_model}, "
            f"params={params:,})"
        )


# ===========================================================================
# VectorBaseModel — for cell-vector classifiers (MLP, DeepSets)
# ===========================================================================

class VectorBaseModel(ABC, nn.Module):
    """Abstract base for supervised cell-vector classifiers.

    Input is a raw marker-value vector (batch, n_markers); output is class
    logits (batch, n_classes).  The :meth:`encode` method returns the
    d_model-dimensional intermediate embedding for downstream quality metrics.
    """

    def __init__(self, n_markers: int, d_model: int, n_classes: int) -> None:
        super().__init__()
        self.n_markers = n_markers
        self.d_model = d_model
        self.n_classes = n_classes

    @abstractmethod
    def forward(self, markers: torch.Tensor) -> torch.Tensor:
        """Compute class logits.

        Args:
            markers: Float tensor, shape ``(batch, n_markers)``.

        Returns:
            Logits of shape ``(batch, n_classes)``.
        """
        ...

    @abstractmethod
    def encode(self, markers: torch.Tensor) -> torch.Tensor:
        """Return d_model-dimensional cell embedding.

        Args:
            markers: Float tensor, shape ``(batch, n_markers)``.

        Returns:
            Embeddings of shape ``(batch, d_model)``.
        """
        ...

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        params = self.count_parameters()
        return (
            f"{self.__class__.__name__}("
            f"n_markers={self.n_markers}, "
            f"d_model={self.d_model}, "
            f"n_classes={self.n_classes}, "
            f"params={params:,})"
        )
