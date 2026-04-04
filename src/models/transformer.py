"""Small causal Transformer language model for CyTOF token sequences.

Architecture:
  - Token embedding + sinusoidal positional encoding
  - N causal (autoregressive) Transformer decoder layers
  - Linear LM head projecting to vocab_size

The causal mask ensures each position can only attend to
earlier positions (standard autoregressive LM objective).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base import BaseModel


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al., 2017).

    Args:
        d_model: Embedding dimensionality.
        max_seq_len: Maximum sequence length.
        dropout: Dropout rate applied to the sum of token + positional embeddings.
    """

    def __init__(self, d_model: int, max_seq_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Pre-compute positional encodings
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_seq_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encodings and apply dropout.

        Args:
            x: Token embeddings, shape ``(batch, seq_len, d_model)``.

        Returns:
            Embeddings with positional encoding added.
        """
        x = x + self.pe[:, : x.size(1), :]  # type: ignore[index]
        return self.dropout(x)


class TransformerLM(BaseModel):
    """Causal Transformer language model for CyTOF marker-token sequences.

    Args:
        vocab_size: Number of tokens in the vocabulary.
        d_model: Model (embedding) dimensionality.
        n_layers: Number of Transformer decoder layers.
        nhead: Number of attention heads.
        dropout: Dropout rate.
        max_seq_len: Maximum supported sequence length.
        dim_feedforward: Hidden size of the feed-forward sublayer.
            Defaults to ``4 * d_model``.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_layers: int = 3,
        nhead: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 128,
        dim_feedforward: Optional[int] = None,
    ) -> None:
        super().__init__(vocab_size=vocab_size, d_model=d_model, max_seq_len=max_seq_len)

        self._causal_mask_cache: dict = {}
        self.n_layers = n_layers
        self.nhead = nhead
        self.dropout_rate = dropout

        if dim_feedforward is None:
            dim_feedforward = 4 * d_model

        # Embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_seq_len, dropout)

        # Transformer encoder (used as decoder with causal mask)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LayerNorm for stability
        )
        encoder_norm = nn.LayerNorm(d_model)
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_layers,
            norm=encoder_norm,
        )

        # Language model head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: tie embedding weights with LM head
        self.lm_head.weight = self.token_embedding.weight

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with small normal distribution."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute next-token logits.

        Args:
            tokens: Token ids, shape ``(batch, seq_len)``.
            attention_mask: Boolean mask ``(batch, seq_len)``.
                ``True`` = valid, ``False`` = padding.

        Returns:
            Logits, shape ``(batch, seq_len, vocab_size)``.
        """
        seq_len = tokens.size(1)
        causal_mask = self._make_causal_mask(seq_len, tokens.device)

        # Key padding mask: True = should be ignored
        key_padding_mask: Optional[torch.Tensor] = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask  # invert: False=valid -> True=ignore

        x = self.token_embedding(tokens)  # (B, T, D)
        x = self.pos_encoding(x)  # (B, T, D)

        x = self.transformer(
            x,
            mask=causal_mask,
            src_key_padding_mask=key_padding_mask,
            is_causal=True,
        )  # (B, T, D)

        logits = self.lm_head(x)  # (B, T, V)
        return logits

    def _make_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Return a cached upper-triangular causal mask (seq_len, seq_len)."""
        key = (seq_len, str(device))
        if key not in self._causal_mask_cache:
            self._causal_mask_cache[key] = torch.triu(
                torch.ones(seq_len, seq_len, device=device), diagonal=1
            ).bool()
        return self._causal_mask_cache[key]

    def get_hidden_states(
        self,
        tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return hidden states before the LM head.

        Args:
            tokens: Token ids, shape ``(batch, seq_len)``.
            attention_mask: Boolean mask ``(batch, seq_len)``.

        Returns:
            Hidden states, shape ``(batch, seq_len, d_model)``.
        """
        seq_len = tokens.size(1)
        causal_mask = self._make_causal_mask(seq_len, tokens.device)

        key_padding_mask: Optional[torch.Tensor] = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask

        x = self.token_embedding(tokens)
        x = self.pos_encoding(x)

        hidden = self.transformer(
            x,
            mask=causal_mask,
            src_key_padding_mask=key_padding_mask,
            is_causal=True,
        )
        return hidden
