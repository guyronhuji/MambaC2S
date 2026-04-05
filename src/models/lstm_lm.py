"""LSTM autoregressive language model for CyTOF token sequences."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from src.models.base import BaseModel


class LSTMLanguageModel(BaseModel):
    """Stacked LSTM language model.

    Args:
        vocab_size: Vocabulary size.
        d_model: Hidden size of the LSTM (= embedding dim).
        n_layers: Number of stacked LSTM layers.
        dropout: Dropout rate between layers (also on embedding).
        max_seq_len: Maximum sequence length (informational only; LSTM is unbounded).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_layers: int = 3,
        dropout: float = 0.1,
        max_seq_len: int = 128,
    ) -> None:
        super().__init__(vocab_size=vocab_size, d_model=d_model, max_seq_len=max_seq_len)

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.embed_dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # weight tying

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        if self.token_embedding.padding_idx is not None:
            self.token_embedding.weight.data[self.token_embedding.padding_idx].zero_()
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)

    def get_hidden_states(
        self,
        tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.token_embedding(tokens)  # (B, T, D)
        x = self.embed_dropout(x)
        hidden, _ = self.lstm(x)          # (B, T, D)
        return self.norm(hidden)

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.lm_head(self.get_hidden_states(tokens, attention_mask))
