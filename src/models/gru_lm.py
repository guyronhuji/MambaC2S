"""GRU autoregressive language model for CyTOF token sequences.

Uses a JIT-compiled GRU scan to avoid PyTorch's MPS fallback path:
  - Input projections computed as a single batched matmul (all T at once).
  - Per-timestep recurrence compiled to native code via @torch.jit.script,
    eliminating Python loop overhead and MPS kernel-launch latency.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base import BaseModel


@torch.jit.script
def _gru_scan(
    x_proj: torch.Tensor,   # (B, T, 3*H) — input projections (pre-computed)
    h: torch.Tensor,        # (B, H) — initial hidden state
    W_hh: torch.Tensor,     # (3*H, H)
    b_hh: torch.Tensor,     # (3*H,)
) -> torch.Tensor:          # (B, T, H)
    """JIT-compiled GRU recurrence.

    Splits hidden-to-hidden into r/z/n gates following the standard GRU
    formulation.  The input-to-hidden contribution (x_proj) is pre-computed
    outside this function in one batched matmul.
    """
    B, T, d3 = x_proj.shape
    H = d3 // 3
    out = torch.empty(B, T, H, device=x_proj.device, dtype=x_proj.dtype)
    for t in range(T):
        g_h = h @ W_hh.t() + b_hh          # (B, 3H)
        g_x = x_proj[:, t]                  # (B, 3H)
        r = torch.sigmoid(g_x[:, :H]     + g_h[:, :H])
        z = torch.sigmoid(g_x[:, H:2*H]  + g_h[:, H:2*H])
        n = torch.tanh(   g_x[:, 2*H:]   + r * g_h[:, 2*H:])
        h = (1 - z) * n + z * h
        out[:, t] = h
    return out


class _GRULayer(nn.Module):
    """Single JIT-compiled GRU layer with optional inter-layer dropout."""

    def __init__(self, d_model: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.d_model = d_model
        self.W_ih = nn.Linear(d_model, 3 * d_model, bias=True)   # input → gates
        self.W_hh = nn.Linear(d_model, 3 * d_model, bias=True)   # hidden → gates
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, D) → (B, T, D)
        x_proj = self.W_ih(x)   # (B, T, 3D) — single batched matmul, no loop
        h0 = torch.zeros(x.size(0), self.d_model, device=x.device, dtype=x.dtype)
        out = _gru_scan(x_proj, h0, self.W_hh.weight, self.W_hh.bias)
        return self.dropout(out)


class GRULanguageModel(BaseModel):
    """Stacked JIT-GRU language model.

    Args:
        vocab_size: Vocabulary size.
        d_model: Hidden size of each GRU layer (= embedding dim).
        n_layers: Number of stacked GRU layers.
        dropout: Dropout rate between layers (also on embedding).
        max_seq_len: Maximum sequence length (informational only).
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

        # Inter-layer dropout only between layers (last layer has no dropout)
        self.layers = nn.ModuleList([
            _GRULayer(d_model, dropout=dropout if i < n_layers - 1 else 0.0)
            for i in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # weight tying

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        if self.token_embedding.padding_idx is not None:
            self.token_embedding.weight.data[self.token_embedding.padding_idx].zero_()
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.W_ih.weight)
            nn.init.xavier_uniform_(layer.W_hh.weight)
            nn.init.zeros_(layer.W_ih.bias)
            nn.init.zeros_(layer.W_hh.bias)

    def get_hidden_states(
        self,
        tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.token_embedding(tokens)   # (B, T, D)
        x = self.embed_dropout(x)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.lm_head(self.get_hidden_states(tokens, attention_mask))
