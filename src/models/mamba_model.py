"""Mamba / SSM autoregressive language model for CyTOF token sequences.

Two implementations are provided:
  1. ``MambaLM``     — wraps the official ``mamba_ssm`` package when available.
  2. ``SimpleMambaLM`` — a pure-PyTorch SSM-inspired recurrent fallback that
     reproduces the selective-state-space *spirit* (input-dependent gating,
     recurrent hidden state, efficient causal computation) without requiring
     the CUDA-specific ``mamba_ssm`` kernels.

At import time the module tries to import ``mamba_ssm``; if that fails it
silently falls back to ``SimpleMambaLM``.  The ``build_mamba_model()`` factory
always returns a :class:`~src.models.base.BaseModel`-compatible object so the
training and evaluation code is architecture-agnostic.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base import BaseModel

logger = logging.getLogger(__name__)

@torch.jit.script
def _ssm_scan(
    x_act: torch.Tensor,   # (B, T, d_inner)
    B_mat: torch.Tensor,   # (B, T, d_state)
    C_mat: torch.Tensor,   # (B, T, d_state)
    dt: torch.Tensor,      # (B, T, d_inner)
    A: torch.Tensor,       # (d_inner, d_state)
) -> torch.Tensor:         # (B, T, d_inner)
    """ZOH-discretised selective state-space scan (TorchScript — no Python loop overhead)."""
    B, T, d_inner = x_act.shape
    d_state = A.shape[1]
    h = torch.zeros(B, d_inner, d_state, device=x_act.device, dtype=x_act.dtype)
    ys = torch.empty(B, T, d_inner, device=x_act.device, dtype=x_act.dtype)
    for t in range(T):
        dt_t = dt[:, t].unsqueeze(-1)                        # (B, d_inner, 1)
        A_bar_t = torch.exp(A * dt_t)                        # (B, d_inner, d_state)
        B_bar_t = B_mat[:, t].unsqueeze(1) * dt_t            # (B, d_inner, d_state)
        u_t = x_act[:, t].unsqueeze(-1) * B_bar_t            # (B, d_inner, d_state)
        h = A_bar_t * h + u_t                                # (B, d_inner, d_state)
        ys[:, t, :] = (h * C_mat[:, t].unsqueeze(1)).sum(-1) # (B, d_inner)
    return ys


# ---------------------------------------------------------------------------
# Try to import official mamba_ssm
# ---------------------------------------------------------------------------
try:
    from mamba_ssm import Mamba as _MambaBlock  # type: ignore[import]
    _MAMBA_SSM_AVAILABLE = True
    logger.info("mamba_ssm package found — using hardware-optimised Mamba kernels.")
except ImportError:
    _MAMBA_SSM_AVAILABLE = False
    logger.info(
        "mamba_ssm not installed — falling back to pure-PyTorch SSM implementation. "
        "Install with: pip install mamba-ssm"
    )


# ===========================================================================
# Pure-PyTorch fallback: SimpleMambaLM
# ===========================================================================

class _S6Layer(nn.Module):
    """Selective State Space (S6) layer — simplified causal implementation.

    This approximates the key ideas of Mamba:
    - Input-dependent (selective) A, B, C matrices
    - Efficient causal convolution via a 1-D depthwise conv (parallel scan
      approximation; exact recurrence used at inference if needed)
    - Gated output projection

    Args:
        d_model: Model dimensionality.
        d_state: SSM state dimension.
        d_conv: Width of the local depthwise convolution.
        expand: Expansion factor for the inner state.
        dropout: Dropout applied to output.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand

        # Input projection (x and z gate)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Depthwise causal convolution (short-range dependency)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=True,
        )

        # Selective SSM parameters: Δ, B, C
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)  # B, C, log_dt
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        # Fixed log A (diagonal HiPPO-style initialisation)
        A = torch.arange(1, d_state + 1, dtype=torch.float).unsqueeze(0).expand(self.d_inner, -1)
        self.log_A = nn.Parameter(torch.log(A))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, D)
        residual = x
        B, T, D = x.shape

        # Split into value and gate
        xz = self.in_proj(x)  # (B, T, 2*d_inner)
        x_in, z = xz.chunk(2, dim=-1)  # each (B, T, d_inner)

        # Causal depthwise conv: transpose to (B, C, T), apply, crop, transpose back
        x_conv = self.conv1d(x_in.transpose(1, 2))[:, :, :T].transpose(1, 2)  # (B, T, d_inner)
        x_act = F.silu(x_conv)

        # Selective SSM parameters per token
        bcdt = self.x_proj(x_act)  # (B, T, 2*d_state + 1)
        B_mat = bcdt[..., : self.d_state]                        # (B, T, d_state)
        C_mat = bcdt[..., self.d_state: 2 * self.d_state]        # (B, T, d_state)
        log_dt = bcdt[..., -1:]                                   # (B, T, 1)
        dt = F.softplus(self.dt_proj(log_dt))                    # (B, T, d_inner)

        # Selective recurrence via TorchScript scan (eliminates Python loop overhead).
        #   Ā_t = exp(A * dt_t),  B̄_t = B_mat_t * dt_t
        #   h_t  = Ā_t * h_{t-1} + B̄_t * x_t,  y_t = C_t · h_t
        A = -torch.exp(self.log_A)                               # (d_inner, d_state)
        y_ssm = _ssm_scan(x_act, B_mat, C_mat, dt, A)           # (B, T, d_inner)

        # Gate
        y_gated = y_ssm * F.silu(z)

        out = self.out_proj(y_gated)  # (B, T, D)
        out = self.dropout(out)
        return self.norm(out + residual)


class SimpleMambaLM(BaseModel):
    """Pure-PyTorch SSM language model (Mamba-style fallback).

    Used automatically when ``mamba_ssm`` is not installed.

    Args:
        vocab_size: Vocabulary size.
        d_model: Model dimensionality.
        n_layers: Number of S6 layers.
        d_state: SSM state dimension.
        d_conv: Depthwise conv width.
        expand: Inner expansion factor.
        dropout: Dropout rate.
        max_seq_len: Maximum sequence length.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_layers: int = 3,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 128,
    ) -> None:
        super().__init__(vocab_size=vocab_size, d_model=d_model, max_seq_len=max_seq_len)

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.embed_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            _S6Layer(d_model=d_model, d_state=d_state, d_conv=d_conv,
                     expand=expand, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # weight tying

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()

    def get_hidden_states(
        self,
        tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.token_embedding(tokens)  # (B, T, D)
        x = self.embed_dropout(x)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden = self.get_hidden_states(tokens, attention_mask)
        return self.lm_head(hidden)


# ===========================================================================
# Official mamba_ssm wrapper: MambaLM
# ===========================================================================

if _MAMBA_SSM_AVAILABLE:
    class MambaLM(BaseModel):
        """Mamba LM using the official ``mamba_ssm`` CUDA kernels.

        Args:
            vocab_size: Vocabulary size.
            d_model: Model dimensionality.
            n_layers: Number of Mamba blocks.
            d_state: SSM state dimension.
            d_conv: Depthwise conv width.
            expand: Inner expansion factor.
            dropout: Dropout rate (applied to embeddings only; Mamba
                blocks handle internal normalisation).
            max_seq_len: Maximum sequence length.
        """

        def __init__(
            self,
            vocab_size: int,
            d_model: int = 128,
            n_layers: int = 3,
            d_state: int = 16,
            d_conv: int = 4,
            expand: int = 2,
            dropout: float = 0.1,
            max_seq_len: int = 128,
        ) -> None:
            super().__init__(vocab_size=vocab_size, d_model=d_model, max_seq_len=max_seq_len)

            self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
            self.embed_dropout = nn.Dropout(dropout)

            self.layers = nn.ModuleList([
                _MambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
                for _ in range(n_layers)
            ])

            self.norm = nn.LayerNorm(d_model)
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
            self.lm_head.weight = self.token_embedding.weight

        def get_hidden_states(
            self,
            tokens: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            x = self.token_embedding(tokens)
            x = self.embed_dropout(x)
            for layer in self.layers:
                x = layer(x)
            return self.norm(x)

        def forward(
            self,
            tokens: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            hidden = self.get_hidden_states(tokens, attention_mask)
            return self.lm_head(hidden)


# ===========================================================================
# Factory
# ===========================================================================

def build_mamba_model(
    vocab_size: int,
    d_model: int = 128,
    n_layers: int = 3,
    d_state: int = 16,
    d_conv: int = 4,
    expand: int = 2,
    dropout: float = 0.1,
    max_seq_len: int = 128,
) -> BaseModel:
    """Return a Mamba language model.

    Uses the official ``mamba_ssm`` CUDA implementation when available;
    falls back to :class:`SimpleMambaLM` otherwise.

    Args:
        vocab_size: Vocabulary size.
        d_model: Model dimensionality.
        n_layers: Number of Mamba blocks.
        d_state: SSM state dimension.
        d_conv: Depthwise conv kernel size.
        expand: Expansion factor for inner dimension.
        dropout: Dropout rate.
        max_seq_len: Maximum sequence length.

    Returns:
        A :class:`~src.models.base.BaseModel`-compatible Mamba LM.
    """
    kwargs = dict(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        dropout=dropout,
        max_seq_len=max_seq_len,
    )
    if _MAMBA_SSM_AVAILABLE:
        return MambaLM(**kwargs)
    return SimpleMambaLM(**kwargs)
