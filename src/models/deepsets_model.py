"""DeepSets classifier for raw CyTOF marker vectors.

Scaffold implementation — each marker is treated as a permutation-invariant
element in a set.  The phi network maps each scalar marker value to a latent
vector; the rho network maps the pooled representation to class logits.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.base import VectorBaseModel


class DeepSetsClassifier(VectorBaseModel):
    """DeepSets permutation-invariant classifier.

    Each of the ``n_markers`` scalar values is processed independently by
    ``phi``, the results are mean-pooled across markers, then ``rho_enc``
    maps the pooled vector to a ``d_model``-dimensional embedding which
    the classification head projects to class logits.

    Args:
        n_markers: Number of input marker features.
        d_model: Embedding dimensionality (exposed via :meth:`encode`).
        n_classes: Number of cell-type classes.
        d_phi: Hidden dimensionality of the per-element phi network.
        dropout: Dropout in the rho encoder.
    """

    def __init__(
        self,
        n_markers: int,
        d_model: int = 128,
        n_classes: int = 14,
        d_phi: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(n_markers=n_markers, d_model=d_model, n_classes=n_classes)

        # phi: scalar per marker → d_phi
        self.phi = nn.Sequential(
            nn.Linear(1, d_phi), nn.GELU(),
            nn.Linear(d_phi, d_phi), nn.GELU(),
        )

        # rho encoder: pooled d_phi → d_model embedding
        self.rho_enc = nn.Sequential(
            nn.Linear(d_phi, d_model), nn.GELU(), nn.Dropout(dropout),
        )

        # classification head
        self.head = nn.Linear(d_model, n_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, markers: torch.Tensor) -> torch.Tensor:
        """Return d_model-dimensional cell embedding.

        Args:
            markers: Float tensor of shape ``(batch, n_markers)``.

        Returns:
            Embeddings of shape ``(batch, d_model)``.
        """
        x = markers.unsqueeze(-1)          # (B, n_markers, 1)
        phi_out = self.phi(x)              # (B, n_markers, d_phi)
        pooled = phi_out.mean(dim=1)       # (B, d_phi)
        return self.rho_enc(pooled)        # (B, d_model)

    def forward(self, markers: torch.Tensor) -> torch.Tensor:
        """Return class logits of shape (batch, n_classes)."""
        return self.head(self.encode(markers))
