"""MLP supervised classifier for raw CyTOF marker vectors."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.base import VectorBaseModel


class MLPClassifier(VectorBaseModel):
    """Feedforward MLP that maps raw marker vectors to cell-type logits.

    Architecture: n_markers → hidden_dims → d_model → n_classes

    The d_model-dimensional layer is exposed as the cell embedding via
    :meth:`encode`, enabling downstream ARI/NMI evaluation.

    Args:
        n_markers: Number of input marker features.
        d_model: Embedding dimensionality (final hidden layer before head).
        n_classes: Number of cell-type classes.
        dropout: Dropout applied after each hidden layer.
        hidden_dims: Sizes of intermediate hidden layers.
    """

    def __init__(
        self,
        n_markers: int,
        d_model: int = 128,
        n_classes: int = 14,
        dropout: float = 0.1,
        hidden_dims: tuple[int, ...] = (256, 128),
    ) -> None:
        super().__init__(n_markers=n_markers, d_model=d_model, n_classes=n_classes)

        layers: list[nn.Module] = []
        in_dim = n_markers
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.GELU(), nn.Dropout(dropout)]
            in_dim = h
        layers += [nn.Linear(in_dim, d_model), nn.GELU()]
        self.encoder = nn.Sequential(*layers)

        self.head = nn.Linear(d_model, n_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, markers: torch.Tensor) -> torch.Tensor:
        """Return d_model-dimensional cell embedding."""
        return self.encoder(markers)

    def forward(self, markers: torch.Tensor) -> torch.Tensor:
        """Return class logits of shape (batch, n_classes)."""
        return self.head(self.encode(markers))
