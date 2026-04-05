"""MLP autoencoder for unsupervised representation learning on CyTOF marker vectors."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.base import VectorBaseModel


class MLPAutoencoder(VectorBaseModel):
    """Symmetric MLP autoencoder that learns unsupervised cell embeddings.

    Architecture:
        Encoder: n_markers → hidden_dims → latent_dim (= d_model)
        Decoder: latent_dim → reversed(hidden_dims) → n_markers

    The latent code is exposed via :meth:`encode` for downstream ARI/NMI
    evaluation.  :meth:`forward` returns ``(reconstruction, latent_code)``
    for use by the reconstruction trainer.

    Args:
        n_markers: Number of input marker features.
        d_model: Latent dimensionality (the cell embedding).
        dropout: Dropout applied after each hidden layer.
        hidden_dims: Intermediate hidden layer sizes (encoder order).
    """

    def __init__(
        self,
        n_markers: int,
        d_model: int = 32,
        dropout: float = 0.1,
        hidden_dims: tuple[int, ...] = (256, 128),
    ) -> None:
        super().__init__(n_markers=n_markers, d_model=d_model, n_classes=0)

        # Encoder: n_markers → hidden_dims → d_model
        enc_layers: list[nn.Module] = []
        in_dim = n_markers
        for h in hidden_dims:
            enc_layers += [nn.Linear(in_dim, h), nn.GELU(), nn.Dropout(dropout)]
            in_dim = h
        enc_layers.append(nn.Linear(in_dim, d_model))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder: d_model → reversed(hidden_dims) → n_markers
        dec_layers: list[nn.Module] = []
        in_dim = d_model
        for h in reversed(hidden_dims):
            dec_layers += [nn.Linear(in_dim, h), nn.GELU(), nn.Dropout(dropout)]
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, n_markers))
        self.decoder = nn.Sequential(*dec_layers)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, markers: torch.Tensor) -> torch.Tensor:
        """Return latent embedding of shape ``(batch, d_model)``."""
        return self.encoder(markers)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct markers from latent code, shape ``(batch, n_markers)``."""
        return self.decoder(z)

    def forward(self, markers: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode then decode, returning ``(reconstruction, latent_code)``.

        Args:
            markers: Float tensor of shape ``(batch, n_markers)``.

        Returns:
            Tuple of reconstruction ``(batch, n_markers)`` and latent
            code ``(batch, d_model)``.
        """
        z = self.encode(markers)
        recon = self.decode(z)
        return recon, z
