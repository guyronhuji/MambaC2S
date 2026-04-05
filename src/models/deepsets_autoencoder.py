"""DeepSets autoencoder for unsupervised, permutation-invariant representation learning."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.base import VectorBaseModel


class DeepSetsAutoencoder(VectorBaseModel):
    """Permutation-invariant DeepSets autoencoder.

    Encoder (permutation-invariant):
        phi: per-marker scalar → d_phi embedding (shared weights across markers)
        mean-pool: (B, n_markers, d_phi) → (B, d_phi)
        rho_enc: d_phi → latent_dim (= d_model)

    Decoder (marker-order fixed):
        Simple MLP: latent_dim → decoder_hidden_dims → n_markers

    The encoder is permutation-invariant because phi is applied independently
    to each marker scalar and the results are mean-pooled.  The decoder maps
    back to a fixed-length vector of the same size as the input, making it
    appropriate for CyTOF data with fixed marker identity.

    Args:
        n_markers: Number of input marker features.
        d_model: Latent dimensionality (the cell embedding).
        d_phi: Hidden dim of the per-marker phi network.
        dropout: Dropout in encoder and decoder.
        decoder_hidden_dims: Hidden dims for the decoder MLP.
    """

    def __init__(
        self,
        n_markers: int,
        d_model: int = 32,
        d_phi: int = 64,
        dropout: float = 0.1,
        decoder_hidden_dims: tuple[int, ...] = (128, 256),
    ) -> None:
        super().__init__(n_markers=n_markers, d_model=d_model, n_classes=0)

        # phi: scalar per marker → d_phi (permutation invariant, shared weights)
        self.phi = nn.Sequential(
            nn.Linear(1, d_phi), nn.GELU(),
            nn.Linear(d_phi, d_phi), nn.GELU(),
        )

        # rho encoder: pooled d_phi → latent d_model
        self.rho_enc = nn.Sequential(
            nn.Linear(d_phi, d_model), nn.GELU(), nn.Dropout(dropout),
        )

        # Decoder: latent → n_markers reconstruction
        dec_layers: list[nn.Module] = []
        in_dim = d_model
        for h in decoder_hidden_dims:
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
        x = markers.unsqueeze(-1)          # (B, n_markers, 1)
        phi_out = self.phi(x)              # (B, n_markers, d_phi)
        pooled = phi_out.mean(dim=1)       # (B, d_phi)
        return self.rho_enc(pooled)        # (B, d_model)

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
