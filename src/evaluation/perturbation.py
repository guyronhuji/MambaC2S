"""Perturbation analysis for CyTOF sequence models.

Tests model sensitivity to local token changes:
  - Rank swaps: exchange two rank-adjacent marker tokens
  - Strength bin edits: shift a strength token one bin up or down

For each perturbation, the cosine distance between the original and
perturbed cell embeddings is measured.  Results are aggregated and saved
as ``perturbation.json`` inside the experiment output directory.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from src.models.base import BaseModel

logger = logging.getLogger(__name__)


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance (1 - cosine_similarity) between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return float(1.0 - np.dot(a, b) / (norm_a * norm_b))


def _swap_two_tokens(seq: list[int], idx_a: int, idx_b: int) -> list[int]:
    """Return a copy of *seq* with positions *idx_a* and *idx_b* swapped."""
    s = list(seq)
    s[idx_a], s[idx_b] = s[idx_b], s[idx_a]
    return s


def _shift_strength_token(
    token_id: int,
    vocab: dict[str, int],  # token→id
    bins: list[str],
) -> Optional[int]:
    """Try to shift a strength token one bin up or down.

    Returns the perturbed token id or None if no valid shift exists.
    """
    id_to_token = {v: k for k, v in vocab.items()}
    token = id_to_token.get(token_id)
    if token is None:
        return None
    for bin_label in bins:
        if token.endswith(f"_{bin_label}"):
            prefix = token[: -len(bin_label) - 1]
            current_idx = bins.index(bin_label)
            # Try shifting up first, then down
            for delta in (1, -1):
                new_idx = current_idx + delta
                if 0 <= new_idx < len(bins):
                    new_token = f"{prefix}_{bins[new_idx]}"
                    if new_token in vocab:
                        return vocab[new_token]
    return None


@torch.no_grad()
def run_perturbation_analysis(
    model: BaseModel,
    sequences: list[list[int]],
    vocab: dict[str, int],
    bins: list[str] = None,
    n_cells: int = 200,
    n_perturbations_per_cell: int = 5,
    device: str | torch.device = "cpu",
    pooling: str = "mean",
    seed: int = 42,
    output_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Measure embedding sensitivity to small local token edits.

    For each sampled cell, the function:
    1. Computes the baseline embedding.
    2. Applies up to *n_perturbations_per_cell* random valid token swaps or
       strength-bin shifts.
    3. Records the cosine distance before and after each perturbation.

    Args:
        model: Trained model.
        sequences: List of integer token-id sequences (one per cell).
        vocab: Token-to-id mapping (for strength edits).
        bins: Ordered strength bin labels (e.g. ``["LOW", "MED", "HIGH"]``).
        n_cells: Number of cells to sample.
        n_perturbations_per_cell: Maximum perturbations per cell.
        device: Compute device.
        pooling: Embedding pooling strategy.
        seed: Random seed.
        output_path: If provided, results are saved as JSON here.

    Returns:
        Dictionary with aggregate statistics::

            {
                "n_cells_evaluated": int,
                "n_perturbations": int,
                "mean_cosine_distance": float,
                "std_cosine_distance": float,
                "max_cosine_distance": float,
                "per_cell": [...]
            }
    """
    if bins is None:
        bins = ["LOW", "MED", "HIGH"]

    device = torch.device(device) if isinstance(device, str) else device
    model.eval()
    model.to(device)

    rng = random.Random(seed)
    n_cells = min(n_cells, len(sequences))
    sampled_indices = rng.sample(range(len(sequences)), n_cells)

    all_distances: list[float] = []
    per_cell_results: list[dict[str, Any]] = []

    for cell_idx in sampled_indices:
        orig_seq = sequences[cell_idx]
        if len(orig_seq) < 3:
            continue

        # Baseline embedding
        orig_tensor = torch.tensor([orig_seq], dtype=torch.long, device=device)
        orig_emb = model.encode(orig_tensor, pooling=pooling).cpu().numpy()[0]

        cell_distances: list[float] = []
        for _ in range(n_perturbations_per_cell):
            perturb_type = rng.choice(["swap", "strength"])
            perturbed_seq: Optional[list[int]] = None

            if perturb_type == "swap":
                # Swap two random adjacent positions (skip BOS/EOS)
                positions = list(range(1, len(orig_seq) - 1))
                if len(positions) >= 2:
                    a, b = rng.sample(positions, 2)
                    perturbed_seq = _swap_two_tokens(orig_seq, a, b)

            elif perturb_type == "strength":
                # Shift one random token's strength bin
                positions = list(range(1, len(orig_seq) - 1))
                rng.shuffle(positions)
                for pos in positions:
                    new_id = _shift_strength_token(orig_seq[pos], vocab, bins)
                    if new_id is not None:
                        perturbed_seq = list(orig_seq)
                        perturbed_seq[pos] = new_id
                        break

            if perturbed_seq is None:
                continue

            pert_tensor = torch.tensor([perturbed_seq], dtype=torch.long, device=device)
            pert_emb = model.encode(pert_tensor, pooling=pooling).cpu().numpy()[0]
            dist = _cosine_distance(orig_emb, pert_emb)
            cell_distances.append(dist)
            all_distances.append(dist)

        per_cell_results.append({
            "cell_idx": cell_idx,
            "n_perturbations": len(cell_distances),
            "mean_cosine_distance": float(np.mean(cell_distances)) if cell_distances else 0.0,
        })

    results: dict[str, Any] = {
        "n_cells_evaluated": len(per_cell_results),
        "n_perturbations": len(all_distances),
        "mean_cosine_distance": float(np.mean(all_distances)) if all_distances else 0.0,
        "std_cosine_distance": float(np.std(all_distances)) if all_distances else 0.0,
        "max_cosine_distance": float(np.max(all_distances)) if all_distances else 0.0,
        "per_cell": per_cell_results,
    }

    logger.info(
        "Perturbation analysis: %d cells, %d perturbations, "
        "mean cosine dist=%.4f ± %.4f",
        results["n_cells_evaluated"],
        results["n_perturbations"],
        results["mean_cosine_distance"],
        results["std_cosine_distance"],
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Perturbation results saved to %s.", output_path)

    return results
