"""Reproducibility helpers: seed setting and environment logging."""

from __future__ import annotations

import json
import logging
import platform
import random
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        # MPS does not expose a per-seed API; manual_seed covers it.
        pass
    logger.info("Random seed set to %d.", seed)


def log_environment(output_dir: Optional[str | Path] = None) -> dict[str, Any]:
    """Collect and optionally save environment metadata.

    Args:
        output_dir: If provided, writes ``environment.json`` to this directory.

    Returns:
        Dictionary of environment metadata.
    """
    env: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "mps_available": torch.backends.mps.is_available(),
        "mps_built": torch.backends.mps.is_built(),
    }

    for pkg in ("anndata", "fcsparser", "umap", "sklearn"):
        try:
            mod = __import__(pkg)
            env[pkg] = getattr(mod, "__version__", "installed")
        except ImportError:
            env[pkg] = "not installed"

    if output_dir is not None:
        p = Path(output_dir)
        p.mkdir(parents=True, exist_ok=True)
        with open(p / "environment.json", "w") as f:
            json.dump(env, f, indent=2)
        logger.info("Environment metadata saved to %s.", p / "environment.json")

    return env


def resolve_device(device_str: str) -> torch.device:
    """Resolve a device string with automatic fallback.

    Priority for ``"auto"``: CUDA → MPS → CPU.

    Args:
        device_str: ``"cpu"``, ``"mps"``, ``"cuda"``, ``"cuda:0"``, or ``"auto"``.

    Returns:
        A :class:`torch.device` object.
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            logger.info("Auto-selected device: cuda")
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            logger.info("Auto-selected device: mps")
            return torch.device("mps")
        logger.info("No GPU available — using cpu.")
        return torch.device("cpu")

    if device_str.startswith("cuda"):
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but not available — falling back to CPU.")
            return torch.device("cpu")
        return torch.device(device_str)

    if device_str == "mps":
        if not torch.backends.mps.is_available():
            logger.warning("MPS requested but not available — falling back to CPU.")
            return torch.device("cpu")
        return torch.device("mps")

    return torch.device(device_str)
