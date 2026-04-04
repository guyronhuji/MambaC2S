"""YAML configuration loading and merging.

Usage::

    cfg = load_config("configs/base.yaml", "configs/transformer.yaml")
    # CLI overrides: "model.d_model=256 training.lr=1e-3"
    cfg = merge_config_overrides(cfg, ["model.d_model=256", "training.lr=1e-3"])
    print(cfg["model"]["d_model"])   # 256
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_config(*paths: str | Path) -> dict[str, Any]:
    """Load one or more YAML files and deep-merge them left-to-right.

    Later files override earlier ones (i.e., *base* first, *experiment* last).

    Args:
        *paths: One or more YAML file paths.

    Returns:
        Merged configuration dictionary.

    Raises:
        FileNotFoundError: If any path does not exist.
    """
    merged: dict[str, Any] = {}
    for path in paths:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")
        with open(p) as f:
            data = yaml.safe_load(f) or {}
        merged = _deep_merge(merged, data)
    return merged


def merge_config_overrides(
    cfg: dict[str, Any],
    overrides: list[str],
) -> dict[str, Any]:
    """Apply ``key=value`` CLI overrides to a config dict.

    Supports dotpath keys (e.g., ``model.d_model=256``).
    Values are parsed with YAML so ``true/false``, integers, and floats
    are handled correctly.

    Args:
        cfg: Base configuration dictionary.
        overrides: List of strings in ``"dotpath.key=value"`` format.

    Returns:
        New config dict with overrides applied (original not mutated).

    Example::

        cfg = merge_config_overrides(cfg, ["model.n_layers=4", "seed=0"])
    """
    cfg = copy.deepcopy(cfg)
    for override in overrides:
        if "=" not in override:
            raise ValueError(
                f"Override must be in 'key=value' format, got: {override!r}"
            )
        key_path, raw_value = override.split("=", 1)
        value = yaml.safe_load(raw_value)
        _set_nested(cfg, key_path.split("."), value)
    return cfg


def save_config(cfg: dict[str, Any], path: str | Path) -> None:
    """Save a configuration dictionary to a YAML file.

    Args:
        cfg: Configuration dictionary.
        path: Destination file path. Parent directories are created if needed.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into *base* (non-destructive)."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _set_nested(d: dict[str, Any], keys: list[str], value: Any) -> None:
    """Set a nested dict value via a list of keys (mutates *d* in-place)."""
    for key in keys[:-1]:
        if key not in d or not isinstance(d[key], dict):
            d[key] = {}
        d = d[key]
    d[keys[-1]] = value
