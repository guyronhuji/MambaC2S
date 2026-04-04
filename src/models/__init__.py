"""Model implementations: Transformer and Mamba."""

from src.models.base import BaseModel
from src.models.transformer import TransformerLM
from src.models.mamba_model import SimpleMambaLM, build_mamba_model

__all__ = ["BaseModel", "TransformerLM", "SimpleMambaLM", "build_mamba_model", "build_model"]


def build_model(config: dict, vocab_size: int) -> "BaseModel":
    """Factory: instantiate a model from config.

    Args:
        config: Full experiment config dict. Reads ``config["model"]``.
        vocab_size: Vocabulary size (from :class:`~src.data.vocab.Vocabulary`).

    Returns:
        An initialized :class:`BaseModel` subclass.

    Raises:
        ValueError: If ``model.type`` is not recognized.
    """
    model_cfg = config.get("model", {})
    model_type = model_cfg.get("type", "transformer").lower()

    if model_type == "transformer":
        return TransformerLM(
            vocab_size=vocab_size,
            d_model=model_cfg.get("d_model", 128),
            n_layers=model_cfg.get("n_layers", 3),
            nhead=model_cfg.get("nhead", 4),
            dropout=model_cfg.get("dropout", 0.1),
            max_seq_len=model_cfg.get("max_seq_len", 128),
            dim_feedforward=model_cfg.get("dim_feedforward", 4 * model_cfg.get("d_model", 128)),
        )
    elif model_type == "mamba":
        return build_mamba_model(
            vocab_size=vocab_size,
            d_model=model_cfg.get("d_model", 128),
            n_layers=model_cfg.get("n_layers", 3),
            dropout=model_cfg.get("dropout", 0.1),
            max_seq_len=model_cfg.get("max_seq_len", 128),
            d_state=model_cfg.get("d_state", 16),
            d_conv=model_cfg.get("d_conv", 4),
            expand=model_cfg.get("expand", 2),
        )
    else:
        raise ValueError(
            f"Unknown model type: {model_type!r}. Choose 'transformer' or 'mamba'."
        )
