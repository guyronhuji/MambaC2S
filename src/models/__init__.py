"""Model implementations: Transformer, LSTM, GRU, MLP, DeepSets."""

from src.models.base import BaseModel, VectorBaseModel
from src.models.transformer import TransformerLM
from src.models.lstm_lm import LSTMLanguageModel
from src.models.gru_lm import GRULanguageModel
from src.models.mlp_model import MLPClassifier
from src.models.deepsets_model import DeepSetsClassifier
from src.models.mlp_autoencoder import MLPAutoencoder
from src.models.deepsets_autoencoder import DeepSetsAutoencoder

__all__ = [
    "BaseModel",
    "VectorBaseModel",
    "TransformerLM",
    "LSTMLanguageModel",
    "GRULanguageModel",
    "MLPClassifier",
    "DeepSetsClassifier",
    "MLPAutoencoder",
    "DeepSetsAutoencoder",
    "build_model",
    "build_vector_model",
]


def build_model(config: dict, vocab_size: int) -> BaseModel:
    """Factory: instantiate a sequence model from config.

    Args:
        config: Full experiment config dict. Reads ``config["model"]``.
        vocab_size: Vocabulary size.

    Returns:
        An initialized :class:`BaseModel` subclass.

    Raises:
        ValueError: If ``model.type`` is not a recognised sequence model.
    """
    model_cfg = config.get("model", {})
    model_type = model_cfg.get("type", "transformer").lower()
    d_model = model_cfg.get("d_model", 128)
    n_layers = model_cfg.get("n_layers", 3)
    dropout = model_cfg.get("dropout", 0.1)
    max_seq_len = model_cfg.get("max_seq_len", 128)

    if model_type == "transformer":
        return TransformerLM(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            nhead=model_cfg.get("nhead", 4),
            dropout=dropout,
            max_seq_len=max_seq_len,
            dim_feedforward=model_cfg.get("dim_feedforward", 4 * d_model),
        )
    elif model_type == "lstm":
        return LSTMLanguageModel(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )
    elif model_type == "gru":
        return GRULanguageModel(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )
    else:
        raise ValueError(
            f"Unknown sequence model type: {model_type!r}. "
            "Choose 'transformer', 'lstm', or 'gru'."
        )


def build_vector_model(
    config: dict,
    n_markers: int,
    n_classes: int,
) -> VectorBaseModel:
    """Factory: instantiate a vector (cell-level) model from config.

    Args:
        config: Full experiment config dict. Reads ``config["model"]``.
        n_markers: Number of marker features (derived from data).
        n_classes: Number of cell-type classes (derived from data).

    Returns:
        An initialized :class:`VectorBaseModel` subclass.

    Raises:
        ValueError: If ``model.type`` is not a recognised vector model.
    """
    model_cfg = config.get("model", {})
    model_type = model_cfg.get("type", "mlp").lower()
    d_model = model_cfg.get("d_model", 128)
    dropout = model_cfg.get("dropout", 0.1)

    if model_type == "mlp":
        return MLPClassifier(
            n_markers=n_markers,
            d_model=d_model,
            n_classes=n_classes,
            dropout=dropout,
            hidden_dims=tuple(model_cfg.get("hidden_dims", [256, 128])),
        )
    elif model_type == "deepsets":
        return DeepSetsClassifier(
            n_markers=n_markers,
            d_model=d_model,
            n_classes=n_classes,
            d_phi=model_cfg.get("d_phi", 64),
            dropout=dropout,
        )
    elif model_type == "mlp_autoencoder":
        return MLPAutoencoder(
            n_markers=n_markers,
            d_model=d_model,
            dropout=dropout,
            hidden_dims=tuple(model_cfg.get("hidden_dims", [256, 128])),
        )
    elif model_type == "deepsets_autoencoder":
        return DeepSetsAutoencoder(
            n_markers=n_markers,
            d_model=d_model,
            d_phi=model_cfg.get("d_phi", 64),
            dropout=dropout,
            decoder_hidden_dims=tuple(model_cfg.get("decoder_hidden_dims", [128, 256])),
        )
    else:
        raise ValueError(
            f"Unknown vector model type: {model_type!r}. "
            "Choose 'mlp', 'deepsets', 'mlp_autoencoder', or 'deepsets_autoencoder'."
        )
