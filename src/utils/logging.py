"""Structured logging utilities.

Provides:
- ``setup_logging()``  — configures root logger with console + optional file handler.
- ``TrainingLogger``   — writes per-epoch metrics to ``training_log.csv``.
"""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path
from typing import Any, Optional


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str | Path] = None,
) -> None:
    """Configure the root logger.

    Args:
        level: Logging level (e.g., ``logging.DEBUG``).
        log_file: Optional path to a ``.log`` file. If provided, messages
            are written to both the console and the file.
    """
    fmt = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
    ]
    if log_file is not None:
        p = Path(log_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(p, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
        force=True,
    )


class TrainingLogger:
    """CSV-backed per-epoch metric logger.

    Writes one row per epoch to ``training_log.csv`` inside *output_dir*.
    Metrics are also forwarded to Python's standard ``logging``.

    Args:
        output_dir: Directory where ``training_log.csv`` is written.
        fields: Column names for the CSV.  Defaults to the standard set.

    Example::

        logger = TrainingLogger(output_dir=Path("outputs/exp01"))
        logger.log(epoch=1, train_loss=2.3, val_loss=2.1, val_perplexity=8.1)
    """

    DEFAULT_FIELDS = [
        "epoch",
        "train_loss",
        "val_loss",
        "val_perplexity",
        "lr",
    ]

    def __init__(
        self,
        output_dir: str | Path,
        fields: Optional[list[str]] = None,
    ) -> None:
        self._dir = Path(output_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / "training_log.csv"
        self._fields = fields if fields is not None else list(self.DEFAULT_FIELDS)
        self._logger = logging.getLogger(__name__)
        self._initialised = False

    def _ensure_header(self) -> None:
        if not self._initialised:
            with open(self._path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._fields, extrasaction="ignore")
                writer.writeheader()
            self._initialised = True

    def log(self, **metrics: Any) -> None:
        """Append a row of metrics to the CSV and log them.

        Args:
            **metrics: Keyword arguments matching the field names.
                Extra keys are silently ignored.
        """
        self._ensure_header()
        with open(self._path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fields, extrasaction="ignore")
            writer.writerow(metrics)

        parts = "  ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in sorted(metrics.items())
        )
        self._logger.info("Epoch metrics — %s", parts)

    @property
    def path(self) -> Path:
        """Path to the CSV file."""
        return self._path
