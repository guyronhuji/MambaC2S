"""Vocabulary for cell token sequences.

Builds a token-to-id mapping from training sequences.
Supports serialization to/from JSON.

Special tokens:
  <PAD>  padding (id=0)
  <UNK>  unknown (id=1)
  <BOS>  beginning of sequence (id=2)
  <EOS>  end of sequence (id=3)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3


class Vocabulary:
    """Token vocabulary for CyTOF marker-token sequences.

    Attributes:
        token2id: Mapping from token string to integer id.
        id2token: Mapping from integer id to token string.
        size: Total number of tokens in the vocabulary.
    """

    def __init__(self) -> None:
        self.token2id: dict[str, int] = {}
        self.id2token: dict[int, str] = {}
        self._initialized = False

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def build(self, sequences: list[list[str]]) -> "Vocabulary":
        """Build vocabulary from a list of token sequences.

        Special tokens are always assigned the first four IDs regardless
        of whether they appear in ``sequences``.

        Args:
            sequences: List of token sequences (training split only —
                do NOT include val/test to prevent leakage).

        Returns:
            ``self`` for chaining.
        """
        token2id: dict[str, int] = {}

        # Reserve special token slots
        for special in SPECIAL_TOKENS:
            token2id[special] = len(token2id)

        # Count and sort regular tokens for deterministic ordering
        seen: set[str] = set()
        for seq in sequences:
            for tok in seq:
                seen.add(tok)

        # Remove special tokens from regular set
        regular = sorted(seen - set(SPECIAL_TOKENS))
        for tok in regular:
            token2id[tok] = len(token2id)

        self.token2id = token2id
        self.id2token = {v: k for k, v in token2id.items()}
        self._initialized = True

        logger.info(
            "Built vocabulary: %d special + %d regular = %d total tokens.",
            len(SPECIAL_TOKENS), len(regular), len(token2id),
        )
        return self

    # ------------------------------------------------------------------
    # Encoding / decoding
    # ------------------------------------------------------------------

    def encode(
        self,
        tokens: list[str],
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        """Convert a token sequence to a list of integer ids.

        Args:
            tokens: Sequence of token strings.
            add_bos: Prepend BOS token before encoding.
            add_eos: Append EOS token after encoding.

        Returns:
            List of integer ids. Unknown tokens map to UNK_ID.
        """
        self._check_initialized()
        if add_bos:
            tokens = [BOS_TOKEN] + list(tokens)
        if add_eos:
            tokens = list(tokens) + [EOS_TOKEN]
        return [self.token2id.get(t, UNK_ID) for t in tokens]

    def decode(self, ids: list[int], skip_special: bool = False) -> list[str]:
        """Convert integer ids back to token strings.

        Args:
            ids: List of integer ids.
            skip_special: If True, omit PAD, BOS, and EOS tokens from output.

        Returns:
            List of token strings.
        """
        self._check_initialized()
        tokens = [self.id2token.get(i, UNK_TOKEN) for i in ids]
        if skip_special:
            skip = {PAD_TOKEN, BOS_TOKEN, EOS_TOKEN}
            tokens = [t for t in tokens if t not in skip]
        return tokens

    def encode_batch(
        self,
        sequences: list[list[str]],
        max_length: Optional[int] = None,
        pad: bool = True,
    ) -> list[list[int]]:
        """Encode a batch of sequences, optionally padding to the same length.

        Args:
            sequences: List of token sequences.
            max_length: Maximum length; sequences longer than this are truncated.
            pad: Whether to pad shorter sequences to ``max_length``.

        Returns:
            List of id lists, all of the same length if ``pad=True``.
        """
        self._check_initialized()
        encoded = [self.encode(seq) for seq in sequences]

        if max_length is not None:
            encoded = [ids[:max_length] for ids in encoded]

        if pad and max_length is not None:
            encoded = [
                ids + [PAD_ID] * (max_length - len(ids)) for ids in encoded
            ]

        return encoded

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save vocabulary to a JSON file.

        Args:
            path: Destination file path.
        """
        self._check_initialized()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "size": self.size,
            "special_tokens": {
                "PAD": PAD_TOKEN,
                "UNK": UNK_TOKEN,
                "BOS": BOS_TOKEN,
                "EOS": EOS_TOKEN,
            },
            "token2id": self.token2id,
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info("Vocabulary saved to %s (%d tokens).", path, self.size)

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        """Load a vocabulary from a JSON file.

        Args:
            path: Path to the JSON file produced by :meth:`save`.

        Returns:
            A fully initialized :class:`Vocabulary` instance.
        """
        path = Path(path)
        with open(path) as f:
            payload = json.load(f)

        vocab = cls()
        vocab.token2id = {k: int(v) for k, v in payload["token2id"].items()}
        vocab.id2token = {v: k for k, v in vocab.token2id.items()}
        vocab._initialized = True
        logger.info("Vocabulary loaded from %s (%d tokens).", path, vocab.size)
        return vocab

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Total number of tokens in the vocabulary."""
        return len(self.token2id)

    @property
    def pad_id(self) -> int:
        """ID of the PAD token."""
        return PAD_ID

    @property
    def unk_id(self) -> int:
        """ID of the UNK token."""
        return UNK_ID

    @property
    def bos_id(self) -> int:
        """ID of the BOS token."""
        return BOS_ID

    @property
    def eos_id(self) -> int:
        """ID of the EOS token."""
        return EOS_ID

    def __len__(self) -> int:
        return self.size

    def __contains__(self, token: str) -> bool:
        return token in self.token2id

    def __repr__(self) -> str:
        return f"Vocabulary(size={self.size}, initialized={self._initialized})"

    def _check_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError(
                "Vocabulary is not initialized. Call build() or load() first."
            )
