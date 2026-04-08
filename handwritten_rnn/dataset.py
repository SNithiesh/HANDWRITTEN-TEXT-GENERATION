"""
TextDataLoader: loads text from a local file or HuggingFace dataset,
preprocesses it, splits into train/val, and returns PyTorch DataLoaders.
"""

import os
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset, random_split

from handwritten_rnn.preprocessor import Preprocessor


class _CharSequenceDataset(Dataset):
    """Flat character-index tensor sliced into (input, target) pairs."""

    def __init__(self, encoded: list[int], seq_len: int) -> None:
        self.seq_len = seq_len
        # Number of complete (input, target) pairs we can form.
        # Each pair needs seq_len + 1 consecutive tokens.
        n_sequences = (len(encoded) - 1) // seq_len
        # Keep only the tokens we actually need.
        total = n_sequences * seq_len + 1
        self.data = torch.tensor(encoded[:total], dtype=torch.long)
        self.n_sequences = n_sequences

    def __len__(self) -> int:
        return self.n_sequences

    def __getitem__(self, idx: int):
        start = idx * self.seq_len
        inputs = self.data[start : start + self.seq_len]
        targets = self.data[start + 1 : start + self.seq_len + 1]
        return inputs, targets


class TextDataLoader:
    """Loads a text corpus and returns train/val DataLoaders plus a Preprocessor.

    Parameters
    ----------
    source:
        Local file path (must end with a readable text file) or a HuggingFace
        dataset identifier (e.g. ``"roneneldan/TinyStories"``).
    seq_len:
        Length of each character sequence fed to the model.
    batch_size:
        Number of sequences per batch.
    val_ratio:
        Fraction of sequences reserved for validation (default 0.1).
    seed:
        Random seed used for the train/val split (default 42).
    """

    def __init__(
        self,
        source: str,
        seq_len: int,
        batch_size: int,
        val_ratio: float = 0.1,
        seed: int = 42,
    ) -> None:
        self.source = source
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.val_ratio = val_ratio
        self.seed = seed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_texts(self) -> list[str]:
        """Return a list of raw text strings from *source*.

        Tries local file first; falls back to HuggingFace ``datasets``.

        Raises
        ------
        ValueError
            If the source cannot be resolved as either a local file or a
            HuggingFace dataset identifier.
        """
        # --- local file ---
        if os.path.exists(self.source):
            try:
                with open(self.source, "r", encoding="utf-8") as fh:
                    return [fh.read()]
            except OSError as exc:
                raise ValueError(
                    f"Cannot read local file '{self.source}': {exc}"
                ) from exc

        # --- HuggingFace dataset ---
        try:
            from datasets import load_dataset  # type: ignore
        except ImportError as exc:
            raise ValueError(
                f"Source '{self.source}' is not a local file and the "
                "'datasets' package is not installed."
            ) from exc

        try:
            ds = load_dataset(self.source)
        except Exception as exc:
            raise ValueError(
                f"Cannot load HuggingFace dataset '{self.source}': {exc}"
            ) from exc

        # Collect text from the first available split.
        split_name = "train" if "train" in ds else next(iter(ds))
        split = ds[split_name]

        # Identify the text column (prefer "text", fall back to first string col).
        text_col: Optional[str] = None
        for col in split.column_names:
            if col == "text":
                text_col = col
                break
        if text_col is None:
            for col in split.column_names:
                if split.features[col].dtype == "string":
                    text_col = col
                    break
        if text_col is None:
            raise ValueError(
                f"HuggingFace dataset '{self.source}' has no string column "
                "to use as text."
            )

        return [row[text_col] for row in split]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> tuple[DataLoader, DataLoader, Preprocessor]:
        """Load the corpus and return ``(train_loader, val_loader, preprocessor)``.

        Each batch yielded by the loaders is a pair ``(inputs, targets)`` of
        shape ``(batch_size, seq_len)`` where ``targets`` is ``inputs`` shifted
        by one position.

        Raises
        ------
        ValueError
            If *source* is invalid or unreachable.
        """
        texts = self._load_texts()

        # Build vocabulary and encode the full corpus.
        preprocessor = Preprocessor()
        preprocessor.build_vocab(texts)

        # Concatenate all normalized texts into one long token stream.
        encoded: list[int] = []
        for text in texts:
            normalized = preprocessor._normalize(text)
            encoded.extend(preprocessor.encode(normalized))

        if len(encoded) < self.seq_len + 1:
            raise ValueError(
                f"Source '{self.source}' produced only {len(encoded)} tokens "
                f"after encoding, which is not enough for seq_len={self.seq_len}."
            )

        full_dataset = _CharSequenceDataset(encoded, self.seq_len)

        n_total = len(full_dataset)
        n_val = max(1, int(n_total * self.val_ratio))
        n_train = n_total - n_val

        generator = torch.Generator().manual_seed(self.seed)
        train_ds, val_ds = random_split(
            full_dataset, [n_train, n_val], generator=generator
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            generator=torch.Generator().manual_seed(self.seed),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
        )

        return train_loader, val_loader, preprocessor
