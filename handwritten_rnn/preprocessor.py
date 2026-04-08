"""
Preprocessor: text normalization, vocabulary building, and serialization.
"""

import json
import re
import unicodedata


class Preprocessor:
    """Normalizes raw text and manages the character vocabulary."""

    char_to_idx: dict[str, int]
    idx_to_char: dict[int, str]
    vocab_size: int

    def _normalize(self, text: str) -> str:
        """Strip non-printable characters and normalize whitespace."""
        # Remove non-printable characters (keep printable ones)
        cleaned = "".join(ch for ch in text if ch.isprintable())
        # Normalize whitespace: collapse runs of whitespace to a single space
        cleaned = re.sub(r"\s+", " ", cleaned)
        # Strip leading/trailing whitespace
        cleaned = cleaned.strip()
        return cleaned

    def build_vocab(self, texts: list[str]) -> None:
        """Build vocabulary from a list of texts.

        Normalizes each text, collects unique characters, and assigns
        sorted integer indices.
        """
        unique_chars: set[str] = set()
        for text in texts:
            normalized = self._normalize(text)
            unique_chars.update(normalized)

        sorted_chars = sorted(unique_chars)
        self.char_to_idx = {ch: idx for idx, ch in enumerate(sorted_chars)}
        self.idx_to_char = {idx: ch for idx, ch in enumerate(sorted_chars)}
        self.vocab_size = len(sorted_chars)

    def encode(self, text: str) -> list[int]:
        """Encode a string to a list of integer indices.

        Raises KeyError if a character is not in the vocabulary.
        """
        return [self.char_to_idx[ch] for ch in text]

    def decode(self, indices: list[int]) -> str:
        """Decode a list of integer indices back to a string."""
        return "".join(self.idx_to_char[idx] for idx in indices)

    def save_vocab(self, path: str) -> None:
        """Serialize the vocabulary to a JSON file.

        Writes {"char_to_idx": {...}, "idx_to_char": {...}}.
        """
        data = {
            "char_to_idx": self.char_to_idx,
            # JSON keys must be strings; store int keys as strings
            "idx_to_char": {str(k): v for k, v in self.idx_to_char.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    @classmethod
    def load_vocab(cls, path: str) -> "Preprocessor":
        """Load a vocabulary from a JSON file.

        Raises:
            FileNotFoundError: if the file does not exist.
            ValueError: if the file is not valid JSON or is missing required keys.
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Vocabulary file not found: '{path}'"
            )
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Vocabulary file '{path}' contains malformed JSON: {exc}"
            )

        if "char_to_idx" not in data or "idx_to_char" not in data:
            raise ValueError(
                f"Vocabulary file '{path}' is missing required keys "
                "'char_to_idx' and/or 'idx_to_char'."
            )

        instance = cls()
        instance.char_to_idx = data["char_to_idx"]
        # Convert string keys back to integers
        try:
            instance.idx_to_char = {int(k): v for k, v in data["idx_to_char"].items()}
        except (ValueError, AttributeError) as exc:
            raise ValueError(
                f"Vocabulary file '{path}' has malformed 'idx_to_char' mapping: {exc}"
            )
        instance.vocab_size = len(instance.char_to_idx)
        return instance
