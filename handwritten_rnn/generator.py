"""Generator: load a trained CharRNN and sample new text sequences."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from handwritten_rnn.model import CharRNN
from handwritten_rnn.preprocessor import Preprocessor


class Generator:
    """Loads a trained CharRNN checkpoint and generates text."""

    def __init__(self, checkpoint_path: str, vocab_path: str) -> None:
        """Load model and vocabulary from disk.

        Args:
            checkpoint_path: Path to a `.pt` checkpoint file produced by Trainer.
            vocab_path: Path to the vocabulary JSON file produced by Preprocessor.

        Raises:
            FileNotFoundError: If either file does not exist.
            ValueError: If the checkpoint is missing required keys.
        """
        # Load vocabulary
        self.preprocessor = Preprocessor.load_vocab(vocab_path)

        # Load checkpoint
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Reconstruct model from saved config (preferred) or fall back to defaults
        if "model_config" in state:
            cfg = state["model_config"]
        else:
            # Legacy checkpoints without model_config: infer vocab_size, use defaults
            cfg = {
                "vocab_size": self.preprocessor.vocab_size,
                "embed_dim": 64,
                "hidden_dim": 256,
                "num_layers": 2,
                "dropout": 0.2,
                "cell_type": "lstm",
            }

        self.model = CharRNN(
            vocab_size=cfg["vocab_size"],
            embed_dim=cfg["embed_dim"],
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
            cell_type=cfg["cell_type"],
        )
        self.model.load_state_dict(state["model_state"])
        self.model.eval()

    def generate(
        self,
        seed_text: str,
        num_chars: int,
        temperature: float = 1.0,
    ) -> str:
        """Generate a sequence of characters.

        Args:
            seed_text: Optional priming text. Pass an empty string for no seed.
            num_chars: Number of characters to generate.
            temperature: Controls sampling randomness (τ > 0).
                         Lower values make output more deterministic.

        Returns:
            A string of exactly num_chars generated characters.

        Raises:
            ValueError: If temperature ≤ 0.
            ValueError: If any character in seed_text is not in the vocabulary.
        """
        if temperature <= 0:
            raise ValueError(
                f"Temperature must be a positive value, got {temperature!r}."
            )

        # Validate seed_text characters before doing any work
        if seed_text:
            for ch in seed_text:
                if ch not in self.preprocessor.char_to_idx:
                    raise ValueError(
                        f"Character {ch!r} in seed_text is not in the vocabulary."
                    )

        with torch.no_grad():
            # Initialize hidden state to zeros (batch_size=1)
            hidden = self.model.init_hidden(batch_size=1)

            # Prime hidden state by processing seed_text
            if seed_text:
                seed_indices = self.preprocessor.encode(seed_text)
                seed_tensor = torch.tensor(
                    [seed_indices], dtype=torch.long
                )  # (1, len(seed))
                _, hidden = self.model(seed_tensor, hidden)
                # The last character of seed_text becomes the first input token
                last_idx = seed_indices[-1]
            else:
                # No seed: pick a random starting character index
                last_idx = 0  # will be overridden by the first sampled char

            generated: list[str] = []

            for i in range(num_chars):
                # Input: single character, shape (1, 1)
                x = torch.tensor([[last_idx]], dtype=torch.long)
                logits, hidden = self.model(x, hidden)
                # logits shape: (1, 1, vocab_size) -> (vocab_size,)
                logits_1d = logits[0, 0]

                # Temperature-scaled sampling
                probs = F.softmax(logits_1d / temperature, dim=-1)
                last_idx = torch.multinomial(probs, num_samples=1).item()

                generated.append(self.preprocessor.idx_to_char[last_idx])

        return "".join(generated)
