"""CharRNN model and TrainingConfig for character-level text generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

import torch
import torch.nn as nn
from torch import Tensor

# Hidden state type: tuple[Tensor, Tensor] for LSTM, Tensor for GRU
Hidden = Union[tuple[Tensor, Tensor], Tensor]


class CharRNN(nn.Module):
    """Character-level RNN with embedding + LSTM/GRU + linear head."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        cell_type: str = "lstm",
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cell_type = cell_type.lower()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Dropout between stacked layers is handled by PyTorch's built-in
        # dropout parameter (applied between layers, not after the final layer).
        rnn_dropout = dropout if num_layers > 1 else 0.0
        if self.cell_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=embed_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=rnn_dropout,
            )
        elif self.cell_type == "gru":
            self.rnn = nn.GRU(
                input_size=embed_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=rnn_dropout,
            )
        else:
            raise ValueError(f"Unsupported cell_type '{cell_type}'. Use 'lstm' or 'gru'.")

        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: Tensor, hidden: Hidden) -> tuple[Tensor, Hidden]:
        """Forward pass.

        Args:
            x: LongTensor of shape (B, T) — token indices.
            hidden: Hidden state from previous step or init_hidden.

        Returns:
            logits: FloatTensor of shape (B, T, vocab_size).
            hidden: Updated hidden state.
        """
        # x: (B, T) -> embedded: (B, T, embed_dim)
        embedded = self.embedding(x)
        # rnn_out: (B, T, hidden_dim)
        rnn_out, hidden = self.rnn(embedded, hidden)
        # logits: (B, T, vocab_size)
        logits = self.linear(rnn_out)
        return logits, hidden

    def init_hidden(self, batch_size: int) -> Hidden:
        """Return zero-initialized hidden state.

        Returns:
            tuple[Tensor, Tensor] for LSTM (h_0, c_0),
            Tensor for GRU (h_0).
        """
        weight = next(self.parameters())
        shape = (self.num_layers, batch_size, self.hidden_dim)
        if self.cell_type == "lstm":
            return (
                weight.new_zeros(shape),
                weight.new_zeros(shape),
            )
        else:  # gru
            return weight.new_zeros(shape)


@dataclass
class TrainingConfig:
    """Configuration for training a CharRNN model."""

    data_source: str
    output_dir: str
    seq_len: int = 100
    batch_size: int = 64
    epochs: int = 20
    lr: float = 1e-3
    hidden_dim: int = 256
    embed_dim: int = 64
    num_layers: int = 2
    cell_type: str = "lstm"       # "lstm" | "gru"
    dropout: float = 0.2
    grad_clip: float = 5.0
    log_interval: int = 1
    val_ratio: float = 0.1
    seed: int = 42
    checkpoint_dir: str = "checkpoints"
