"""Trainer: training loop, checkpointing, and resume logic for CharRNN."""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from handwritten_rnn.model import CharRNN, TrainingConfig

logger = logging.getLogger(__name__)


class Trainer:
    """Trains a CharRNN model with checkpointing and resume support."""

    def __init__(
        self,
        model: CharRNN,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        self.criterion = nn.CrossEntropyLoss()

        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

        os.makedirs(config.checkpoint_dir, exist_ok=True)
        logger.info("Using device: %s", self.device)

    # ------------------------------------------------------------------
    # Checkpoint paths
    # ------------------------------------------------------------------

    def _checkpoint_path(self) -> str:
        return os.path.join(self.config.checkpoint_dir, "checkpoint.pt")

    def _best_checkpoint_path(self) -> str:
        return os.path.join(self.config.checkpoint_dir, "best.pt")

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def _model_config(self) -> dict:
        """Return model hyperparameters needed to reconstruct the architecture."""
        return {
            "vocab_size": self.model.vocab_size,
            "embed_dim": self.model.embedding.embedding_dim,
            "hidden_dim": self.model.hidden_dim,
            "num_layers": self.model.num_layers,
            "dropout": (
                self.model.rnn.dropout
                if self.model.num_layers > 1
                else 0.0
            ),
            "cell_type": self.model.cell_type,
        }

    def _save_checkpoint(self, epoch: int, val_loss: float) -> None:
        """Save training state to disk."""
        state = {
            "epoch": epoch,
            "model_config": self._model_config(),
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "train_losses": list(self.train_losses),
            "val_losses": list(self.val_losses),
        }
        torch.save(state, self._checkpoint_path())
        logger.info("Checkpoint saved at epoch %d (val_loss=%.4f)", epoch, val_loss)

    def _load_checkpoint(self) -> int:
        """Load checkpoint if it exists; return the starting epoch.

        Returns 0 (start from scratch) if no checkpoint is found.
        """
        path = self._checkpoint_path()
        if not os.path.exists(path):
            logger.warning(
                "No checkpoint found at '%s'. Starting training from scratch.", path
            )
            return 0

        state = torch.load(path, weights_only=False)
        self.model.load_state_dict(state["model_state"])
        self.optimizer.load_state_dict(state["optimizer_state"])
        self.train_losses = state.get("train_losses", [])
        self.val_losses = state.get("val_losses", [])
        start_epoch = state["epoch"] + 1
        logger.info("Resumed from checkpoint at epoch %d.", state["epoch"])
        return start_epoch

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _evaluate(self) -> float:
        """Compute average cross-entropy loss on the validation set."""
        self.model.eval()
        total_loss = 0.0
        total_batches = 0
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                batch_size = inputs.size(0)
                hidden = self.model.init_hidden(batch_size)
                logits, _ = self.model(inputs, hidden)
                # logits: (B, T, V) -> (B*T, V); targets: (B, T) -> (B*T,)
                loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                )
                total_loss += loss.item()
                total_batches += 1
        self.model.train()
        return total_loss / total_batches if total_batches > 0 else float("inf")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Run the full training loop."""
        # Set random seeds for reproducibility
        if self.config.seed is not None:
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)
            torch.manual_seed(self.config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.config.seed)

        start_epoch = _load_checkpoint_if_exists(self)

        self.model.train()
        best_val_loss = min(self.val_losses) if self.val_losses else float("inf")

        for epoch in range(start_epoch, self.config.epochs):
            epoch_loss = 0.0
            num_batches = 0

            for inputs, targets in self.train_loader:
                batch_size = inputs.size(0)
                hidden = self.model.init_hidden(batch_size)

                self.optimizer.zero_grad()
                logits, _ = self.model(inputs, hidden)
                loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                )
                loss.backward()

                # Gradient clipping
                if self.config.grad_clip > 0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip
                    )

                self.optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1

            avg_train_loss = epoch_loss / num_batches if num_batches > 0 else float("inf")
            self.train_losses.append(avg_train_loss)

            # Evaluate on validation set every epoch
            val_loss = self._evaluate()
            self.val_losses.append(val_loss)

            # Log at configured interval
            if (epoch + 1) % self.config.log_interval == 0:
                logger.info(
                    "Epoch %d/%d — train_loss=%.4f  val_loss=%.4f",
                    epoch + 1,
                    self.config.epochs,
                    avg_train_loss,
                    val_loss,
                )

            # Save regular checkpoint
            self._save_checkpoint(epoch, val_loss)

            # Track best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model_config": self._model_config(),
                        "model_state": self.model.state_dict(),
                        "optimizer_state": self.optimizer.state_dict(),
                        "train_losses": list(self.train_losses),
                        "val_losses": list(self.val_losses),
                    },
                    self._best_checkpoint_path(),
                )
                logger.info(
                    "New best model saved at epoch %d (val_loss=%.4f)",
                    epoch + 1,
                    val_loss,
                )

        logger.info(
            "Training complete. Best val_loss=%.4f", best_val_loss
        )


def _load_checkpoint_if_exists(trainer: Trainer) -> int:
    """Delegate to trainer._load_checkpoint(); extracted for testability."""
    return trainer._load_checkpoint()
