"""Integration tests for the train → generate pipeline.

Requirements: 4.1, 5.1, 5.5
"""

import os
import tempfile

import pytest

from handwritten_rnn.dataset import TextDataLoader
from handwritten_rnn.generator import Generator
from handwritten_rnn.model import CharRNN, TrainingConfig
from handwritten_rnn.trainer import Trainer


def test_train_then_generate_pipeline():
    """Train on a small synthetic corpus for 2 epochs, then generate 50 chars.

    Asserts output is a non-empty string of exactly 50 characters.

    Requirements: 4.1, 5.1, 5.5
    """
    corpus = "hello world " * 200  # ~2400 chars, plenty for a tiny model

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write corpus to a temp file
        corpus_path = os.path.join(tmpdir, "corpus.txt")
        with open(corpus_path, "w", encoding="utf-8") as f:
            f.write(corpus)

        vocab_path = os.path.join(tmpdir, "vocab.json")
        checkpoint_dir = os.path.join(tmpdir, "checkpoints")

        # Load data
        data_loader = TextDataLoader(
            source=corpus_path,
            seq_len=20,
            batch_size=16,
            val_ratio=0.1,
            seed=42,
        )
        train_loader, val_loader, preprocessor = data_loader.load()

        # Save vocabulary
        preprocessor.save_vocab(vocab_path)

        # Build a small model
        model = CharRNN(
            vocab_size=preprocessor.vocab_size,
            embed_dim=8,
            hidden_dim=16,
            num_layers=1,
            dropout=0.0,
            cell_type="lstm",
        )

        # Train for 2 epochs
        config = TrainingConfig(
            data_source=corpus_path,
            output_dir=tmpdir,
            checkpoint_dir=checkpoint_dir,
            seq_len=20,
            batch_size=16,
            epochs=2,
            lr=1e-3,
            embed_dim=8,
            hidden_dim=16,
            num_layers=1,
            dropout=0.0,
            grad_clip=5.0,
            log_interval=1,
            seed=42,
        )
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
        )
        trainer.train()

        assert len(trainer.train_losses) == 2, "Expected 2 epochs of training"

        # Generate 50 characters using the saved checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
        assert os.path.exists(checkpoint_path), "Checkpoint file must exist after training"

        generator = Generator(
            checkpoint_path=checkpoint_path,
            vocab_path=vocab_path,
        )
        output = generator.generate(seed_text="", num_chars=50)

        assert isinstance(output, str), "Generated output must be a string"
        assert len(output) == 50, f"Expected 50 chars, got {len(output)}"
        assert len(output) > 0, "Generated output must be non-empty"
