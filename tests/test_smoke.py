"""Smoke tests: verify core modules are importable and instantiable."""

import os
import tempfile

import torch
from torch.utils.data import DataLoader, TensorDataset

from handwritten_rnn.preprocessor import Preprocessor
from handwritten_rnn.dataset import TextDataLoader
from handwritten_rnn.model import CharRNN, TrainingConfig
from handwritten_rnn.trainer import Trainer


def test_preprocessor_importable():
    p = Preprocessor()
    assert p is not None


def test_preprocessor_build_vocab():
    p = Preprocessor()
    p.build_vocab(["hello world"])
    assert p.vocab_size > 0
    assert " " in p.char_to_idx
    assert "h" in p.char_to_idx


def test_preprocessor_encode_decode_roundtrip():
    p = Preprocessor()
    p.build_vocab(["abcdef"])
    encoded = p.encode("abc")
    assert p.decode(encoded) == "abc"


def test_text_data_loader_importable():
    loader = TextDataLoader(source="dummy", seq_len=10, batch_size=2)
    assert loader is not None


# ---------------------------------------------------------------------------
# CharRNN / model.py smoke tests
# ---------------------------------------------------------------------------

def test_charrnn_lstm_instantiation():
    model = CharRNN(vocab_size=20, embed_dim=8, hidden_dim=16, num_layers=1, dropout=0.0, cell_type="lstm")
    assert model is not None
    assert model.cell_type == "lstm"


def test_charrnn_gru_instantiation():
    model = CharRNN(vocab_size=20, embed_dim=8, hidden_dim=16, num_layers=1, dropout=0.0, cell_type="gru")
    assert model is not None
    assert model.cell_type == "gru"


def test_charrnn_forward_output_shape():
    vocab_size, B, T = 20, 4, 10
    model = CharRNN(vocab_size=vocab_size, embed_dim=8, hidden_dim=16, num_layers=1, dropout=0.0)
    x = torch.randint(0, vocab_size, (B, T))
    hidden = model.init_hidden(B)
    logits, hidden_out = model(x, hidden)
    assert logits.shape == (B, T, vocab_size)


def test_charrnn_init_hidden_lstm():
    model = CharRNN(vocab_size=10, embed_dim=8, hidden_dim=16, num_layers=2, dropout=0.0, cell_type="lstm")
    h, c = model.init_hidden(3)
    assert h.shape == (2, 3, 16)
    assert c.shape == (2, 3, 16)


def test_charrnn_init_hidden_gru():
    model = CharRNN(vocab_size=10, embed_dim=8, hidden_dim=16, num_layers=2, dropout=0.0, cell_type="gru")
    h = model.init_hidden(3)
    assert h.shape == (2, 3, 16)


def test_training_config_defaults():
    cfg = TrainingConfig(data_source="data.txt", output_dir="out/")
    assert cfg.seq_len == 100
    assert cfg.batch_size == 64
    assert cfg.cell_type == "lstm"


# ---------------------------------------------------------------------------
# Trainer smoke tests
# ---------------------------------------------------------------------------

def _make_tiny_loaders(vocab_size: int = 10, seq_len: int = 5, n: int = 8, batch_size: int = 4):
    """Build minimal DataLoaders with random integer sequences."""
    inputs = torch.randint(0, vocab_size, (n, seq_len))
    targets = torch.randint(0, vocab_size, (n, seq_len))
    ds = TensorDataset(inputs, targets)
    loader = DataLoader(ds, batch_size=batch_size, drop_last=True)
    return loader


def test_trainer_instantiation():
    vocab_size = 10
    model = CharRNN(vocab_size=vocab_size, embed_dim=8, hidden_dim=16, num_layers=1, dropout=0.0)
    loader = _make_tiny_loaders(vocab_size=vocab_size)
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = TrainingConfig(
            data_source="dummy",
            output_dir=tmpdir,
            checkpoint_dir=tmpdir,
            epochs=1,
            log_interval=1,
        )
        trainer = Trainer(model=model, train_loader=loader, val_loader=loader, config=cfg)
        assert trainer is not None


def test_trainer_runs_one_epoch():
    vocab_size = 10
    model = CharRNN(vocab_size=vocab_size, embed_dim=8, hidden_dim=16, num_layers=1, dropout=0.0)
    loader = _make_tiny_loaders(vocab_size=vocab_size)
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = TrainingConfig(
            data_source="dummy",
            output_dir=tmpdir,
            checkpoint_dir=tmpdir,
            epochs=1,
            log_interval=1,
        )
        trainer = Trainer(model=model, train_loader=loader, val_loader=loader, config=cfg)
        trainer.train()
        assert len(trainer.train_losses) == 1
        assert len(trainer.val_losses) == 1


def test_trainer_saves_checkpoint():
    vocab_size = 10
    model = CharRNN(vocab_size=vocab_size, embed_dim=8, hidden_dim=16, num_layers=1, dropout=0.0)
    loader = _make_tiny_loaders(vocab_size=vocab_size)
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = TrainingConfig(
            data_source="dummy",
            output_dir=tmpdir,
            checkpoint_dir=tmpdir,
            epochs=1,
            log_interval=1,
        )
        trainer = Trainer(model=model, train_loader=loader, val_loader=loader, config=cfg)
        trainer.train()
        assert os.path.exists(os.path.join(tmpdir, "checkpoint.pt"))
