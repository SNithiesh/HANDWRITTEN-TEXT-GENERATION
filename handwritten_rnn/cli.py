"""Command-line interface for handwritten_rnn: train and generate subcommands."""

from __future__ import annotations

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m handwritten_rnn",
        description="Character-level RNN for handwritten text generation.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.required = True

    # ------------------------------------------------------------------
    # train subcommand
    # ------------------------------------------------------------------
    train_p = subparsers.add_parser(
        "train",
        help="Train a CharRNN model on a text corpus.",
    )
    train_p.add_argument(
        "--data", required=True, metavar="SOURCE",
        help="Local file path or HuggingFace dataset identifier.",
    )
    train_p.add_argument(
        "--output-dir", required=True, metavar="DIR",
        help="Directory where checkpoints and vocabulary are saved.",
    )
    train_p.add_argument(
        "--seq-len", type=int, default=100, metavar="N",
        help="Sequence length for training (default: 100).",
    )
    train_p.add_argument(
        "--batch-size", type=int, default=64, metavar="N",
        help="Batch size (default: 64).",
    )
    train_p.add_argument(
        "--epochs", type=int, default=20, metavar="N",
        help="Number of training epochs (default: 20).",
    )
    train_p.add_argument(
        "--lr", type=float, default=1e-3, metavar="LR",
        help="Learning rate (default: 0.001).",
    )
    train_p.add_argument(
        "--hidden-dim", type=int, default=256, metavar="N",
        help="Hidden state dimension (default: 256).",
    )
    train_p.add_argument(
        "--num-layers", type=int, default=2, metavar="N",
        help="Number of stacked RNN layers (default: 2).",
    )
    train_p.add_argument(
        "--cell-type", default="lstm", choices=["lstm", "gru"], metavar="TYPE",
        help="Recurrent cell type: lstm or gru (default: lstm).",
    )
    train_p.add_argument(
        "--dropout", type=float, default=0.2, metavar="P",
        help="Dropout rate between recurrent layers (default: 0.2).",
    )
    train_p.add_argument(
        "--grad-clip", type=float, default=5.0, metavar="NORM",
        help="Gradient clipping threshold (default: 5.0).",
    )
    train_p.add_argument(
        "--seed", type=int, default=42, metavar="SEED",
        help="Random seed for reproducibility (default: 42).",
    )

    # ------------------------------------------------------------------
    # generate subcommand
    # ------------------------------------------------------------------
    gen_p = subparsers.add_parser(
        "generate",
        help="Generate text from a trained CharRNN checkpoint.",
    )
    gen_p.add_argument(
        "--checkpoint", required=True, metavar="PATH",
        help="Path to a .pt checkpoint file.",
    )
    gen_p.add_argument(
        "--vocab", required=True, metavar="PATH",
        help="Path to the vocabulary JSON file.",
    )
    gen_p.add_argument(
        "--seed-text", default="", metavar="TEXT",
        help="Optional priming text (default: empty string).",
    )
    gen_p.add_argument(
        "--num-chars", type=int, default=500, metavar="N",
        help="Number of characters to generate (default: 500).",
    )
    gen_p.add_argument(
        "--temperature", type=float, default=1.0, metavar="T",
        help="Sampling temperature > 0 (default: 1.0).",
    )

    return parser


def _cmd_train(args: argparse.Namespace) -> None:
    import os
    from handwritten_rnn.dataset import TextDataLoader
    from handwritten_rnn.model import CharRNN, TrainingConfig
    from handwritten_rnn.trainer import Trainer

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    loader = TextDataLoader(
        source=args.data,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        val_ratio=0.1,
        seed=args.seed,
    )
    train_loader, val_loader, preprocessor = loader.load()

    # Save vocabulary alongside checkpoints
    vocab_path = os.path.join(args.output_dir, "vocab.json")
    preprocessor.save_vocab(vocab_path)

    # Build model
    model = CharRNN(
        vocab_size=preprocessor.vocab_size,
        embed_dim=64,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        cell_type=args.cell_type,
    )

    config = TrainingConfig(
        data_source=args.data,
        output_dir=args.output_dir,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        cell_type=args.cell_type,
        dropout=args.dropout,
        grad_clip=args.grad_clip,
        seed=args.seed,
        checkpoint_dir=args.output_dir,
    )

    trainer = Trainer(model, train_loader, val_loader, config)
    trainer.train()


def _cmd_generate(args: argparse.Namespace) -> None:
    from handwritten_rnn.generator import Generator

    gen = Generator(checkpoint_path=args.checkpoint, vocab_path=args.vocab)
    text = gen.generate(
        seed_text=args.seed_text,
        num_chars=args.num_chars,
        temperature=args.temperature,
    )
    print(text)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        _cmd_train(args)
    elif args.command == "generate":
        _cmd_generate(args)
    else:
        parser.print_usage(sys.stderr)
        sys.exit(2)
