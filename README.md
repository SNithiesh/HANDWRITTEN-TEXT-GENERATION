# Handwritten Text Generation

A character-level Recurrent Neural Network (RNN) that learns patterns from handwritten text and generates new handwritten-like text sequences.

## Live Demo

Try it live on HuggingFace Spaces:
**https://huggingface.co/spaces/nithieshs/handwritten-text-generation**

## GitHub Repository

**https://github.com/SNithiesh/HANDWRITTEN-TEXT-GENERATION**

## How it works

The model trains on the [IAM Handwriting Dataset](https://huggingface.co/datasets/Teklia/IAM-line) — a benchmark dataset of handwritten English text from 657 writers. It learns character-by-character transition probabilities using an LSTM or GRU architecture, then generates new text by sampling from the learned distribution.

## Example Output

```
The Health in Anglesey a more that the home. Allock of the come to the history
of the temporary of brother-ing the political men in the controw must be had been
being changes, in an eventary. A policy of the numerous change. And the conference,
and the entroursed and we should be a young...
```

## Installation

```bash
pip install -e .
```

## Usage

### Train

```bash
python -m handwritten_rnn train --data Teklia/IAM-line --output-dir ./output --epochs 50 --seq-len 80 --batch-size 32 --hidden-dim 128 --num-layers 1 --cell-type gru --seed 42
```

### Generate

```bash
python -m handwritten_rnn generate --checkpoint ./output/best.pt --vocab ./output/vocab.json --seed-text "The" --num-chars 300 --temperature 0.5
```

### CLI Options

**train**
| Argument | Default | Description |
|---|---|---|
| `--data` | required | Local file path or HuggingFace dataset ID |
| `--output-dir` | required | Directory to save checkpoints and vocab |
| `--epochs` | 20 | Number of training epochs |
| `--seq-len` | 100 | Sequence length |
| `--batch-size` | 64 | Batch size |
| `--hidden-dim` | 256 | RNN hidden dimension |
| `--num-layers` | 2 | Number of stacked RNN layers |
| `--cell-type` | lstm | `lstm` or `gru` |
| `--dropout` | 0.2 | Dropout rate |
| `--lr` | 0.001 | Learning rate |
| `--grad-clip` | 5.0 | Gradient clipping threshold |
| `--seed` | 42 | Random seed |

**generate**
| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | required | Path to `.pt` checkpoint file |
| `--vocab` | required | Path to `vocab.json` file |
| `--seed-text` | `""` | Priming text |
| `--num-chars` | 500 | Number of characters to generate |
| `--temperature` | 1.0 | Sampling temperature (lower = more predictable) |

## Project Structure

```
handwritten_rnn/
  preprocessor.py   # Vocabulary building and text normalization
  dataset.py        # Data loading (local file or HuggingFace)
  model.py          # CharRNN architecture + TrainingConfig
  trainer.py        # Training loop with checkpointing and resume
  generator.py      # Text generation with temperature sampling
  cli.py            # Command-line interface
  __main__.py       # Entry point for python -m handwritten_rnn
tests/
  test_smoke.py     # Unit and smoke tests
  test_integration.py  # End-to-end pipeline test
app.py              # Gradio web demo
pyproject.toml      # Dependencies and project config
```

## Dependencies

- Python >= 3.10
- PyTorch
- HuggingFace `datasets`
- Gradio
- `hypothesis` (property-based testing)
- `pytest`

## Dataset

Trained on [Teklia/IAM-line](https://huggingface.co/datasets/Teklia/IAM-line) — the IAM Handwriting Database, the standard benchmark for handwriting research containing ~13,000 lines from 657 writers.
