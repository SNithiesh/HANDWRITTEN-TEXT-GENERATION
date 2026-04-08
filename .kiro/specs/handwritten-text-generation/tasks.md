# Implementation Plan: Handwritten Text Generation

## Overview

Implement a character-level RNN text generation pipeline in Python using PyTorch. The implementation follows the module structure defined in the design: `preprocessor.py`, `dataset.py`, `model.py`, `trainer.py`, `generator.py`, and `cli.py`, packaged as `handwritten_rnn`.

## Tasks

- [x] 1. Set up project structure and dependencies
  - Create the `handwritten_rnn/` package directory with `__init__.py`
  - Create `pyproject.toml` with dependencies: `torch`, `datasets`, `hypothesis`, `pytest`
  - Add `[tool.pytest.ini_options]` and `[tool.hypothesis]` config sections (max_examples=100, integration marker)
  - Create `tests/` directory with `__init__.py`
  - _Requirements: 7.1_

- [x] 2. Implement Preprocessor
  - [x] 2.1 Implement `Preprocessor` class in `handwritten_rnn/preprocessor.py`
    - Implement `build_vocab(texts)`: normalize each text (strip non-printable chars, normalize whitespace), collect unique chars, assign sorted integer indices
    - Implement `encode(text) -> list[int]` and `decode(indices) -> str`
    - Implement `save_vocab(path)`: write `{"char_to_idx": ..., "idx_to_char": ...}` to JSON
    - Implement `load_vocab(path)` classmethod: reconstruct mappings, raise `FileNotFoundError` for missing file, `ValueError` for malformed JSON
    - _Requirements: 1.2, 1.3, 1.4, 2.1, 2.2, 2.3_

  - [ ]* 2.2 Write property test for text normalization (Property 1)
    - **Property 1: Text normalization removes non-printable characters**
    - **Validates: Requirements 1.2**
    - Use `st.text()` with unicode categories; assert no non-printable chars and no leading/trailing/consecutive whitespace in output

  - [ ]* 2.3 Write property test for vocabulary bijectivity (Property 2)
    - **Property 2: Vocabulary is a complete bijection over the corpus**
    - **Validates: Requirements 1.3, 1.4**
    - Use `st.lists(st.text(min_size=1))`; assert char_to_idx and idx_to_char are inverses and cover exactly the unique chars in corpus

  - [ ]* 2.4 Write property test for vocabulary serialization round-trip (Property 5)
    - **Property 5: Vocabulary serialization round-trip**
    - **Validates: Requirements 2.1, 2.2**
    - Use `st.sets(st.characters())`; assert save then load_vocab produces identical mappings

  - [ ]* 2.5 Write unit tests for Preprocessor
    - Test normalization examples (tabs, newlines, non-printable chars)
    - Test OOV character handling in encode
    - Test `FileNotFoundError` on missing vocab file
    - Test `ValueError` on malformed vocab JSON
    - _Requirements: 1.2, 1.3, 1.4, 2.1, 2.2, 2.3_

- [x] 3. Implement DataLoader
  - [x] 3.1 Implement `TextDataLoader` class in `handwritten_rnn/dataset.py`
    - Accept `source` (local file path or HuggingFace dataset identifier), `seq_len`, `batch_size`, `val_ratio`, `seed`
    - Implement `load() -> tuple[DataLoader, DataLoader, Preprocessor]`
    - Build and apply Preprocessor, split into train/val, produce `(inputs, targets)` batches of shape `(batch_size, seq_len)` where targets is inputs shifted by one
    - Raise `ValueError` with source info for invalid/unreachable sources
    - _Requirements: 1.1, 1.5, 1.6, 1.7_

  - [ ]* 3.2 Write property test for train/val split ratio (Property 3)
    - **Property 3: Train/validation split preserves total size and respects ratio**
    - **Validates: Requirements 1.5**
    - Use `st.integers(10, 10000)` and `st.floats(0.05, 0.5)`; assert train+val == N and train fraction is within one sample of expected

  - [ ]* 3.3 Write property test for sequence length (Property 4)
    - **Property 4: All produced sequences have exactly the configured length**
    - **Validates: Requirements 1.7**
    - Use `st.integers(5, 200)` for seq_len; assert every sequence in every batch has exactly seq_len tokens

  - [ ]* 3.4 Write unit tests for DataLoader
    - Test local file loading
    - Test HuggingFace dataset identifier loading (mark `@pytest.mark.integration`)
    - Test `ValueError` raised for invalid source
    - _Requirements: 1.1, 1.6_

- [x] 4. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Implement CharRNN model
  - [x] 5.1 Implement `CharRNN` and `TrainingConfig` in `handwritten_rnn/model.py`
    - Implement `CharRNN(nn.Module)` with embedding + LSTM/GRU + linear head
    - `forward(x, hidden) -> (logits, hidden)`: x shape `(B, T)`, logits shape `(B, T, vocab_size)`
    - `init_hidden(batch_size)`: return zero-initialized hidden state (tuple for LSTM, tensor for GRU)
    - Apply dropout between stacked recurrent layers only (not after final layer)
    - Define `TrainingConfig` dataclass with all fields from design
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

  - [ ]* 5.2 Write property test for RNN output distribution (Property 6)
    - **Property 6: RNN output is a valid logit tensor over the vocabulary**
    - **Validates: Requirements 3.3**
    - Use `st.integers(5, 50)` for dims; assert output shape is `(B, T, vocab_size)` and softmax sums to 1.0 per time step

  - [ ]* 5.3 Write property test for hidden state threading (Property 7)
    - **Property 7: Step-by-step hidden state threading is equivalent to full-sequence forward pass**
    - **Validates: Requirements 3.4**
    - Use `st.integers(1, 20)` for seq_len; assert character-by-character logits match full-sequence logits

  - [ ]* 5.4 Write unit tests for CharRNN
    - Test LSTM and GRU instantiation
    - Test dropout applied in train mode, not in eval mode
    - Test `init_hidden` returns correct shapes for both cell types
    - _Requirements: 3.1, 3.2, 3.5_

- [x] 6. Implement Trainer
  - [x] 6.1 Implement `Trainer` class in `handwritten_rnn/trainer.py`
    - Implement `train()`: cross-entropy loss loop over epochs, gradient clipping, log at `log_interval`, evaluate on val set each epoch
    - Implement `_save_checkpoint(epoch, val_loss)`: save `{epoch, model_state, optimizer_state, train_losses, val_losses}` via `torch.save`
    - Implement `_load_checkpoint() -> int`: load checkpoint if exists, log warning and return 0 if missing
    - Track best model by minimum val loss and save designated best checkpoint on training completion
    - Set random seeds (Python `random`, `numpy`, `torch`) at start of training when seed is configured
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 7.1, 7.2, 7.3_

  - [ ]* 6.2 Write property test for gradient clipping (Property 8)
    - **Property 8: Gradient clipping bounds the global norm**
    - **Validates: Requirements 4.3**
    - Use `st.floats(0.1, 10.0)` for threshold; assert global L2 norm of clipped gradients ≤ threshold

  - [ ]* 6.3 Write property test for checkpoint round-trip (Property 9)
    - **Property 9: Checkpoint round-trip preserves full training state**
    - **Validates: Requirements 4.6**
    - Use `st.integers(1, 100)` for epoch; assert save then load recovers identical model weights, optimizer state, and loss history

  - [ ]* 6.4 Write property test for best checkpoint selection (Property 10)
    - **Property 10: Best checkpoint corresponds to minimum validation loss**
    - **Validates: Requirements 4.8**
    - Use `st.lists(st.floats(0.1, 10.0), min_size=2)`; assert best checkpoint epoch matches argmin of val loss list

  - [ ]* 6.5 Write property test for reproducibility (Property 13)
    - **Property 13: Identical seeds produce identical weights after training**
    - **Validates: Requirements 7.3**
    - Use `st.integers(0, 2**31)` for seed; assert two runs with same seed produce bit-identical weights after epoch 1

  - [ ]* 6.6 Write unit tests for Trainer
    - Test loss logged at configured interval
    - Test val loss recorded each epoch
    - Test resume from checkpoint restores epoch and state
    - Test best-model designation after training
    - _Requirements: 4.4, 4.5, 4.7, 4.8_

- [x] 7. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. Implement Generator
  - [x] 8.1 Implement `Generator` class in `handwritten_rnn/generator.py`
    - Implement `__init__(checkpoint_path, vocab_path)`: load model and vocabulary from disk
    - Implement `generate(seed_text, num_chars, temperature=1.0) -> str`
    - Prime hidden state by encoding and processing seed_text if provided; initialize to zeros otherwise
    - Sample characters using temperature-scaled softmax: `P(c) = softmax(logits / τ)`
    - Raise `ValueError` for OOV characters in seed_text and for temperature ≤ 0
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7_

  - [ ]* 8.2 Write property test for temperature scaling (Property 11)
    - **Property 11: Temperature scaling produces the correct sampling distribution**
    - **Validates: Requirements 5.4**
    - Use `st.floats(0.01, 10.0)` for τ; assert output distribution equals `softmax(logits / τ)` element-wise within float tolerance

  - [ ]* 8.3 Write property test for output length (Property 12)
    - **Property 12: Generated output length equals num_chars**
    - **Validates: Requirements 5.5**
    - Use `st.integers(1, 1000)` for num_chars; assert `len(Generator.generate(...))` == num_chars

  - [ ]* 8.4 Write unit tests for Generator
    - Test seed text priming changes output vs no-seed baseline
    - Test no-seed initializes hidden to zeros
    - Test `ValueError` for OOV character in seed text
    - Test `ValueError` for temperature ≤ 0
    - _Requirements: 5.2, 5.3, 5.6, 5.7_

- [x] 9. Implement CLI
  - [x] 9.1 Implement CLI in `handwritten_rnn/cli.py` and `handwritten_rnn/__main__.py`
    - Implement `train` subcommand with all required arguments from design (data, output-dir, seq-len, batch-size, epochs, lr, hidden-dim, num-layers, cell-type, dropout, grad-clip, seed)
    - Implement `generate` subcommand with all required arguments (checkpoint, vocab, seed-text, num-chars, temperature)
    - Wire `train` to `TextDataLoader` + `Trainer`; wire `generate` to `Generator` printing output to stdout
    - Use `argparse`; missing required args print usage to stderr and exit with code 2
    - _Requirements: 6.1, 6.2, 6.3, 6.4_

  - [ ]* 9.2 Write unit tests for CLI
    - Test all required arguments are parsed correctly for both subcommands
    - Test missing required argument exits with non-zero status
    - Test `generate` output is printed to stdout
    - _Requirements: 6.1, 6.2, 6.3, 6.4_

- [x] 10. Integration wiring and end-to-end test
  - [x] 10.1 Write integration test for train → generate pipeline
    - Train on a small synthetic corpus for 2 epochs, then generate 50 chars; assert output is a non-empty string of correct length
    - _Requirements: 4.1, 5.1, 5.5_

  - [ ]* 10.2 Write integration test for checkpoint resume
    - Train 2 epochs, save checkpoint, resume and train 2 more epochs; assert final weights match an uninterrupted 4-epoch run
    - Mark `@pytest.mark.integration`
    - _Requirements: 4.7_

  - [ ]* 10.3 Write integration test for HuggingFace dataset loading
    - Load a small HuggingFace dataset, run one training step; assert no errors
    - Mark `@pytest.mark.integration`
    - _Requirements: 1.1_

- [x] 11. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- Each task references specific requirements for traceability
- Property tests use Hypothesis with `max_examples=100` as configured in `pyproject.toml`
- Integration tests requiring network access are marked `@pytest.mark.integration` and can be excluded with `-m "not integration"`
