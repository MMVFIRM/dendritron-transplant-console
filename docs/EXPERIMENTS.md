# Architecture and training experiments

This records the study behind the recommended training recipe. All code and
raw results are in `experiments/` (`harness.py`, `variants.py`, `batch*.py`,
`results/*.jsonl`).

## Protocol

- Data: the packaged CPython-docstring benchmark, rebuilt from
  `assets/definitions` (tokenizer fingerprint matches the packaged one).
  4,000 training records (132,623 tokens), 1,000 validation records.
- Windows of 32 tokens, batch 32, AdamW (β = 0.9, 0.98), 50 warmup steps,
  cosine decay, gradient clip 1.0, weight decay on matrices only.
- Metric: full-vocabulary validation NLL on all 1,045 validation windows
  (33,440 tokens), the same measure as the packaged `dense_nll`.
- Models are trained from scratch in each run. Seeds: 7, 42, 314, 1618.
  Seed-to-seed standard deviation is about 0.045 NLL with the sparse loss and
  about 0.02 with the full-vocabulary loss.

The released benchmark trained for roughly 100 steps; these runs train for
1,500–6,000 steps, so absolute numbers are not comparable with the released
five-seed table.

## Findings

### 1. The expert update was nearly inert in the packaged recipient

In the packaged checkpoint the Dendritron MoE update has RMS ≈ 0.003 against a
residual of RMS ≈ 2.8 (α = 2.83 scaling, 0.18 initialisation gain, product of
two small `tanh` terms, 0.1 output gate). Under the original recipe, removing
the MoE cost only 0.11 NLL and parameter-matched MLP branches performed
identically.

### 2. The training objective dominated everything else

The packaged loss trains only over the ~256 candidate tokens of the selected
clusters. Scoring more of the vocabulary during training improves
full-vocabulary NLL monotonically (prenorm, lr 5e-2, seed 7):

| Vocabulary scored in training | Val NLL |
|---|---|
| 12.5% (2 of 16 clusters) | 6.260 |
| 25% (4 of 16) | 5.999 |
| 50% (8 of 16) | 4.939 |
| 100% (`training_loss="full"`) | 4.002 (4-seed mean 3.987 ± 0.016) |

Inference still scores two clusters: sparse-path accuracy is 32.0% against
32.2% for the full vocabulary. Many earlier "architecture wins" were
compensating for this objective and vanished once it was fixed: prenorm,
width 64, output temperature, candidate-cluster count, the second recurrent
round, LNGram and the learning rate all moved results by less than noise.

### 3. The benchmark is data-limited; regularisation helps

With the full loss the recipient memorises its 133K training tokens: the
unregularised model reaches train NLL 2.35 after 6,000 steps while validation
worsens (4.00 → 4.16). The Dendritron experts memorise the most. Residual
dropout 0.2 and weight decay 0.1 give a consistent gain:

| Recipe (full loss, prenorm) | Val NLL |
|---|---|
| No regularisation, 1,500 steps | 3.984 (2 seeds) |
| Dropout 0.2, 1,500 steps | 3.907 (2 seeds) |
| Dropout 0.2 + wd 0.1, 1,500 steps | 3.899 (3 seeds) |
| Dropout 0.2 + wd 0.1, 3,000 steps | 3.872 (2 seeds) |
| Weight decay 1.0 | 4.019 (underfits) |

### 4. More training data makes the experts pay off

Adding non-validation standard-library docstrings (`--corpus`), with the
recommended recipe at 3,000 steps (2 seeds each):

| Expert layer | Benchmark corpus (133K tokens) | Full docstrings (311K tokens) |
|---|---|---|
| Dendritron experts (533K params, 3% active) | 3.872 | **3.719** |
| MLP-branch experts (532K params, 3% active) | 3.864 | **3.717** |
| Dense gated MLP (195K params) | 3.871 | 3.779 |
| No expert layer | 3.904 (1 seed) | 3.800 |

On the small corpus the sparse experts only tie a dense MLP a third their
size. With 2.3× the data they beat it by 0.06 and beat no experts by 0.08.

More data also makes longer training useful again. On the small corpus 6,000
steps overfit (validation got worse); on the full-docstring corpus:

| Full-docstring corpus | Val NLL |
|---|---|
| Dendritron, 3,000 steps | 3.719 (2 seeds) |
| Dendritron, 6,000 steps | **3.670** (2 seeds: 3.671, 3.669) |
| MLP branches, 6,000 steps | 3.665 (1 seed) |

The extension adds 6,135 non-validation docstrings (311K training tokens).
1,414 scanned docstrings were excluded as possible validation matches, a
stricter filter than the 1,000 validation records require. Validation 8-gram
overlap with training text rises only from 11.3% to 13.5%, in line with the
added volume of shared docstring phrasing.

### 5. Scaling on the larger corpus: width, batch size, learning rate

With the full-docstring corpus the recipient is no longer mainly memorising,
so some capacity changes that failed on the small corpus now help (6,000
steps, dropout 0.2, weight decay 0.1):

| Change (seed 7 unless noted) | Val NLL |
|---|---|
| Width 48, lr 5e-2, batch 32 (previous recipe) | 3.670 (2 seeds) |
| Learning rate 3e-2 | 3.655 (2 seeds) |
| Width 64, lr 3e-2 | 3.618 (3 seeds) |
| Width 80, lr 3e-2 | 3.598 (2 seeds) |
| Width 96, lr 3e-2 | 3.566 |
| Width 128, lr 2e-2 | 3.561 (memorisation gap widens to 0.67) |
| Width 64, lr 3e-2, batch 64 | 3.588 (3 seeds) |
| **Width 80, lr 3e-2, batch 64** | **3.565 (3 seeds: 3.563, 3.565, 3.567)** |
| Width 96, lr 2e-2, batch 64 | 3.573 (overfits: gap 0.83) |

Batch 64 beats twice as many steps at batch 32 (3.628 against 3.657 at
width 48). Neutral or worse on this corpus: 24 experts (3.670), wider
branches (3.687), dropout 0.1 / 0.3 (3.670 / 3.687), no evidence axis
(3.684), GLU branches (3.673), three recurrent rounds (3.664, 1.5× compute).
One recurrent round costs 0.03 (3.701), so the second round now earns its
place.

### 6. Dendritron branches versus MLP branches

Across nearly every setting the multiplicative Dendritron branch and a
parameter-matched MLP branch are within noise of each other. At the final
configuration (width 64, lr 3e-2, batch 64) the MLP branch is slightly but
consistently better:

| Width 64, batch 64 | Seed 7 | Seed 42 | Seed 314 | Mean |
|---|---|---|---|---|
| MLP branches | 3.577 | 3.579 | 3.580 | 3.579 |
| Dendritron branches | 3.593 | 3.588 | 3.584 | 3.588 |

This benchmark does not show an advantage for the Dendritron branch form.

### Changes that did not help

Learnable router temperature, softmax branch weights, load-balancing loss,
GLU branch interaction, larger input gains, layer scale, longer causal kernel
(9), extra causal attention, three recurrent rounds, 24 experts, wider
branches, three active branches, learnable output temperature (diverged).

## Recommended recipes

All use `residual_mode="prenorm"`, `residual_dropout=0.2`,
`training_loss="full"`, weight decay 0.1 and `--corpus stdlib_full`.
`scripts/train_recipient.py` provides them as presets:

| Preset | Width | Batch | LR | Steps | Parameters | Val NLL |
|---|---|---|---|---|---|---|
| `compact` (default) | 64 | 64 | 3e-2 | 6,000 | 706K | 3.588 (3 seeds) |
| `quality` | 80 | 64 | 3e-2 | 6,000 | 885K | 3.565 (3 seeds) |
| `benchmark-corpus` | 48 | 32 | 5e-2 | 3,000 | 533K | 3.872 (2 seeds, packaged data only) |

The package trainer reproduces the harness bit-for-bit on one thread (seed 7,
`benchmark-corpus`: 3.87691 NLL); multi-threaded runs differ by about 0.002.

## Checkpoint benchmark

`python scripts/benchmark_recipient.py` on the fixed validation set (seed 7
checkpoints). Latency is one thread, 16 tokens, median of seven interleaved
rounds on an otherwise idle CPU.

| | Packaged | `compact` | `quality` |
|---|---|---|---|
| Parameters (active conditional fraction) | 533K (3.0%) | 706K (3.0%) | 885K (3.0%) |
| Val NLL / perplexity | 6.793 / 892 | 3.584 / 36.0 | **3.568 / 35.5** |
| Accuracy (dense / cluster-sparse) | 8.2% / 15.1% | 35.6% / 35.1% | **35.7% / 35.3%** |
| NLL cost of shuffled / opposite / random memory | +0.05 / +0.06 / +0.05 | **+0.17 / +0.19 / +0.17** | +0.14 / +0.16 / +0.14 |
| NLL cost of zero memory / hash-only | +0.04 / +0.04 | +0.08 / +0.12 | +0.06 / +0.11 |
| Readable validation phrases: target at rank 1 | 5 / 148 | 79 / 148 | 74 / 148 |
| Readable validation phrases: correct memory beats every control | 93 / 148 | 56 / 148 | 62 / 148 |
| 16-token CPU latency | 8.9 ms | 8.9 ms | 9.4 ms |

Retrained recipients predict far better and depend on the transplanted memory
about three times as much across the validation set, at the same latency.
Because they predict well from context alone, a smaller share of individual
phrases changes rank under corrupted memory. `scripts/select_examples.py`
re-selects the ten demo examples for a checkpoint with the packaged rule
(correct memory must beat every control); for both new checkpoints all ten
selected examples have the target at rank 1.

Release v1.1.0 packages the `compact` seed-7 checkpoint with the Dendritron
branch. The v1.0.0 checkpoint remains available in git history:
`git show 6fe723c:assets/model/recipient.pt > recipient_v1.0.pt`.
