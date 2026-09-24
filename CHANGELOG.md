# Changelog

## 1.1.0 — 2026-09-24

- Retrained packaged recipient (706,063 parameters): held-out NLL 3.584
  (perplexity 36.0) versus 6.793 (891.8), next-token accuracy 35.6% versus
  8.2%, three times the NLL cost under corrupted memory, same CPU latency.
- Added `residual_mode` (pre-norm residual), `residual_dropout`, and
  `training_loss` ("full" vocabulary) recipient options; defaults keep v1.0.0
  checkpoints bit-identical.
- Added reproducible training (`scripts/train_recipient.py`, presets
  `compact`, `quality`, `benchmark-corpus`, `packaged`) with a validation-safe
  extension to non-validation CPython standard-library docstrings.
- Added `scripts/benchmark_recipient.py` and `scripts/select_examples.py`.
- Re-selected the ten diagnostic examples for the new recipient with the
  packaged rule; every example ranks its target first and beats all controls.
- Added a recipient benchmark panel; the VM32 panel is labelled as v1.0.0
  phrase-memory evidence.
- Added the architecture and training study (`docs/EXPERIMENTS.md`,
  `experiments/`).

## 1.0.0 — 2026-08-04

- Added live CPU Dendritron recipient console.
- Added mechanistic phrase-memory, definition-memory, recurrent, expert,
  branch, and vocabulary traces.
- Added six fixed-model memory interventions.
- Added centered MACSL/VIVERE rank-8 packaged memory.
- Added five-seed Gate 2C-VM32 benchmark panel.
- Added one-click Windows and Unix launchers, Docker, CI, API docs, manifests,
  provenance, third-party notices, and clean-extraction verification.
