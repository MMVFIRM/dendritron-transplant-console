# Dendritron Transplant Console v1.1.0

## Included

- Live local CPU inference with the frozen 706,063-parameter recipient
  (pre-norm residual, dropout 0.2, full-vocabulary training loss).
- 2,000-row MACSL/VIVERE rank-8 phrase memory (unchanged).
- 5,000-row frozen definition memory (unchanged).
- Correct, shuffled, semantic-opposite, random, zero, and hash-only controls.
- Recurrent expert/branch routing visualization.
- Sparse vocabulary candidate visualization.
- Recipient benchmark (v1.0.0 versus v1.1.0) and five-seed VM32 phrase-memory
  evidence.
- Reproducible training, benchmarking, and example-selection scripts.
- Windows, Unix, Docker, API, tests, CI, manifests, provenance, and required
  third-party notices.

## Recipient on the fixed 1,000-record held-out set

| | v1.0.0 | v1.1.0 |
|---|---:|---:|
| Full-vocabulary NLL | 6.793 | 3.584 |
| Perplexity | 891.8 | 36.0 |
| Next-token accuracy | 8.2% | 35.6% |
| NLL cost of corrupted memory | +0.05 | +0.17 |
| CPU latency (16 tokens, 1 thread) | 8.9 ms | 8.9 ms |

## Validation

- 14/14 release tests pass.
- 20/20 frozen assets verify by SHA-256.
- Every packaged diagnostic example ranks its target first under correct
  memory and produces higher NLL under every packaged corruption; the default
  `the path → to` example gives the target probability 0.69, against
  0.02–0.08 under the controls.

## Boundary

This release is a mechanistic research demo. It is not a Qwen-equivalent
compressed model or a production chatbot. On this benchmark the multiplicative
Dendritron branch shows no advantage over a parameter-matched MLP branch; see
`docs/EXPERIMENTS.md`.
