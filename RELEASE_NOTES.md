# Dendritron Transplant Console v1.0.0

## Included

- Live local CPU inference with the frozen 532,639-parameter recipient.
- 2,000-row MACSL/VIVERE rank-8 phrase memory.
- 5,000-row frozen definition memory.
- Correct, shuffled, semantic-opposite, random, zero, and hash-only controls.
- Recurrent expert/branch routing visualization.
- Sparse vocabulary candidate visualization.
- Five-seed VM32 benchmark evidence.
- Windows, Unix, Docker, API, tests, CI, manifests, provenance, and required third-party notices.

## Validation

- 7/7 release tests pass.
- 19/19 frozen assets verify by SHA-256.
- 80/80 release files verify by SHA-256.
- Clean-extraction tests pass.
- The packaged `true if → the` diagnostic ranks the expected token first under
  correct memory and produces higher NLL under every packaged corruption.

## Boundary

This release is a mechanistic research demo. It is not a Qwen-equivalent
compressed model or a production chatbot.
