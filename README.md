# Dendritron Transplant Console

A self-contained, GitHub-ready mechanistic demonstration of a **small sparse
recurrent Dendritron recipient** using frozen representations extracted from a
larger pretrained donor.

The demo runs locally on CPU and makes the architecture visible:

```text
Qwen2.5-0.5B used offline
        ↓
frozen phrase + definition representations
        ↓
MACSL-owned VIVERE rank-8 memory cards
        ↓
two recurrent physical blocks
        ↓
top-2 Dendritron experts
        ↓
top-2 nonlinear branches inside each selected expert
        ↓
sparse vocabulary candidates
```

The donor checkpoint is **not included** and is **not loaded at runtime**.

## What the live demo shows

- Exact bigram and trigram address resolution.
- The frozen value row and MACSL card bound to each phrase address.
- Correct, shuffled, semantic-opposite, random, zero, and hash-only memory.
- A fixed recipient evaluated against every memory intervention.
- Two recurrent rounds through two physical blocks.
- Top-2 expert routing and top-2 branch routing inside each selected expert.
- Memory gates, recurrent-state change, sparse output clusters, and candidates.
- Live CPU latency and active-versus-stored parameter accounting.
- Held-out benchmark evidence for the packaged recipient and the five-seed
  phrase-memory (VM32) gate.

The default **File path** example is intentionally diagnostic:

```text
Prompt: the path
Expected next token: to
```

With correct transplanted memory, the expected token is rank 1 in the packaged
recipient with probability 0.69. Replacing only the memory values drops it to
0.02–0.08 and changes the ranking while all recipient weights remain frozen.
Every packaged example satisfies the same test: target at rank 1 and correct
memory beats all five controls.

## One-command launch

### Windows

Double-click:

```text
start_windows.bat
```

The script creates `.venv`, installs the CPU build of PyTorch and the small
remaining dependencies, verifies the packaged assets, and opens the console.

### Linux or macOS

```bash
chmod +x start_unix.sh
./start_unix.sh
```

### Existing Python environment

Python 3.10–3.13 is recommended.

```bash
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
python -m pip install -r requirements.txt
python scripts/verify_assets.py
python launch.py
```

Then open:

```text
http://127.0.0.1:8765
```

Use a different port with:

```bash
python launch.py --port 9000
```

The server binds to `127.0.0.1` by default. Binding to another interface should
be treated as an explicit deployment decision; the demo has no authentication.

## Docker

```bash
docker compose up --build
```

Open `http://127.0.0.1:8765`.

## Test

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The test suite verifies:

- donor provenance and the absence of a live donor;
- exact-address control invariants;
- model execution and recurrent routing;
- the diagnostic fixed-model memory intervention for every packaged example;
- the residual modes, dropout, full-vocabulary loss, and validation-safe
  training-corpus extension;
- HTTP API behavior;
- every packaged asset hash.

## Train and benchmark a recipient

The package can retrain the recipient from the packaged assets on CPU:

```bash
# Recommended compact recipient (706K parameters, prenorm residual, dropout 0.2,
# full-vocabulary loss, non-validation stdlib docstrings as extra training text)
python scripts/train_recipient.py --out build/recipient.pt

# Larger, slightly better recipient (885K parameters)
python scripts/train_recipient.py --preset quality --out build/recipient_quality.pt

# Compare checkpoints on the fixed validation set, memory controls and latency
python scripts/benchmark_recipient.py assets/model/recipient.pt build/recipient.pt

# Re-select the demo's diagnostic examples for a checkpoint
python scripts/select_examples.py build/recipient.pt --out build/examples.json
```

The validation set is always the packaged 1,000 held-out records. Extended
corpora exclude every docstring that matches a validation record. See
`docs/EXPERIMENTS.md` for the results behind the recommended recipe.

## Packaged architecture

| Component | Packaged value |
|---|---:|
| Recipient parameters | 706,063 |
| Model width | 64 |
| Conditional expert parameters | 467,522 |
| Active conditional coefficients/token | 14,225 |
| Conditional activity | 3.0426% |
| Experts | 12, top-2 active |
| Branches per expert | 6, top-2 active |
| Recurrent core | 2 blocks × 2 rounds, pre-norm residual |
| Vocabulary | 2,048 tokens |
| Active vocabulary fraction | 12.5% |
| Phrase rows | 2,000 |
| Definition rows | 5,000 |
| MACSL cards | 16 |
| VIVERE rank | 8 |
| Phrase-memory payload | 257,352 bytes |

The packaged seed-7 recipient (v1.1.0, `compact` preset) was trained with the
centered MACSL/VIVERE bank on the 4,000 benchmark training records plus
non-validation CPython 3.13 standard-library docstrings. On the fixed 1,000
held-out records (33,440 tokens) it reports:

| Metric | v1.0.0 recipient | v1.1.0 recipient |
|---|---:|---:|
| Full-vocabulary NLL | 6.793 | **3.584** |
| Perplexity | 891.8 | **36.0** |
| Next-token accuracy (full / cluster-sparse) | 8.2% / 15.1% | **35.6% / 35.1%** |
| NLL cost of shuffled / opposite / random memory | +0.05 / +0.06 / +0.05 | **+0.17 / +0.19 / +0.17** |
| NLL cost of zero memory / hash-only | +0.04 / +0.04 | +0.08 / +0.12 |
| CPU latency, 16 tokens, one thread | 8.9 ms | 8.9 ms |

Three independent seeds of the v1.1.0 recipe give 3.593, 3.588 and 3.584 NLL.
The retrained recipient depends on its transplanted memory about three times
as much as v1.0.0. These measurements characterize the controlled research
fixture. They are not a general natural-language benchmark.

## Released five-seed VM32 result

The phrase-memory panel is sourced from the Gate 2C-VM32 five-seed report,
measured with the v1.0.0 recipient:

- phrase-memory payload reduction: **2.60×** versus full 32D rows;
- measured perplexity reduction: **3.91%** versus the original uncentered rows;
- MACSL local cards beat one global VIVERE card in **4/5 seeds**;
- reference-runtime latency did not improve, so the runtime-speed subgate is
  reported as **FAIL**, not hidden.

VM32 is a **post-JTD** transfer/compression result. It does not establish that
Qwen has been reproduced or compressed into an equivalent small model.

## Repository layout

```text
assets/                  Frozen recipient and memory artifacts
  data/                  Examples, controls, and benchmark evidence
  definitions/           Frozen definition bank
  memory/                MACSL/VIVERE phrase bank
  model/                 Frozen sparse recipient checkpoint

demo_server/             Local JSON API and inference engine
dendritron_transplant/   Reference model/runtime implementation
web/                     Static console frontend
docs/                    Architecture, demo script, experiments, claims boundary
experiments/             Architecture/training study harness and raw results
scripts/                 Training, benchmarking, verification, packaging
tests/                   Falsification-oriented tests
launch.py                 Local entrypoint
```

## Claims boundary

This repository demonstrates that a smaller sparse recipient can execute on CPU
without its donor, retrieve donor-derived representations by exact phrase
address, and respond differently when the values bound to those addresses are
corrupted.

It does **not** establish:

- Qwen-equivalent language quality;
- a general-purpose chatbot;
- quality-matched superiority over Transformers;
- a native optimized ARM kernel;
- a latency win from VIVERE reconstruction;
- successful raw-896D MACSL transfer, which remains a separate gate;
- an advantage of the multiplicative Dendritron branch over a
  parameter-matched MLP branch (on this benchmark MLP branches are 0.01 NLL
  better; see `docs/EXPERIMENTS.md`).

Read [`docs/CLAIMS_BOUNDARY.md`](docs/CLAIMS_BOUNDARY.md) before publishing
performance claims. Upload instructions are in
[`docs/GITHUB_UPLOAD.md`](docs/GITHUB_UPLOAD.md).

## Third-party licenses and provenance

No license is granted for the original code in this repository. Qwen-derived
numerical assets and CPython-derived definition metadata retain their required
third-party notices. See:

- [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)
- [`assets/official_qwen_provenance.json`](assets/official_qwen_provenance.json)
- [`licenses/APACHE-2.0-QWEN.txt`](licenses/APACHE-2.0-QWEN.txt)
- [`licenses/PSF-LICENSE.txt`](licenses/PSF-LICENSE.txt)
