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
- Five-seed released benchmark evidence for quality, compression, and latency.

The default **Boolean condition** example is intentionally diagnostic:

```text
Prompt: true if
Expected next token: the
```

With correct transplanted memory, the expected token is rank 1 in the packaged
recipient. Replacing only the memory values changes the ranking and output while
all recipient weights remain frozen.

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
- the diagnostic fixed-model memory intervention;
- HTTP API behavior;
- every packaged asset hash.

## Packaged architecture

| Component | Packaged value |
|---|---:|
| Recipient parameters | 532,639 |
| Conditional expert parameters | 351,938 |
| Active conditional coefficients/token | 10,705 |
| Conditional activity | 3.0417% |
| Experts | 12, top-2 active |
| Branches per expert | 6, top-2 active |
| Recurrent core | 2 blocks × 2 rounds |
| Vocabulary | 2,048 tokens |
| Active vocabulary fraction | 12.5% |
| Phrase rows | 2,000 |
| Definition rows | 5,000 |
| MACSL cards | 16 |
| VIVERE rank | 8 |
| Phrase-memory payload | 257,352 bytes |

The packaged seed-7 recipient was trained for this demonstration with the
centered MACSL/VIVERE bank. Its retained validation record reports:

- dense NLL: `6.742524`;
- sparse next-token accuracy: `14.624%`;
- exact phrase-hit rate: `26.733%`;
- exact-hit sparse accuracy: `11.781%`.

These measurements characterize the controlled research fixture. They are not
a general natural-language benchmark.

## Released five-seed VM32 result

The included benchmark panel is sourced from the Gate 2C-VM32 five-seed report:

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
docs/                    Architecture, demo script, and claims boundary
scripts/                 Verification and packaging utilities
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
- successful raw-896D MACSL transfer, which remains a separate gate.

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
