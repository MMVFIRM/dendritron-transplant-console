# Architecture

## Runtime boundary

The console ships no Qwen checkpoint. At runtime it loads only:

1. a 706,063-parameter recipient checkpoint;
2. a 2,000-row VIVERE phrase bank;
3. a 5,000-row frozen definition bank;
4. deterministic address/value corruption controls;
5. a static web interface and dependency-light local HTTP server.

The official donor provenance remains visible, but the donor process is absent.

## Memory path

```text
recipient token IDs
    → longest exact trigram/bigram lookup
        → correct or controlled value binding
            → VIVERE card ID + row coefficients
                → reconstructed early and late 32D vectors
                    → recurrent memory injection

miss
    → deterministic hash address
        → trainable recipient hash memory
```

For every exact control except `hash_only`, phrase keys, exact-hit positions,
address rows, and phrase orders remain fixed. Only the retrieved values change.

## Recurrent Dendritron path

Each of the two physical blocks is visited twice. Every visit records:

- selected experts and signed routing weights;
- selected nonlinear branches inside each expert;
- exact, hash, and definition reads;
- early, late, hash, and definition gates;
- recurrent relative state change;
- active conditional parameter fraction.

The packed Dendritron branch computes multiplicative nonlinear interactions
before integrating branch outputs. Only selected expert/branch slices execute.

### Residual modes

`residual_mode` selects how each visit updates the recurrent state:

- `postnorm` (config default, v1.0.0): `x ← norm(α · norm(α · x + u) + moe)`
  with α = √(2 · depth). The expert update is small relative to the scaled
  state.
- `prenorm` (packaged v1.1.0 recipient): `x ← x + u(norm(x))`, then
  `x ← x + moe(norm(x))`. Every update is added to an unnormalised residual
  stream.

`residual_dropout` applies dropout to each sublayer update during training
only. The config defaults reproduce v1.0.0 checkpoints bit-identically.

## Training objective

`training_loss` selects the vocabulary loss used by `DendritronRecipientLM.loss`:

- `sparse` (config default, v1.0.0): cross-entropy over the candidate tokens of the
  selected clusters only. Tokens outside those clusters are never pushed down,
  so the full-vocabulary distribution is poorly calibrated.
- `full` (packaged v1.1.0 recipient): cross-entropy over the whole vocabulary
  during training. Inference is unchanged and still scores only the selected
  clusters.

Both add the cluster-routing loss. `scripts/train_recipient.py` trains a
recipient; `scripts/benchmark_recipient.py` compares checkpoints. See
`docs/EXPERIMENTS.md` for the study behind the recommended settings.

## Sparse output

The vocabulary head first selects two of sixteen clusters, then computes exact
scores only for tokens in those clusters. The packaged configuration therefore
activates 12.5% of vocabulary clusters per token.

## Falsification view

`POST /api/compare` holds the model checkpoint and input fixed while evaluating:

- correct memory;
- shuffled row binding;
- semantic-opposite binding;
- scale-matched random values;
- zero values with the exact hit retained;
- hash fallback with exact retrieval disabled.

The API reports full-vocabulary target probability, NLL, rank, hidden-state
cosine similarity, Jensen–Shannon divergence, and the exact value row/card used.
