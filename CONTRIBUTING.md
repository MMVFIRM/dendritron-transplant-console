# Contributing

1. Preserve the factual claims boundary.
2. Add or update tests for every change to memory controls, routing traces, or
   reported metrics.
3. Never replace an official provenance field with an inferred value.
4. Never include donor model weights in this repository.
5. Rebuild and verify manifests before publishing a release:

```bash
python scripts/build_manifests.py
python scripts/verify_assets.py
python scripts/verify_release.py
python -m pytest -q
```

Changes that improve visual polish must not convert candidate-normalized
probabilities into full-vocabulary probabilities or otherwise relabel metrics.
