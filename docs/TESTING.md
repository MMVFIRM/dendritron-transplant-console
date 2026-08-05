# Testing guide

## Fast integrity check

```bash
python scripts/verify_assets.py
```

## Diagnostic command-line smoke

```bash
python scripts/smoke_demo.py
```

Expected final line:

```text
SMOKE PASS: correct memory is rank 1 and beats every packaged corruption.
```

## Full test suite

```bash
python -m pytest -q
```

## Manual web test

1. Launch `python launch.py`.
2. Confirm the header reports **Donor offline** and **CPU recipient live**.
3. Leave **Boolean condition** selected.
4. Run **Correct transplanted memory**.
5. Confirm the expected token `the` is full-vocabulary rank 1.
6. Confirm one exact phrase hit for `true if`, address row 34, MACSL card 10.
7. Confirm four recurrent visits and two active experts per visit.
8. Click **Compare all controls**.
9. Confirm all five alternatives have higher target NLL than correct memory.
10. Inspect the benchmark panel and claims boundary.

## GitHub CI

The included workflow verifies all frozen asset hashes and runs the full suite on
a GitHub-hosted CPU runner with Python 3.12.
