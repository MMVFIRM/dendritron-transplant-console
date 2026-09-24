"""Training length on the full-docstring corpus (regularised recipe)."""
from experiments.sweep import sweep
R = ["prenorm", "dense_loss", "drop0.2"]
F = ["--corpus", "stdlib_full", "--lr", "5e-2", "--wd", "0.1"]
S = [
    ("d13/stdlib_full/best_12k", R, {}, F + ["--steps", "12000"]),
    ("d13/stdlib_full/best_6k", R, {}, F + ["--steps", "6000", "--seed", "42"]),
    ("d13/stdlib_full/mlp_kind_6k", R, {"expert_kind": "mlp"}, F + ["--steps", "6000"]),
]
sweep(S, "experiments/results/batch13.jsonl", threads=2, workers=3)
