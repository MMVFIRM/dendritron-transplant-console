"""More training data (validation fixed). Regularised config at 3000 steps.

Benchmark corpus reference (3000 steps): seed 7 -> 3.877, seed 42 -> 3.866.
"""
import subprocess, sys
from experiments.sweep import sweep, ROOT
R = ["prenorm", "dense_loss", "drop0.2"]
L3 = ["--lr", "5e-2", "--wd", "0.1", "--steps", "3000"]
S = []
for corpus in ("stdlib", "stdlib_full"):
    C = ["--corpus", corpus]
    for seed in ("7", "42"):
        S.append((f"d12/{corpus}/best_3k", R, {}, L3 + C + ["--seed", seed]))
F = ["--corpus", "stdlib_full"]
S.append(("d12/stdlib_full/best_6k", R, {}, ["--lr", "5e-2", "--wd", "0.1", "--steps", "6000"] + F))
for seed in ("7", "42"):
    S.append(("d12/stdlib_full/mlp_kind_3k", R, {"expert_kind": "mlp"}, L3 + F + ["--seed", seed]))
    S.append(("d12/stdlib_full/dense_kind_3k", R, {"expert_kind": "dense"}, L3 + F + ["--seed", seed]))
    S.append(("d12/stdlib_full/no_moe_3k", R + ["no_moe"], {}, L3 + F + ["--seed", seed]))
sweep(S, "experiments/results/batch12.jsonl", threads=1, workers=12)
