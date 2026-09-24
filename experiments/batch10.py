"""Confirm the regularised configuration and settle the expert question under it.

Config: prenorm + full-vocabulary loss + residual dropout 0.2 + weight decay 0.1, lr 5e-2.
"""
from experiments.sweep import sweep
R = ["prenorm", "dense_loss", "drop0.2"]
L = ["--lr", "5e-2", "--wd", "0.1"]
S = [
    ("x10/best", R, {}, L + ["--seed", "42"]),
    ("x10/best", R, {}, L + ["--seed", "314"]),
    ("x10/drop0.2_wd0.3", R, {}, ["--lr", "5e-2", "--wd", "0.3"]),
    ("x10/drop0.3_wd0.1", ["prenorm", "dense_loss", "drop0.3"], {}, L),
    ("x10/best_3k", R, {}, L + ["--steps", "3000"]),
    ("x10/mlp_kind", R, {"expert_kind": "mlp"}, L),
    ("x10/mlp_kind", R, {"expert_kind": "mlp"}, L + ["--seed", "42"]),
    ("x10/dense_kind", R, {"expert_kind": "dense"}, L),
    ("x10/dense_kind", R, {"expert_kind": "dense"}, L + ["--seed", "42"]),
    ("x10/no_moe", R + ["no_moe"], {}, L),
    ("x10/no_moe", R + ["no_moe"], {}, L + ["--seed", "42"]),
]
sweep(S, "experiments/results/batch10.jsonl", threads=1, workers=11)
