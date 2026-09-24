"""Longer regularised training: does expert capacity pay off with more steps?

Config: prenorm + full-vocabulary loss + dropout 0.2 + weight decay 0.1, lr 5e-2.
1500 steps seed 7 -> 3.903; 3000 steps seed 7 -> 3.877.
"""
from experiments.sweep import sweep
R = ["prenorm", "dense_loss", "drop0.2"]
L3 = ["--lr", "5e-2", "--wd", "0.1", "--steps", "3000"]
S = [
    ("x11/best_3k", R, {}, L3 + ["--seed", "42"]),
    ("x11/best_6k", R, {}, ["--lr", "5e-2", "--wd", "0.1", "--steps", "6000"]),
    ("x11/drop0.3_3k", ["prenorm", "dense_loss", "drop0.3"], {}, L3),
    ("x11/mlp_kind_3k", R, {"expert_kind": "mlp"}, L3),
    ("x11/dense_kind_3k", R, {"expert_kind": "dense"}, L3),
    ("x11/no_moe_3k", R + ["no_moe"], {}, L3),
    ("x11/mlp_kind_3k", R, {"expert_kind": "mlp"}, L3 + ["--seed", "42"]),
    ("x11/dense_kind_3k", R, {"expert_kind": "dense"}, L3 + ["--seed", "42"]),
]
sweep(S, "experiments/results/batch11.jsonl", threads=1, workers=8)
