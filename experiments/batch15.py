"""Combine width 64 and lr 3e-2 on the full-docstring corpus; push width and lr.

Recipe: prenorm + full loss + dropout 0.2, wd 0.1, 6000 steps.
References (seed 7): base 3.671, width64 3.640, lr3e-2 3.651.
"""
from experiments.sweep import sweep
R = ["prenorm", "dense_loss", "drop0.2"]
C = ["--corpus", "stdlib_full", "--wd", "0.1", "--steps", "6000"]
L3, L2 = ["--lr", "3e-2"], ["--lr", "2e-2"]
W64, W80, W96 = {"model_width": 64}, {"model_width": 80}, {"model_width": 96}
S = [
    ("d15/w64_lr3e-2", R, W64, C + L3),
    ("d15/w64_lr3e-2", R, W64, C + L3 + ["--seed", "42"]),
    ("d15/w64_lr3e-2", R, W64, C + L3 + ["--seed", "314"]),
    ("d15/w80_lr3e-2", R, W80, C + L3),
    ("d15/w96_lr3e-2", R, W96, C + L3),
    ("d15/w64_lr2e-2", R, W64, C + L2),
    ("d15/w48_lr3e-2", R, {}, C + L3 + ["--seed", "42"]),
    ("d15/w48_lr2e-2", R, {}, C + L2),
    ("d15/w64_lr3e-2_mlp", R, {**W64, "expert_kind": "mlp"}, C + L3),
    ("d15/w64_lr3e-2_dense", R, {**W64, "expert_kind": "dense"}, C + L3),
    ("d15/w64_lr3e-2_12k", R, W64, ["--corpus", "stdlib_full", "--wd", "0.1", "--steps", "12000"] + L3),
]
sweep(S, "experiments/results/batch15.jsonl", threads=1, workers=11)
