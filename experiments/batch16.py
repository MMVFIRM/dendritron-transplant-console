"""Combine width, lr 3e-2 and batch 64 on the full-docstring corpus.

Recipe: prenorm + full loss + dropout 0.2, wd 0.1, 6000 steps.
References: w64 lr3e-2 batch32 -> 3.618 (3 seeds); w80 lr3e-2 -> 3.598 (seed 7);
w48 lr5e-2 batch64 -> 3.628 (seed 7).
"""
from experiments.sweep import sweep
R = ["prenorm", "dense_loss", "drop0.2"]
C = ["--corpus", "stdlib_full", "--wd", "0.1", "--steps", "6000"]
B64 = ["--batch", "64"]
S = [
    ("d16/w64_lr3e-2_b64", R, {"model_width": 64}, C + ["--lr", "3e-2"] + B64),
    ("d16/w64_lr3e-2_b64", R, {"model_width": 64}, C + ["--lr", "3e-2", "--seed", "42"] + B64),
    ("d16/w80_lr3e-2_b64", R, {"model_width": 80}, C + ["--lr", "3e-2"] + B64),
    ("d16/w96_lr2e-2_b64", R, {"model_width": 96}, C + ["--lr", "2e-2"] + B64),
    ("d16/w80_lr3e-2", R, {"model_width": 80}, C + ["--lr", "3e-2", "--seed", "42"]),
    ("d16/w80_lr2e-2", R, {"model_width": 80}, C + ["--lr", "2e-2"]),
    ("d16/w128_lr2e-2", R, {"model_width": 128}, C + ["--lr", "2e-2"]),
    ("d16/w64_lr3e-2_b64_mlp", R, {"model_width": 64, "expert_kind": "mlp"}, C + ["--lr", "3e-2"] + B64),
]
sweep(S, "experiments/results/batch16.jsonl", threads=1, workers=8)
