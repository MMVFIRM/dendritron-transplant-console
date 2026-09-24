"""Seed check: Dendritron vs MLP branches at width 64, lr 3e-2, batch 64.

Seed 7: MLP 3.577, Dendritron 3.593. Dendritron seed 42: 3.588.
"""
from experiments.sweep import sweep
R = ["prenorm", "dense_loss", "drop0.2"]
C = ["--corpus", "stdlib_full", "--wd", "0.1", "--steps", "6000", "--lr", "3e-2", "--batch", "64"]
W, M = {"model_width": 64}, {"model_width": 64, "expert_kind": "mlp"}
S = [
    ("d16/w64_lr3e-2_b64_mlp", R, M, C + ["--seed", "42"]),
    ("d16/w64_lr3e-2_b64_mlp", R, M, C + ["--seed", "314"]),
    ("d16/w64_lr3e-2_b64", R, W, C + ["--seed", "314"]),
]
sweep(S, "experiments/results/batch17.jsonl", threads=2, workers=3)
