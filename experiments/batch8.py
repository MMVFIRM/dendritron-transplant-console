"""Is the model overfitting or undertrained? Train/val gap and longer training.

All runs use the full-vocabulary loss (dense_loss) + prenorm at lr 5e-2 unless noted.
"""
from experiments.sweep import sweep
PD = ["prenorm", "dense_loss"]
L = ["--lr", "5e-2"]
LONG = L + ["--steps", "6000"]
S = [
    # generalisation gap at the standard budget
    ("g8/ref", PD, {}, L),
    ("g8/no_moe", PD + ["no_moe"], {}, L),
    ("g8/dense_kind", PD, {"expert_kind": "dense"}, L),
    ("g8/width32", PD, {"model_width": 32}, L),
    ("g8/wd0.1", PD, {}, L + ["--wd", "0.1"]),
    # 4x longer training: does expert capacity pay off with more steps?
    ("g8/ref_6k", PD, {}, LONG),
    ("g8/ref_6k", PD, {}, LONG + ["--seed", "42"]),
    ("g8/no_moe_6k", PD + ["no_moe"], {}, LONG),
    ("g8/dense_kind_6k", PD, {"expert_kind": "dense"}, LONG),
]
sweep(S, "experiments/results/batch8.jsonl", threads=1, workers=9)
