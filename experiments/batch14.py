"""Revisit capacity, branch form and regularisation now that data is less limiting.

Full-docstring corpus, recipe (prenorm + full loss + dropout 0.2, wd 0.1, lr 5e-2),
6000 steps, seed 7. Reference: 3.671 (seed 7), 3.669 (seed 42).
"""
from experiments.sweep import sweep
R = ["prenorm", "dense_loss", "drop0.2"]
F = ["--corpus", "stdlib_full", "--lr", "5e-2", "--wd", "0.1", "--steps", "6000"]
S = [
    # capacity
    ("d14/width64", R, {"model_width": 64}, F),
    ("d14/experts24", R, {"expert_count": 24}, F),
    ("d14/hidden24", R, {"branch_hidden_width": 24}, F),
    ("d14/rounds1", R, {"loop_rounds": 1}, F),
    ("d14/rounds3", R, {"loop_rounds": 3}, F),
    # branch form
    ("d14/glu", R + ["glu"], {}, F),
    ("d14/no_evidence", R + ["no_evidence"], {}, F),
    ("d14/mlp_kind", R, {"expert_kind": "mlp"}, F + ["--seed", "42"]),
    # regularisation and optimisation
    ("d14/drop0.1", ["prenorm", "dense_loss", "drop0.1"], {}, F),
    ("d14/drop0.3", ["prenorm", "dense_loss", "drop0.3"], {}, F),
    ("d14/lr3e-2", R, {}, F[:-6] + ["--lr", "3e-2", "--wd", "0.1", "--steps", "6000"]),
    ("d14/batch64", R, {}, F + ["--batch", "64"]),
]
sweep(S, "experiments/results/batch14.jsonl", threads=1, workers=12)
