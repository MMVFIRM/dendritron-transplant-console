"""Regularisation against memorisation (full-vocabulary loss, prenorm, lr 5e-2).

Reference seed 7: val 4.002, train 3.024 (gap 0.98). 4-seed val mean 3.987.
"""
from experiments.sweep import sweep
PD = ["prenorm", "dense_loss"]
L = ["--lr", "5e-2"]
S = [
    ("r9/drop0.1", PD + ["drop0.1"], {}, L),
    ("r9/drop0.2", PD + ["drop0.2"], {}, L),
    ("r9/drop0.3", PD + ["drop0.3"], {}, L),
    ("r9/drop0.2", PD + ["drop0.2"], {}, L + ["--seed", "42"]),
    ("r9/wd0.3", PD, {}, L + ["--wd", "0.3"]),
    ("r9/wd1.0", PD, {}, L + ["--wd", "1.0"]),
    ("r9/drop0.2_wd0.1", PD + ["drop0.2"], {}, L + ["--wd", "0.1"]),
    # with memorisation controlled, do the experts pull ahead?
    ("r9/drop0.2_dense_kind", PD + ["drop0.2"], {"expert_kind": "dense"}, L),
    ("r9/drop0.2_no_moe", PD + ["drop0.2", "no_moe"], {}, L),
    ("r9/drop0.2_mlp_kind", PD + ["drop0.2"], {"expert_kind": "mlp"}, L),
]
sweep(S, "experiments/results/batch9.jsonl", threads=1, workers=10)
