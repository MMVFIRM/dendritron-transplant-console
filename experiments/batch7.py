"""Re-baseline on the full-vocabulary training loss (dense_loss).

Reference: prenorm + dense_loss, lr 5e-2, seed 7 -> 4.002.
"""
from experiments.sweep import sweep
PD = ["prenorm", "dense_loss"]
L = ["--lr", "5e-2"]
S = [
    # multi-seed confirmation of the new reference
    ("d7/ref", PD, {}, L + ["--seed", "42"]),
    ("d7/ref", PD, {}, L + ["--seed", "314"]),
    ("d7/ref", PD, {}, L + ["--seed", "1618"]),
    # does prenorm still matter under the new loss? (original residual, its tuned lr)
    ("d7/postnorm", ["dense_loss"], {}, ["--lr", "3e-2"]),
    ("d7/postnorm", ["dense_loss"], {}, ["--lr", "3e-2", "--seed", "42"]),
    # expert-path questions under the new loss
    ("d7/mlp_kind", PD, {"expert_kind": "mlp"}, L),
    ("d7/dense_kind", PD, {"expert_kind": "dense"}, L),
    ("d7/no_moe", PD + ["no_moe"], {}, L),
    ("d7/rounds1", PD, {"loop_rounds": 1}, L),
    ("d7/no_lngram", PD, {"use_lngram": False}, L),
    # learning rate and head settings under the new loss
    ("d7/lr3e-2", PD, {}, ["--lr", "3e-2"]),
    ("d7/lr1e-1", PD, {}, ["--lr", "1e-1"]),
    ("d7/temp4", PD, {"output_temperature": 4.0}, L + ["--seed", "42"]),
    ("d7/clusters4", PD, {"vocabulary_top_k_clusters": 4}, L),
]
sweep(S, "experiments/results/batch7.jsonl", threads=1, workers=9)
