"""Rerun of the batch-6 dense_loss runs after fixing the loss attribute bug."""
from experiments.sweep import sweep
P = ["prenorm", "dense_loss"]
L = ["--lr", "5e-2"]
S = [
    ("p6/dense_loss", P, {}, L),
    ("p6/dense_loss_temp4", P, {"output_temperature": 4.0}, L),
    ("p6/dense_loss_temp8", P, {"output_temperature": 8.0}, L),
    ("p6/dense_loss_w64", P, {"model_width": 64}, L),
]
sweep(S, "experiments/results/batch6.jsonl", threads=1, workers=4)
