"""Output-head training signal and width, on top of tuned prenorm (lr 5e-2)."""
from experiments.sweep import sweep
P = ["prenorm"]
L = ["--lr", "5e-2"]
C4 = {"vocabulary_top_k_clusters": 4}
S = [
    # confirm clusters4 and width64 on more seeds
    ("p6/clusters4", P, C4, L + ["--seed", "42"]),
    ("p6/clusters4", P, C4, L + ["--seed", "314"]),
    ("p6/width64", P, {"model_width": 64}, L + ["--seed", "42"]),
    # how far does the candidate-set effect go?
    ("p6/clusters8", P, {"vocabulary_top_k_clusters": 8}, L),
    ("p6/c32_top8", P, {"vocabulary_clusters": 32, "vocabulary_top_k_clusters": 8}, L),
    ("p6/dense_loss", P + ["dense_loss"], {}, L),
    # output temperature below 6
    ("p6/clusters4_temp4", P, {**C4, "output_temperature": 4.0}, L),
    ("p6/dense_loss_temp4", P + ["dense_loss"], {"output_temperature": 4.0}, L),
    ("p6/dense_loss_temp8", P + ["dense_loss"], {"output_temperature": 8.0}, L),
    # combine with width
    ("p6/clusters4_w64", P, {**C4, "model_width": 64}, L),
    ("p6/dense_loss_w64", P + ["dense_loss"], {"model_width": 64}, L),
    ("p6/clusters4_w96", P, {**C4, "model_width": 96}, L),
    # attention at a gentler lr
    ("p6/clusters4_attn_lr2e-2", P + ["attn"], C4, ["--lr", "2e-2"]),
    ("p6/clusters4_lr3e-2", P, C4, ["--lr", "3e-2"]),
]
sweep(S, "experiments/results/batch6.jsonl", threads=1, workers=14)
