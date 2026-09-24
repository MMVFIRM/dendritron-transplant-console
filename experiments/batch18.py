"""Seed confirmation of the quality configuration: width 80, lr 3e-2, batch 64."""
from experiments.sweep import sweep
R = ["prenorm", "dense_loss", "drop0.2"]
C = ["--corpus", "stdlib_full", "--wd", "0.1", "--steps", "6000", "--lr", "3e-2", "--batch", "64"]
S = [("d16/w80_lr3e-2_b64", R, {"model_width": 80}, C + ["--seed", s]) for s in ("42", "314")]
sweep(S, "experiments/results/batch18.jsonl", threads=3, workers=2)
