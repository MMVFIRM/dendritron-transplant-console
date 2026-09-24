from experiments.sweep import sweep
P = ["prenorm"]
S = [
    ("lr5e-2", [], {}, ["--lr", "5e-2"]),
    ("lr1e-1", [], {}, ["--lr", "1e-1"]),
    ("lr2e-2/prenorm", P, {}, ["--lr", "2e-2"]),
    ("lr3e-2/prenorm", P, {}, ["--lr", "3e-2"]),
    ("lr5e-2/prenorm", P, {}, ["--lr", "5e-2"]),
    ("lr1e-1/prenorm", P, {}, ["--lr", "1e-1"]),
    # second seed at 3e-2 for both, to gauge seed-to-seed variation
    ("lr3e-2", [], {}, ["--lr", "3e-2", "--seed", "42"]),
    ("lr3e-2/prenorm", P, {}, ["--lr", "3e-2", "--seed", "42"]),
]
sweep(S, "experiments/results/batch3.jsonl", threads=1, workers=8)
