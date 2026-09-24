from experiments.sweep import sweep
B = ["--lr", "3e-2"]   # tuned lr for the released architecture
P = ["--lr", "5e-2"]   # tuned lr for prenorm
S = [
    # multi-seed confirmation at each design's tuned lr
    ("tuned/baseline", [], {}, B + ["--seed", "314"]),
    ("tuned/baseline", [], {}, B + ["--seed", "1618"]),
    ("tuned/prenorm", ["prenorm"], {}, P + ["--seed", "42"]),
    ("tuned/prenorm", ["prenorm"], {}, P + ["--seed", "314"]),
    ("tuned/prenorm", ["prenorm"], {}, P + ["--seed", "1618"]),
    # design questions on top of tuned prenorm (seed 7)
    ("tuned/prenorm+mlp_kind", ["prenorm"], {"expert_kind": "mlp"}, P),
    ("tuned/prenorm+dense_kind", ["prenorm"], {"expert_kind": "dense"}, P),
    ("tuned/prenorm+no_moe", ["prenorm", "no_moe"], {}, P),
    ("tuned/prenorm+rounds1", ["prenorm"], {"loop_rounds": 1}, P),
    ("tuned/prenorm+no_evidence", ["prenorm", "no_evidence"], {}, P),
    ("tuned/prenorm+glu", ["prenorm", "glu"], {}, P),
    ("tuned/prenorm+memctrl", ["prenorm"], {}, P + ["--controls"]),
]
sweep(S, "experiments/results/batch4.jsonl", threads=1, workers=12)
