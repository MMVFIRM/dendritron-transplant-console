from experiments.sweep import sweep
LR = ["--lr", "1e-2"]
S = [
    # reruns of batch-1 runs that died silently (default lr 3e-3)
    ("baseline", [], {}, []),
    ("dense_kind", [], {"expert_kind": "dense"}, []),
    ("gate1", ["gate1"], {}, []),
    ("no_evidence", ["no_evidence"], {}, []),
    # learning-rate sweep on the unchanged architecture
    ("lr2e-2", [], {}, ["--lr", "2e-2"]),
    ("lr3e-2", [], {}, ["--lr", "3e-2"]),
    # leading variants and controls re-tested at lr 1e-2
    ("lr1e-2/prenorm", ["prenorm"], {}, LR),
    ("lr1e-2/glu_fg_gate1", ["glu", "full_gain", "gate1"], {}, LR),
    ("lr1e-2/glu", ["glu"], {}, LR),
    ("lr1e-2/prenorm_glu_fg_gate1", ["prenorm", "glu", "full_gain", "gate1"], {}, LR),
    ("lr1e-2/mlp_kind", [], {"expert_kind": "mlp"}, LR),
    ("lr1e-2/no_moe", ["no_moe"], {}, LR),
    ("lr1e-2/rounds1", [], {"loop_rounds": 1}, LR),
    ("lr1e-2/zero_mem_ctrl", [], {}, LR + ["--controls"]),
]
sweep(S, "experiments/results/batch2.jsonl", threads=1, workers=14)
