from experiments.sweep import sweep
S = [
    ("baseline", [], {}, []),
    ("no_moe", ["no_moe"], {}, []),
    ("mlp_kind", [], {"expert_kind": "mlp"}, []),
    ("dense_kind", [], {"expert_kind": "dense"}, []),
    ("full_gain", ["full_gain"], {}, []),
    ("glu", ["glu"], {}, []),
    ("gate1", ["gate1"], {}, []),
    ("prenorm", ["prenorm"], {}, []),
    ("router_temp", ["router_temp"], {}, []),
    ("branch_softmax", ["branch_softmax"], {}, []),
    ("balance", ["balance"], {}, []),
    ("no_evidence", ["no_evidence"], {}, []),
    ("learn_temp", ["learn_temp"], {}, []),
    ("fg_gate1", ["full_gain", "gate1"], {}, []),
    ("glu_fg_gate1", ["glu", "full_gain", "gate1"], {}, []),
    ("rounds3", [], {"loop_rounds": 3}, []),
    ("rounds1", [], {"loop_rounds": 1}, []),
    ("lr1e-2", [], {}, ["--lr", "1e-2"]),
    ("lr1e-3", [], {}, ["--lr", "1e-3"]),
    ("zero_mem_ctrl", [], {}, ["--controls"]),
]
sweep(S, "experiments/results/batch1.jsonl", workers=12)
