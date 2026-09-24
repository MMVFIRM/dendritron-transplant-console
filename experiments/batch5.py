"""Screen further changes on top of tuned prenorm (lr 5e-2, seed 7 -> 6.260)."""
from experiments.sweep import sweep
P = ["prenorm"]
L = ["--lr", "5e-2"]
S = [
    ("p5/temp10", P, {"output_temperature": 10.0}, L),
    ("p5/temp16", P, {"output_temperature": 16.0}, L),
    ("p5/clusters4", P, {"vocabulary_top_k_clusters": 4}, L),
    ("p5/kernel9", P, {"causal_kernel_size": 9}, L),
    ("p5/attn", P + ["attn"], {}, L),
    ("p5/layerscale", P + ["layerscale"], {}, L),
    ("p5/rounds3", P, {"loop_rounds": 3}, L),
    ("p5/experts24", P, {"expert_count": 24}, L),
    ("p5/branch_top3", P, {"branch_top_k": 3}, L),
    ("p5/hidden24", P, {"branch_hidden_width": 24}, L),
    ("p5/no_lngram", P, {"use_lngram": False}, L),
    ("p5/no_defmem", P, {"use_definition_memory": False}, L),
    ("p5/router_temp", P + ["router_temp"], {}, L),
    ("p5/width64", P, {"model_width": 64}, L),
    ("p5/balance", P + ["balance"], {}, L),
]
sweep(S, "experiments/results/batch5.jsonl", threads=1, workers=15)
