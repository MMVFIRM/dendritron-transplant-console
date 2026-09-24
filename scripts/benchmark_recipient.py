"""Benchmark recipient checkpoints on the fixed CPython-docstring validation set.

For each checkpoint this reports:

* full-vocabulary validation NLL, perplexity and accuracy, plus the accuracy
  and target recall of the cluster-sparse runtime path;
* validation NLL under every packaged memory control (the model is fixed and
  only the retrieved phrase values change, or exact retrieval is disabled);
* the ten packaged diagnostic examples: target rank with correct memory and
  whether correct memory beats every control;
* single-thread CPU latency for one 16-token forward pass.

Usage:

    python scripts/benchmark_recipient.py assets/model/recipient.pt build/recipient.pt
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from demo_server.controls import CONTROL_MODES  # noqa: E402
from demo_server.engine import DemoEngine  # noqa: E402
from dendritron_transplant.model import DendritronRecipientLM  # noqa: E402
from dendritron_transplant.training import (  # noqa: E402
    build_data,
    evaluate,
    windows_with_payloads,
)


def control_nlls(model, engine: DemoEngine, validation: dict) -> dict[str, float]:
    stream = torch.cat([validation["x"].flatten()[:1], validation["y"].flatten()])
    length = validation["x"].shape[1]
    result = {}
    for mode in CONTROL_MODES:
        split = windows_with_payloads(stream, engine.builders[mode], length)
        result[mode] = evaluate(model, split)["nll"]
    return result


def example_diagnostics(engine: DemoEngine) -> dict:
    rows = []
    for example in engine.examples:
        comparison = engine.compare({"example_id": example["id"]})
        correct, alternatives = comparison["modes"][0], comparison["modes"][1:]
        rows.append({
            "id": example["id"],
            "target_rank": correct["target"]["rank"],
            "target_nll": correct["target"]["nll"],
            "beats_all_controls": all(correct["target"]["nll"] < item["target"]["nll"]
                                      for item in alternatives),
        })
    return {
        "examples": rows,
        "rank1_count": sum(row["target_rank"] == 1 for row in rows),
        "beats_all_controls_count": sum(row["beats_all_controls"] for row in rows),
        "median_target_rank": statistics.median(row["target_rank"] for row in rows),
    }


@torch.no_grad()
def latency_ms(model, engine: DemoEngine, repeats: int = 200) -> dict[str, float]:
    input_ids = torch.randint(5, model.config.vocab_size, (1, 16),
                              generator=torch.Generator().manual_seed(0))
    payloads = engine.builders["correct"].build(input_ids)
    model.eval()
    for _ in range(20):
        model(input_ids, memory_payloads=payloads)
    samples = []
    for _ in range(repeats):
        began = time.perf_counter()
        model(input_ids, memory_payloads=payloads)
        samples.append((time.perf_counter() - began) * 1000)
    samples.sort()
    return {"median_ms": samples[len(samples) // 2], "p95_ms": samples[int(len(samples) * 0.95)]}


def benchmark(path: Path, engine: DemoEngine, validation: dict) -> dict:
    model, record = DendritronRecipientLM.from_checkpoint(path)
    model.eval()
    engine.model = model
    torch.set_num_threads(4)
    report = {
        "checkpoint": str(path),
        "recipe": record.get("metadata", {}).get("recipe", "packaged"),
        "corpus": record.get("metadata", {}).get("corpus", "benchmark"),
        "config": {key: record["config"].get(key) for key in (
            "residual_mode", "residual_dropout", "training_loss", "expert_kind", "loop_rounds")},
        "parameters": model.parameter_report(),
        "validation": evaluate(model, validation),
        "control_nll": control_nlls(model, engine, validation),
    }
    torch.set_num_threads(1)
    report["diagnostics"] = example_diagnostics(engine)
    report["latency_16_tokens"] = latency_ms(model, engine)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--json", type=Path, help="write the full report here")
    args = parser.parse_args()

    engine = DemoEngine(ROOT, threads=1)
    data = build_data(ROOT / "assets", engine.model.config, corpus="benchmark",
                      cache_dir=ROOT / "build" / "cache")
    reports = [benchmark(path, engine, data["val"]) for path in args.checkpoints]

    header = (f"{'checkpoint':40s} {'val NLL':>8s} {'ppl':>7s} {'acc':>6s} {'sparse':>6s} "
              f"{'zero-mem':>8s} {'hash-only':>9s} {'rank1':>5s} {'beats':>5s} {'ms':>6s}")
    print(header)
    for r in reports:
        v, c, d = r["validation"], r["control_nll"], r["diagnostics"]
        print(f"{Path(r['checkpoint']).name[:40]:40s} {v['nll']:8.3f} {v['perplexity']:7.1f} "
              f"{v['dense_accuracy']:6.3f} {v['sparse_accuracy']:6.3f} {c['zero']:8.3f} "
              f"{c['hash_only']:9.3f} {d['rank1_count']:>3d}/10 {d['beats_all_controls_count']:>3d}/10 "
              f"{r['latency_16_tokens']['median_ms']:6.2f}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
