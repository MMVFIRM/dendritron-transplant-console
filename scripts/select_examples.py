"""Select demo diagnostic examples for a recipient checkpoint.

Candidates come from the fixed validation corpus: every exact phrase-bank hit
(bigram or trigram) paired with the token that follows it there. Each
candidate prompt is the phrase alone, as in the packaged examples. A candidate
qualifies when, for the given checkpoint, correct transplanted memory gives
the target a lower full-vocabulary NLL than every packaged control. Qualifying
examples are ranked by target rank, then by mean control NLL penalty; no two
selected phrases share two or more tokens, so the set stays varied. By default only readable candidates are used:
every phrase token and the target must be alphabetic words.

Usage:

    python scripts/select_examples.py build/recipient.pt --out build/examples.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from demo_server.engine import DemoEngine  # noqa: E402
from dendritron_transplant.model import DendritronRecipientLM  # noqa: E402
from dendritron_transplant.training import build_data  # noqa: E402

CONTROL_KEYS = ("hidden_cosine_to_correct", "jensen_shannon_to_correct", "next_token")


def candidates(engine: DemoEngine, validation: dict, minimum_count: int) -> list[tuple[tuple[int, ...], int, int]]:
    """(phrase token IDs, target ID, occurrence count) for exact hits in validation."""
    payloads = validation["p"]
    counts: Counter[tuple[tuple[int, ...], int]] = Counter()
    inputs, targets = validation["x"].tolist(), validation["y"].tolist()
    masks, orders = payloads["phrase_mask"].tolist(), payloads["phrase_orders"].tolist()
    for row, (tokens, following) in enumerate(zip(inputs, targets)):
        for position, hit in enumerate(masks[row]):
            if not hit:
                continue
            order = orders[row][position]
            phrase = tuple(tokens[position - order + 1: position + 1])
            counts[(phrase, following[position])] += 1
    special = {0, 1, 2, 3, 4}
    return [(phrase, target, count) for (phrase, target), count in counts.most_common()
            if count >= minimum_count and target not in special]


def evaluate_candidate(engine: DemoEngine, phrase: tuple[int, ...], target: int) -> dict:
    tokens = engine.tokenizer.tokens
    comparison = engine.compare({"prompt": engine.tokenizer.decode(phrase),
                                 "target": tokens[target]})
    modes = {item["mode"]: item for item in comparison["modes"]}
    correct = modes["correct"]["target"]
    penalties = [modes[m]["target"]["nll"] - correct["nll"] for m in modes if m != "correct"]
    return {
        "phrase": phrase,
        "target": target,
        "rank": correct["rank"],
        "correct_nll": correct["nll"],
        "beats_all": min(penalties) > 0,
        "mean_penalty": statistics.mean(penalties),
        "metrics": {
            mode: {**{key: item[key] for key in CONTROL_KEYS},
                   "nll": item["target"]["nll"], "probability": item["target"]["probability"],
                   "rank": item["target"]["rank"]}
            for mode, item in modes.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--min-occurrences", type=int, default=3)
    parser.add_argument("--default", help="example ID to list first (the demo's default)")
    parser.add_argument("--allow-symbols", action="store_true",
                        help="also consider phrases or targets containing punctuation")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    engine = DemoEngine(ROOT, threads=2)
    engine.model, _ = DendritronRecipientLM.from_checkpoint(args.checkpoint)
    engine.model.eval()
    data = build_data(ROOT / "assets", engine.model.config, corpus="benchmark",
                      cache_dir=ROOT / "build" / "cache")
    pool = candidates(engine, data["val"], args.min_occurrences)
    words = engine.tokenizer.tokens
    if not args.allow_symbols:
        pool = [(phrase, target, count) for phrase, target, count in pool
                if all(words[t].isalpha() for t in (*phrase, target))]
    bank = engine.base_bank
    results = [evaluate_candidate(engine, phrase, target) for phrase, target, _ in pool]
    qualifying = sorted((r for r in results if r["beats_all"]),
                        key=lambda r: (r["rank"], -r["mean_penalty"]))
    chosen = []
    for result in qualifying:
        if any(len(set(result["phrase"]) & set(other["phrase"])) >= 2 for other in chosen):
            continue
        chosen.append(result)
        if len(chosen) == args.count:
            break

    tokens = engine.tokenizer.tokens
    examples = []
    for result in chosen:
        phrase_text = engine.tokenizer.decode(result["phrase"])
        target = tokens[result["target"]]
        examples.append({
            "address_row": int(bank.index[result["phrase"]]) if hasattr(bank, "index") else None,
            "display_prompt": phrase_text,
            "exact_phrase": phrase_text,
            "id": "_".join([*phrase_text.split(), target]).replace("'", ""),
            "mean_control_nll_penalty": result["mean_penalty"],
            "mean_recorded_control_nll_penalty": result["mean_penalty"],
            "order": len(result["phrase"]),
            "recorded_live_control_metrics": result["metrics"],
            "selection_note": ("Selected deterministically from the fixed validation corpus: "
                               "correct memory beats every control on target NLL."),
            "target": target,
            "target_id": result["target"],
            "title": f"{phrase_text} → {target}",
            "token_ids": list(result["phrase"]),
        })
    if args.default:
        ids = [example["id"] for example in examples]
        if args.default not in ids:
            raise SystemExit(f"--default {args.default!r} is not among the selected examples: {ids}")
        examples.sort(key=lambda example: example["id"] != args.default)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(examples, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    ranks = [r["rank"] for r in results]
    print(json.dumps({
        "candidates": len(results),
        "qualifying": len(qualifying),
        "qualifying_fraction": len(qualifying) / max(1, len(results)),
        "candidates_target_rank1": sum(rank == 1 for rank in ranks),
        "selected": [(e["title"], e["recorded_live_control_metrics"]["correct"]["rank"],
                      round(e["mean_control_nll_penalty"], 3)) for e in examples],
    }, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
