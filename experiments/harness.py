"""Matched-budget training harness for Dendritron architecture experiments.

Rebuilds the packaged CPython-docstring corpus from assets/definitions, trains a
recipient from scratch under one fixed protocol, and reports full-vocabulary
validation NLL. Variants are expressed as config overrides plus optional
module patches registered in ``variants.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dendritron_transplant.config import DendritronRecipientConfig  # noqa: E402
from dendritron_transplant.definition import FrozenDefinitionBank  # noqa: E402
from dendritron_transplant.memory import MemoryPayloadBuilder, MemoryPayloads  # noqa: E402
from dendritron_transplant.model import DendritronRecipientLM  # noqa: E402
from dendritron_transplant.public_fixture import (  # noqa: E402
    DefinitionRecord,
    build_public_corpus,
    stream_windows,
)
from dendritron_transplant.vivere_macsl import ViverePhraseBank  # noqa: E402

CACHE = ROOT / "experiments" / ".cache"


def released_config() -> dict:
    """The v1.0.0 recipient config that every experiment starts from."""
    import copy
    from dendritron_transplant.training import V1_RECIPIENT_CONFIG
    return copy.deepcopy(V1_RECIPIENT_CONFIG)


def _records() -> tuple[DefinitionRecord, ...]:
    items = []
    with (ROOT / "assets/definitions/metadata.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            headword, _, definition = row["text"].partition(": ")
            items.append(DefinitionRecord(row["sense_id"], headword, definition, "", ""))
    return tuple(items)


def _stack_payloads(parts: list[MemoryPayloads]) -> dict:
    out = {}
    for name in ("phrase_early", "phrase_late", "phrase_mask", "phrase_rows",
                 "phrase_orders", "definitions", "definition_mask", "definition_rows"):
        out[name] = torch.cat([getattr(p, name) for p in parts])
    orders = parts[0].hash_addresses.keys()
    out["hash"] = {o: torch.cat([p.hash_addresses[o] for p in parts]) for o in orders}
    return out


def load_data(seq_len: int, max_senses: int = 4, corpus: str = "benchmark") -> dict:
    if corpus != "benchmark":
        # Extended training corpora are assembled by the package (validation
        # records are excluded there); the validation windows are identical.
        from dendritron_transplant.training import build_data
        cfg = DendritronRecipientConfig.from_dict(released_config())
        return build_data(ROOT / "assets", cfg, corpus=corpus, sequence_length=seq_len,
                          cache_dir=ROOT / "build" / "cache")
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"data_T{seq_len}_S{max_senses}.pt"
    if path.exists():
        return torch.load(path, weights_only=False)
    corpus = build_public_corpus(_records(), vocabulary_size=2048)
    cfg = DendritronRecipientConfig.from_dict(released_config())
    builder = MemoryPayloadBuilder(
        ViverePhraseBank(ROOT / "assets/memory/centered_macsl_vivere32_r8"),
        cfg.hash_memory,
        FrozenDefinitionBank(ROOT / "assets/definitions"),
        max_definition_senses=max_senses,
    )
    data = {}
    for split, stream in (("train", corpus.training_stream), ("val", corpus.validation_stream)):
        x, y = stream_windows(stream, sequence_length=seq_len, stride=seq_len)
        parts = [builder.build(x[i:i + 256]) for i in range(0, len(x), 256)]
        data[split] = {"x": x, "y": y, "p": _stack_payloads(parts)}
    torch.save(data, path)
    return data


def payload_slice(p: dict, idx: torch.Tensor, *, mode: str = "correct") -> MemoryPayloads:
    early, late = p["phrase_early"][idx], p["phrase_late"][idx]
    if mode == "zero":
        early, late = torch.zeros_like(early), torch.zeros_like(late)
    return MemoryPayloads(
        phrase_early=early, phrase_late=late, phrase_mask=p["phrase_mask"][idx],
        phrase_rows=p["phrase_rows"][idx], phrase_orders=p["phrase_orders"][idx],
        hash_addresses={o: t[idx] for o, t in p["hash"].items()},
        definitions=p["definitions"][idx], definition_mask=p["definition_mask"][idx],
        definition_rows=p["definition_rows"][idx],
    )


def _index_payloads(p: dict, idx: torch.Tensor) -> dict:
    out = {k: v[idx] for k, v in p.items() if k != "hash"}
    out["hash"] = {o: t[idx] for o, t in p["hash"].items()}
    return out


@torch.no_grad()
def evaluate(model, split: dict, *, batch: int = 128, mode: str = "correct") -> dict:
    model.eval()
    nll_sum = correct = sparse_correct = recall = tokens = 0.0
    n = len(split["x"])
    for start in range(0, n, batch):
        idx = torch.arange(start, min(n, start + batch))
        x, y = split["x"][idx], split["y"][idx]
        out = model(x, memory_payloads=payload_slice(split["p"], idx, mode=mode))
        dense = model.vocabulary_head.dense_scores(out.hidden, model.token_embeddings.weight)
        logp = F.log_softmax(dense, dim=-1)
        nll_sum += float(-logp.gather(-1, y.unsqueeze(-1)).sum())
        correct += float((dense.argmax(-1) == y).sum())
        sparse_correct += float((out.next_token_ids == y).sum())
        recall += float((out.vocabulary.candidate_token_ids == y.unsqueeze(-1)).any(-1).sum())
        tokens += y.numel()
    model.train()
    return {
        "nll": nll_sum / tokens,
        "ppl": math.exp(nll_sum / tokens),
        "dense_acc": correct / tokens,
        "sparse_acc": sparse_correct / tokens,
        "cluster_recall": recall / tokens,
    }


def run(name: str, *, seed: int, steps: int, seq_len: int, batch: int, lr: float,
        overrides: dict, patches: list[str], weight_decay: float = 0.01,
        warmup: int = 50, eval_every: int = 0, controls: bool = False,
        corpus: str = "benchmark") -> dict:
    from experiments import variants

    torch.manual_seed(seed)
    data = load_data(seq_len, corpus=corpus)
    cfg_dict = released_config()
    cfg_dict.update(overrides)
    cfg_dict["max_sequence_length"] = max(seq_len, int(cfg_dict["max_sequence_length"]))
    config = DendritronRecipientConfig.from_dict(cfg_dict)
    with variants.applied(patches):
        model = DendritronRecipientLM(config)
        variants.post_init(model, patches)
        params = sum(p.numel() for p in model.parameters())
        decay = [p for p in model.parameters() if p.ndim >= 2]
        no_decay = [p for p in model.parameters() if p.ndim < 2]
        opt = torch.optim.AdamW(
            [{"params": decay, "weight_decay": weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=lr, betas=(0.9, 0.98),
        )
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / warmup)
            * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))),
        )
        train = data["train"]
        gen = torch.Generator().manual_seed(seed)
        history = []
        t0 = time.perf_counter()
        for step in range(1, steps + 1):
            idx = torch.randint(0, len(train["x"]), (batch,), generator=gen)
            x, y = train["x"][idx], train["y"][idx]
            out = model(x, memory_payloads=payload_slice(train["p"], idx), target_ids=y,
                        include_target_cluster=True)
            vocab_loss = model.loss(out, y)
            if "dense_loss" in patches:
                # Train against the full vocabulary; inference stays cluster-sparse.
                dense = model.vocabulary_head.dense_scores(out.hidden, model.token_embeddings.weight)
                loss = (F.cross_entropy(dense.reshape(-1, dense.shape[-1]), y.reshape(-1))
                        + config.cluster_loss_weight * vocab_loss.vocabulary.cluster_loss)
            else:
                loss = vocab_loss.total
            loss = loss + variants.aux_loss(model)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            if eval_every and step % eval_every == 0 and step != steps:
                history.append({"step": step, **evaluate(model, data["val"])})
        seconds = time.perf_counter() - t0
        result = evaluate(model, data["val"])
        # Same metric on a fixed training subset (size of the validation set) to
        # show the generalisation gap.
        sub = torch.arange(0, len(train["x"]), max(1, len(train["x"]) // len(data["val"]["x"])))
        sub = sub[: len(data["val"]["x"])]
        train_eval = evaluate(model, {"x": train["x"][sub], "y": train["y"][sub],
                                      "p": _index_payloads(train["p"], sub)})
        result["train_nll"] = train_eval["nll"]
        if controls:
            result["zero_memory_nll"] = evaluate(model, data["val"], mode="zero")["nll"]
        report = model.parameter_report()
    return {
        "name": name, "corpus": corpus, "seed": seed, "steps": steps, "seq_len": seq_len, "batch": batch,
        "lr": lr, "overrides": overrides, "patches": patches, "params": params,
        "active_fraction": report["conditional_active_fraction"],
        "active_conditional": report["active_conditional_parameters_per_token"],
        "train_seconds": seconds, "history": history, **result,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--overrides", default="{}")
    parser.add_argument("--patches", default="")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--controls", action="store_true")
    parser.add_argument("--corpus", default="benchmark",
                        choices=["benchmark", "stdlib", "stdlib_full"])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    result = run(
        args.name, seed=args.seed, steps=args.steps, seq_len=args.seq_len,
        batch=args.batch, lr=args.lr, overrides=json.loads(args.overrides),
        patches=[p for p in args.patches.split(",") if p], weight_decay=args.wd,
        eval_every=args.eval_every, controls=args.controls, corpus=args.corpus,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result) + "\n")
    print(json.dumps({k: result[k] for k in ("name", "seed", "nll", "train_nll", "dense_acc",
                                             "params", "active_fraction", "train_seconds")}))


if __name__ == "__main__":
    main()
