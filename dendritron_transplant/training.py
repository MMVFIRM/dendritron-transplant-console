"""Reproducible recipient training on the CPython-docstring benchmark.

The benchmark split is fixed by the packaged definition bank: 5,000 docstring
records, ordered by SHA-256 of their record ID, with the first 4,000 used for
training and the last 1,000 for validation. Training can optionally be
extended with further standard-library docstrings. Every docstring that
corresponds to a validation record is excluded from the extension, so the
validation set is never seen during training.
"""

from __future__ import annotations

import hashlib
import json
import math
import sysconfig
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Literal

import torch
import torch.nn.functional as F
from torch import Tensor

from .config import DendritronRecipientConfig
from .definition import FrozenDefinitionBank
from .memory import MemoryPayloadBuilder, MemoryPayloads
from .model import DendritronRecipientLM
from .public_fixture import (
    DefinitionRecord,
    WordTokenizer,
    scan_python_stdlib,
    stream_windows,
)
from .vivere_macsl import ViverePhraseBank

Corpus = Literal["benchmark", "stdlib", "stdlib_full"]

# Configuration of the v1.0.0 packaged recipient. Training presets and the
# experiment harness build on this fixed base, independent of whichever
# checkpoint is currently packaged.
V1_RECIPIENT_CONFIG: dict = {
    "vocab_size": 2048, "model_width": 48, "donor_width": 32, "max_sequence_length": 64,
    "loop_rounds": 2, "adaptive_threshold": None, "expert_kind": "dendritron",
    "expert_count": 12, "expert_top_k": 2, "branches_per_expert": 6, "branch_top_k": 2,
    "branch_hidden_width": 12, "use_hash_memory": True, "use_exact_early": True,
    "use_exact_late": True, "use_definition_memory": True, "definition_max_senses": 4,
    "use_lngram": True, "causal_kernel_size": 5, "residual_epsilon": 1e-06,
    "route_epsilon": 1e-06, "vocabulary_clusters": 16, "vocabulary_top_k_clusters": 2,
    "output_temperature": 6.0, "cluster_loss_weight": 0.25,
    "hash_memory": {"orders": (2, 3), "heads": 2, "rows": (1024, 4096), "memory_width": 8,
                    "modulus": 2147483647, "multiplier": 1000003},
    "lngram": {"orders": (2, 3), "bits_per_route": 2, "route_memory_width": 2,
               "surrogate_scale": 0.25},
}
BENCHMARK_RECORD_TOKENS = 64
PHRASE_BANK = "memory/centered_macsl_vivere32_r8"


def benchmark_records(assets: Path) -> tuple[tuple[DefinitionRecord, ...], tuple[DefinitionRecord, ...]]:
    """Return the fixed (train, validation) records of the packaged benchmark."""
    records = []
    with (assets / "definitions" / "metadata.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            headword, _, definition = row["text"].partition(": ")
            records.append(DefinitionRecord(row["sense_id"], headword, definition, "", ""))
    ordered = sorted(records, key=lambda record: hashlib.sha256(record.record_id.encode()).digest())
    return tuple(ordered[:4000]), tuple(ordered[4000:])


@dataclass(frozen=True)
class ExtensionReport:
    corpus: str
    stdlib_root: str
    scanned_docstrings: int
    excluded_as_validation: int
    added_records: int
    unmatched_validation_records: int


def extended_training_records(
    train: tuple[DefinitionRecord, ...],
    validation: tuple[DefinitionRecord, ...],
    *,
    full_text: bool,
    stdlib_root: Path | None = None,
) -> tuple[tuple[DefinitionRecord, ...], ExtensionReport]:
    """Benchmark training records plus every non-validation stdlib docstring.

    Validation docstrings are excluded when any of these match: the record ID,
    the source location (module + qualified name), the definition text, or,
    for the few validation records absent from this Python version, the
    headword. ``full_text`` keeps whole docstrings instead of the 800-character
    fixture clip.
    """
    root = Path(stdlib_root or sysconfig.get_paths()["stdlib"])
    reference = scan_python_stdlib(root, limit=None)
    by_id = {record.record_id: record for record in reference}
    validation_ids = {record.record_id for record in validation}
    validation_locations = {
        (by_id[rid].source_path, by_id[rid].qualified_name)
        for rid in validation_ids if rid in by_id
    }
    unmatched = [record for record in validation if record.record_id not in by_id]
    unmatched_headwords = {record.headword.casefold() for record in unmatched}
    validation_texts = [record.definition.casefold() for record in validation]
    validation_text_set = set(validation_texts)
    train_ids = {record.record_id for record in train}
    train_locations = {
        (by_id[rid].source_path, by_id[rid].qualified_name)
        for rid in train_ids if rid in by_id
    }

    def is_validation(record: DefinitionRecord) -> bool:
        text = record.definition.casefold()
        if (record.source_path, record.qualified_name) in validation_locations:
            return True
        if record.headword.casefold() in unmatched_headwords:
            return True
        if text in validation_text_set:
            return True
        # A full-length docstring whose 800-character clip is a validation text.
        return full_text and text[:800].strip() in validation_text_set

    candidates = scan_python_stdlib(root, limit=None, definition_limit=10**9) if full_text else reference
    added: list[DefinitionRecord] = []
    excluded = 0
    for record in candidates:
        if is_validation(record):
            excluded += 1
            continue
        location = (record.source_path, record.qualified_name)
        if not full_text and (record.record_id in train_ids or location in train_locations):
            continue  # already present as a benchmark training record
        added.append(record)
    if full_text:
        # Full docstrings replace the clipped benchmark copies; keep only the
        # benchmark records that this Python version does not contain.
        base = tuple(record for record in train if record.record_id not in by_id)
    else:
        base = train
    records = tuple(sorted((*base, *added),
                           key=lambda r: hashlib.sha256(r.record_id.encode()).digest()))
    return records, ExtensionReport(
        corpus="stdlib_full" if full_text else "stdlib",
        stdlib_root=str(root),
        scanned_docstrings=len(candidates),
        excluded_as_validation=excluded,
        added_records=len(added),
        unmatched_validation_records=len(unmatched),
    )


def encode_stream(tokenizer: WordTokenizer, records, maximum_record_tokens: int) -> Tensor:
    values: list[int] = []
    for record in records:
        values.extend(tokenizer.encode_record(record, maximum_record_tokens))
    return torch.tensor(values, dtype=torch.long)


def _stack_payloads(parts: list[MemoryPayloads]) -> dict:
    out = {}
    for name in ("phrase_early", "phrase_late", "phrase_mask", "phrase_rows",
                 "phrase_orders", "definitions", "definition_mask", "definition_rows"):
        values = [getattr(part, name) for part in parts]
        out[name] = None if values[0] is None else torch.cat(values)
    out["hash"] = {order: torch.cat([part.hash_addresses[order] for part in parts])
                   for order in parts[0].hash_addresses}
    return out


def payload_slice(payloads: dict, index: Tensor, *, zero_phrase_values: bool = False) -> MemoryPayloads:
    def take(name: str) -> Tensor | None:
        value = payloads[name]
        return None if value is None else value[index]

    early, late = take("phrase_early"), take("phrase_late")
    if zero_phrase_values and early is not None:
        early, late = torch.zeros_like(early), torch.zeros_like(late)
    return MemoryPayloads(
        phrase_early=early, phrase_late=late,
        phrase_mask=take("phrase_mask"),
        phrase_rows=take("phrase_rows"),
        phrase_orders=take("phrase_orders"),
        hash_addresses={order: tensor[index] for order, tensor in payloads["hash"].items()},
        definitions=take("definitions"),
        definition_mask=take("definition_mask"),
        definition_rows=take("definition_rows"),
    )


def windows_with_payloads(stream: Tensor, builder: MemoryPayloadBuilder, sequence_length: int) -> dict:
    inputs, targets = stream_windows(stream, sequence_length=sequence_length, stride=sequence_length)
    parts = [builder.build(inputs[start:start + 256]) for start in range(0, len(inputs), 256)]
    return {"x": inputs, "y": targets, "p": _stack_payloads(parts)}


def build_data(
    assets: Path,
    config: DendritronRecipientConfig,
    *,
    corpus: Corpus = "benchmark",
    sequence_length: int = 32,
    cache_dir: Path | None = None,
) -> dict:
    """Tokenised train/validation windows with precomputed memory payloads."""
    cache = None
    if cache_dir is not None:
        cache = Path(cache_dir) / f"data_{corpus}_T{sequence_length}_S{config.definition_max_senses}.pt"
        if cache.exists():
            return torch.load(cache, weights_only=False)
    tokenizer = WordTokenizer.load(assets / "tokenizer.json")
    train, validation = benchmark_records(assets)
    report = None
    if corpus == "benchmark":
        train_stream = encode_stream(tokenizer, train, BENCHMARK_RECORD_TOKENS)
    else:
        full = corpus == "stdlib_full"
        records, report = extended_training_records(train, validation, full_text=full)
        train_stream = encode_stream(tokenizer, records, 10**9 if full else BENCHMARK_RECORD_TOKENS)
    validation_stream = encode_stream(tokenizer, validation, BENCHMARK_RECORD_TOKENS)
    builder = MemoryPayloadBuilder(
        ViverePhraseBank(assets / PHRASE_BANK),
        config.hash_memory,
        FrozenDefinitionBank(assets / "definitions"),
        max_definition_senses=config.definition_max_senses,
    )
    data = {
        "train": windows_with_payloads(train_stream, builder, sequence_length),
        "val": windows_with_payloads(validation_stream, builder, sequence_length),
        "corpus": corpus,
        "train_tokens": int(train_stream.numel()),
        "extension": None if report is None else asdict(report),
    }
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, cache)
    return data


@torch.no_grad()
def evaluate(model: DendritronRecipientLM, split: dict, *, batch: int = 128,
             zero_phrase_values: bool = False, limit: int | None = None) -> dict:
    """Full-vocabulary NLL/accuracy plus cluster-sparse accuracy and recall."""
    was_training = model.training
    model.eval()
    nll = correct = sparse_correct = recall = tokens = 0.0
    count = len(split["x"]) if limit is None else min(limit, len(split["x"]))
    for start in range(0, count, batch):
        index = torch.arange(start, min(count, start + batch))
        inputs, targets = split["x"][index], split["y"][index]
        output = model(inputs, memory_payloads=payload_slice(
            split["p"], index, zero_phrase_values=zero_phrase_values))
        dense = model.vocabulary_head.dense_scores(output.hidden, model.token_embeddings.weight)
        log_probabilities = F.log_softmax(dense, dim=-1)
        nll += float(-log_probabilities.gather(-1, targets.unsqueeze(-1)).sum())
        correct += float((dense.argmax(-1) == targets).sum())
        sparse_correct += float((output.next_token_ids == targets).sum())
        recall += float((output.vocabulary.candidate_token_ids == targets.unsqueeze(-1)).any(-1).sum())
        tokens += targets.numel()
    model.train(was_training)
    return {
        "nll": nll / tokens,
        "perplexity": math.exp(nll / tokens),
        "dense_accuracy": correct / tokens,
        "sparse_accuracy": sparse_correct / tokens,
        "cluster_recall": recall / tokens,
        "tokens": int(tokens),
    }


@dataclass(frozen=True)
class TrainingSettings:
    steps: int = 3000
    batch_size: int = 32
    learning_rate: float = 5e-2
    weight_decay: float = 0.1
    warmup_steps: int = 50
    gradient_clip: float = 1.0
    seed: int = 7


def train(
    model: DendritronRecipientLM,
    data: dict,
    settings: TrainingSettings,
    *,
    extra_loss: Callable[[DendritronRecipientLM], Tensor | float] | None = None,
    log_every: int = 0,
    log: Callable[[str], None] = lambda message: print(message, flush=True),
) -> dict:
    """AdamW with linear warmup and cosine decay; decay only on matrices."""
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": settings.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=settings.learning_rate, betas=(0.9, 0.98),
    )
    steps, warmup = settings.steps, settings.warmup_steps
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / warmup)
        * 0.5 * (1 + math.cos(math.pi * min(1.0, step / steps))),
    )
    split = data["train"]
    generator = torch.Generator().manual_seed(settings.seed)
    model.train()
    began = time.perf_counter()
    history = []
    for step in range(1, steps + 1):
        index = torch.randint(0, len(split["x"]), (settings.batch_size,), generator=generator)
        inputs, targets = split["x"][index], split["y"][index]
        output = model(inputs, memory_payloads=payload_slice(split["p"], index),
                       target_ids=targets, include_target_cluster=True)
        loss = model.loss(output, targets).total
        if extra_loss is not None:
            loss = loss + extra_loss(model)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), settings.gradient_clip)
        optimizer.step()
        scheduler.step()
        if log_every and (step % log_every == 0 or step == steps):
            value = float(loss.detach())
            history.append({"step": step, "loss": value})
            log(f"step {step}/{steps} loss {value:.4f} "
                f"({time.perf_counter() - began:.0f}s)")
    return {"seconds": time.perf_counter() - began, "history": history}
