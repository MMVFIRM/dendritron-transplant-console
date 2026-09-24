"""Live inference engine for the Dendritron Transplant Console."""
from __future__ import annotations

from dataclasses import asdict
import json
import math
import platform
from pathlib import Path
import time
import threading
from typing import Any

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from dendritron_transplant.definition import FrozenDefinitionBank
from dendritron_transplant.memory import MemoryPayloadBuilder
from dendritron_transplant.model import DendritronRecipientLM, RecipientOutput
from dendritron_transplant.public_fixture import WordTokenizer
from dendritron_transplant.vivere_macsl import ViverePhraseBank

from .controls import CONTROL_LABELS, CONTROL_MODES, ControlledPhraseBank


def _float(value: Any) -> float:
    if isinstance(value, Tensor):
        return float(value.detach().float().mean().cpu())
    return float(value)


def _softmax_records(
    scores: Tensor,
    token_ids: Tensor,
    tokenizer: WordTokenizer,
    *,
    limit: int = 10,
) -> list[dict[str, Any]]:
    finite = torch.isfinite(scores)
    valid_scores = scores[finite]
    valid_ids = token_ids[finite]
    if valid_scores.numel() == 0:
        return []
    probability = F.softmax(valid_scores.float(), dim=-1)
    ordering = torch.argsort(valid_scores, descending=True)[:limit]
    result: list[dict[str, Any]] = []
    for rank, local in enumerate(ordering.tolist(), start=1):
        token_id = int(valid_ids[local])
        result.append(
            {
                "rank": rank,
                "token_id": token_id,
                "token": tokenizer.tokens[token_id],
                "score": float(valid_scores[local]),
                "candidate_probability": float(probability[local]),
            }
        )
    return result


def _dense_records(
    scores: Tensor,
    tokenizer: WordTokenizer,
    *,
    limit: int = 10,
) -> list[dict[str, Any]]:
    probability = F.softmax(scores.float(), dim=-1)
    ordering = torch.argsort(scores, descending=True)[:limit]
    return [
        {
            "rank": rank,
            "token_id": int(token_id),
            "token": tokenizer.tokens[int(token_id)],
            "score": float(scores[int(token_id)]),
            "probability": float(probability[int(token_id)]),
        }
        for rank, token_id in enumerate(ordering.tolist(), start=1)
    ]


class DemoEngine:
    """Load one frozen sparse recipient and expose mechanistic traces."""

    def __init__(self, root: str | Path, *, threads: int = 2) -> None:
        self.root = Path(root).resolve()
        self.assets = self.root / "assets"
        torch.set_num_threads(max(1, int(threads)))
        self.threads = max(1, int(threads))
        self._inference_lock = threading.RLock()
        self.tokenizer = WordTokenizer.load(self.assets / "tokenizer.json")
        self.model, self.checkpoint = DendritronRecipientLM.from_checkpoint(
            self.assets / "model" / "recipient.pt",
            map_location="cpu",
        )
        self.model.eval()
        self.base_bank = ViverePhraseBank(
            self.assets / "memory" / "centered_macsl_vivere32_r8"
        )
        self.definitions = FrozenDefinitionBank(self.assets / "definitions")
        control_arrays = self.assets / "data" / "control_arrays.npz"
        self.banks: dict[str, ControlledPhraseBank | None] = {
            mode: (
                None
                if mode == "hash_only"
                else ControlledPhraseBank(self.base_bank, mode, control_arrays)
            )
            for mode in CONTROL_MODES
        }
        self.builders = {
            mode: MemoryPayloadBuilder(
                bank,
                self.model.config.hash_memory,
                self.definitions,
                max_definition_senses=self.model.config.definition_max_senses,
                use_hash=True,
            )
            for mode, bank in self.banks.items()
        }
        self.provenance = json.loads(
            (self.assets / "official_qwen_provenance.json").read_text(encoding="utf-8")
        )
        self.benchmark = json.loads(
            (self.assets / "data" / "benchmark_summary.json").read_text(encoding="utf-8")
        )
        recipient_benchmark = self.assets / "data" / "recipient_benchmark.json"
        self.recipient_benchmark = (
            json.loads(recipient_benchmark.read_text(encoding="utf-8"))
            if recipient_benchmark.is_file()
            else None
        )
        self.examples = json.loads(
            (self.assets / "data" / "examples.json").read_text(encoding="utf-8")
        )
        self.examples_by_id = {item["id"]: item for item in self.examples}
        self.control_report = json.loads(
            (self.assets / "data" / "control_report.json").read_text(encoding="utf-8")
        )

    def status(self) -> dict[str, Any]:
        parameter_report = self.model.parameter_report()
        manifest = self.base_bank.manifest
        metadata = self.checkpoint.get("metadata", {})
        evaluation = metadata.get("evaluation", {})
        weight_file = self.provenance.get("weight_file", {})
        model_record = self.provenance.get("model", {})
        return {
            "engine": "live",
            "ready": True,
            "donor_loaded": False,
            "recipient_loaded": True,
            "title": "Dendritron Transplant Console",
            "version": "1.1.0",
            "donor": {
                "model_id": self.provenance.get("model_id", "Qwen/Qwen2.5-0.5B"),
                "revision": self.provenance.get("requested_revision")
                or self.provenance.get("revision"),
                "weight_sha256": (
                    weight_file.get("sha256")
                    if isinstance(weight_file, dict)
                    else self.provenance.get("model_file_sha256")
                ),
                "weight_bytes": (
                    weight_file.get("bytes")
                    if isinstance(weight_file, dict)
                    else None
                ),
                "parameters": (
                    model_record.get("actual_base_model_parameters", 494_032_768)
                    if isinstance(model_record, dict)
                    else 494_032_768
                ),
                "runtime_role": "offline knowledge factory",
            },
            "recipient": {
                **parameter_report,
                "expert_count": self.model.config.expert_count,
                "expert_top_k": self.model.config.expert_top_k,
                "branches_per_expert": self.model.config.branches_per_expert,
                "branch_top_k": self.model.config.branch_top_k,
                "loop_rounds": self.model.config.loop_rounds,
                "physical_blocks": 2,
                "donor_width": self.model.config.donor_width,
                "model_width": self.model.config.model_width,
                "vocab_size": self.model.config.vocab_size,
                "checkpoint_validation_nll": evaluation.get("dense_nll"),
                "checkpoint_sparse_accuracy": evaluation.get("sparse_next_token_accuracy"),
            },
            "memory": {
                "rows": self.base_bank.rows,
                "orders": list(self.base_bank.orders),
                "width": self.base_bank.width,
                "rank": self.base_bank.rank,
                "card_count": manifest.get("card_count"),
                "stored_bytes": sum(
                    path.stat().st_size
                    for path in self.base_bank.root.iterdir()
                    if path.is_file()
                ),
                "resident_card_cache_bytes": self.base_bank.resident_card_cache_bytes,
                "definition_rows": self.definitions.rows,
                "control_invariants": self.control_report,
            },
            "modes": [
                {"id": mode, "label": CONTROL_LABELS[mode]}
                for mode in CONTROL_MODES
            ],
            "examples": [
                {
                    key: item[key]
                    for key in ("id", "title", "display_prompt", "target")
                    if key in item
                }
                for item in self.examples
            ],
            "environment": {
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "numpy": str(np.__version__),
                "platform": platform.platform(),
                "threads": self.threads,
            },
            "claim_boundary": (
                "This is a mechanistic research demo of addressable donor-state transfer, "
                "sparse Dendritron routing, and fixed-model memory falsification. It is not "
                "a Qwen-equivalent compressed chatbot."
            ),
        }

    def _request_ids(self, request: dict[str, Any]) -> tuple[list[int], dict[str, Any] | None]:
        example = None
        example_id = request.get("example_id")
        if example_id:
            example = self.examples_by_id.get(str(example_id))
            if example is None:
                raise ValueError(f"unknown example_id: {example_id}")
            token_ids = [int(value) for value in example["token_ids"]]
        else:
            prompt = str(request.get("prompt", "")).strip()
            token_ids = list(self.tokenizer.encode_text(prompt))
            if not token_ids:
                token_ids = [self.tokenizer.bos_id]
        return token_ids[-self.model.config.max_sequence_length :], example

    def _target_id(
        self,
        request: dict[str, Any],
        example: dict[str, Any] | None,
    ) -> int | None:
        if example is not None and example.get("target_id") is not None:
            return int(example["target_id"])
        target = request.get("target")
        if target is None or not str(target).strip():
            return None
        encoded = self.tokenizer.encode_text(str(target))
        return int(encoded[0]) if encoded else None

    def _phrase_trace(
        self,
        input_ids: Tensor,
        mode: str,
    ) -> list[dict[str, Any]]:
        reference = self.base_bank.resolve(input_ids)
        bank = self.banks[mode]
        result: list[dict[str, Any]] = []
        for position, row in enumerate(reference.row_indices[0].tolist()):
            if row < 0:
                continue
            record = self.base_bank.metadata[int(row)]
            if bank is None:
                value_row = None
                card_id = None
            else:
                resolved = bank.resolved_value(int(row))
                value_row = resolved.value_row
                card_id = resolved.card_id
            early, late = (
                (np.zeros(self.base_bank.width), np.zeros(self.base_bank.width))
                if bank is None
                else bank.get(int(row))
            )
            result.append(
                {
                    "position": position,
                    "address_row": int(row),
                    "value_row": value_row,
                    "phrase": str(record.get("text", "")),
                    "order": int(record.get("order", reference.match_orders[0, position])),
                    "frequency": int(record.get("frequency", 0)),
                    "card_id": card_id,
                    "early_norm": float(np.linalg.norm(early)),
                    "late_norm": float(np.linalg.norm(late)),
                    "exact_enabled": mode != "hash_only",
                }
            )
        return result

    def _definition_trace(self, payloads: Any) -> list[dict[str, Any]]:
        rows = payloads.definition_rows
        if rows is None:
            return []
        result: list[dict[str, Any]] = []
        seen: set[int] = set()
        for position in range(rows.shape[1]):
            for row in rows[0, position].tolist():
                if row < 0 or int(row) in seen:
                    continue
                seen.add(int(row))
                record = self.definitions.metadata[int(row)]
                result.append(
                    {
                        "position": position,
                        "row": int(row),
                        "token_id": int(record["token_id"]),
                        "token": self.tokenizer.tokens[int(record["token_id"])],
                        "sense_id": str(record.get("sense_id", "")),
                        "text": str(record.get("text", ""))[:240],
                    }
                )
                if len(result) >= 12:
                    return result
        return result

    def _routing_trace(self, output: RecipientOutput) -> list[dict[str, Any]]:
        stats = output.core_stats
        if stats is None:
            return []
        result: list[dict[str, Any]] = []
        for visit in stats.visits:
            experts = visit.moe.selected_expert_indices[0, -1].tolist()
            expert_weights = visit.moe.selected_expert_weights[0, -1].tolist()
            branches = visit.moe.selected_branch_indices[0, -1].tolist()
            branch_weights = visit.moe.selected_branch_weights[0, -1].tolist()
            result.append(
                {
                    "round": visit.round_index + 1,
                    "block": visit.block_index + 1,
                    "relative_change": _float(visit.relative_change),
                    "selected_experts": [
                        {
                            "expert": int(expert),
                            "weight": float(expert_weights[index]),
                            "branches": [
                                {
                                    "branch": int(branch),
                                    "weight": float(branch_weights[index][slot]),
                                }
                                for slot, branch in enumerate(branches[index])
                            ],
                        }
                        for index, expert in enumerate(experts)
                    ],
                    "memory": {
                        "exact_hits": int(visit.memory.exact_hits),
                        "hash_reads": int(visit.memory.hash_reads),
                        "definition_reads": int(visit.memory.definition_reads),
                        "early_gate": _float(visit.memory.early_gate),
                        "late_gate": _float(visit.memory.late_gate),
                        "hash_gate": _float(visit.memory.hash_gate),
                        "definition_gate": _float(visit.memory.definition_gate),
                    },
                    "active_conditional_fraction": visit.moe.active_parameter_fraction,
                }
            )
        return result

    @torch.no_grad()
    def _forward(
        self,
        token_ids: list[int],
        *,
        mode: str,
        target_id: int | None = None,
        return_trace: bool = True,
    ) -> dict[str, Any]:
        if mode not in CONTROL_MODES:
            raise ValueError(f"unknown memory mode: {mode}")
        input_ids = torch.tensor([token_ids], dtype=torch.long)
        builder = self.builders[mode]
        payloads = builder.build(input_ids)
        began = time.perf_counter()
        with self._inference_lock:
            output = self.model(
                input_ids,
                memory_payloads=payloads,
                return_stats=return_trace,
            )
            sparse_scores = output.vocabulary.scores[0, -1]
            sparse_ids = output.vocabulary.candidate_token_ids[0, -1]
            dense_scores = self.model.vocabulary_head.dense_scores(
                output.hidden[:, -1:], self.model.token_embeddings.weight
            )[0, 0]
        elapsed_ms = (time.perf_counter() - began) * 1000.0
        dense_probability = F.softmax(dense_scores.float(), dim=-1)
        next_token = int(output.next_token_ids[0, -1])
        target_record = None
        if target_id is not None:
            target_id = int(target_id)
            target_score = float(dense_scores[target_id])
            rank = int((dense_scores > dense_scores[target_id]).sum()) + 1
            probability = float(dense_probability[target_id])
            target_record = {
                "token_id": target_id,
                "token": self.tokenizer.tokens[target_id],
                "score": target_score,
                "probability": probability,
                "nll": -math.log(max(probability, 1e-30)),
                "rank": rank,
            }
        return {
            "token_ids": token_ids,
            "tokens": [self.tokenizer.tokens[value] for value in token_ids],
            "decoded_input": self.tokenizer.decode(token_ids),
            "mode": mode,
            "mode_label": CONTROL_LABELS[mode],
            "latency_ms": elapsed_ms,
            "next_token_id": next_token,
            "next_token": self.tokenizer.tokens[next_token],
            "sparse_candidates": _softmax_records(
                sparse_scores, sparse_ids, self.tokenizer
            ),
            "dense_candidates": _dense_records(dense_scores, self.tokenizer),
            "selected_clusters": [
                int(value)
                for value in output.vocabulary.selected_clusters[0, -1].tolist()
            ],
            "active_vocabulary_fraction": output.vocabulary.mean_active_vocabulary_fraction,
            "target": target_record,
            "hidden": output.hidden[0, -1].detach().float().cpu().numpy(),
            "dense_probability": dense_probability.detach().cpu().numpy(),
            "phrase_hits": self._phrase_trace(input_ids, mode),
            "definition_hits": self._definition_trace(payloads),
            "routing": self._routing_trace(output) if return_trace else [],
        }

    def analyze(self, request: dict[str, Any]) -> dict[str, Any]:
        token_ids, example = self._request_ids(request)
        target_id = self._target_id(request, example)
        mode = str(request.get("mode", "correct"))
        max_new_tokens = max(0, min(int(request.get("max_new_tokens", 1)), 8))
        generated = list(token_ids)
        steps: list[dict[str, Any]] = []
        for step_index in range(max_new_tokens or 1):
            window = generated[-self.model.config.max_sequence_length :]
            step = self._forward(
                window,
                mode=mode,
                target_id=target_id if step_index == 0 else None,
                return_trace=True,
            )
            hidden = step.pop("hidden")
            probability = step.pop("dense_probability")
            del hidden, probability
            step["step"] = step_index + 1
            steps.append(step)
            if step_index < max_new_tokens:
                next_id = int(step["next_token_id"])
                generated.append(next_id)
                if next_id == self.tokenizer.eos_id:
                    break
            if max_new_tokens == 0:
                break
        return {
            "engine": "live",
            "example": example,
            "input": {
                "token_ids": token_ids,
                "tokens": [self.tokenizer.tokens[value] for value in token_ids],
                "decoded": self.tokenizer.decode(token_ids),
            },
            "mode": mode,
            "mode_label": CONTROL_LABELS[mode],
            "generated_ids": generated,
            "generated_text": self.tokenizer.decode(generated),
            "steps": steps,
            "resource_summary": {
                **self.model.parameter_report(),
                "phrase_bank_bytes": sum(
                    path.stat().st_size
                    for path in self.base_bank.root.iterdir()
                    if path.is_file()
                ),
                "definition_bank_bytes": sum(
                    path.stat().st_size
                    for path in self.definitions.root.iterdir()
                    if path.is_file()
                ),
                "donor_loaded": False,
            },
        }

    def compare(self, request: dict[str, Any]) -> dict[str, Any]:
        token_ids, example = self._request_ids(request)
        target_id = self._target_id(request, example)
        raw: dict[str, dict[str, Any]] = {}
        for mode in CONTROL_MODES:
            raw[mode] = self._forward(
                token_ids,
                mode=mode,
                target_id=target_id,
                return_trace=(mode == "correct"),
            )
        reference_probability = raw["correct"]["dense_probability"]
        reference_hidden = raw["correct"]["hidden"]
        modes: list[dict[str, Any]] = []
        for mode in CONTROL_MODES:
            item = raw[mode]
            probability = item.pop("dense_probability")
            hidden = item.pop("hidden")
            midpoint = 0.5 * (reference_probability + probability)
            epsilon = 1e-30
            js = 0.5 * float(
                np.sum(reference_probability * np.log((reference_probability + epsilon) / (midpoint + epsilon)))
                + np.sum(probability * np.log((probability + epsilon) / (midpoint + epsilon)))
            )
            cosine = float(
                np.dot(reference_hidden, hidden)
                / max(np.linalg.norm(reference_hidden) * np.linalg.norm(hidden), 1e-12)
            )
            modes.append(
                {
                    "mode": mode,
                    "label": CONTROL_LABELS[mode],
                    "next_token": item["next_token"],
                    "latency_ms": item["latency_ms"],
                    "target": item["target"],
                    "hidden_cosine_to_correct": cosine,
                    "jensen_shannon_to_correct": js,
                    "phrase_hits": item["phrase_hits"],
                    "sparse_candidates": item["sparse_candidates"][:5],
                }
            )
        return {
            "engine": "live",
            "example": example,
            "input": {
                "token_ids": token_ids,
                "tokens": [self.tokenizer.tokens[value] for value in token_ids],
                "decoded": self.tokenizer.decode(token_ids),
            },
            "target_id": target_id,
            "target": None if target_id is None else self.tokenizer.tokens[target_id],
            "modes": modes,
            "correct_routing": raw["correct"]["routing"],
            "interpretation": (
                "All exact-memory modes retain the same phrase addresses. Only the values "
                "bound to those addresses change; hash-only disables exact retrieval."
            ),
        }

    def benchmark_report(self) -> dict[str, Any]:
        return {**self.benchmark, "recipient_benchmark": self.recipient_benchmark}
