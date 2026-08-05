"""Memory-aware CPU runtime for the transplanted Dendritron recipient."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .memory import MemoryPayloadBuilder
from .model import DendritronRecipientLM, RecipientLoss, RecipientOutput


@dataclass(frozen=True)
class GenerationTrace:
    generated_ids: Tensor
    phrase_rows_by_step: tuple[tuple[int, ...], ...]
    phrase_orders_by_step: tuple[tuple[int, ...], ...]
    output_candidate_counts: tuple[int, ...]


class CPUTransplantRuntime:
    """Join token addressing, sparse row materialization, and recipient inference."""

    def __init__(
        self,
        model: DendritronRecipientLM,
        payload_builder: MemoryPayloadBuilder,
    ) -> None:
        self.model = model
        self.payload_builder = payload_builder
        bank = payload_builder.phrase_bank
        if bank is not None and bank.width != model.config.donor_width:
            raise ValueError(
                f"phrase-bank width {bank.width} differs from recipient donor width "
                f"{model.config.donor_width}"
            )
        if payload_builder.hash_config != model.config.hash_memory:
            raise ValueError(
                "runtime hash addressor and recipient hash tables must share one config"
            )
        definition_bank = payload_builder.definition_bank
        if definition_bank is not None and definition_bank.width != model.config.donor_width:
            raise ValueError("definition bank width differs from recipient donor width")

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def forward_ids(
        self,
        input_ids: Tensor,
        *,
        target_ids: Tensor | None = None,
        include_target_cluster: bool = False,
        return_stats: bool = False,
        rounds: int | None = None,
    ) -> RecipientOutput:
        input_ids = input_ids.to(self.device, dtype=torch.long)
        target_ids = (
            None
            if target_ids is None
            else target_ids.to(self.device, dtype=torch.long)
        )
        payloads = self.payload_builder.build(input_ids)
        return self.model(
            input_ids,
            memory_payloads=payloads,
            target_ids=target_ids,
            include_target_cluster=include_target_cluster,
            return_stats=return_stats,
            rounds=rounds,
        )

    def loss(self, output: RecipientOutput, targets: Tensor) -> RecipientLoss:
        return self.model.loss(output, targets.to(self.device, dtype=torch.long))

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        *,
        max_new_tokens: int,
        eos_id: int | None = None,
        rounds: int | None = None,
    ) -> GenerationTrace:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be [B,T]")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be nonnegative")
        generated = input_ids.to(self.device, dtype=torch.long)
        phrase_rows: list[tuple[int, ...]] = []
        phrase_orders: list[tuple[int, ...]] = []
        candidate_counts: list[int] = []
        self.model.eval()
        for _ in range(max_new_tokens):
            window = generated[:, -self.model.config.max_sequence_length :]
            payloads = self.payload_builder.build(window)
            output = self.model(
                window,
                memory_payloads=payloads,
                rounds=rounds,
            )
            next_token = output.next_token_ids[:, -1].unsqueeze(-1)
            generated = torch.cat([generated, next_token], dim=1)
            if payloads.phrase_rows is None:
                phrase_rows.append(())
                phrase_orders.append(())
            else:
                phrase_rows.append(
                    tuple(int(value) for value in payloads.phrase_rows[:, -1].tolist())
                )
                phrase_orders.append(
                    tuple(int(value) for value in payloads.phrase_orders[:, -1].tolist())
                )
            candidate_counts.append(
                int(output.vocabulary.valid_candidate_count[:, -1].float().mean())
            )
            if eos_id is not None and bool((next_token == int(eos_id)).all()):
                break
        return GenerationTrace(
            generated_ids=generated,
            phrase_rows_by_step=tuple(phrase_rows),
            phrase_orders_by_step=tuple(phrase_orders),
            output_candidate_counts=tuple(candidate_counts),
        )

    def storage_report(self) -> dict[str, int | float | str]:
        report: dict[str, int | float | str] = dict(self.model.parameter_report())
        report["model_parameter_bytes"] = sum(
            parameter.numel() * parameter.element_size()
            for parameter in self.model.parameters()
        )
        bank = self.payload_builder.phrase_bank
        if bank is None:
            report["phrase_bank_rows"] = 0
            report["phrase_bank_bytes"] = 0
        else:
            report["phrase_bank_rows"] = bank.rows
            report["phrase_bank_bytes"] = sum(
                path.stat().st_size
                for path in bank.root.iterdir()
                if path.is_file()
            )
            report["phrase_bank_root"] = str(bank.root)
        definition_bank = self.payload_builder.definition_bank
        if definition_bank is None:
            report["definition_bank_rows"] = 0
            report["definition_bank_bytes"] = 0
        else:
            report["definition_bank_rows"] = definition_bank.rows
            report["definition_bank_bytes"] = sum(
                path.stat().st_size for path in definition_bank.root.iterdir() if path.is_file()
            )
            report["definition_bank_root"] = str(definition_bank.root)
        report["expert_kind"] = self.model.config.expert_kind
        return report
