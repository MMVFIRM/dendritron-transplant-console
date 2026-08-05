"""Dense offline donor and hidden-state extraction into frozen phrase memory."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .memory import PhraseBankBuilder, PhraseRecord


@dataclass(frozen=True)
class DonorOutput:
    logits: Tensor
    layer_states: tuple[Tensor, ...]


class DenseDonorLM(nn.Module):
    """Small dense stacked-GRU donor used to exercise the transplant pipeline.

    Production use can replace this class with a Qwen adapter as long as it
    exposes same-width intermediate states for isolated phrases.
    """

    def __init__(
        self,
        vocab_size: int,
        *,
        width: int = 96,
        layer_count: int = 4,
    ) -> None:
        super().__init__()
        if min(vocab_size, width, layer_count) < 1:
            raise ValueError("donor dimensions must be positive")
        self.vocab_size = int(vocab_size)
        self.width = int(width)
        self.layer_count = int(layer_count)
        self.token_embeddings = nn.Embedding(vocab_size, width)
        self.layers = nn.ModuleList(
            [nn.GRU(width, width, num_layers=1, batch_first=True) for _ in range(layer_count)]
        )
        self.output = nn.Linear(width, vocab_size, bias=False)
        self.output.weight = self.token_embeddings.weight

    def forward(self, input_ids: Tensor) -> DonorOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be [B,T]")
        if input_ids.numel() and (
            int(input_ids.min()) < 0 or int(input_ids.max()) >= self.vocab_size
        ):
            raise ValueError("donor input token is outside the vocabulary")
        hidden = self.token_embeddings(input_ids)
        states: list[Tensor] = []
        for layer in self.layers:
            hidden, _ = layer(hidden)
            states.append(hidden)
        return DonorOutput(logits=self.output(hidden), layer_states=tuple(states))

    def parameter_report(self) -> dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        return {"stored_parameters": int(total), "active_parameters_per_token": int(total)}


class DonorTransplanter:
    """Run each phrase once through a donor and freeze two intermediate views."""

    def __init__(
        self,
        donor: DenseDonorLM,
        *,
        early_layer: int = 1,
        late_layer: int = 3,
    ) -> None:
        if not 0 <= early_layer < donor.layer_count:
            raise ValueError("early_layer is outside the donor")
        if not 0 <= late_layer < donor.layer_count:
            raise ValueError("late_layer is outside the donor")
        if early_layer >= late_layer:
            raise ValueError("early_layer must precede late_layer")
        self.donor = donor
        self.early_layer = int(early_layer)
        self.late_layer = int(late_layer)

    @torch.no_grad()
    def build_phrase_bank(
        self,
        phrases: Sequence[tuple[int, ...]],
        root: str | Path,
        *,
        frequencies: Mapping[tuple[int, ...], int] | None = None,
        texts: Mapping[tuple[int, ...], str] | None = None,
        dtype: str = "float32",
        overwrite: bool = False,
        device: str | torch.device = "cpu",
    ) -> dict[str, object]:
        self.donor.eval().to(device)
        records: list[PhraseRecord] = []
        for phrase in phrases:
            if len(phrase) not in {2, 3}:
                raise ValueError("transplant inventory supports only 2/3-grams")
            values = torch.tensor([phrase], dtype=torch.long, device=device)
            output = self.donor(values)
            early = output.layer_states[self.early_layer][0, -1]
            late = output.layer_states[self.late_layer][0, -1]
            records.append(
                PhraseRecord(
                    token_ids=tuple(int(value) for value in phrase),
                    layer_early=early.detach().to("cpu", dtype=torch.float32).numpy(),
                    layer_late=late.detach().to("cpu", dtype=torch.float32).numpy(),
                    text="" if texts is None else str(texts.get(phrase, "")),
                    frequency=(
                        0 if frequencies is None else int(frequencies.get(phrase, 0))
                    ),
                )
            )
        manifest = PhraseBankBuilder.write(
            root,
            records,
            dtype=dtype,
            overwrite=overwrite,
        )
        metadata = {
            "donor_class": type(self.donor).__name__,
            "donor_width": self.donor.width,
            "donor_layers": self.donor.layer_count,
            "early_layer": self.early_layer,
            "late_layer": self.late_layer,
            "phrase_rows": len(records),
        }
        Path(root, "transplant.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return {"phrase_bank": manifest, "transplant": metadata}
