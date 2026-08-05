"""Deterministic fixed-address memory controls for the live demo."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from dendritron_transplant.memory import PhraseLookup
from dendritron_transplant.vivere_macsl import ViverePhraseBank

CONTROL_MODES = (
    "correct",
    "shuffled",
    "semantic_opposite",
    "random_frozen",
    "zero",
    "hash_only",
)

CONTROL_LABELS = {
    "correct": "Correct transplanted memory",
    "shuffled": "Shuffled phrase binding",
    "semantic_opposite": "Semantic-opposite binding",
    "random_frozen": "Scale-matched random memory",
    "zero": "Zero memory (exact hit retained)",
    "hash_only": "Hash fallback only",
}


@dataclass(frozen=True)
class ResolvedValue:
    address_row: int
    value_row: int | None
    card_id: int | None
    mode: str


class ControlledPhraseBank:
    """Wrap one VIVERE bank while changing only values bound to its addresses.

    Exact hit locations, phrase orders, and keys are identical across all
    non-hash controls.  Shuffled and semantic-opposite modes preserve the full
    value multiset; random and zero retain the exact-hit mask while replacing
    the payload.
    """

    def __init__(
        self,
        base: ViverePhraseBank,
        mode: str,
        control_arrays: str | Path,
    ) -> None:
        if mode not in CONTROL_MODES[:-1]:
            raise ValueError(f"unsupported exact-memory control: {mode}")
        self.base = base
        self.mode = mode
        self.root = base.root
        self.rows = base.rows
        self.width = base.width
        self.orders = base.orders
        self.index = base.index
        self.metadata = base.metadata
        self.card_ids = base.card_ids
        arrays = np.load(Path(control_arrays), mmap_mode="r")
        self.shuffled_mapping = arrays["shuffled_mapping"]
        self.semantic_mapping = arrays["semantic_mapping"]
        self.random_early = arrays["random_early"]
        self.random_late = arrays["random_late"]

    def value_row(self, address_row: int) -> int | None:
        row = int(address_row)
        if self.mode == "shuffled":
            return int(self.shuffled_mapping[row])
        if self.mode == "semantic_opposite":
            return int(self.semantic_mapping[row])
        if self.mode in {"random_frozen", "zero"}:
            return None
        return row

    def resolved_value(self, address_row: int) -> ResolvedValue:
        value_row = self.value_row(address_row)
        card_id = None
        if value_row is not None:
            card_id = int(self.card_ids[value_row])
        return ResolvedValue(
            address_row=int(address_row),
            value_row=value_row,
            card_id=card_id,
            mode=self.mode,
        )

    def get_many(self, rows: Any) -> tuple[np.ndarray, np.ndarray]:
        indices = np.asarray(rows, dtype=np.int64)
        if indices.ndim != 1:
            raise ValueError("rows must be one-dimensional")
        if len(indices) and (indices.min() < 0 or indices.max() >= self.rows):
            raise IndexError("row outside phrase bank")
        if self.mode == "zero":
            shape = (len(indices), self.width)
            return np.zeros(shape, np.float32), np.zeros(shape, np.float32)
        if self.mode == "random_frozen":
            return (
                np.asarray(self.random_early[indices], dtype=np.float32),
                np.asarray(self.random_late[indices], dtype=np.float32),
            )
        if self.mode == "shuffled":
            indices = np.asarray(self.shuffled_mapping[indices], dtype=np.int64)
        elif self.mode == "semantic_opposite":
            indices = np.asarray(self.semantic_mapping[indices], dtype=np.int64)
        return self.base.get_many(indices)

    def get(self, row: int) -> tuple[np.ndarray, np.ndarray]:
        early, late = self.get_many([int(row)])
        return early[0], late[0]

    def resolve(self, input_ids: Tensor) -> PhraseLookup:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be [B,T]")
        batch, length = input_ids.shape
        mask = np.zeros((batch, length), np.bool_)
        rows = np.full((batch, length), -1, np.int64)
        orders = np.zeros((batch, length), np.int64)
        selected_rows: list[int] = []
        positions: list[tuple[int, int]] = []
        sequences = input_ids.detach().to("cpu", dtype=torch.long).tolist()
        for batch_index, sequence in enumerate(sequences):
            for end in range(length):
                for order in self.orders:
                    start = end - order + 1
                    if start < 0:
                        continue
                    row = self.index.get(tuple(sequence[start : end + 1]))
                    if row is None:
                        continue
                    mask[batch_index, end] = True
                    rows[batch_index, end] = row
                    orders[batch_index, end] = order
                    selected_rows.append(row)
                    positions.append((batch_index, end))
                    break
        early = np.zeros((batch, length, self.width), np.float32)
        late = np.zeros_like(early)
        if selected_rows:
            early_values, late_values = self.get_many(selected_rows)
            batch_indices = np.asarray([item[0] for item in positions])
            token_indices = np.asarray([item[1] for item in positions])
            early[batch_indices, token_indices] = early_values
            late[batch_indices, token_indices] = late_values
        device = input_ids.device
        return PhraseLookup(
            layer_early=torch.from_numpy(early).to(device),
            layer_late=torch.from_numpy(late).to(device),
            mask=torch.from_numpy(mask).to(device),
            row_indices=torch.from_numpy(rows).to(device),
            match_orders=torch.from_numpy(orders).to(device),
        )
