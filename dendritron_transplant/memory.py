"""Exact donor phrase banks, unified payload building, and trainable hash memory."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .config import HashMemoryConfig
from .hashing import HashAddressor
from .definition import FrozenDefinitionBank


def _sha256(path: Path, chunk_bytes: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class PhraseRecord:
    token_ids: tuple[int, ...]
    layer_early: np.ndarray
    layer_late: np.ndarray
    text: str = ""
    frequency: int = 0

    def __post_init__(self) -> None:
        if len(self.token_ids) not in {2, 3}:
            raise ValueError("phrase records must be bigrams or trigrams")
        if any(int(value) < 0 for value in self.token_ids):
            raise ValueError("phrase token IDs must be nonnegative")
        early = np.asarray(self.layer_early)
        late = np.asarray(self.layer_late)
        if early.ndim != 1 or late.ndim != 1 or early.shape != late.shape:
            raise ValueError("early and late donor vectors must be equal-width vectors")
        if self.frequency < 0:
            raise ValueError("phrase frequency must be nonnegative")


class PhraseBankBuilder:
    """Write a small or large immutable phrase bank with NumPy memmap payloads."""

    @staticmethod
    def write(
        root: str | Path,
        records: Iterable[PhraseRecord],
        *,
        dtype: str = "float32",
        overwrite: bool = False,
    ) -> dict[str, object]:
        destination = Path(root)
        if destination.exists() and any(destination.iterdir()) and not overwrite:
            raise FileExistsError(f"phrase bank is not empty: {destination}")
        destination.mkdir(parents=True, exist_ok=True)
        items = list(records)
        if not items:
            raise ValueError("phrase bank requires at least one record")
        width = int(np.asarray(items[0].layer_early).shape[0])
        seen: set[tuple[int, ...]] = set()
        for item in items:
            if item.token_ids in seen:
                raise ValueError(f"duplicate phrase key: {item.token_ids}")
            seen.add(item.token_ids)
            if np.asarray(item.layer_early).shape != (width,):
                raise ValueError("phrase bank contains inconsistent donor widths")

        early_path = destination / "layer_early.npy"
        late_path = destination / "layer_late.npy"
        early = np.lib.format.open_memmap(
            early_path,
            mode="w+",
            dtype=np.dtype(dtype),
            shape=(len(items), width),
        )
        late = np.lib.format.open_memmap(
            late_path,
            mode="w+",
            dtype=np.dtype(dtype),
            shape=(len(items), width),
        )
        keys_path = destination / "keys.jsonl"
        with keys_path.open("w", encoding="utf-8") as handle:
            for row, item in enumerate(items):
                early[row] = np.asarray(item.layer_early, dtype=dtype)
                late[row] = np.asarray(item.layer_late, dtype=dtype)
                handle.write(
                    json.dumps(
                        {
                            "row": row,
                            "token_ids": list(item.token_ids),
                            "order": len(item.token_ids),
                            "text": item.text,
                            "frequency": int(item.frequency),
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        early.flush()
        late.flush()
        del early, late

        manifest: dict[str, object] = {
            "schema_version": 1,
            "rows": len(items),
            "width": width,
            "dtype": str(np.dtype(dtype)),
            "orders": sorted({len(item.token_ids) for item in items}),
            "files": {
                "keys": {
                    "path": keys_path.name,
                    "bytes": keys_path.stat().st_size,
                    "sha256": _sha256(keys_path),
                },
                "layer_early": {
                    "path": early_path.name,
                    "bytes": early_path.stat().st_size,
                    "sha256": _sha256(early_path),
                },
                "layer_late": {
                    "path": late_path.name,
                    "bytes": late_path.stat().st_size,
                    "sha256": _sha256(late_path),
                },
            },
        }
        temporary = destination / "manifest.json.tmp"
        temporary.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination / "manifest.json")
        return manifest


@dataclass(frozen=True)
class PhraseLookup:
    layer_early: Tensor
    layer_late: Tensor
    mask: Tensor
    row_indices: Tensor
    match_orders: Tensor


class FrozenPhraseBank:
    """Memory-map exact donor phrase values and resolve longest suffix matches."""

    def __init__(self, root: str | Path, *, validate: bool = True) -> None:
        self.root = Path(root)
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.rows = int(self.manifest["rows"])
        self.width = int(self.manifest["width"])
        self.dtype = np.dtype(self.manifest["dtype"])
        self.orders = tuple(sorted((int(v) for v in self.manifest["orders"]), reverse=True))
        if not self.orders or any(order not in {2, 3} for order in self.orders):
            raise ValueError("phrase bank orders must be a subset of {2,3}")

        files = self.manifest["files"]
        if validate:
            for record in files.values():
                path = self.root / record["path"]
                if path.stat().st_size != int(record["bytes"]):
                    raise ValueError(f"phrase-bank size mismatch: {path}")
                if _sha256(path) != str(record["sha256"]):
                    raise ValueError(f"phrase-bank hash mismatch: {path}")

        self.layer_early = np.load(
            self.root / files["layer_early"]["path"], mmap_mode="r"
        )
        self.layer_late = np.load(
            self.root / files["layer_late"]["path"], mmap_mode="r"
        )
        if self.layer_early.shape != (self.rows, self.width):
            raise ValueError("early phrase payload shape differs from manifest")
        if self.layer_late.shape != (self.rows, self.width):
            raise ValueError("late phrase payload shape differs from manifest")

        self.index: dict[tuple[int, ...], int] = {}
        self.metadata: list[dict[str, object]] = []
        keys_path = self.root / files["keys"]["path"]
        with keys_path.open(encoding="utf-8") as handle:
            for expected_row, line in enumerate(handle):
                if not line.strip():
                    continue
                record = json.loads(line)
                row = int(record["row"])
                if row != expected_row:
                    raise ValueError("phrase rows are not contiguous")
                key = tuple(int(value) for value in record["token_ids"])
                if len(key) != int(record["order"]):
                    raise ValueError("phrase key order differs from its metadata")
                if key in self.index:
                    raise ValueError(f"duplicate phrase key in bank: {key}")
                self.index[key] = row
                self.metadata.append(record)
        if len(self.metadata) != self.rows:
            raise ValueError("phrase key row count differs from manifest")

    def get(self, row: int) -> tuple[np.ndarray, np.ndarray]:
        if not 0 <= int(row) < self.rows:
            raise IndexError(f"phrase row {row} falls outside the bank")
        return self.layer_early[int(row)], self.layer_late[int(row)]

    def resolve(self, input_ids: Tensor) -> PhraseLookup:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be [B,T]")
        if input_ids.numel() and int(input_ids.min()) < 0:
            raise ValueError("input_ids must be nonnegative")
        batch, length = input_ids.shape
        early = np.zeros((batch, length, self.width), dtype=np.float32)
        late = np.zeros((batch, length, self.width), dtype=np.float32)
        mask = np.zeros((batch, length), dtype=np.bool_)
        rows = np.full((batch, length), -1, dtype=np.int64)
        orders = np.zeros((batch, length), dtype=np.int64)
        values = input_ids.detach().to("cpu", dtype=torch.long).tolist()
        for batch_index, sequence in enumerate(values):
            for end in range(length):
                for order in self.orders:
                    start = end - order + 1
                    if start < 0:
                        continue
                    key = tuple(sequence[start : end + 1])
                    row = self.index.get(key)
                    if row is None:
                        continue
                    early[batch_index, end] = np.asarray(
                        self.layer_early[row], dtype=np.float32
                    )
                    late[batch_index, end] = np.asarray(
                        self.layer_late[row], dtype=np.float32
                    )
                    mask[batch_index, end] = True
                    rows[batch_index, end] = row
                    orders[batch_index, end] = order
                    break
        device = input_ids.device
        return PhraseLookup(
            layer_early=torch.from_numpy(early).to(device),
            layer_late=torch.from_numpy(late).to(device),
            mask=torch.from_numpy(mask).to(device),
            row_indices=torch.from_numpy(rows).to(device),
            match_orders=torch.from_numpy(orders).to(device),
        )


@dataclass
class MemoryPayloads:
    phrase_early: Tensor | None = None
    phrase_late: Tensor | None = None
    phrase_mask: Tensor | None = None
    phrase_rows: Tensor | None = None
    phrase_orders: Tensor | None = None
    hash_addresses: Mapping[int, Tensor] | None = None
    definitions: Tensor | None = None
    definition_mask: Tensor | None = None
    definition_rows: Tensor | None = None

    def to(self, device: torch.device | str) -> "MemoryPayloads":
        def move(value: Tensor | None) -> Tensor | None:
            return None if value is None else value.to(device)

        return MemoryPayloads(
            phrase_early=move(self.phrase_early),
            phrase_late=move(self.phrase_late),
            phrase_mask=move(self.phrase_mask),
            phrase_rows=move(self.phrase_rows),
            phrase_orders=move(self.phrase_orders),
            hash_addresses=(None if self.hash_addresses is None else {
                order: tensor.to(device) for order, tensor in self.hash_addresses.items()
            }),
            definitions=move(self.definitions),
            definition_mask=move(self.definition_mask),
            definition_rows=move(self.definition_rows),
        )


class MemoryPayloadBuilder:
    """Bridge token IDs to exact, hash, and definition memory payloads."""

    def __init__(
        self,
        phrase_bank: FrozenPhraseBank | None,
        hash_config: HashMemoryConfig,
        definition_bank: FrozenDefinitionBank | None = None,
        *,
        max_definition_senses: int = 4,
        use_hash: bool = True,
    ) -> None:
        self.phrase_bank = phrase_bank
        self.hash_config = hash_config
        self.hash_addressor = HashAddressor(hash_config)
        self.definition_bank = definition_bank
        self.max_definition_senses = int(max_definition_senses)
        self.use_hash = bool(use_hash)

    def build(self, input_ids: Tensor) -> MemoryPayloads:
        lookup = None if self.phrase_bank is None else self.phrase_bank.resolve(input_ids)
        addresses = self.hash_addressor.build(input_ids) if self.use_hash else None
        if lookup is not None and addresses is not None:
            for order, tensor in addresses.items():
                addresses[order] = tensor.masked_fill(lookup.mask.unsqueeze(-1), -1)
                self.hash_addressor.validate_order(addresses[order], order)
        definitions = None if self.definition_bank is None else self.definition_bank.resolve(
            input_ids, max_senses=self.max_definition_senses
        )
        return MemoryPayloads(
            phrase_early=None if lookup is None else lookup.layer_early,
            phrase_late=None if lookup is None else lookup.layer_late,
            phrase_mask=None if lookup is None else lookup.mask,
            phrase_rows=None if lookup is None else lookup.row_indices,
            phrase_orders=None if lookup is None else lookup.match_orders,
            hash_addresses=addresses,
            definitions=None if definitions is None else definitions.vectors,
            definition_mask=None if definitions is None else definitions.mask,
            definition_rows=None if definitions is None else definitions.row_indices,
        )


class TrainableHashMemory(nn.Module):
    """Sparse table reads whose dimensions are locked to the addressor config."""

    def __init__(
        self,
        config: HashMemoryConfig,
        model_width: int,
        *,
        initialization_gain: float = 1.0,
    ) -> None:
        super().__init__()
        self.config = config
        self.model_width = int(model_width)
        self.tables = nn.ModuleDict(
            {
                str(order): nn.Embedding(rows, config.memory_width)
                for order, rows in config.rows_by_order.items()
            }
        )
        self.projections = nn.ModuleDict(
            {
                str(order): nn.Linear(config.memory_width, model_width, bias=False)
                for order in config.orders
            }
        )
        self.order_gates = nn.ParameterDict(
            {str(order): nn.Parameter(torch.tensor(1e-3)) for order in config.orders}
        )
        for table in self.tables.values():
            nn.init.normal_(table.weight, std=config.memory_width**-0.5)
        with torch.no_grad():
            for projection in self.projections.values():
                projection.weight.mul_(float(initialization_gain))

    def forward(self, addresses: Mapping[int, Tensor] | None) -> Tensor | None:
        if not addresses:
            return None
        output: Tensor | None = None
        for order in self.config.orders:
            raw = addresses.get(order)
            if raw is None:
                continue
            if raw.ndim != 3 or raw.shape[-1] != self.config.heads:
                raise ValueError(
                    f"order-{order} hash addresses must be [B,T,{self.config.heads}]"
                )
            rows = self.config.rows_by_order[order]
            valid = raw >= 0
            if bool((raw[valid] >= rows).any()):
                maximum = int(raw[valid].max()) if bool(valid.any()) else -1
                raise ValueError(
                    f"order-{order} hash address {maximum} exceeds {rows} rows"
                )
            safe = raw.masked_fill(~valid, 0)
            values = self.tables[str(order)](safe) * valid.unsqueeze(-1)
            denominator = valid.sum(dim=-1, keepdim=True).clamp_min(1)
            pooled = values.sum(dim=-2) / denominator
            projected = self.projections[str(order)](pooled)
            gated = torch.tanh(self.order_gates[str(order)]) * projected
            output = gated if output is None else output + gated
        return output


@dataclass(frozen=True)
class MemoryFusionStats:
    early_gate: Tensor
    late_gate: Tensor
    hash_gate: Tensor
    definition_gate: Tensor
    exact_hits: int
    hash_reads: int
    definition_reads: int
    definition_entropy: Tensor | None = None


class SparseMemoryFusion(nn.Module):
    """Fuse exact donor views, trainable hash misses, and definition senses."""

    def __init__(
        self,
        model_width: int,
        donor_width: int,
        hash_config: HashMemoryConfig,
        *,
        initialization_gain: float = 1.0,
        use_hash_memory: bool = True,
        use_exact_early: bool = True,
        use_exact_late: bool = True,
        use_definition_memory: bool = False,
    ) -> None:
        super().__init__()
        self.model_width = int(model_width)
        self.donor_width = int(donor_width)
        self.use_hash_memory = bool(use_hash_memory)
        self.use_exact_early = bool(use_exact_early)
        self.use_exact_late = bool(use_exact_late)
        self.use_definition_memory = bool(use_definition_memory)
        self.early_projection = nn.Linear(donor_width, model_width, bias=False)
        self.late_projection = nn.Linear(donor_width, model_width, bias=False)
        self.hash_memory = TrainableHashMemory(hash_config, model_width,
                                                initialization_gain=initialization_gain)
        self.live_to_definition = nn.Linear(model_width, donor_width, bias=False)
        self.definition_projection = nn.Linear(donor_width, model_width, bias=False)
        self.initial_early_gate = nn.Parameter(torch.tensor(1e-3))
        self.early_gates = nn.Parameter(torch.full((2,), 1e-3))
        self.late_gates = nn.Parameter(torch.full((2,), 1e-3))
        self.hash_gates = nn.Parameter(torch.full((2,), 1e-3))
        self.definition_gates = nn.Parameter(torch.full((2,), 1e-3))
        with torch.no_grad():
            for module in (self.early_projection, self.late_projection,
                           self.definition_projection):
                module.weight.mul_(float(initialization_gain))

    def _phrase(self, values: Tensor, hidden: Tensor, mask: Tensor | None,
                projection: nn.Linear) -> Tensor:
        if values.shape != (*hidden.shape[:2], self.donor_width):
            raise ValueError(f"phrase payload must be [B,T,{self.donor_width}]")
        update = projection(values.to(hidden))
        if mask is not None:
            update = update * mask.to(hidden.dtype).unsqueeze(-1)
        return update

    def _definitions(self, hidden: Tensor, payloads: MemoryPayloads) -> tuple[Tensor | None, Tensor | None, int]:
        values = payloads.definitions
        if values is None:
            return None, None, 0
        if values.ndim != 4 or values.shape[:2] != hidden.shape[:2] or values.shape[-1] != self.donor_width:
            raise ValueError("definitions must be [B,T,S,donor_width]")
        mask = payloads.definition_mask
        if mask is None:
            mask = torch.ones(values.shape[:-1], dtype=torch.bool, device=values.device)
        query = torch.nn.functional.normalize(self.live_to_definition(hidden), dim=-1)
        anchors = torch.nn.functional.normalize(values.to(hidden), dim=-1)
        scores = 4.0 * (query.unsqueeze(-2) * anchors).sum(dim=-1)
        scores = scores.masked_fill(~mask, -torch.inf)
        active = mask.any(dim=-1)
        safe_scores = torch.where(active.unsqueeze(-1), scores, torch.zeros_like(scores))
        weights = torch.softmax(safe_scores, dim=-1) * mask
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        centroid = (weights.unsqueeze(-1) * values.to(hidden)).sum(dim=-2)
        update = self.definition_projection(centroid) * active.unsqueeze(-1)
        entropy = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(dim=-1)
        return update, entropy, int(mask.sum())

    def initial_update(self, hidden: Tensor, payloads: MemoryPayloads | None) -> Tensor:
        if not self.use_exact_early or payloads is None or payloads.phrase_early is None:
            return torch.zeros_like(hidden)
        return torch.tanh(self.initial_early_gate) * self._phrase(
            payloads.phrase_early, hidden, payloads.phrase_mask, self.early_projection
        )

    def forward(self, hidden: Tensor, payloads: MemoryPayloads | None, *, block_index: int,
                return_stats: bool = False) -> Tensor | tuple[Tensor, MemoryFusionStats]:
        if block_index not in {0,1}:
            raise ValueError("block_index must be 0 or 1")
        update = torch.zeros_like(hidden)
        exact_hits = hash_reads = definition_reads = 0
        definition_entropy = None
        if payloads is not None:
            if self.use_exact_early and payloads.phrase_early is not None:
                update = update + torch.tanh(self.early_gates[block_index]) * self._phrase(
                    payloads.phrase_early, hidden, payloads.phrase_mask, self.early_projection)
            if self.use_exact_late and block_index == 1 and payloads.phrase_late is not None:
                update = update + torch.tanh(self.late_gates[block_index]) * self._phrase(
                    payloads.phrase_late, hidden, payloads.phrase_mask, self.late_projection)
            if self.use_hash_memory:
                hashed = self.hash_memory(payloads.hash_addresses)
                if hashed is not None:
                    update = update + torch.tanh(self.hash_gates[block_index]) * hashed
            if self.use_definition_memory:
                definition_update, definition_entropy, definition_reads = self._definitions(hidden, payloads)
                if definition_update is not None:
                    update = update + torch.tanh(self.definition_gates[block_index]) * definition_update
            if payloads.phrase_mask is not None:
                exact_hits = int(payloads.phrase_mask.sum())
            if self.use_hash_memory and payloads.hash_addresses:
                hash_reads = sum(int((tensor >= 0).sum()) for tensor in payloads.hash_addresses.values())
        if return_stats:
            zero = hidden.new_tensor(0.0)
            return update, MemoryFusionStats(
                early_gate=torch.tanh(self.early_gates[block_index]) if self.use_exact_early else zero,
                late_gate=torch.tanh(self.late_gates[block_index]) if self.use_exact_late else zero,
                hash_gate=torch.tanh(self.hash_gates[block_index]) if self.use_hash_memory else zero,
                definition_gate=torch.tanh(self.definition_gates[block_index]) if self.use_definition_memory else zero,
                exact_hits=exact_hits, hash_reads=hash_reads,
                definition_reads=definition_reads, definition_entropy=definition_entropy,
            )
        return update

