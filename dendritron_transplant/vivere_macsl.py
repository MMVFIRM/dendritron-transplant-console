"""Task-conditioned VIVERE/MACSL transfer cards for exact phrase memory.

The Gate 2C implementation is intentionally separated into two stages:

1. A transfer mapper predicts recipient-owned phrase vectors from an existing
   donor-space phrase bank.  Global ridge/MLP controls and locally owned MACSL
   residual cards share the same train/validation/test split.
2. VIVERE cards compress the predicted vectors into small per-row coefficient
   records.  Exact phrase addresses store a card ID and FP16 coefficients;
   shared card means and bases reconstruct values only when a row is read.

The module is NumPy/PyTorch only so the reference benchmark remains runnable
on ordinary CPU machines without scikit-learn.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
import torch
from torch import Tensor, nn

from .memory import PhraseLookup


def _sha256(path: Path, chunk_bytes: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_scale(values: np.ndarray, axis: int = 0) -> np.ndarray:
    scale = values.std(axis=axis)
    return np.where(scale < 1e-6, 1.0, scale).astype(np.float32)


def canonicalize_svd_signs(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Remove arbitrary SVD sign flips for deterministic artifacts."""
    left = np.asarray(left).copy()
    right = np.asarray(right).copy()
    for column in range(left.shape[1]):
        pivot = int(np.argmax(np.abs(left[:, column])))
        if left[pivot, column] < 0:
            left[:, column] *= -1
            right[column] *= -1
    return left, right


@dataclass(frozen=True)
class PhraseSplit:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray

    @classmethod
    def deterministic(
        cls,
        orders: np.ndarray,
        *,
        train_per_order: int = 700,
        validation_per_order: int = 150,
        salt: str = "gate2c-vm32-v1",
    ) -> "PhraseSplit":
        order_values = np.asarray(orders, dtype=np.int64)
        train: list[int] = []
        validation: list[int] = []
        test: list[int] = []
        for order in sorted(set(order_values.tolist())):
            rows = np.flatnonzero(order_values == order).tolist()
            rows.sort(
                key=lambda row: hashlib.sha256(
                    f"{salt}:{order}:{row}".encode("utf-8")
                ).digest()
            )
            required = train_per_order + validation_per_order
            if len(rows) <= required:
                raise ValueError(f"order {order} has only {len(rows)} rows")
            train.extend(rows[:train_per_order])
            validation.extend(rows[train_per_order:required])
            test.extend(rows[required:])
        return cls(
            train=np.asarray(sorted(train), dtype=np.int64),
            validation=np.asarray(sorted(validation), dtype=np.int64),
            test=np.asarray(sorted(test), dtype=np.int64),
        )


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray, rows: np.ndarray) -> "Standardizer":
        selected = np.asarray(values, dtype=np.float32)[rows]
        return cls(selected.mean(axis=0).astype(np.float32), _safe_scale(selected))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((np.asarray(values, dtype=np.float32) - self.mean) / self.scale).astype(np.float32)

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=np.float32) * self.scale + self.mean).astype(np.float32)


@dataclass(frozen=True)
class RidgeMap:
    source: Standardizer
    target_mean: np.ndarray
    weight: np.ndarray
    ridge: float

    @classmethod
    def fit(
        cls,
        source: np.ndarray,
        target: np.ndarray,
        rows: np.ndarray,
        *,
        ridge: float,
    ) -> "RidgeMap":
        scaler = Standardizer.fit(source, rows)
        x = scaler.transform(source)[rows].astype(np.float64)
        y_all = np.asarray(target, dtype=np.float32)
        target_mean = y_all[rows].mean(axis=0).astype(np.float32)
        y = (y_all[rows] - target_mean).astype(np.float64)
        gram = x.T @ x
        weight = np.linalg.solve(
            gram + float(ridge) * np.eye(gram.shape[0], dtype=np.float64),
            x.T @ y,
        ).astype(np.float32)
        return cls(scaler, target_mean, weight, float(ridge))

    def predict(self, source: np.ndarray) -> np.ndarray:
        return (self.source.transform(source) @ self.weight + self.target_mean).astype(np.float32)


class TinyTransferMLP(nn.Module):
    def __init__(self, source_width: int, target_width: int, hidden_width: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(source_width, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, target_width),
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.network(values)


@dataclass
class FittedMLPMap:
    source: Standardizer
    target_mean: np.ndarray
    target_scale: np.ndarray
    model: TinyTransferMLP
    best_validation_mse: float
    epochs: int

    def predict(self, source: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            x = torch.from_numpy(self.source.transform(source))
            output = self.model(x).cpu().numpy()
        return (output * self.target_scale + self.target_mean).astype(np.float32)


def fit_mlp_map(
    source: np.ndarray,
    target: np.ndarray,
    split: PhraseSplit,
    *,
    seed: int,
    hidden_width: int = 64,
    maximum_epochs: int = 600,
    patience: int = 60,
) -> FittedMLPMap:
    torch.manual_seed(int(seed))
    source_scaler = Standardizer.fit(source, split.train)
    x = torch.from_numpy(source_scaler.transform(source))
    target_values = np.asarray(target, dtype=np.float32)
    target_mean = target_values[split.train].mean(axis=0).astype(np.float32)
    target_scale = _safe_scale(target_values[split.train])
    y = torch.from_numpy(((target_values - target_mean) / target_scale).astype(np.float32))
    model = TinyTransferMLP(source.shape[1], target.shape[1], int(hidden_width))
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-3, weight_decay=2e-4)
    train_rows = torch.from_numpy(split.train)
    validation_rows = torch.from_numpy(split.validation)
    best_state: dict[str, Tensor] | None = None
    best = math.inf
    best_epoch = 0
    stale = 0
    for epoch in range(1, int(maximum_epochs) + 1):
        model.train()
        prediction = model(x.index_select(0, train_rows))
        loss = torch.nn.functional.mse_loss(prediction, y.index_select(0, train_rows))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if epoch % 5:
            continue
        model.eval()
        with torch.no_grad():
            observed = torch.nn.functional.mse_loss(
                model(x.index_select(0, validation_rows)),
                y.index_select(0, validation_rows),
            ).item()
        if observed < best - 1e-6:
            best = float(observed)
            best_epoch = epoch
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 5
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("MLP transfer did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    return FittedMLPMap(
        source=source_scaler,
        target_mean=target_mean,
        target_scale=target_scale,
        model=model,
        best_validation_mse=best,
        epochs=best_epoch,
    )


@dataclass(frozen=True)
class KMeansRouter:
    standardizer: Standardizer
    centers: np.ndarray

    @classmethod
    def fit(
        cls,
        values: np.ndarray,
        rows: np.ndarray,
        *,
        card_count: int,
        seed: int,
        maximum_iterations: int = 80,
    ) -> "KMeansRouter":
        if card_count < 1 or card_count > len(rows):
            raise ValueError("invalid card_count")
        scaler = Standardizer.fit(values, rows)
        normalized = scaler.transform(values)
        train_values = normalized[rows]
        rng = np.random.default_rng(int(seed))
        centers = np.empty((card_count, values.shape[1]), dtype=np.float32)
        first_local = int(rng.integers(0, len(rows)))
        centers[0] = train_values[first_local]
        nearest = np.square(train_values - centers[0]).sum(axis=1)
        for card in range(1, card_count):
            probability = nearest / max(float(nearest.sum()), 1e-12)
            local = int(rng.choice(len(rows), p=probability))
            centers[card] = train_values[local]
            nearest = np.minimum(
                nearest,
                np.square(train_values - centers[card]).sum(axis=1),
            )
        prior: np.ndarray | None = None
        for _ in range(int(maximum_iterations)):
            distances = np.square(
                train_values[:, None, :] - centers[None, :, :]
            ).sum(axis=-1)
            assignment = distances.argmin(axis=1)
            updated = centers.copy()
            for card in range(card_count):
                members = train_values[assignment == card]
                if len(members):
                    updated[card] = members.mean(axis=0)
            if prior is not None and np.array_equal(assignment, prior):
                centers = updated
                break
            centers = updated
            prior = assignment
        return cls(scaler, centers.astype(np.float32))

    def distances(self, values: np.ndarray) -> np.ndarray:
        normalized = self.standardizer.transform(values)
        distances = (np.square(normalized).sum(axis=1, keepdims=True)
                     + np.square(self.centers).sum(axis=1, keepdims=True).T
                     - 2.0 * (normalized @ self.centers.T))
        return np.maximum(distances, 0.0).astype(np.float32)

    def route(self, values: np.ndarray, *, top_k: int = 1) -> tuple[np.ndarray, np.ndarray]:
        distances = self.distances(values)
        if not 1 <= top_k <= self.centers.shape[0]:
            raise ValueError("top_k falls outside the card bank")
        indices = np.argpartition(distances, top_k - 1, axis=1)[:, :top_k]
        selected = np.take_along_axis(distances, indices, axis=1)
        ordering = np.argsort(selected, axis=1)
        indices = np.take_along_axis(indices, ordering, axis=1)
        selected = np.take_along_axis(selected, ordering, axis=1)
        temperature = max(float(np.median(selected[:, 0])), 1e-6)
        weights = np.exp(-selected / temperature)
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-8)
        return indices.astype(np.int64), weights.astype(np.float32)


@dataclass(frozen=True)
class ResidualCard:
    residual_mean: np.ndarray
    basis: np.ndarray
    coefficient_source_mean: np.ndarray
    coefficient_mean: np.ndarray
    coefficient_weight: np.ndarray
    shrinkage: float

    def predict(self, normalized_source: np.ndarray) -> np.ndarray:
        coefficients = (
            (normalized_source - self.coefficient_source_mean) @ self.coefficient_weight
            + self.coefficient_mean
        )
        residual = self.residual_mean + coefficients @ self.basis
        return (float(self.shrinkage) * residual).astype(np.float32)


@dataclass
class MacslMap:
    global_map: RidgeMap
    router: KMeansRouter
    cards: tuple[ResidualCard, ...]
    source_for_router: np.ndarray
    top_k: int

    def predict(self, source: np.ndarray, router_source: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        routed_source = self.source_for_router if router_source is None else router_source
        indices, weights = self.router.route(routed_source, top_k=self.top_k)
        normalized = self.router.standardizer.transform(routed_source)
        local = np.empty((len(source), self.top_k, self.global_map.weight.shape[1]), dtype=np.float32)
        for slot in range(self.top_k):
            for card in range(len(self.cards)):
                rows = np.flatnonzero(indices[:, slot] == card)
                if len(rows):
                    local[rows, slot] = self.cards[card].predict(normalized[rows])
        correction = (local * weights[..., None]).sum(axis=1)
        return (
            self.global_map.predict(source) + correction,
            indices,
            weights,
        )


def fit_macsl_map(
    source: np.ndarray,
    router_source: np.ndarray,
    target: np.ndarray,
    split: PhraseSplit,
    *,
    card_count: int,
    residual_rank: int,
    top_k: int,
    ridge: float,
    local_ridge: float,
    shrinkage_prior: float,
    seed: int,
) -> MacslMap:
    global_map = RidgeMap.fit(source, target, split.train, ridge=ridge)
    global_prediction = global_map.predict(source)
    router = KMeansRouter.fit(
        router_source,
        split.train,
        card_count=card_count,
        seed=seed,
    )
    distances = router.distances(router_source)
    assignment = distances.argmin(axis=1)
    normalized_router = router.standardizer.transform(router_source)
    cards: list[ResidualCard] = []
    for card in range(card_count):
        rows = split.train[assignment[split.train] == card]
        if len(rows) < 4:
            cards.append(
                ResidualCard(
                    residual_mean=np.zeros(target.shape[1], dtype=np.float32),
                    basis=np.zeros((1, target.shape[1]), dtype=np.float32),
                    coefficient_source_mean=np.zeros(router_source.shape[1], dtype=np.float32),
                    coefficient_mean=np.zeros(1, dtype=np.float32),
                    coefficient_weight=np.zeros((router_source.shape[1], 1), dtype=np.float32),
                    shrinkage=0.0,
                )
            )
            continue
        residual = target[rows] - global_prediction[rows]
        residual_mean = residual.mean(axis=0).astype(np.float32)
        centered = (residual - residual_mean).astype(np.float64)
        _, _, right = np.linalg.svd(centered, full_matrices=False)
        rank = max(1, min(int(residual_rank), right.shape[0], len(rows) - 1))
        basis = right[:rank].astype(np.float32)
        coefficients = centered @ basis.T
        x = normalized_router[rows].astype(np.float64)
        x_mean = x.mean(axis=0).astype(np.float32)
        y_mean = coefficients.mean(axis=0).astype(np.float32)
        xc = x - x_mean
        yc = coefficients - y_mean
        weight = np.linalg.solve(
            xc.T @ xc + float(local_ridge) * np.eye(xc.shape[1]),
            xc.T @ yc,
        ).astype(np.float32)
        shrinkage = float(len(rows) / (len(rows) + float(shrinkage_prior)))
        cards.append(
            ResidualCard(
                residual_mean=residual_mean,
                basis=basis,
                coefficient_source_mean=x_mean,
                coefficient_mean=y_mean,
                coefficient_weight=weight,
                shrinkage=shrinkage,
            )
        )
    return MacslMap(
        global_map=global_map,
        router=router,
        cards=tuple(cards),
        source_for_router=np.asarray(router_source, dtype=np.float32),
        top_k=int(top_k),
    )


@dataclass(frozen=True)
class VivereEncoding:
    card_ids: np.ndarray
    coefficients: np.ndarray
    means: np.ndarray
    bases: np.ndarray
    original_width: int
    rank: int

    def reconstruct(self) -> np.ndarray:
        card_ids = np.asarray(self.card_ids, dtype=np.int64)
        coefficients = np.asarray(self.coefficients, dtype=np.float32)
        return (
            self.means[card_ids].astype(np.float32)
            + np.einsum(
                "nr,nrd->nd",
                coefficients,
                self.bases[card_ids].astype(np.float32),
                optimize=True,
            )
        ).astype(np.float32)

    @property
    def theoretical_bf16_bytes(self) -> int:
        # Card IDs use uint16; means, bases, and row coefficients use BF16/FP16.
        return int(
            self.card_ids.size * 2
            + self.coefficients.size * 2
            + self.means.size * 2
            + self.bases.size * 2
        )


def fit_vivere_encoding(
    values: np.ndarray,
    split: PhraseSplit,
    *,
    rank: int,
    card_ids: np.ndarray | None = None,
) -> VivereEncoding:
    matrix = np.asarray(values, dtype=np.float32)
    ids = np.zeros(len(matrix), dtype=np.int64) if card_ids is None else np.asarray(card_ids, dtype=np.int64)
    if ids.shape != (len(matrix),):
        raise ValueError("card_ids must contain one ID per row")
    card_count = int(ids.max()) + 1
    means = np.zeros((card_count, matrix.shape[1]), dtype=np.float32)
    bases = np.zeros((card_count, rank, matrix.shape[1]), dtype=np.float32)
    coefficients = np.zeros((len(matrix), rank), dtype=np.float32)
    global_rows = split.train
    global_mean = matrix[global_rows].mean(axis=0)
    global_centered = matrix[global_rows] - global_mean
    _, _, global_right = np.linalg.svd(global_centered, full_matrices=False)
    global_basis = global_right[:rank].astype(np.float32)
    for card in range(card_count):
        train_rows = split.train[ids[split.train] == card]
        if len(train_rows) <= rank:
            mean = global_mean
            basis = global_basis
        else:
            mean = matrix[train_rows].mean(axis=0)
            centered = matrix[train_rows] - mean
            _, _, right = np.linalg.svd(centered, full_matrices=False)
            available = min(rank, right.shape[0])
            basis = np.zeros((rank, matrix.shape[1]), dtype=np.float32)
            basis[:available] = right[:available]
        means[card] = mean
        bases[card] = basis
        rows = np.flatnonzero(ids == card)
        coefficients[rows] = (matrix[rows] - mean) @ basis.T
    return VivereEncoding(
        card_ids=ids.astype(np.uint16),
        coefficients=coefficients.astype(np.float16),
        means=means.astype(np.float16),
        bases=bases.astype(np.float16),
        original_width=matrix.shape[1],
        rank=int(rank),
    )


def write_vivere_phrase_bank(
    root: str | Path,
    *,
    keys_path: str | Path,
    early: VivereEncoding,
    late: VivereEncoding,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    destination = Path(root)
    if destination.exists():
        for child in destination.iterdir():
            if child.is_file():
                child.unlink()
    destination.mkdir(parents=True, exist_ok=True)
    if not np.array_equal(early.card_ids, late.card_ids):
        raise ValueError("early and late encodings must share card IDs")
    keys_source = Path(keys_path)
    keys_destination = destination / "keys.jsonl"
    keys_destination.write_bytes(keys_source.read_bytes())
    arrays = {
        "card_ids": early.card_ids,
        "early_coefficients": early.coefficients,
        "early_means": early.means,
        "early_bases": early.bases,
        "late_coefficients": late.coefficients,
        "late_means": late.means,
        "late_bases": late.bases,
    }
    files: dict[str, dict[str, Any]] = {}
    for name, values in arrays.items():
        path = destination / f"{name}.npy"
        np.save(path, values, allow_pickle=False)
        files[name] = {"path": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)}
    files["keys"] = {
        "path": keys_destination.name,
        "bytes": keys_destination.stat().st_size,
        "sha256": _sha256(keys_destination),
    }
    rows = int(len(early.card_ids))
    manifest = {
        "schema_version": 1,
        "kind": "vivere_phrase_bank",
        "rows": rows,
        "width": int(early.original_width),
        "rank": int(early.rank),
        "card_count": int(early.means.shape[0]),
        "orders": [2, 3],
        "theoretical_bf16_payload_bytes": int(
            early.theoretical_bf16_bytes + late.theoretical_bf16_bytes - early.card_ids.size * 2
        ),
        "metadata": metadata or {},
        "files": files,
    }
    temporary = destination / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination / "manifest.json")
    return manifest


class ViverePhraseBank:
    """Exact phrase bank with vectorized VIVERE reconstruction."""
    def __init__(self, root: str | Path, *, validate: bool = True) -> None:
        self.root=Path(root); self.manifest=json.loads((self.root/'manifest.json').read_text(encoding='utf-8'))
        if self.manifest.get('kind')!='vivere_phrase_bank': raise ValueError('not a VIVERE phrase bank')
        self.rows=int(self.manifest['rows']); self.width=int(self.manifest['width']); self.rank=int(self.manifest['rank']); self.orders=tuple(sorted((int(v) for v in self.manifest['orders']),reverse=True)); files=self.manifest['files']
        if validate:
            for record in files.values():
                path=self.root/record['path']
                if path.stat().st_size!=int(record['bytes']) or _sha256(path)!=str(record['sha256']): raise ValueError(f'VIVERE bank integrity mismatch: {path}')
        self.card_ids=np.load(self.root/files['card_ids']['path'],mmap_mode='r'); self.early_coefficients=np.load(self.root/files['early_coefficients']['path'],mmap_mode='r'); self.late_coefficients=np.load(self.root/files['late_coefficients']['path'],mmap_mode='r')
        self.early_means=np.asarray(np.load(self.root/files['early_means']['path'],mmap_mode='r'),dtype=np.float32).copy(); self.early_bases=np.asarray(np.load(self.root/files['early_bases']['path'],mmap_mode='r'),dtype=np.float32).copy(); self.late_means=np.asarray(np.load(self.root/files['late_means']['path'],mmap_mode='r'),dtype=np.float32).copy(); self.late_bases=np.asarray(np.load(self.root/files['late_bases']['path'],mmap_mode='r'),dtype=np.float32).copy()
        self.resident_card_cache_bytes=int(self.early_means.nbytes+self.early_bases.nbytes+self.late_means.nbytes+self.late_bases.nbytes)
        self.index={}; self.metadata=[]
        with (self.root/files['keys']['path']).open(encoding='utf-8') as handle:
            for expected,line in enumerate(handle):
                if not line.strip(): continue
                record=json.loads(line); row=int(record['row'])
                if row!=expected: raise ValueError('VIVERE rows not contiguous')
                key=tuple(int(v) for v in record['token_ids']); self.index[key]=row; self.metadata.append(record)
        if len(self.metadata)!=self.rows: raise ValueError('VIVERE key count differs')
    def get_many(self, rows) -> tuple[np.ndarray,np.ndarray]:
        indices=np.asarray(rows,dtype=np.int64)
        if indices.ndim!=1: raise ValueError('rows must be 1D')
        if len(indices) and (indices.min()<0 or indices.max()>=self.rows): raise IndexError('row outside bank')
        cards=np.asarray(self.card_ids[indices],dtype=np.int64); ec=np.asarray(self.early_coefficients[indices],dtype=np.float32); lc=np.asarray(self.late_coefficients[indices],dtype=np.float32)
        early=self.early_means[cards]+np.einsum('nr,nrd->nd',ec,self.early_bases[cards],optimize=True); late=self.late_means[cards]+np.einsum('nr,nrd->nd',lc,self.late_bases[cards],optimize=True)
        return early.astype(np.float32),late.astype(np.float32)
    def get(self,row:int)->tuple[np.ndarray,np.ndarray]:
        if not 0<=int(row)<self.rows: raise IndexError(row)
        e,l=self.get_many([int(row)]); return e[0],l[0]
    def resolve(self,input_ids:Tensor)->PhraseLookup:
        if input_ids.ndim!=2: raise ValueError('input_ids must be [B,T]')
        batch,length=input_ids.shape; mask=np.zeros((batch,length),np.bool_); row_matrix=np.full((batch,length),-1,np.int64); orders=np.zeros((batch,length),np.int64); selected_rows=[]; positions=[]
        for b,sequence in enumerate(input_ids.detach().to('cpu',dtype=torch.long).tolist()):
            for end in range(length):
                for order in self.orders:
                    start=end-order+1
                    if start<0: continue
                    row=self.index.get(tuple(sequence[start:end+1]))
                    if row is None: continue
                    mask[b,end]=True;row_matrix[b,end]=row;orders[b,end]=order;selected_rows.append(row);positions.append((b,end));break
        early=np.zeros((batch,length,self.width),np.float32);late=np.zeros_like(early)
        if selected_rows:
            ev,lv=self.get_many(selected_rows); bi=np.asarray([p[0] for p in positions]);ti=np.asarray([p[1] for p in positions]);early[bi,ti]=ev;late[bi,ti]=lv
        device=input_ids.device
        return PhraseLookup(torch.from_numpy(early).to(device),torch.from_numpy(late).to(device),torch.from_numpy(mask).to(device),torch.from_numpy(row_matrix).to(device),torch.from_numpy(orders).to(device))

