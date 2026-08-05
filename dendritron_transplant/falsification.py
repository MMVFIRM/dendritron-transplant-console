"""Deterministic phrase-memory falsification controls for transplant experiments."""
from __future__ import annotations
import hashlib, json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import numpy as np
from .memory import FrozenPhraseBank, PhraseBankBuilder, PhraseRecord
CONTROL_NAMES = ('correct', 'shuffled', 'semantic_opposite', 'random_frozen', 'zero')

def _array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).view(np.uint8)).hexdigest()

def _safe_cosine_mean(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    return float(np.mean(np.sum(left * right, axis=1) / np.maximum(np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1), 1e-12)))

def _derangement(indices: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if indices.ndim != 1 or len(indices) < 2:
        raise ValueError('derangement requires at least two row indices')
    for _ in range(128):
        candidate = rng.permutation(indices)
        if not np.any(candidate == indices):
            return candidate
    return np.roll(indices, 1)

def _shuffled_mapping(bank: FrozenPhraseBank, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    mapping = np.arange(bank.rows, dtype=np.int64)
    for order in bank.orders:
        indices = np.array([row for row, item in enumerate(bank.metadata) if int(item['order']) == order], dtype=np.int64)
        mapping[indices] = _derangement(indices, rng)
    return mapping

def _semantic_opposite_mapping(bank: FrozenPhraseBank) -> np.ndarray:
    source = np.concatenate([np.asarray(bank.layer_early, dtype=np.float64), np.asarray(bank.layer_late, dtype=np.float64)], axis=1)
    mapping = np.arange(bank.rows, dtype=np.int64)
    for order in bank.orders:
        indices = np.array([row for row, item in enumerate(bank.metadata) if int(item['order']) == order], dtype=np.int64)
        values = source[indices]
        centered = values - values.mean(axis=0, keepdims=True)
        _, _, right = np.linalg.svd(centered, full_matrices=False)
        scores = centered @ right[0]
        ordered = indices[np.argsort(scores, kind='mergesort')]
        targets = ordered[::-1].copy()
        if np.any(targets == ordered):
            targets = np.roll(targets, 1)
        mapping[ordered] = targets
    return mapping

def _matched_random(source: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    source64 = np.asarray(source, dtype=np.float64)
    mean = source64.mean(0, keepdims=True)
    std = source64.std(0, keepdims=True)
    sampled = rng.normal(size=source64.shape) * np.maximum(std, 1e-08) + mean
    sampled *= np.linalg.norm(source64, axis=1, keepdims=True) / np.maximum(np.linalg.norm(sampled, axis=1, keepdims=True), 1e-12)
    return sampled.astype(np.float32)

@dataclass(frozen=True)
class ControlBuildReport:
    name: str
    rows: int
    width: int
    mapping_sha256: str | None
    fixed_point_fraction: float | None
    source_value_multiset_preserved: bool
    mean_early_cosine_to_correct: float
    mean_late_cosine_to_correct: float
    early_mean_absolute_drift: float
    late_mean_absolute_drift: float
    early_std_absolute_drift: float
    late_std_absolute_drift: float

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()

def _report(name, ce, cl, e, l, mapping, *, multiset_preserved) -> ControlBuildReport:
    fixed = None if mapping is None else float(np.mean(mapping == np.arange(len(mapping))))
    return ControlBuildReport(name, int(ce.shape[0]), int(ce.shape[1]), None if mapping is None else _array_sha256(mapping), fixed, bool(multiset_preserved), _safe_cosine_mean(ce, e), _safe_cosine_mean(cl, l), float(np.mean(np.abs(e.mean(0) - ce.mean(0)))), float(np.mean(np.abs(l.mean(0) - cl.mean(0)))), float(np.mean(np.abs(e.std(0) - ce.std(0)))), float(np.mean(np.abs(l.std(0) - cl.std(0)))))

def build_phrase_control_banks(source_root: str | Path, destination_root: str | Path, *, seed: int=20260804, overwrite: bool=False) -> dict[str, Mapping[str, Any]]:
    source = FrozenPhraseBank(source_root)
    destination = Path(destination_root)
    destination.mkdir(parents=True, exist_ok=True)
    ce = np.asarray(source.layer_early, dtype=np.float32)
    cl = np.asarray(source.layer_late, dtype=np.float32)
    shuffled = _shuffled_mapping(source, seed)
    opposite = _semantic_opposite_mapping(source)
    rng = np.random.default_rng(int(seed) + 1)
    controls = {'correct': (ce, cl, np.arange(source.rows, dtype=np.int64), True), 'shuffled': (ce[shuffled], cl[shuffled], shuffled, True), 'semantic_opposite': (ce[opposite], cl[opposite], opposite, True), 'random_frozen': (_matched_random(ce, rng), _matched_random(cl, rng), None, False), 'zero': (np.zeros_like(ce), np.zeros_like(cl), None, False)}
    reports = {}
    for name, (early, late, mapping, preserved) in controls.items():
        root = destination / name
        if not (root.exists() and any(root.iterdir()) and (not overwrite)):
            records = [PhraseRecord(tuple((int(v) for v in item['token_ids'])), early[row], late[row], str(item.get('text', '')), int(item.get('frequency', 0))) for row, item in enumerate(source.metadata)]
            PhraseBankBuilder.write(root, records, overwrite=True)
        result = _report(name, ce, cl, early, late, mapping, multiset_preserved=preserved)
        (root / 'control.json').write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + '\n')
        reports[name] = result.to_dict()
    (destination / 'controls_report.json').write_text(json.dumps(reports, indent=2, sort_keys=True) + '\n')
    return reports
