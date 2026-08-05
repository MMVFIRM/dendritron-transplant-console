"""Fitted joint geometry connecting donor depths to a shallow concept frame."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal
import torch
from torch import Tensor

Source = Literal["early", "late"]


@dataclass(frozen=True)
class JointTransferStats:
    source_width: int
    joint_width: int
    anchor_count: int
    explained_variance_fraction: float
    early_mse: float
    late_mse: float
    early_neighbor_recall_at_8: float
    late_neighbor_recall_at_8: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class JointTransferCheckpoint:
    reference_mean: Tensor
    reference_basis: Tensor
    early_weight: Tensor
    early_bias: Tensor
    late_weight: Tensor
    late_bias: Tensor
    stats: JointTransferStats

    def transform_reference(self, values: Tensor) -> Tensor:
        return (values.float() - self.reference_mean) @ self.reference_basis

    def transform_source(self, values: Tensor, source: Source) -> Tensor:
        if source == "early":
            return values.float() @ self.early_weight + self.early_bias
        if source == "late":
            return values.float() @ self.late_weight + self.late_bias
        raise ValueError(f"unknown source {source}")

    def save(self, path: str | Path, *, metadata: dict[str, Any] | None = None) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "reference_mean": self.reference_mean,
            "reference_basis": self.reference_basis,
            "early_weight": self.early_weight,
            "early_bias": self.early_bias,
            "late_weight": self.late_weight,
            "late_bias": self.late_bias,
            "stats": self.stats.to_dict(),
            "metadata": metadata or {},
        }, destination)

    @classmethod
    def load(cls, path: str | Path) -> tuple["JointTransferCheckpoint", dict[str, Any]]:
        record = torch.load(path, map_location="cpu", weights_only=True)
        result = cls(
            reference_mean=record["reference_mean"],
            reference_basis=record["reference_basis"],
            early_weight=record["early_weight"],
            early_bias=record["early_bias"],
            late_weight=record["late_weight"],
            late_bias=record["late_bias"],
            stats=JointTransferStats(**record["stats"]),
        )
        return result, dict(record.get("metadata", {}))


def _ridge(source: Tensor, target: Tensor, ridge: float) -> tuple[Tensor, Tensor]:
    source = source.float()
    target = target.float()
    mean = source.mean(dim=0)
    centered = source - mean
    gram = centered.T @ centered
    identity = torch.eye(gram.shape[0], dtype=gram.dtype)
    weight = torch.linalg.solve(gram + float(ridge) * identity, centered.T @ target)
    bias = target.mean(dim=0) - mean @ weight
    return weight, bias


def _neighbor_recall(predicted: Tensor, reference: Tensor, k: int = 8, limit: int = 512) -> float:
    count = min(limit, reference.shape[0])
    predicted = predicted[:count]
    reference = reference[:count]
    target_distance = torch.cdist(reference, reference)
    predicted_distance = torch.cdist(predicted, predicted)
    diagonal = torch.eye(count, dtype=torch.bool)
    target = target_distance.masked_fill(diagonal, torch.inf).topk(k, largest=False).indices
    observed = predicted_distance.masked_fill(diagonal, torch.inf).topk(k, largest=False).indices
    return float((observed.unsqueeze(-1) == target.unsqueeze(-2)).any(dim=-1).float().mean())


def fit_joint_transfer(early: Tensor, late: Tensor, reference: Tensor, *, joint_width: int = 48,
                       ridge: float = 1e-2) -> JointTransferCheckpoint:
    if early.shape != late.shape or early.shape != reference.shape or early.ndim != 2:
        raise ValueError("joint-transfer anchors must be equal-shape [N,D] tensors")
    if not 1 <= joint_width <= reference.shape[1]:
        raise ValueError("joint_width is invalid")
    reference = reference.float()
    mean = reference.mean(dim=0)
    centered = reference - mean
    _, singular, vectors = torch.pca_lowrank(centered, q=joint_width, center=False)
    basis = vectors[:, :joint_width].contiguous()
    joint = centered @ basis
    early_weight, early_bias = _ridge(early, joint, ridge)
    late_weight, late_bias = _ridge(late, joint, ridge)
    early_pred = early.float() @ early_weight + early_bias
    late_pred = late.float() @ late_weight + late_bias
    total_variance = centered.square().sum().clamp_min(1e-12)
    explained = joint.square().sum() / total_variance
    stats = JointTransferStats(
        source_width=reference.shape[1],
        joint_width=joint_width,
        anchor_count=reference.shape[0],
        explained_variance_fraction=float(explained),
        early_mse=float((early_pred-joint).square().mean()),
        late_mse=float((late_pred-joint).square().mean()),
        early_neighbor_recall_at_8=_neighbor_recall(early_pred, joint),
        late_neighbor_recall_at_8=_neighbor_recall(late_pred, joint),
    )
    return JointTransferCheckpoint(mean, basis, early_weight, early_bias,
                                   late_weight, late_bias, stats)
