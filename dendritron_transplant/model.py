"""Runnable sparse Dendritron recipient assembled from transplant subsystems."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .config import DendritronRecipientConfig
from .memory import MemoryPayloads, SparseMemoryFusion
from .output import (
    ClusteredVocabularyHead,
    SparseVocabularyLoss,
    SparseVocabularyOutput,
    sparse_vocabulary_loss,
)
from .recurrent import SparseCoreStats, TwoBlockSparseCore


def _sinusoidal_geometry(length: int, width: int) -> Tensor:
    positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, width, 2, dtype=torch.float32)
        * (-math.log(10_000.0) / max(width, 1))
    )
    result = torch.zeros(length, width, dtype=torch.float32)
    result[:, 0::2] = torch.sin(positions * frequencies)
    if width > 1:
        result[:, 1::2] = torch.cos(
            positions * frequencies[: result[:, 1::2].shape[1]]
        )
    return result


@dataclass(frozen=True)
class RecipientOutput:
    vocabulary: SparseVocabularyOutput
    hidden: Tensor
    core_stats: SparseCoreStats | None = None

    @property
    def next_token_ids(self) -> Tensor:
        return self.vocabulary.argmax_token_ids()


@dataclass(frozen=True)
class RecipientLoss:
    total: Tensor
    vocabulary: SparseVocabularyLoss


class DendritronRecipientLM(nn.Module):
    """Small recurrent recipient with sparse memory, experts, branches, and output."""

    def __init__(self, config: DendritronRecipientConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding(config.vocab_size, config.model_width)
        nn.init.normal_(
            self.token_embeddings.weight,
            std=config.model_width**-0.5,
        )
        self.register_buffer(
            "position_geometry",
            _sinusoidal_geometry(
                config.max_sequence_length,
                config.model_width,
            ),
            persistent=True,
        )
        self.memory_fusion = SparseMemoryFusion(
            config.model_width,
            config.donor_width,
            config.hash_memory,
            initialization_gain=config.deep_loop_beta,
            use_hash_memory=config.use_hash_memory,
            use_exact_early=config.use_exact_early,
            use_exact_late=config.use_exact_late,
            use_definition_memory=config.use_definition_memory,
        )
        self.core = TwoBlockSparseCore(
            config,
            memory_fusion=self.memory_fusion,
        )
        self.vocabulary_head = ClusteredVocabularyHead(
            config.model_width,
            config.vocab_size,
            cluster_count=config.vocabulary_clusters,
            top_k_clusters=config.vocabulary_top_k_clusters,
            temperature=config.output_temperature,
            epsilon=config.route_epsilon,
        )

    def forward(
        self,
        input_ids: Tensor,
        *,
        memory_payloads: MemoryPayloads | None = None,
        target_ids: Tensor | None = None,
        include_target_cluster: bool = False,
        rounds: int | None = None,
        adaptive_threshold: float | None = None,
        minimum_rounds: int = 1,
        return_stats: bool = False,
    ) -> RecipientOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be [B,T]")
        if input_ids.shape[1] > self.config.max_sequence_length:
            raise ValueError("input exceeds configured maximum sequence length")
        if input_ids.numel() and (
            int(input_ids.min()) < 0
            or int(input_ids.max()) >= self.config.vocab_size
        ):
            raise ValueError("input_ids contain an out-of-range token")
        if target_ids is not None and target_ids.shape != input_ids.shape:
            raise ValueError("target_ids must match input_ids")

        length = input_ids.shape[1]
        hidden = self.token_embeddings(input_ids)
        hidden = hidden + self.position_geometry[:length].to(hidden).unsqueeze(0)
        hidden = hidden + self.memory_fusion.initial_update(hidden, memory_payloads)
        if return_stats:
            hidden, core_stats = self.core(
                hidden,
                payloads=memory_payloads,
                rounds=rounds,
                adaptive_threshold=adaptive_threshold,
                minimum_rounds=minimum_rounds,
                return_stats=True,
            )
        else:
            hidden = self.core(
                hidden,
                payloads=memory_payloads,
                rounds=rounds,
                adaptive_threshold=adaptive_threshold,
                minimum_rounds=minimum_rounds,
                return_stats=False,
            )
            core_stats = None
        vocabulary = self.vocabulary_head(
            hidden,
            self.token_embeddings.weight,
            target_ids=target_ids,
            include_target_cluster=include_target_cluster,
        )
        return RecipientOutput(vocabulary=vocabulary, hidden=hidden, core_stats=core_stats)

    def loss(self, output: RecipientOutput, targets: Tensor) -> RecipientLoss:
        vocabulary = sparse_vocabulary_loss(
            output.vocabulary,
            targets,
            cluster_loss_weight=self.config.cluster_loss_weight,
        )
        return RecipientLoss(total=vocabulary.total, vocabulary=vocabulary)

    def parameter_report(self) -> dict[str, int | float]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        conditional_total = self.core.total_conditional_parameters
        conditional_active = self.core.active_conditional_parameters_per_token
        return {
            "stored_parameters": int(total),
            "trainable_parameters": int(trainable),
            "conditional_parameters": int(conditional_total),
            "active_conditional_parameters_per_token": int(conditional_active),
            "conditional_active_fraction": float(
                conditional_active / conditional_total
                if conditional_total
                else 0.0
            ),
            "maximum_output_candidates": int(
                self.vocabulary_head.maximum_candidates
            ),
            "output_candidate_fraction": float(
                self.vocabulary_head.maximum_candidates / self.config.vocab_size
            ),
        }

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self.state_dict(),
                "config": self.config.to_dict(),
                "metadata": metadata or {},
            },
            destination,
        )

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
    ) -> tuple["DendritronRecipientLM", dict[str, Any]]:
        record = torch.load(path, map_location=map_location, weights_only=True)
        if "model" not in record or "config" not in record:
            raise ValueError("checkpoint must contain model and config")
        config = DendritronRecipientConfig.from_dict(record["config"])
        model = cls(config).to(map_location)
        model.load_state_dict(record["model"])
        return model, record
