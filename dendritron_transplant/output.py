"""Hierarchical sparse vocabulary routing without a full dense vocabulary scan."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .recurrent import RMSNorm


@dataclass(frozen=True)
class SparseVocabularyOutput:
    scores: Tensor
    candidate_token_ids: Tensor
    cluster_logits: Tensor
    selected_clusters: Tensor
    valid_candidate_count: Tensor
    vocabulary_size: int

    @property
    def mean_active_vocabulary_fraction(self) -> float:
        return float(
            self.valid_candidate_count.float().mean() / max(self.vocabulary_size, 1)
        )

    def argmax_token_ids(self) -> Tensor:
        local = self.scores.argmax(dim=-1, keepdim=True)
        return self.candidate_token_ids.gather(-1, local).squeeze(-1)


@dataclass(frozen=True)
class SparseVocabularyLoss:
    total: Tensor
    candidate_loss: Tensor
    cluster_loss: Tensor


class ClusteredVocabularyHead(nn.Module):
    """Route to a few token clusters, then rank only those token embeddings.

    Unlike the upstream reference scorer, scores are not divided by model
    width. Unit-vector Euclidean distance therefore has a usable dynamic range
    and does not impose a hinge-loss floor near the requested margin.
    """

    def __init__(
        self,
        width: int,
        vocab_size: int,
        *,
        cluster_count: int,
        top_k_clusters: int,
        temperature: float = 8.0,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        if not 1 <= top_k_clusters <= cluster_count <= vocab_size:
            raise ValueError("invalid clustered vocabulary geometry")
        self.width = int(width)
        self.vocab_size = int(vocab_size)
        self.cluster_count = int(cluster_count)
        self.top_k_clusters = int(top_k_clusters)
        self.temperature = float(temperature)
        self.epsilon = float(epsilon)
        self.state_norm = RMSNorm(width, epsilon)
        self.state_projection = nn.Linear(width, width, bias=False)
        nn.init.eye_(self.state_projection.weight)
        self.cluster_anchors = nn.Parameter(torch.empty(cluster_count, width))
        nn.init.normal_(self.cluster_anchors, std=width**-0.5)

        cluster_size = math.ceil(vocab_size / cluster_count)
        cluster_tokens = torch.full(
            (cluster_count, cluster_size),
            -1,
            dtype=torch.long,
        )
        token_to_cluster = torch.empty(vocab_size, dtype=torch.long)
        for token_id in range(vocab_size):
            cluster_id = min(token_id // cluster_size, cluster_count - 1)
            local = token_id - cluster_id * cluster_size
            cluster_tokens[cluster_id, local] = token_id
            token_to_cluster[token_id] = cluster_id
        self.register_buffer("cluster_tokens", cluster_tokens)
        self.register_buffer("token_to_cluster", token_to_cluster)

    @property
    def maximum_candidates(self) -> int:
        return self.top_k_clusters * self.cluster_tokens.shape[1]

    def project_state(self, hidden: Tensor) -> Tensor:
        if hidden.shape[-1] != self.width:
            raise ValueError("hidden width differs from vocabulary head width")
        return F.normalize(self.state_projection(self.state_norm(hidden)), dim=-1)

    def dense_scores(self, hidden: Tensor, embedding_weight: Tensor) -> Tensor:
        """Full-vocabulary scores used only for unbiased validation NLL."""
        state = self.project_state(hidden)
        vocabulary = F.normalize(embedding_weight, dim=-1)
        return self.temperature * (2.0 * torch.einsum("...d,vd->...v", state, vocabulary) - 2.0)

    def forward(
        self,
        hidden: Tensor,
        embedding_weight: Tensor,
        *,
        target_ids: Tensor | None = None,
        include_target_cluster: bool = False,
    ) -> SparseVocabularyOutput:
        if hidden.shape[-1] != self.width:
            raise ValueError("hidden width differs from vocabulary head width")
        if embedding_weight.shape != (self.vocab_size, self.width):
            raise ValueError("embedding weight shape differs from vocabulary contract")
        if target_ids is not None and target_ids.shape != hidden.shape[:-1]:
            raise ValueError("target_ids must match hidden without the width axis")
        if include_target_cluster and target_ids is None:
            raise ValueError("include_target_cluster requires target_ids")

        leading_shape = hidden.shape[:-1]
        state = self.project_state(hidden)
        flat_state = state.reshape(-1, self.width)
        cluster_logits = self.temperature * (
            flat_state @ F.normalize(self.cluster_anchors, dim=-1).T
        )
        selected = cluster_logits.topk(self.top_k_clusters, dim=-1).indices

        if include_target_cluster:
            flat_targets = target_ids.reshape(-1)
            if flat_targets.numel() and (
                int(flat_targets.min()) < 0 or int(flat_targets.max()) >= self.vocab_size
            ):
                raise ValueError("target token is outside the vocabulary")
            target_clusters = self.token_to_cluster.index_select(0, flat_targets)
            present = (selected == target_clusters.unsqueeze(-1)).any(dim=-1)
            if bool((~present).any()):
                selected = selected.clone()
                selected[~present, -1] = target_clusters[~present]

        candidate_ids = self.cluster_tokens[selected].flatten(start_dim=-2)
        valid = candidate_ids >= 0
        safe_ids = candidate_ids.masked_fill(~valid, 0)
        candidate_embeddings = F.embedding(safe_ids, embedding_weight)
        candidate_embeddings = F.normalize(candidate_embeddings, dim=-1)
        cosine = torch.einsum("nd,ncd->nc", flat_state, candidate_embeddings)
        scores = self.temperature * (2.0 * cosine - 2.0)
        scores = scores.masked_fill(~valid, -torch.inf)

        return SparseVocabularyOutput(
            scores=scores.view(*leading_shape, candidate_ids.shape[-1]),
            candidate_token_ids=candidate_ids.view(
                *leading_shape,
                candidate_ids.shape[-1],
            ),
            cluster_logits=cluster_logits.view(*leading_shape, self.cluster_count),
            selected_clusters=selected.view(
                *leading_shape,
                self.top_k_clusters,
            ),
            valid_candidate_count=valid.sum(dim=-1).view(*leading_shape),
            vocabulary_size=self.vocab_size,
        )


def sparse_vocabulary_loss(
    output: SparseVocabularyOutput,
    targets: Tensor,
    *,
    cluster_loss_weight: float = 0.25,
    ignore_index: int = -100,
) -> SparseVocabularyLoss:
    if targets.shape != output.scores.shape[:-1]:
        raise ValueError("targets must match sparse scores without candidate axis")
    valid = targets != ignore_index
    safe_targets = targets.masked_fill(~valid, 0)
    matches = output.candidate_token_ids == safe_targets.unsqueeze(-1)
    if bool((~matches.any(dim=-1) & valid).any()):
        missing = int((~matches.any(dim=-1) & valid).sum())
        raise ValueError(
            f"{missing} targets are absent from candidate clusters; "
            "train with include_target_cluster=True"
        )
    local_targets = matches.to(torch.long).argmax(dim=-1)
    flat_scores = output.scores.reshape(-1, output.scores.shape[-1])
    flat_local = local_targets.reshape(-1)
    flat_valid = valid.reshape(-1)
    if bool(flat_valid.any()):
        candidate_loss = F.cross_entropy(
            flat_scores[flat_valid],
            flat_local[flat_valid],
        )
        target_clusters = (
            safe_targets.reshape(-1)[flat_valid]
            * output.cluster_logits.shape[-1]
            // output.vocabulary_size
        ).clamp(max=output.cluster_logits.shape[-1] - 1)
        # Use the exact registered contiguous cluster geometry. The arithmetic
        # above is equivalent to floor(token / ceil(V/C)) only when divisible,
        # so recover from candidate metadata when needed below.
        cluster_size = math.ceil(
            output.vocabulary_size / output.cluster_logits.shape[-1]
        )
        target_clusters = (
            safe_targets.reshape(-1)[flat_valid] // cluster_size
        ).clamp(max=output.cluster_logits.shape[-1] - 1)
        cluster_loss = F.cross_entropy(
            output.cluster_logits.reshape(-1, output.cluster_logits.shape[-1])[
                flat_valid
            ],
            target_clusters,
        )
    else:
        candidate_loss = output.scores.sum() * 0.0
        cluster_loss = output.cluster_logits.sum() * 0.0
    total = candidate_loss + float(cluster_loss_weight) * cluster_loss
    return SparseVocabularyLoss(total, candidate_loss, cluster_loss)


def sparse_candidate_recall(output: SparseVocabularyOutput, targets: Tensor, *, ignore_index: int=-100) -> float:
    valid=targets != ignore_index
    if not bool(valid.any()): return 0.0
    matches=(output.candidate_token_ids == targets.masked_fill(~valid,0).unsqueeze(-1)).any(dim=-1)
    return float(matches[valid].float().mean())
