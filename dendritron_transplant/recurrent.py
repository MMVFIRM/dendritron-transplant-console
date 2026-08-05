"""Two physical recurrent blocks reused for adaptive sparse CPU depth."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .config import DendritronRecipientConfig
from .memory import MemoryFusionStats, MemoryPayloads, SparseMemoryFusion
from .moe import DenseMLPControl, SparseDendritronMoE, SparseMLPMoE, SparseMoEStats
from .lngram import LNGramMemory, LNGramStats


class RMSNorm(nn.Module):
    def __init__(self, width: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.epsilon = float(epsilon)

    def forward(self, values: Tensor) -> Tensor:
        energy = values.square().mean(dim=-1, keepdim=True)
        return values * torch.rsqrt(energy + self.epsilon) * self.weight


class CausalDepthwiseMixer(nn.Module):
    """Cheap causal local context without dense Q/K/V attention."""

    def __init__(
        self,
        width: int,
        kernel_size: int,
        *,
        initialization_gain: float,
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.kernel_size = int(kernel_size)
        self.depthwise = nn.Conv1d(
            width,
            width,
            kernel_size=kernel_size,
            groups=width,
            padding=kernel_size - 1,
            bias=False,
        )
        self.channel_gate = nn.Linear(width, width, bias=False)
        with torch.no_grad():
            self.depthwise.weight.mul_(float(initialization_gain))
            self.channel_gate.weight.mul_(float(initialization_gain))

    def forward(self, hidden: Tensor) -> Tensor:
        sequence = hidden.transpose(1, 2)
        mixed = self.depthwise(sequence)[..., : hidden.shape[1]].transpose(1, 2)
        return F.silu(mixed) * torch.sigmoid(self.channel_gate(hidden))


@dataclass(frozen=True)
class RecurrentVisitStats:
    round_index: int
    block_index: int
    relative_change: Tensor
    memory: MemoryFusionStats
    lngram: LNGramStats | None
    moe: SparseMoEStats


@dataclass(frozen=True)
class SparseCoreStats:
    visits: tuple[RecurrentVisitStats, ...]
    rounds_executed: int
    final_relative_change: Tensor
    alpha: float
    beta: float
    active_conditional_parameters_per_token: int
    total_conditional_parameters: int

    @property
    def conditional_active_fraction(self) -> float:
        if self.total_conditional_parameters == 0:
            return 0.0
        return self.active_conditional_parameters_per_token / self.total_conditional_parameters


def _relative_change(current: Tensor, previous: Tensor, epsilon: float) -> Tensor:
    numerator = (current - previous).square().mean(dim=(-2, -1)).sqrt()
    denominator = previous.square().mean(dim=(-2, -1)).sqrt()
    return numerator / (denominator + float(epsilon))


class SparseRecurrentBlock(nn.Module):
    def __init__(self, config: DendritronRecipientConfig) -> None:
        super().__init__()
        self.context_norm = RMSNorm(config.model_width, config.residual_epsilon)
        self.compute_norm = RMSNorm(config.model_width, config.residual_epsilon)
        self.mixer = CausalDepthwiseMixer(
            config.model_width,
            config.causal_kernel_size,
            initialization_gain=config.deep_loop_beta,
        )
        common = dict(
            expert_count=config.expert_count,
            expert_top_k=config.expert_top_k,
            branches_per_expert=config.branches_per_expert,
            branch_top_k=config.branch_top_k,
            branch_hidden_width=config.branch_hidden_width,
            initialization_gain=config.deep_loop_beta,
            epsilon=config.route_epsilon,
        )
        if config.expert_kind == "dense":
            self.moe = DenseMLPControl(
                config.model_width,
                branch_hidden_width=config.branch_hidden_width,
                initialization_gain=config.deep_loop_beta,
            )
        elif config.expert_kind == "mlp":
            self.moe = SparseMLPMoE(config.model_width, **common)
        else:
            self.moe = SparseDendritronMoE(config.model_width, **common)
        self.lngram = (
            LNGramMemory(config.model_width, config.lngram,
                         initialization_gain=config.deep_loop_beta)
            if config.use_lngram else None
        )


class TwoBlockSparseCore(nn.Module):
    """Reuse two stored blocks over multiple rounds.

    Each visit performs causal mixing + memory injection, then a globally and
    locally sparse Dendritron MoE update. The same physical weights are reused
    on later rounds, so effective depth can grow without storing more blocks.
    """

    def __init__(
        self,
        config: DendritronRecipientConfig,
        *,
        memory_fusion: SparseMemoryFusion,
    ) -> None:
        super().__init__()
        self.config = config
        self.memory_fusion = memory_fusion
        self.alpha = config.deep_loop_alpha
        self.beta = config.deep_loop_beta
        self.blocks = nn.ModuleList([SparseRecurrentBlock(config) for _ in range(2)])

    @property
    def total_conditional_parameters(self) -> int:
        return sum(block.moe.total_conditional_parameters for block in self.blocks)

    @property
    def active_conditional_parameters_per_token(self) -> int:
        # One block is active at a time. Per visit, only top-k experts and
        # top-k branches inside those experts execute.
        return self.blocks[0].moe.active_conditional_parameters_per_position

    def forward(
        self,
        hidden: Tensor,
        *,
        payloads: MemoryPayloads | None,
        rounds: int | None = None,
        adaptive_threshold: float | None = None,
        minimum_rounds: int = 1,
        return_stats: bool = False,
    ) -> Tensor | tuple[Tensor, SparseCoreStats]:
        total_rounds = self.config.loop_rounds if rounds is None else int(rounds)
        if total_rounds < 1:
            raise ValueError("rounds must be positive")
        if not 1 <= minimum_rounds <= total_rounds:
            raise ValueError("minimum_rounds must be inside the round budget")
        threshold = (
            self.config.adaptive_threshold
            if adaptive_threshold is None
            else adaptive_threshold
        )
        if threshold is not None and threshold <= 0:
            raise ValueError("adaptive_threshold must be positive")

        visits: list[RecurrentVisitStats] = []
        final_change = hidden.new_full(hidden.shape[:1], float("inf"))
        rounds_executed = 0
        for round_index in range(total_rounds):
            for block_index, block in enumerate(self.blocks):
                previous = hidden
                context = block.mixer(block.context_norm(hidden))
                memory_update, memory_stats = self.memory_fusion(
                    hidden + context,
                    payloads,
                    block_index=block_index,
                    return_stats=True,
                )
                if block.lngram is None:
                    lngram_update = torch.zeros_like(hidden)
                    lngram_stats = None
                else:
                    lngram_update, lngram_stats = block.lngram(
                        hidden + context + memory_update,
                        return_stats=True,
                    )
                contracted = block.context_norm(
                    self.alpha * hidden + context + memory_update + lngram_update
                )
                moe_update, moe_stats = block.moe(contracted, return_stats=True)
                hidden = block.compute_norm(self.alpha * contracted + moe_update)
                final_change = _relative_change(
                    hidden,
                    previous,
                    self.config.residual_epsilon,
                )
                if return_stats:
                    visits.append(
                        RecurrentVisitStats(
                            round_index=round_index,
                            block_index=block_index,
                            relative_change=final_change,
                            memory=memory_stats,
                            lngram=lngram_stats,
                            moe=moe_stats,
                        )
                    )
            rounds_executed = round_index + 1
            if (
                threshold is not None
                and rounds_executed >= minimum_rounds
                and bool((final_change < threshold).all())
            ):
                break

        if return_stats:
            return hidden, SparseCoreStats(
                visits=tuple(visits),
                rounds_executed=rounds_executed,
                final_relative_change=final_change,
                alpha=self.alpha,
                beta=self.beta,
                active_conditional_parameters_per_token=(
                    self.active_conditional_parameters_per_token
                ),
                total_conditional_parameters=self.total_conditional_parameters,
            )
        return hidden
