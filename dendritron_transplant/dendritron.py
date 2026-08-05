"""Sparse Dendritron cells with nonlinear branch interactions and local routing."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class DendritronCellStats:
    selected_branch_indices: Tensor
    selected_branch_weights: Tensor
    executed_branches: tuple[int, ...]
    router_parameter_count: int
    active_branch_parameter_count_per_position: int
    total_branch_parameter_count: int

    @property
    def active_parameter_fraction(self) -> float:
        denominator = self.router_parameter_count + self.total_branch_parameter_count
        numerator = self.router_parameter_count + self.active_branch_parameter_count_per_position
        return float(numerator / denominator) if denominator else 0.0


class DendriticBranch(nn.Module):
    """One nonlinear conditional branch.

    Two independently learned subconditions are multiplied before output
    integration. This gives each branch an internal conditional interaction
    rather than reducing it to a single affine perceptron path.
    """

    def __init__(
        self,
        width: int,
        hidden_width: int,
        *,
        initialization_gain: float = 1.0,
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.hidden_width = int(hidden_width)
        self.left = nn.Linear(width, hidden_width)
        self.right = nn.Linear(width, hidden_width)
        self.gate = nn.Linear(width, hidden_width)
        self.output = nn.Linear(hidden_width, width, bias=False)
        self.evidence_axis = nn.Parameter(torch.empty(width))
        nn.init.normal_(self.evidence_axis, std=width**-0.5)
        with torch.no_grad():
            self.left.weight.mul_(float(initialization_gain))
            self.right.weight.mul_(float(initialization_gain))
            self.gate.weight.mul_(float(initialization_gain))
            self.output.weight.mul_(float(initialization_gain))

    def forward(self, values: Tensor) -> tuple[Tensor, Tensor]:
        left = torch.tanh(self.left(values))
        right = torch.tanh(self.right(values))
        gate = torch.sigmoid(self.gate(values))
        interaction = left * right * gate
        output = self.output(interaction)
        evidence = (
            F.normalize(output, dim=-1)
            * F.normalize(self.evidence_axis, dim=-1)
        ).sum(dim=-1)
        return output, evidence

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class DendritronCell(nn.Module):
    """A locally routed multi-branch Dendritron primitive.

    Cheap branch scouts choose a small subset. Only selected branches execute
    their nonlinear conditional transforms, so inactive branches receive no
    branch-transform gradient for that position.
    """

    def __init__(
        self,
        width: int,
        *,
        branch_count: int,
        branch_top_k: int,
        hidden_width: int,
        initialization_gain: float = 1.0,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        if min(width, branch_count, branch_top_k, hidden_width) < 1:
            raise ValueError("Dendritron dimensions must be positive")
        if branch_top_k > branch_count:
            raise ValueError("branch_top_k cannot exceed branch_count")
        self.width = int(width)
        self.branch_count = int(branch_count)
        self.branch_top_k = int(branch_top_k)
        self.hidden_width = int(hidden_width)
        self.epsilon = float(epsilon)
        self.branch_scouts = nn.Parameter(torch.empty(branch_count, width))
        nn.init.normal_(self.branch_scouts, std=width**-0.5)
        self.branches = nn.ModuleList(
            [
                DendriticBranch(
                    width,
                    hidden_width,
                    initialization_gain=initialization_gain,
                )
                for _ in range(branch_count)
            ]
        )
        self.output_gate = nn.Parameter(torch.tensor(0.1))

    @property
    def branch_parameter_count(self) -> int:
        return self.branches[0].parameter_count

    @property
    def router_parameter_count(self) -> int:
        return self.branch_scouts.numel() + self.output_gate.numel()

    @property
    def total_branch_parameter_count(self) -> int:
        return sum(branch.parameter_count for branch in self.branches)

    @property
    def active_parameter_count_per_position(self) -> int:
        return self.router_parameter_count + self.branch_top_k * self.branch_parameter_count

    @property
    def total_parameter_count(self) -> int:
        return self.router_parameter_count + self.total_branch_parameter_count

    def forward(
        self,
        values: Tensor,
        *,
        return_stats: bool = False,
    ) -> Tensor | tuple[Tensor, DendritronCellStats]:
        if values.shape[-1] != self.width:
            raise ValueError(
                f"Dendritron expected width {self.width}, found {values.shape[-1]}"
            )
        leading_shape = values.shape[:-1]
        flat = values.reshape(-1, self.width)
        if flat.shape[0] == 0:
            empty = torch.zeros_like(values)
            if return_stats:
                indices = torch.empty(
                    *leading_shape,
                    self.branch_top_k,
                    dtype=torch.long,
                    device=values.device,
                )
                weights = values.new_zeros(*leading_shape, self.branch_top_k)
                return empty, DendritronCellStats(
                    selected_branch_indices=indices,
                    selected_branch_weights=weights,
                    executed_branches=(),
                    router_parameter_count=self.router_parameter_count,
                    active_branch_parameter_count_per_position=(
                        self.branch_top_k * self.branch_parameter_count
                    ),
                    total_branch_parameter_count=self.total_branch_parameter_count,
                )
            return empty

        normalized_values = F.normalize(flat, dim=-1)
        normalized_scouts = F.normalize(self.branch_scouts, dim=-1)
        scores = normalized_values @ normalized_scouts.T
        selected_indices = scores.abs().topk(
            self.branch_top_k,
            dim=-1,
        ).indices
        selected_scores = scores.gather(-1, selected_indices)
        selected_weights = torch.tanh(selected_scores)
        selected_weights = selected_weights / (
            selected_weights.abs().sum(dim=-1, keepdim=True) + self.epsilon
        )
        dense_weights = torch.zeros_like(scores).scatter(
            -1,
            selected_indices,
            selected_weights,
        )

        output = torch.zeros_like(flat)
        executed = tuple(int(value) for value in torch.unique(selected_indices).tolist())
        for branch_id in executed:
            weights = dense_weights[:, branch_id]
            active = weights != 0
            if not bool(active.any()):
                continue
            positions = active.nonzero(as_tuple=False).flatten()
            branch_input = flat.index_select(0, positions)
            branch_output, evidence = self.branches[branch_id](branch_input)
            # Branch evidence modulates magnitude but does not create a dense
            # cross-branch normalization step.
            evidence_scale = torch.sigmoid(evidence).unsqueeze(-1)
            weighted = (
                branch_output
                * evidence_scale
                * weights.index_select(0, positions).unsqueeze(-1)
            )
            output = output.index_add(0, positions, weighted)
        output = torch.tanh(self.output_gate) * output
        reshaped = output.view(*leading_shape, self.width)
        if return_stats:
            return reshaped, DendritronCellStats(
                selected_branch_indices=selected_indices.view(
                    *leading_shape,
                    self.branch_top_k,
                ),
                selected_branch_weights=selected_weights.view(
                    *leading_shape,
                    self.branch_top_k,
                ),
                executed_branches=executed,
                router_parameter_count=self.router_parameter_count,
                active_branch_parameter_count_per_position=(
                    self.branch_top_k * self.branch_parameter_count
                ),
                total_branch_parameter_count=self.total_branch_parameter_count,
            )
        return reshaped
