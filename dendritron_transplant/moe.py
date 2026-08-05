"""Packed two-level sparse Dendritron MoE for CPU-friendly gathered execution."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class SparseMoEStats:
    selected_expert_indices: Tensor
    selected_expert_weights: Tensor
    selected_branch_indices: Tensor
    selected_branch_weights: Tensor
    executed_experts: tuple[int, ...]
    active_conditional_parameters_per_position: int
    total_conditional_parameters: int

    @property
    def active_parameter_fraction(self) -> float:
        if self.total_conditional_parameters == 0:
            return 0.0
        return (
            self.active_conditional_parameters_per_position
            / self.total_conditional_parameters
        )


class SparseDendritronMoE(nn.Module):
    """Top-k experts and top-k nonlinear branches using gathered parameter tensors.

    The implementation avoids Python dispatch over experts and branches. It
    gathers only the selected branch matrices and evaluates them in one packed
    tensor program. This keeps the semantic sparsity contract while making the
    reference path materially more suitable for CPU execution.
    """

    def __init__(
        self,
        width: int,
        *,
        expert_count: int,
        expert_top_k: int,
        branches_per_expert: int,
        branch_top_k: int,
        branch_hidden_width: int,
        initialization_gain: float = 1.0,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        if min(
            width,
            expert_count,
            expert_top_k,
            branches_per_expert,
            branch_top_k,
            branch_hidden_width,
        ) < 1:
            raise ValueError("sparse Dendritron dimensions must be positive")
        if expert_top_k > expert_count:
            raise ValueError("expert_top_k cannot exceed expert_count")
        if branch_top_k > branches_per_expert:
            raise ValueError("branch_top_k cannot exceed branches_per_expert")
        self.width = int(width)
        self.expert_count = int(expert_count)
        self.expert_top_k = int(expert_top_k)
        self.branches_per_expert = int(branches_per_expert)
        self.branch_top_k = int(branch_top_k)
        self.branch_hidden_width = int(branch_hidden_width)
        self.epsilon = float(epsilon)

        self.expert_scouts = nn.Parameter(torch.empty(expert_count, width))
        self.branch_scouts = nn.Parameter(
            torch.empty(expert_count, branches_per_expert, width)
        )
        shape_weight = (
            expert_count,
            branches_per_expert,
            branch_hidden_width,
            width,
        )
        shape_bias = (expert_count, branches_per_expert, branch_hidden_width)
        self.left_weight = nn.Parameter(torch.empty(shape_weight))
        self.left_bias = nn.Parameter(torch.zeros(shape_bias))
        self.right_weight = nn.Parameter(torch.empty(shape_weight))
        self.right_bias = nn.Parameter(torch.zeros(shape_bias))
        self.gate_weight = nn.Parameter(torch.empty(shape_weight))
        self.gate_bias = nn.Parameter(torch.zeros(shape_bias))
        self.output_weight = nn.Parameter(
            torch.empty(
                expert_count,
                branches_per_expert,
                width,
                branch_hidden_width,
            )
        )
        self.evidence_axis = nn.Parameter(
            torch.empty(expert_count, branches_per_expert, width)
        )
        self.output_gate = nn.Parameter(torch.tensor(0.1))

        nn.init.normal_(self.expert_scouts, std=width**-0.5)
        nn.init.normal_(self.branch_scouts, std=width**-0.5)
        nn.init.normal_(self.left_weight, std=width**-0.5)
        nn.init.normal_(self.right_weight, std=width**-0.5)
        nn.init.normal_(self.gate_weight, std=width**-0.5)
        nn.init.normal_(self.output_weight, std=branch_hidden_width**-0.5)
        nn.init.normal_(self.evidence_axis, std=width**-0.5)
        with torch.no_grad():
            for parameter in (
                self.left_weight,
                self.right_weight,
                self.gate_weight,
                self.output_weight,
            ):
                parameter.mul_(float(initialization_gain))

    @property
    def branch_transform_parameter_count(self) -> int:
        d = self.width
        h = self.branch_hidden_width
        return 3 * (h * d + h) + d * h + d

    @property
    def total_conditional_parameters(self) -> int:
        return (
            self.expert_scouts.numel()
            + self.branch_scouts.numel()
            + self.expert_count
            * self.branches_per_expert
            * self.branch_transform_parameter_count
            + self.output_gate.numel()
        )

    @property
    def active_conditional_parameters_per_position(self) -> int:
        return (
            self.expert_scouts.numel()
            + self.expert_top_k * self.branches_per_expert * self.width
            + self.expert_top_k
            * self.branch_top_k
            * self.branch_transform_parameter_count
            + self.output_gate.numel()
        )

    def forward(
        self,
        hidden: Tensor,
        *,
        return_stats: bool = False,
    ) -> Tensor | tuple[Tensor, SparseMoEStats]:
        if hidden.shape[-1] != self.width:
            raise ValueError(f"MoE expected width {self.width}")
        leading_shape = hidden.shape[:-1]
        flat = hidden.reshape(-1, self.width)
        normalized = F.normalize(flat, dim=-1)

        expert_scores = normalized @ F.normalize(self.expert_scouts, dim=-1).T
        selected_experts = expert_scores.topk(self.expert_top_k, dim=-1).indices
        selected_expert_scores = expert_scores.gather(-1, selected_experts)
        expert_weights = torch.softmax(selected_expert_scores, dim=-1)

        selected_scouts = F.normalize(self.branch_scouts, dim=-1)[selected_experts]
        branch_scores = torch.einsum("nd,nkbd->nkb", normalized, selected_scouts)
        selected_branches = branch_scores.abs().topk(
            self.branch_top_k,
            dim=-1,
        ).indices
        selected_branch_scores = branch_scores.gather(-1, selected_branches)
        branch_weights = torch.tanh(selected_branch_scores)
        branch_weights = branch_weights / (
            branch_weights.abs().sum(dim=-1, keepdim=True) + self.epsilon
        )

        expert_index = selected_experts.unsqueeze(-1).expand(
            -1,
            -1,
            self.branch_top_k,
        )
        branch_index = selected_branches

        left_weight = self.left_weight[expert_index, branch_index]
        right_weight = self.right_weight[expert_index, branch_index]
        gate_weight = self.gate_weight[expert_index, branch_index]
        output_weight = self.output_weight[expert_index, branch_index]
        left_bias = self.left_bias[expert_index, branch_index]
        right_bias = self.right_bias[expert_index, branch_index]
        gate_bias = self.gate_bias[expert_index, branch_index]
        evidence_axis = self.evidence_axis[expert_index, branch_index]

        left = torch.tanh(
            torch.einsum("nkjhd,nd->nkjh", left_weight, flat) + left_bias
        )
        right = torch.tanh(
            torch.einsum("nkjhd,nd->nkjh", right_weight, flat) + right_bias
        )
        gate = torch.sigmoid(
            torch.einsum("nkjhd,nd->nkjh", gate_weight, flat) + gate_bias
        )
        interaction = left * right * gate
        branch_output = torch.einsum(
            "nkjdh,nkjh->nkjd",
            output_weight,
            interaction,
        )
        evidence = (
            F.normalize(branch_output, dim=-1)
            * F.normalize(evidence_axis, dim=-1)
        ).sum(dim=-1)
        branch_output = branch_output * torch.sigmoid(evidence).unsqueeze(-1)
        expert_output = (
            branch_weights.unsqueeze(-1) * branch_output
        ).sum(dim=-2)
        output = (expert_weights.unsqueeze(-1) * expert_output).sum(dim=-2)
        output = torch.tanh(self.output_gate) * output
        output = output.view(*leading_shape, self.width)

        if return_stats:
            return output, SparseMoEStats(
                selected_expert_indices=selected_experts.view(
                    *leading_shape,
                    self.expert_top_k,
                ),
                selected_expert_weights=expert_weights.view(
                    *leading_shape,
                    self.expert_top_k,
                ),
                selected_branch_indices=selected_branches.view(
                    *leading_shape,
                    self.expert_top_k,
                    self.branch_top_k,
                ),
                selected_branch_weights=branch_weights.view(
                    *leading_shape,
                    self.expert_top_k,
                    self.branch_top_k,
                ),
                executed_experts=tuple(
                    int(value) for value in torch.unique(selected_experts).tolist()
                ),
                active_conditional_parameters_per_position=(
                    self.active_conditional_parameters_per_position
                ),
                total_conditional_parameters=self.total_conditional_parameters,
            )
        return output


class DenseMLPControl(nn.Module):
    """Always-active gated MLP control with a comparable active compute budget."""
    def __init__(self, width: int, *, branch_hidden_width: int, initialization_gain: float=1.0) -> None:
        super().__init__()
        hidden=max(width,4*branch_hidden_width)
        self.width=width
        self.up=nn.Linear(width,hidden,bias=False)
        self.gate=nn.Linear(width,hidden,bias=False)
        self.down=nn.Linear(hidden,width,bias=False)
        self.output_gate=nn.Parameter(torch.tensor(0.1))
        with torch.no_grad():
            for module in (self.up,self.gate,self.down): module.weight.mul_(initialization_gain)

    @property
    def total_conditional_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
    @property
    def active_conditional_parameters_per_position(self) -> int:
        return self.total_conditional_parameters

    def forward(self, hidden: Tensor, *, return_stats: bool=False):
        update=self.down(F.silu(self.gate(hidden))*self.up(hidden))*torch.tanh(self.output_gate)
        if return_stats:
            leading=hidden.shape[:-1]
            empty_i=torch.empty(*leading,0,dtype=torch.long,device=hidden.device)
            empty_w=hidden.new_zeros(*leading,0)
            return update,SparseMoEStats(empty_i,empty_w,empty_i.view(*leading,0,0),
                empty_w.view(*leading,0,0),(),self.total_conditional_parameters,
                self.total_conditional_parameters)
        return update


class SparseMLPMoE(nn.Module):
    """Parameter-matched top-k MLP branch control for the Dendritron MoE."""
    def __init__(self,width:int,*,expert_count:int,expert_top_k:int,
                 branches_per_expert:int,branch_top_k:int,branch_hidden_width:int,
                 initialization_gain:float=1.0,epsilon:float=1e-6) -> None:
        super().__init__()
        self.width=width; self.expert_count=expert_count; self.expert_top_k=expert_top_k
        self.branches_per_expert=branches_per_expert; self.branch_top_k=branch_top_k
        self.hidden_width=max(1,round(branch_hidden_width*4/3)); self.epsilon=epsilon
        self.expert_scouts=nn.Parameter(torch.empty(expert_count,width))
        self.branch_scouts=nn.Parameter(torch.empty(expert_count,branches_per_expert,width))
        shape=(expert_count,branches_per_expert,self.hidden_width,width)
        bias=(expert_count,branches_per_expert,self.hidden_width)
        self.content_weight=nn.Parameter(torch.empty(shape)); self.content_bias=nn.Parameter(torch.zeros(bias))
        self.gate_weight=nn.Parameter(torch.empty(shape)); self.gate_bias=nn.Parameter(torch.zeros(bias))
        self.output_weight=nn.Parameter(torch.empty(expert_count,branches_per_expert,width,self.hidden_width))
        self.evidence_axis=nn.Parameter(torch.empty(expert_count,branches_per_expert,width))
        self.output_gate=nn.Parameter(torch.tensor(0.1))
        for parameter,std in ((self.expert_scouts,width**-0.5),(self.branch_scouts,width**-0.5),
                              (self.content_weight,width**-0.5),(self.gate_weight,width**-0.5),
                              (self.output_weight,self.hidden_width**-0.5),(self.evidence_axis,width**-0.5)):
            nn.init.normal_(parameter,std=std)
        with torch.no_grad():
            for parameter in (self.content_weight,self.gate_weight,self.output_weight):
                parameter.mul_(initialization_gain)

    @property
    def branch_transform_parameter_count(self)->int:
        d=self.width; h=self.hidden_width
        return 2*(h*d+h)+d*h+d
    @property
    def total_conditional_parameters(self)->int:
        return self.expert_scouts.numel()+self.branch_scouts.numel()+self.expert_count*self.branches_per_expert*self.branch_transform_parameter_count+1
    @property
    def active_conditional_parameters_per_position(self)->int:
        return self.expert_scouts.numel()+self.expert_top_k*self.branches_per_expert*self.width+self.expert_top_k*self.branch_top_k*self.branch_transform_parameter_count+1

    def forward(self,hidden:Tensor,*,return_stats:bool=False):
        leading=hidden.shape[:-1]; flat=hidden.reshape(-1,self.width); normalized=F.normalize(flat,dim=-1)
        expert_scores=normalized@F.normalize(self.expert_scouts,dim=-1).T
        experts=expert_scores.topk(self.expert_top_k,dim=-1).indices
        expert_weights=torch.softmax(expert_scores.gather(-1,experts),dim=-1)
        scouts=F.normalize(self.branch_scouts,dim=-1)[experts]
        branch_scores=torch.einsum('nd,nkbd->nkb',normalized,scouts)
        branches=branch_scores.abs().topk(self.branch_top_k,dim=-1).indices
        selected_scores=branch_scores.gather(-1,branches)
        branch_weights=torch.tanh(selected_scores)
        branch_weights=branch_weights/(branch_weights.abs().sum(dim=-1,keepdim=True)+self.epsilon)
        e=experts.unsqueeze(-1).expand(-1,-1,self.branch_top_k); b=branches
        content=torch.tanh(torch.einsum('nkjhd,nd->nkjh',self.content_weight[e,b],flat)+self.content_bias[e,b])
        gate=torch.sigmoid(torch.einsum('nkjhd,nd->nkjh',self.gate_weight[e,b],flat)+self.gate_bias[e,b])
        branch_output=torch.einsum('nkjdh,nkjh->nkjd',self.output_weight[e,b],content*gate)
        evidence=(F.normalize(branch_output,dim=-1)*F.normalize(self.evidence_axis[e,b],dim=-1)).sum(dim=-1)
        branch_output=branch_output*torch.sigmoid(evidence).unsqueeze(-1)
        expert_output=(branch_weights.unsqueeze(-1)*branch_output).sum(dim=-2)
        output=(expert_weights.unsqueeze(-1)*expert_output).sum(dim=-2)
        output=(torch.tanh(self.output_gate)*output).view(*leading,self.width)
        if return_stats:
            return output,SparseMoEStats(
                experts.view(*leading,self.expert_top_k),expert_weights.view(*leading,self.expert_top_k),
                branches.view(*leading,self.expert_top_k,self.branch_top_k),
                branch_weights.view(*leading,self.expert_top_k,self.branch_top_k),
                tuple(int(v) for v in torch.unique(experts).tolist()),
                self.active_conditional_parameters_per_position,self.total_conditional_parameters)
        return output
