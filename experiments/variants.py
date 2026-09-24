"""Architecture variants applied by name before model construction.

Each run is its own process, so patches are applied globally. The baseline
(no patches) is bit-for-bit the packaged architecture.

Patch names
-----------
MoE (replace SparseDendritronMoE with ExpMoE):
  router_temp   learnable score scale for expert/branch routers (init 5.0)
  branch_softmax  branch weights = softmax over selected scores (not signed tanh)
  full_gain     left/right/gate input matrices initialised at unit gain
  glu           interaction = silu(left) * right * 2*sigmoid(gate)
  mlp_branch    parameter-matched MLP branch: tanh(content) * sigmoid(gate)
  no_evidence   drop evidence-axis magnitude modulation
  balance       switch-style expert load-balancing auxiliary loss
  gate1         MoE output gate initialised at 1.0 instead of 0.1
  no_moe        zero the MoE update (ablation: does the expert path matter?)
Core:
  prenorm       additive pre-norm residual stream instead of scaled post-norm
  attn          (with prenorm) add a 4-head causal attention sublayer per block
  dropP         (with prenorm) dropout with probability P on every residual update, e.g. drop0.2
  layerscale    (with prenorm) learnable per-channel scale on each residual update
Head:
  dense_loss    train with full-vocabulary cross-entropy (inference stays sparse)
  learn_temp    learnable output temperature
"""

from __future__ import annotations

import contextlib
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from dendritron_transplant import model as model_module
from dendritron_transplant import recurrent as recurrent_module
from dendritron_transplant.moe import SparseMoEStats
from dendritron_transplant.output import ClusteredVocabularyHead

ACTIVE: set[str] = set()
MOE_PATCHES = {"router_temp", "branch_softmax", "full_gain", "glu", "mlp_branch",
               "no_evidence", "balance", "gate1", "no_moe"}
BALANCE_COEF = 0.01


class ExpMoE(nn.Module):
    """Re-implementation of SparseDendritronMoE with switchable components.

    With no active flags its forward pass matches SparseDendritronMoE exactly.
    """

    def __init__(self, width: int, *, expert_count: int, expert_top_k: int,
                 branches_per_expert: int, branch_top_k: int, branch_hidden_width: int,
                 initialization_gain: float = 1.0, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.flags = set(ACTIVE)
        self.width = width
        self.expert_count, self.expert_top_k = expert_count, expert_top_k
        self.branches_per_expert, self.branch_top_k = branches_per_expert, branch_top_k
        self.mlp = "mlp_branch" in self.flags
        h = max(1, round(branch_hidden_width * 4 / 3)) if self.mlp else branch_hidden_width
        self.branch_hidden_width = h
        self.epsilon = epsilon
        E, B = expert_count, branches_per_expert
        self.expert_scouts = nn.Parameter(torch.randn(E, width) * width**-0.5)
        self.branch_scouts = nn.Parameter(torch.randn(E, B, width) * width**-0.5)
        in_gain = 1.0 if "full_gain" in self.flags else initialization_gain

        def mat(*shape, std, gain):
            return nn.Parameter(torch.randn(*shape) * std * gain)

        self.left_weight = mat(E, B, h, width, std=width**-0.5, gain=in_gain)
        self.left_bias = nn.Parameter(torch.zeros(E, B, h))
        if not self.mlp:
            self.right_weight = mat(E, B, h, width, std=width**-0.5, gain=in_gain)
            self.right_bias = nn.Parameter(torch.zeros(E, B, h))
        self.gate_weight = mat(E, B, h, width, std=width**-0.5, gain=in_gain)
        self.gate_bias = nn.Parameter(torch.zeros(E, B, h))
        self.output_weight = mat(E, B, width, h, std=h**-0.5, gain=initialization_gain)
        self.evidence_axis = nn.Parameter(torch.randn(E, B, width) * width**-0.5)
        self.output_gate = nn.Parameter(torch.tensor(1.0 if "gate1" in self.flags else 0.1))
        if "router_temp" in self.flags:
            self.log_router_scale = nn.Parameter(torch.tensor(math.log(5.0)))
        self._aux: Tensor | None = None

    @property
    def _stored_branch(self) -> int:
        d, h = self.width, self.branch_hidden_width
        mats = 2 if self.mlp else 3
        return mats * (h * d + h) + d * h + d

    @property
    def total_conditional_parameters(self) -> int:
        return (self.expert_scouts.numel() + self.branch_scouts.numel()
                + self.expert_count * self.branches_per_expert * self._stored_branch + 1)

    @property
    def active_conditional_parameters_per_position(self) -> int:
        return (self.expert_scouts.numel()
                + self.expert_top_k * self.branches_per_expert * self.width
                + self.expert_top_k * self.branch_top_k * self._stored_branch + 1)

    def forward(self, hidden: Tensor, *, return_stats: bool = False):
        f = self.flags
        leading = hidden.shape[:-1]
        flat = hidden.reshape(-1, self.width)
        normalized = F.normalize(flat, dim=-1)
        scale = self.log_router_scale.exp() if "router_temp" in f else 1.0

        expert_scores = normalized @ F.normalize(self.expert_scouts, dim=-1).T
        experts = expert_scores.topk(self.expert_top_k, dim=-1).indices
        expert_weights = torch.softmax(scale * expert_scores.gather(-1, experts), dim=-1)
        if "balance" in f and self.training:
            probs = torch.softmax(scale * expert_scores, dim=-1).mean(0)
            load = F.one_hot(experts, self.expert_count).float().sum(1).mean(0) / self.expert_top_k
            self._aux = self.expert_count * (probs * load).sum()

        scouts = F.normalize(self.branch_scouts, dim=-1)[experts]
        branch_scores = torch.einsum("nd,nkbd->nkb", normalized, scouts)
        if "branch_softmax" in f:
            branches = branch_scores.topk(self.branch_top_k, dim=-1).indices
            branch_weights = torch.softmax(scale * branch_scores.gather(-1, branches), dim=-1)
        else:
            branches = branch_scores.abs().topk(self.branch_top_k, dim=-1).indices
            branch_weights = torch.tanh(scale * branch_scores.gather(-1, branches))
            branch_weights = branch_weights / (
                branch_weights.abs().sum(dim=-1, keepdim=True) + self.epsilon)

        e = experts.unsqueeze(-1).expand(-1, -1, self.branch_top_k)
        b = branches

        def pre(weight, bias):
            return torch.einsum("nkjhd,nd->nkjh", weight[e, b], flat) + bias[e, b]

        gate = torch.sigmoid(pre(self.gate_weight, self.gate_bias))
        if self.mlp:
            interaction = torch.tanh(pre(self.left_weight, self.left_bias)) * gate
        elif "glu" in f:
            interaction = (F.silu(pre(self.left_weight, self.left_bias))
                           * pre(self.right_weight, self.right_bias) * 2.0 * gate)
        else:
            interaction = (torch.tanh(pre(self.left_weight, self.left_bias))
                           * torch.tanh(pre(self.right_weight, self.right_bias)) * gate)
        branch_output = torch.einsum("nkjdh,nkjh->nkjd", self.output_weight[e, b], interaction)
        if "no_evidence" not in f:
            evidence = (F.normalize(branch_output, dim=-1)
                        * F.normalize(self.evidence_axis[e, b], dim=-1)).sum(-1)
            branch_output = branch_output * torch.sigmoid(evidence).unsqueeze(-1)
        expert_output = (branch_weights.unsqueeze(-1) * branch_output).sum(-2)
        output = (expert_weights.unsqueeze(-1) * expert_output).sum(-2)
        output = (torch.tanh(self.output_gate) * output).view(*leading, self.width)
        if "no_moe" in f:
            output = output * 0.0
        if return_stats:
            return output, SparseMoEStats(
                experts.view(*leading, self.expert_top_k),
                expert_weights.view(*leading, self.expert_top_k),
                branches.view(*leading, self.expert_top_k, self.branch_top_k),
                branch_weights.view(*leading, self.expert_top_k, self.branch_top_k),
                tuple(int(v) for v in torch.unique(experts).tolist()),
                self.active_conditional_parameters_per_position,
                self.total_conditional_parameters)
        return output


class CausalAttention(nn.Module):
    """Small multi-head causal softmax attention (added only by the attn patch)."""

    def __init__(self, width: int, heads: int = 4) -> None:
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.out = nn.Linear(width, width, bias=False)
        nn.init.zeros_(self.out.weight)

    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(y.transpose(1, 2).reshape(B, T, D))


DROPOUT = 0.0


def _drop(update: Tensor, training: bool) -> Tensor:
    return F.dropout(update, DROPOUT, training) if DROPOUT else update


def _scaled(block, name: str, update: Tensor) -> Tensor:
    scale = getattr(block, f"ls_{name}", None)
    return update if scale is None else update * scale


def _prenorm_forward(self, hidden, *, payloads, rounds=None, adaptive_threshold=None,
                     minimum_rounds=1, return_stats=False):
    """Additive residual stream: h += mix+memory+lngram; h += moe(norm(h))."""
    total_rounds = self.config.loop_rounds if rounds is None else int(rounds)
    for _ in range(total_rounds):
        for block_index, block in enumerate(self.blocks):
            normed = block.context_norm(hidden)
            context = block.mixer(normed)
            memory_update = self.memory_fusion(normed + context, payloads,
                                               block_index=block_index)
            update = context + memory_update
            if block.lngram is not None:
                update = update + block.lngram(normed + update)
            hidden = hidden + _drop(_scaled(block, "ctx", update), self.training)
            if hasattr(block, "attn"):
                hidden = hidden + block.attn(block.attn_norm(hidden))
            hidden = hidden + _drop(_scaled(block, "moe", block.moe(block.compute_norm(hidden))),
                                    self.training)
    if return_stats:
        raise NotImplementedError("prenorm variant does not collect stats")
    return hidden


class LearnedTempHead(ClusteredVocabularyHead):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.log_temperature = nn.Parameter(torch.tensor(math.log(self._init_temperature)))

    @property
    def temperature(self):
        if "log_temperature" in self._parameters:
            return self.log_temperature.exp()
        return self._init_temperature

    @temperature.setter
    def temperature(self, value) -> None:
        self._init_temperature = float(value)


@contextlib.contextmanager
def applied(patches: list[str]):
    global DROPOUT
    drops = [p for p in patches if p.startswith("drop")]
    DROPOUT = float(drops[0][4:]) if drops else 0.0
    patches = [p for p in patches if not p.startswith("drop")]
    unknown = set(patches) - MOE_PATCHES - {"prenorm", "learn_temp", "attn", "layerscale", "dense_loss"}
    if unknown:
        raise ValueError(f"unknown patches: {sorted(unknown)}")
    ACTIVE.clear()
    ACTIVE.update(patches)
    if ACTIVE & MOE_PATCHES:
        recurrent_module.SparseDendritronMoE = ExpMoE
    if "prenorm" in ACTIVE:
        recurrent_module.TwoBlockSparseCore.forward = _prenorm_forward
    if "learn_temp" in ACTIVE:
        model_module.ClusteredVocabularyHead = LearnedTempHead
    yield


def post_init(model, patches: list[str]) -> None:
    from dendritron_transplant.recurrent import RMSNorm
    width = model.config.model_width
    for block in model.core.blocks:
        if "attn" in patches:
            block.attn_norm = RMSNorm(width)
            block.attn = CausalAttention(width)
        if "layerscale" in patches:
            block.ls_ctx = nn.Parameter(torch.ones(width))
            block.ls_moe = nn.Parameter(torch.ones(width))


def aux_loss(model) -> Tensor | float:
    total = 0.0
    for module in model.modules():
        if isinstance(module, ExpMoE) and module._aux is not None:
            total = total + BALANCE_COEF * module._aux
            module._aux = None
    return total
