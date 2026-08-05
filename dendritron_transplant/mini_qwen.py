"""Small Qwen2-mechanics donor for the offline public Gate-2 precursor.

The execution host cannot fetch an official checkpoint. This donor preserves
RMSNorm, RoPE, grouped-query causal attention, SwiGLU, pre-norm residuals, and
tied embeddings so the transplant machinery can be tested end to end.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class MiniQwenConfig:
    vocab_size: int
    width: int = 96
    layer_count: int = 6
    attention_heads: int = 4
    key_value_heads: int = 2
    intermediate_width: int = 256
    max_sequence_length: int = 64
    rope_theta: float = 1_000_000.0
    norm_epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if min(self.vocab_size, self.width, self.layer_count, self.attention_heads,
               self.key_value_heads, self.intermediate_width,
               self.max_sequence_length) < 1:
            raise ValueError("MiniQwen dimensions must be positive")
        if self.width % self.attention_heads:
            raise ValueError("width must be divisible by attention_heads")
        if self.attention_heads % self.key_value_heads:
            raise ValueError("attention_heads must be divisible by key_value_heads")

    @property
    def head_width(self) -> int:
        return self.width // self.attention_heads

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> "MiniQwenConfig":
        return cls(**record)


class RMSNorm(nn.Module):
    def __init__(self, width: int, epsilon: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.epsilon = float(epsilon)

    def forward(self, values: Tensor) -> Tensor:
        energy = values.float().square().mean(dim=-1, keepdim=True)
        return values * torch.rsqrt(energy.to(values.dtype) + self.epsilon) * self.weight


def _rotate_half(values: Tensor) -> Tensor:
    half = values.shape[-1] // 2
    return torch.cat((-values[..., half:], values[..., :half]), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_width: int, maximum_length: int, theta: float) -> None:
        super().__init__()
        if head_width % 2:
            raise ValueError("RoPE head width must be even")
        inverse = 1.0 / (float(theta) ** (
            torch.arange(0, head_width, 2, dtype=torch.float32) / head_width
        ))
        positions = torch.arange(maximum_length, dtype=torch.float32)
        frequencies = torch.outer(positions, inverse)
        angles = torch.cat((frequencies, frequencies), dim=-1)
        self.register_buffer("cosine", angles.cos(), persistent=True)
        self.register_buffer("sine", angles.sin(), persistent=True)

    def forward(self, query: Tensor, key: Tensor) -> tuple[Tensor, Tensor]:
        length = query.shape[-2]
        cosine = self.cosine[:length].to(query).view(1, 1, length, -1)
        sine = self.sine[:length].to(query).view(1, 1, length, -1)
        return (
            query * cosine + _rotate_half(query) * sine,
            key * cosine + _rotate_half(key) * sine,
        )


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: MiniQwenConfig) -> None:
        super().__init__()
        self.width = config.width
        self.heads = config.attention_heads
        self.kv_heads = config.key_value_heads
        self.head_width = config.head_width
        self.q = nn.Linear(config.width, self.heads * self.head_width, bias=True)
        self.k = nn.Linear(config.width, self.kv_heads * self.head_width, bias=True)
        self.v = nn.Linear(config.width, self.kv_heads * self.head_width, bias=True)
        self.o = nn.Linear(config.width, config.width, bias=False)
        self.rope = RotaryEmbedding(
            self.head_width, config.max_sequence_length, config.rope_theta
        )

    def forward(self, hidden: Tensor) -> Tensor:
        batch, length, _ = hidden.shape
        query = self.q(hidden).view(batch, length, self.heads, self.head_width).transpose(1, 2)
        key = self.k(hidden).view(batch, length, self.kv_heads, self.head_width).transpose(1, 2)
        value = self.v(hidden).view(batch, length, self.kv_heads, self.head_width).transpose(1, 2)
        query, key = self.rope(query, key)
        if self.kv_heads != self.heads:
            repeat = self.heads // self.kv_heads
            key = key.repeat_interleave(repeat, dim=1)
            value = value.repeat_interleave(repeat, dim=1)
        attended = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=True
        )
        return self.o(attended.transpose(1, 2).contiguous().view(batch, length, self.width))


class SwiGLU(nn.Module):
    def __init__(self, config: MiniQwenConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(config.width, config.intermediate_width, bias=False)
        self.up = nn.Linear(config.width, config.intermediate_width, bias=False)
        self.down = nn.Linear(config.intermediate_width, config.width, bias=False)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(hidden)) * self.up(hidden))


class MiniQwenBlock(nn.Module):
    def __init__(self, config: MiniQwenConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.width, config.norm_epsilon)
        self.attention = GroupedQueryAttention(config)
        self.mlp_norm = RMSNorm(config.width, config.norm_epsilon)
        self.mlp = SwiGLU(config)

    def forward(self, hidden: Tensor) -> Tensor:
        hidden = hidden + self.attention(self.attention_norm(hidden))
        return hidden + self.mlp(self.mlp_norm(hidden))


@dataclass(frozen=True)
class MiniQwenOutput:
    logits: Tensor | None
    hidden: Tensor
    hidden_states: tuple[Tensor, ...]


class MiniQwenLM(nn.Module):
    def __init__(self, config: MiniQwenConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding(config.vocab_size, config.width)
        self.blocks = nn.ModuleList([MiniQwenBlock(config) for _ in range(config.layer_count)])
        self.final_norm = RMSNorm(config.width, config.norm_epsilon)
        self.output = nn.Linear(config.width, config.vocab_size, bias=False)
        self.output.weight = self.token_embeddings.weight
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: Tensor, *, output_hidden_states: bool = True,
                compute_logits: bool = True) -> MiniQwenOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be [B,T]")
        if input_ids.shape[1] > self.config.max_sequence_length:
            raise ValueError("input exceeds MiniQwen maximum sequence length")
        hidden = self.token_embeddings(input_ids)
        states: list[Tensor] = [hidden] if output_hidden_states else []
        for block in self.blocks:
            hidden = block(hidden)
            if output_hidden_states:
                states.append(hidden)
        final = self.final_norm(hidden)
        return MiniQwenOutput(
            logits=self.output(final) if compute_logits else None,
            hidden=final,
            hidden_states=tuple(states),
        )

    def parameter_report(self) -> dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        return {"stored_parameters": int(total), "active_parameters_per_token": int(total)}

    def save_checkpoint(self, path: str | Path, *, metadata: dict[str, Any] | None = None) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self.state_dict(), "config": self.config.to_dict(),
                    "metadata": metadata or {}}, destination)

    @classmethod
    def from_checkpoint(cls, path: str | Path, *, map_location: str | torch.device = "cpu") -> tuple["MiniQwenLM", dict[str, Any]]:
        record = torch.load(path, map_location=map_location, weights_only=True)
        model = cls(MiniQwenConfig.from_dict(record["config"])).to(map_location)
        model.load_state_dict(record["model"])
        return model, record


def next_token_loss(output: MiniQwenOutput, targets: Tensor) -> Tensor:
    if output.logits is None:
        raise ValueError("MiniQwen output does not contain logits")
    return F.cross_entropy(output.logits.reshape(-1, output.logits.shape[-1]), targets.reshape(-1))


@dataclass(frozen=True)
class ExtractedDonorStates:
    layer2: Tensor
    early: Tensor
    late: Tensor


@torch.no_grad()
def extract_final_states(
    donor: MiniQwenLM,
    sequences: list[tuple[int, ...]] | tuple[tuple[int, ...], ...],
    *,
    layer2_index: int = 2,
    early_layer_index: int = 4,
    late_layer_index: int | None = None,
    batch_size: int = 64,
    device: str | torch.device = "cpu",
) -> ExtractedDonorStates:
    """Extract final-real-token states using causally safe right padding."""
    late = donor.config.layer_count if late_layer_index is None else int(late_layer_index)
    if not 0 <= layer2_index < early_layer_index < late <= donor.config.layer_count:
        raise ValueError("depths must satisfy 0 <= layer2 < early < late <= layer_count")
    normalized = [tuple(int(value) for value in item) for item in sequences]
    if not normalized or any(not item for item in normalized):
        raise ValueError("sequences must be nonempty")
    if max(map(len, normalized)) > donor.config.max_sequence_length:
        raise ValueError("sequence exceeds donor maximum length")
    donor.eval().to(device)
    outputs = [torch.empty(len(normalized), donor.config.width) for _ in range(3)]
    ordered = sorted(range(len(normalized)), key=lambda index: len(normalized[index]))
    for start in range(0, len(ordered), batch_size):
        chosen = ordered[start:start + batch_size]
        lengths = torch.tensor([len(normalized[i]) for i in chosen], device=device)
        batch = torch.zeros(len(chosen), int(lengths.max()), dtype=torch.long, device=device)
        for local, source in enumerate(chosen):
            batch[local, :len(normalized[source])] = torch.tensor(normalized[source], device=device)
        result = donor(batch, output_hidden_states=True, compute_logits=False)
        rows = torch.arange(len(chosen), device=device)
        final_positions = lengths - 1
        destination = torch.tensor(chosen, dtype=torch.long)
        for target, depth in zip(outputs, (layer2_index, early_layer_index, late), strict=True):
            target.index_copy_(0, destination, result.hidden_states[depth][rows, final_positions].float().cpu())
    return ExtractedDonorStates(layer2=outputs[0], early=outputs[1], late=outputs[2])
