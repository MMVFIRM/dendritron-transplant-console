"""Validated configuration shared by addressing, memory, and sparse compute."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Literal, Mapping

ExpertKind = Literal["dense", "mlp", "dendritron"]
ResidualMode = Literal["postnorm", "prenorm"]
TrainingLoss = Literal["sparse", "full"]


@dataclass(frozen=True)
class HashMemoryConfig:
    """One source of truth for both hash address generation and table storage."""

    orders: tuple[int, ...] = (2, 3)
    heads: int = 2
    rows: tuple[int, ...] = (4_096, 16_384)
    memory_width: int = 16
    modulus: int = 2_147_483_647
    multiplier: int = 1_000_003

    def __post_init__(self) -> None:
        if not self.orders or any(int(order) < 1 for order in self.orders):
            raise ValueError("hash orders must contain positive integers")
        if len(set(self.orders)) != len(self.orders):
            raise ValueError("hash orders must be unique")
        if len(self.orders) != len(self.rows):
            raise ValueError("hash orders and row counts must have equal length")
        if self.heads < 1 or self.memory_width < 1:
            raise ValueError("hash heads and memory_width must be positive")
        if any(int(value) < 1 for value in self.rows):
            raise ValueError("hash row counts must be positive")
        if self.modulus < 3 or self.multiplier < 2:
            raise ValueError("hash modulus and multiplier are invalid")

    @property
    def rows_by_order(self) -> dict[int, int]:
        return dict(zip(self.orders, self.rows, strict=True))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "HashMemoryConfig":
        values = dict(record)
        for key in ("orders", "rows"):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)


@dataclass(frozen=True)
class LNGramConfig:
    orders: tuple[int, ...] = (2, 3)
    bits_per_route: int = 2
    route_memory_width: int = 2
    surrogate_scale: float = 0.25

    def __post_init__(self) -> None:
        if not self.orders or min(self.orders) < 1:
            raise ValueError("LNGram orders must be positive")
        if self.bits_per_route < 1 or self.bits_per_route > 8:
            raise ValueError("LNGram bits_per_route must be between 1 and 8")
        if self.route_memory_width < 1 or self.surrogate_scale < 0:
            raise ValueError("LNGram dimensions must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "LNGramConfig":
        values = dict(record)
        if "orders" in values:
            values["orders"] = tuple(values["orders"])
        return cls(**values)


@dataclass(frozen=True)
class DendritronRecipientConfig:
    """Configuration for the sparse CPU recipient."""

    vocab_size: int
    model_width: int = 64
    donor_width: int = 96
    max_sequence_length: int = 128
    loop_rounds: int = 2
    adaptive_threshold: float | None = None

    expert_kind: ExpertKind = "dendritron"
    expert_count: int = 24
    expert_top_k: int = 2
    branches_per_expert: int = 8
    branch_top_k: int = 2
    branch_hidden_width: int = 32

    use_hash_memory: bool = True
    use_exact_early: bool = True
    use_exact_late: bool = True
    use_definition_memory: bool = False
    definition_max_senses: int = 4
    use_lngram: bool = False

    causal_kernel_size: int = 5
    # "postnorm" is the original scaled post-norm recurrence (alpha * x, then
    # RMSNorm). "prenorm" adds every sublayer update to an unnormalised
    # residual stream, which lets expert updates change the state.
    residual_mode: ResidualMode = "postnorm"
    residual_dropout: float = 0.0
    residual_epsilon: float = 1e-6
    route_epsilon: float = 1e-6

    vocabulary_clusters: int = 8
    vocabulary_top_k_clusters: int = 2
    output_temperature: float = 8.0
    cluster_loss_weight: float = 0.25
    # "sparse" trains only against tokens in the selected clusters; "full"
    # trains against the whole vocabulary (inference stays cluster-sparse).
    training_loss: TrainingLoss = "sparse"

    hash_memory: HashMemoryConfig = field(default_factory=HashMemoryConfig)
    lngram: LNGramConfig = field(default_factory=LNGramConfig)

    def __post_init__(self) -> None:
        positive = {
            "vocab_size": self.vocab_size,
            "model_width": self.model_width,
            "donor_width": self.donor_width,
            "max_sequence_length": self.max_sequence_length,
            "loop_rounds": self.loop_rounds,
            "expert_count": self.expert_count,
            "expert_top_k": self.expert_top_k,
            "branches_per_expert": self.branches_per_expert,
            "branch_top_k": self.branch_top_k,
            "branch_hidden_width": self.branch_hidden_width,
            "definition_max_senses": self.definition_max_senses,
            "causal_kernel_size": self.causal_kernel_size,
            "vocabulary_clusters": self.vocabulary_clusters,
            "vocabulary_top_k_clusters": self.vocabulary_top_k_clusters,
        }
        for name, value in positive.items():
            if int(value) < 1:
                raise ValueError(f"{name} must be positive")
        if self.expert_kind not in {"dense", "mlp", "dendritron"}:
            raise ValueError("expert_kind must be dense, mlp, or dendritron")
        if self.expert_top_k > self.expert_count:
            raise ValueError("expert_top_k cannot exceed expert_count")
        if self.branch_top_k > self.branches_per_expert:
            raise ValueError("branch_top_k cannot exceed branches_per_expert")
        if self.vocabulary_top_k_clusters > self.vocabulary_clusters:
            raise ValueError("vocabulary_top_k_clusters cannot exceed vocabulary_clusters")
        if self.vocabulary_clusters > self.vocab_size:
            raise ValueError("vocabulary_clusters cannot exceed vocab_size")
        if self.max_sequence_length < 2:
            raise ValueError("max_sequence_length must be at least 2")
        if self.causal_kernel_size % 2 == 0:
            raise ValueError("causal_kernel_size must be odd")
        if self.residual_mode not in {"postnorm", "prenorm"}:
            raise ValueError("residual_mode must be postnorm or prenorm")
        if not 0.0 <= self.residual_dropout < 1.0:
            raise ValueError("residual_dropout must be in [0, 1)")
        if self.training_loss not in {"sparse", "full"}:
            raise ValueError("training_loss must be sparse or full")
        if self.output_temperature <= 0:
            raise ValueError("output_temperature must be positive")
        if not 0 <= self.cluster_loss_weight <= 10:
            raise ValueError("cluster_loss_weight is outside a sensible range")
        if self.adaptive_threshold is not None and self.adaptive_threshold <= 0:
            raise ValueError("adaptive_threshold must be positive when supplied")

    @property
    def block_equivalent_depth(self) -> int:
        return 2 * self.loop_rounds

    @property
    def deep_loop_alpha(self) -> float:
        return float((2.0 * self.block_equivalent_depth) ** 0.5)

    @property
    def deep_loop_beta(self) -> float:
        return float((8.0 * self.block_equivalent_depth) ** -0.5)

    def to_dict(self) -> dict[str, Any]:
        record = asdict(self)
        record["deep_loop_alpha"] = self.deep_loop_alpha
        record["deep_loop_beta"] = self.deep_loop_beta
        return record

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "DendritronRecipientConfig":
        accepted = {item.name for item in fields(cls)}
        values = {key: value for key, value in record.items() if key in accepted}
        if "hash_memory" in values and not isinstance(values["hash_memory"], HashMemoryConfig):
            values["hash_memory"] = HashMemoryConfig.from_dict(values["hash_memory"])
        if "lngram" in values and not isinstance(values["lngram"], LNGramConfig):
            values["lngram"] = LNGramConfig.from_dict(values["lngram"])
        return cls(**values)
