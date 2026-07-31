"""Validated configuration for the maintained DreaMARL architecture."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal


ObservationKind = Literal["vector", "image"]


@dataclass(frozen=True)
class EncoderConfig:
    kind: ObservationKind = "vector"
    embedding_dim: int = 128
    vector_hidden_dim: int = 256
    vector_layers: int = 2
    cnn_depth: int = 32
    cnn_blocks: int = 4

    def __post_init__(self) -> None:
        _positive(self, "embedding_dim", "vector_hidden_dim", "vector_layers")
        _positive(self, "cnn_depth", "cnn_blocks")


@dataclass(frozen=True)
class DynamicsConfig:
    model_dim: int = 256
    num_layers: int = 3
    num_heads: int = 4
    mlp_ratio: int = 4
    context_length: int = 64
    stochastic_variables: int = 16
    stochastic_classes: int = 16
    cross_agent_layers: int = 1
    cross_agent_heads: int = 4
    use_agent_identity: bool = True
    unimix: float = 0.01
    initial_continuation: float = 0.99
    initial_agent_alive: float = 0.99
    initial_action_legal: float = 0.99

    def __post_init__(self) -> None:
        _positive(
            self,
            "model_dim",
            "num_layers",
            "num_heads",
            "mlp_ratio",
            "context_length",
            "stochastic_variables",
            "stochastic_classes",
            "cross_agent_layers",
            "cross_agent_heads",
        )
        if self.model_dim % self.num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        if self.model_dim % self.cross_agent_heads:
            raise ValueError("model_dim must be divisible by cross_agent_heads")
        if not 0.0 <= self.unimix < 1.0:
            raise ValueError("unimix must lie in [0, 1)")
        for name in (
            "initial_continuation",
            "initial_agent_alive",
            "initial_action_legal",
        ):
            value = getattr(self, name)
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must lie in (0, 1), got {value}")


@dataclass(frozen=True)
class WorldModelLossConfig:
    jepa: float = 1.0
    dynamics_kl: float = 0.5
    representation_kl: float = 0.1
    team_reward: float = 1.0
    agent_reward: float = 0.25
    continuation: float = 1.0
    agent_alive: float = 0.5
    action_mask: float = 0.25
    free_nats: float = 1.0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")


@dataclass(frozen=True)
class DistributionConfig:
    """Shared symlog two-hot support for rewards and values."""

    bins: int = 255
    low: float = -20.0
    high: float = 20.0

    def __post_init__(self) -> None:
        if self.bins < 2:
            raise ValueError("distribution bins must be at least 2")
        if self.low >= self.high:
            raise ValueError("distribution support must have low < high")


@dataclass(frozen=True)
class ImaginationConfig:
    horizon: int = 15
    discount: float = 0.99
    lambda_: float = 0.95
    entropy_coefficient: float = 3e-4
    return_percentile_low: float = 5.0
    return_percentile_high: float = 95.0
    return_scale_decay: float = 0.99
    target_critic_decay: float = 0.98

    def __post_init__(self) -> None:
        _positive(self, "horizon")
        if not 0.0 <= self.discount <= 1.0:
            raise ValueError("discount must lie in [0, 1]")
        if not 0.0 <= self.lambda_ <= 1.0:
            raise ValueError("lambda_ must lie in [0, 1]")
        if not 0.0 <= self.target_critic_decay < 1.0:
            raise ValueError("target_critic_decay must lie in [0, 1)")
        if not 0.0 <= self.return_scale_decay < 1.0:
            raise ValueError("return_scale_decay must lie in [0, 1)")
        if not (
            0.0
            <= self.return_percentile_low
            < self.return_percentile_high
            <= 100.0
        ):
            raise ValueError("return percentiles must be ordered within [0, 100]")


@dataclass(frozen=True)
class OptimizerConfig:
    world_model_learning_rate: float = 1e-4
    actor_learning_rate: float = 3e-5
    critic_learning_rate: float = 1e-4
    world_model_grad_clip: float = 100.0
    actor_critic_grad_clip: float = 100.0
    target_encoder_decay: float = 0.99

    def __post_init__(self) -> None:
        _positive(
            self,
            "world_model_learning_rate",
            "actor_learning_rate",
            "critic_learning_rate",
            "world_model_grad_clip",
            "actor_critic_grad_clip",
        )
        if not 0.0 <= self.target_encoder_decay < 1.0:
            raise ValueError("target_encoder_decay must lie in [0, 1)")


@dataclass(frozen=True)
class ReplayConfig:
    capacity: int = 100_000
    sequence_length: int = 64
    batch_size: int = 16
    prefetch_batches: int = 2

    def __post_init__(self) -> None:
        _positive(
            self,
            "capacity",
            "sequence_length",
            "batch_size",
            "prefetch_batches",
        )
        if self.capacity < self.sequence_length:
            raise ValueError("replay capacity must cover at least one sequence")


@dataclass(frozen=True)
class DreaMARLConfig:
    """One static-shape DreaMARL learner configuration."""

    max_agents: int
    action_dim: int
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    world_model_loss: WorldModelLossConfig = field(
        default_factory=WorldModelLossConfig
    )
    distribution: DistributionConfig = field(
        default_factory=DistributionConfig
    )
    imagination: ImaginationConfig = field(default_factory=ImaginationConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)

    def __post_init__(self) -> None:
        _positive(self, "max_agents", "action_dim")

    @property
    def stochastic_dim(self) -> int:
        return (
            self.dynamics.stochastic_variables
            * self.dynamics.stochastic_classes
        )

    @property
    def belief_dim(self) -> int:
        return self.dynamics.model_dim + self.stochastic_dim

    @property
    def temporal_pair_dim(self) -> int:
        return self.stochastic_dim + self.action_dim + self.dynamics.model_dim

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _positive(instance: object, *names: str) -> None:
    for name in names:
        value = getattr(instance, name)
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
