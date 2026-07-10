from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RSSMConfig:
    deterministic_size: int = 128
    stochastic_size: int = 16
    discrete_classes: int = 16
    hidden_size: int = 256

    def __post_init__(self) -> None:
        if self.deterministic_size <= 0:
            raise ValueError("deterministic_size must be positive")
        if self.stochastic_size <= 0:
            raise ValueError("stochastic_size must be positive")
        if self.discrete_classes <= 1:
            raise ValueError("discrete_classes must be greater than one")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")

    @property
    def latent_size(self) -> int:
        return self.deterministic_size + self.stochastic_size * self.discrete_classes


@dataclass(frozen=True)
class EncoderConfig:
    embedding_dim: int = 64
    hidden_dims: tuple[int, ...] = (128, 128)


@dataclass(frozen=True)
class RewardHeadConfig:
    bins: int = 255
    distribution: str = "symlog_two_hot"
    hidden_dims: tuple[int, ...] = (128, 128)


@dataclass(frozen=True)
class ContinueHeadConfig:
    distribution: str = "bernoulli"
    hidden_dims: tuple[int, ...] = (128, 128)


@dataclass(frozen=True)
class ActorCriticConfig:
    hidden_dims: tuple[int, ...] = (128, 128)
    value_bins: int = 255
    imagination_horizon: int = 15
    discount_lambda: float = 0.95
    entropy_scale: float = 3e-4

    def __post_init__(self) -> None:
        if not self.hidden_dims or any(dim <= 0 for dim in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive dimensions")
        if self.value_bins <= 1:
            raise ValueError("value_bins must be greater than one")
        if self.imagination_horizon <= 0:
            raise ValueError("imagination_horizon must be positive")
        if not 0.0 <= self.discount_lambda <= 1.0:
            raise ValueError("discount_lambda must be in [0, 1]")
        if self.entropy_scale < 0.0:
            raise ValueError("entropy_scale must be non-negative")


@dataclass(frozen=True)
class DreamerV3Config:
    action_dim: int
    observation_shape: tuple[int, ...]
    action_mode: str = "discrete"
    rssm: RSSMConfig = field(default_factory=RSSMConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    reward_head: RewardHeadConfig = field(default_factory=RewardHeadConfig)
    continue_head: ContinueHeadConfig = field(default_factory=ContinueHeadConfig)
    actor_critic: ActorCriticConfig = field(default_factory=ActorCriticConfig)
    kl_free_nats: float = 1.0
    dynamics_kl_scale: float = 0.5
    representation_kl_scale: float = 0.1

    def __post_init__(self) -> None:
        if self.action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if self.action_mode not in {"discrete", "continuous"}:
            raise ValueError("action_mode must be 'discrete' or 'continuous'")
        if not self.observation_shape or any(
            dim <= 0 for dim in self.observation_shape
        ):
            raise ValueError("observation_shape must contain positive dimensions")
