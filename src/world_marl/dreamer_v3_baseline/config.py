from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RSSMConfig:
    deterministic_size: int = 2048
    stochastic_size: int = 32
    discrete_classes: int = 16
    hidden_size: int = 256
    blocks: int = 8
    unimix: float = 0.01
    prior_layers: int = 2
    posterior_layers: int = 1

    def __post_init__(self) -> None:
        if self.deterministic_size <= 0:
            raise ValueError("deterministic_size must be positive")
        if self.deterministic_size % self.blocks:
            raise ValueError("deterministic_size must be divisible by blocks")
        if self.stochastic_size <= 0:
            raise ValueError("stochastic_size must be positive")
        if self.discrete_classes <= 1:
            raise ValueError("discrete_classes must be greater than one")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.blocks <= 0:
            raise ValueError("blocks must be positive")
        if not 0.0 <= self.unimix < 1.0:
            raise ValueError("unimix must be in [0, 1)")
        if self.prior_layers <= 0 or self.posterior_layers <= 0:
            raise ValueError("prior_layers and posterior_layers must be positive")

    @property
    def latent_size(self) -> int:
        return self.deterministic_size + self.stochastic_size * self.discrete_classes


@dataclass(frozen=True)
class EncoderConfig:
    embedding_dim: int = 256
    hidden_dims: tuple[int, ...] = (256, 256, 256)
    cnn_depth: int = 16
    cnn_multipliers: tuple[int, ...] = (2, 3, 4, 4)
    cnn_kernel: int = 5
    cnn_outer_stride: int = 2


@dataclass(frozen=True)
class RewardHeadConfig:
    bins: int = 255
    distribution: str = "symlog_two_hot"
    hidden_dims: tuple[int, ...] = (256,)


@dataclass(frozen=True)
class ContinueHeadConfig:
    distribution: str = "bernoulli"
    hidden_dims: tuple[int, ...] = (256,)


@dataclass(frozen=True)
class ActorCriticConfig:
    hidden_dims: tuple[int, ...] = (256, 256, 256)
    value_bins: int = 255
    imagination_horizon: int = 15
    discount_horizon: int = 333
    discount_lambda: float = 0.95
    entropy_scale: float = 3e-4
    actor_unimix: float = 0.01
    min_std: float = 0.1
    max_std: float = 1.0
    return_percentile_low: float = 5.0
    return_percentile_high: float = 95.0
    return_scale_min: float = 1.0
    return_norm_decay: float = 0.99
    critic_imagination_scale: float = 1.0
    critic_replay_scale: float = 0.3
    critic_ema_regularizer_scale: float = 1.0
    critic_ema_decay: float = 0.98

    def __post_init__(self) -> None:
        if not self.hidden_dims or any(dim <= 0 for dim in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive dimensions")
        if self.value_bins <= 1:
            raise ValueError("value_bins must be greater than one")
        if self.imagination_horizon <= 0:
            raise ValueError("imagination_horizon must be positive")
        if self.discount_horizon <= 1:
            raise ValueError("discount_horizon must be greater than one")
        if not 0.0 <= self.discount_lambda <= 1.0:
            raise ValueError("discount_lambda must be in [0, 1]")
        if self.entropy_scale < 0.0:
            raise ValueError("entropy_scale must be non-negative")
        if not 0.0 <= self.actor_unimix < 1.0:
            raise ValueError("actor_unimix must be in [0, 1)")

    @property
    def discount(self) -> float:
        return 1.0 - 1.0 / self.discount_horizon


@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float = 4e-5
    agc: float = 0.3
    epsilon: float = 1e-20
    beta1: float = 0.9
    beta2: float = 0.99


@dataclass(frozen=True)
class ReplayConfig:
    capacity: int = 5_000_000
    batch_size: int = 16
    batch_length: int = 64
    train_ratio: float = 32.0


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
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    kl_free_nats: float = 1.0
    dynamics_kl_scale: float = 1.0
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
        if self.is_image_observation and self.observation_shape[-1] not in {1, 3, 4}:
            raise ValueError("image observations must use HWC channel order")

    @property
    def is_image_observation(self) -> bool:
        return len(self.observation_shape) == 3

    @classmethod
    def debug(
        cls,
        *,
        action_dim: int,
        observation_shape: tuple[int, ...],
        action_mode: str = "discrete",
    ) -> DreamerV3Config:
        return cls(
            action_dim=action_dim,
            observation_shape=observation_shape,
            action_mode=action_mode,
            rssm=RSSMConfig(
                deterministic_size=32,
                stochastic_size=4,
                discrete_classes=4,
                hidden_size=32,
                blocks=4,
            ),
            encoder=EncoderConfig(
                embedding_dim=32,
                hidden_dims=(32, 32, 32),
                cnn_depth=4,
            ),
            reward_head=RewardHeadConfig(hidden_dims=(32,)),
            continue_head=ContinueHeadConfig(hidden_dims=(32,)),
            actor_critic=ActorCriticConfig(hidden_dims=(32, 32, 32)),
            replay=ReplayConfig(capacity=1024, batch_size=2, batch_length=4),
        )
