from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from typing import Any


class DreamerProfile(str, Enum):
    PAPER = "paper"
    UPSTREAM_CURRENT = "upstream-current"


class ObservationMode(str, Enum):
    VISION = "vision"
    PROPRIO = "proprio"


_CANONICAL_AUTHORITY_HASHES = {
    (DreamerProfile.PAPER, ObservationMode.VISION): (
        "0f7b619eeb24d87d2d20b3331cd0a4229d4e322f007af52a85f1eb9028af4ca3"
    ),
    (DreamerProfile.PAPER, ObservationMode.PROPRIO): (
        "6e1b19b9d058ab579980dad3ed8d8caa16e805f0633bb623611d12216c5987c2"
    ),
    (DreamerProfile.UPSTREAM_CURRENT, ObservationMode.VISION): (
        "8973a4665e54be56322a158ddceea785b19850806bfcdb65743a310fbed4b94c"
    ),
    (DreamerProfile.UPSTREAM_CURRENT, ObservationMode.PROPRIO): (
        "ce029c8275701e2a692133b1d4bd1cdabe16144fc5864309d59fe9fdbae69908"
    ),
}


class ModelSize(str, Enum):
    M1 = "1m"
    M12 = "12m"
    M25 = "25m"
    M50 = "50m"
    M100 = "100m"
    M200 = "200m"
    M400 = "400m"

    SIZE_1M = "1m"
    SIZE_12M = "12m"
    SIZE_25M = "25m"
    SIZE_50M = "50m"
    SIZE_100M = "100m"
    SIZE_200M = "200m"
    SIZE_400M = "400m"

    @classmethod
    def _missing_(cls, value: object) -> ModelSize | None:
        if isinstance(value, str):
            normalized = value.lower().removeprefix("size")
            for member in cls:
                if member.value == normalized:
                    return member
        return None

    def resolve(self) -> NetworkSize:
        return {
            ModelSize.M1: NetworkSize(64, 512, 4, 4),
            ModelSize.M12: NetworkSize(256, 2048, 16, 16),
            ModelSize.M25: NetworkSize(384, 3072, 24, 24),
            ModelSize.M50: NetworkSize(512, 4096, 32, 32),
            ModelSize.M100: NetworkSize(768, 6144, 48, 48),
            ModelSize.M200: NetworkSize(1024, 8192, 64, 64),
            ModelSize.M400: NetworkSize(1536, 12288, 96, 96),
        }[self]


@dataclass(frozen=True)
class NetworkSize:
    model_dim: int
    deter: int
    depth: int
    classes: int

    def __post_init__(self) -> None:
        if self.model_dim <= 0:
            raise ValueError("model_dim must be positive")
        if self.deter != 8 * self.model_dim:
            raise ValueError("deter must be eight times model_dim")
        if self.depth != self.model_dim // 16:
            raise ValueError("depth must be model_dim divided by 16")
        if self.classes != self.model_dim // 16:
            raise ValueError("classes must be model_dim divided by 16")

    @property
    def hidden(self) -> int:
        return self.model_dim

    @property
    def units(self) -> int:
        return self.model_dim


@dataclass(frozen=True, init=False)
class RSSMConfig:
    deter: int
    hidden: int
    stoch: int
    classes: int
    blocks: int
    free_nats: float
    unimix: float
    activation: str
    normalization: str
    image_layers: int
    observation_layers: int
    dynamics_layers: int
    absolute: bool
    initializer: str
    output_scale: float
    _legacy: bool = field(default=False, repr=False, compare=False)

    def __init__(
        self,
        deter: int = 8192,
        hidden: int = 1024,
        stoch: int = 32,
        classes: int = 64,
        blocks: int = 8,
        free_nats: float = 1.0,
        unimix: float = 0.01,
        activation: str = "silu",
        normalization: str = "rms",
        image_layers: int = 2,
        observation_layers: int = 1,
        dynamics_layers: int = 1,
        absolute: bool = False,
        initializer: str = "trunc_normal_in",
        output_scale: float = 1.0,
        *,
        deterministic_size: int | None = None,
        hidden_size: int | None = None,
        stochastic_size: int | None = None,
        discrete_classes: int | None = None,
        _legacy: bool = False,
    ) -> None:
        aliases = (
            deterministic_size,
            hidden_size,
            stochastic_size,
            discrete_classes,
        )
        legacy = _legacy or any(value is not None for value in aliases)
        if deterministic_size is not None:
            deter = deterministic_size
        if hidden_size is not None:
            hidden = hidden_size
        if stochastic_size is not None:
            stoch = stochastic_size
        if discrete_classes is not None:
            classes = discrete_classes
        for name, value in (
            ("deter", deter),
            ("hidden", hidden),
            ("stoch", stoch),
            ("classes", classes),
            ("blocks", blocks),
            ("free_nats", free_nats),
            ("unimix", unimix),
            ("activation", activation),
            ("normalization", normalization),
            ("image_layers", image_layers),
            ("observation_layers", observation_layers),
            ("dynamics_layers", dynamics_layers),
            ("absolute", absolute),
            ("initializer", initializer),
            ("output_scale", output_scale),
            ("_legacy", legacy),
        ):
            object.__setattr__(self, name, value)
        self.validate(canonical=not legacy)

    def validate(self, *, canonical: bool = True) -> None:
        if self.deter <= 0 or self.deter % 8:
            raise ValueError("deter must be positive and divisible by eight")
        if self.hidden <= 0:
            raise ValueError("hidden must be positive")
        if self.stoch <= 0:
            raise ValueError("stoch must be positive")
        if canonical and self.stoch != 32:
            raise ValueError("canonical RSSM requires 32 stochastic latents")
        if self.classes <= 0:
            raise ValueError("classes must be positive")
        if self.blocks <= 0:
            raise ValueError("blocks must be positive")
        if canonical and self.blocks != 8:
            raise ValueError("canonical RSSM requires eight blocks")
        if self.free_nats < 0:
            raise ValueError("free_nats must be nonnegative")
        if not 0.0 <= self.unimix < 1.0:
            raise ValueError("unimix must be in [0, 1)")
        if min(self.image_layers, self.observation_layers, self.dynamics_layers) < 0:
            raise ValueError("RSSM layer counts must be nonnegative")

    @property
    def deterministic_size(self) -> int:
        return self.deter

    @property
    def hidden_size(self) -> int:
        return self.hidden

    @property
    def stochastic_size(self) -> int:
        return self.stoch

    @property
    def discrete_classes(self) -> int:
        return self.classes

    @property
    def latent_size(self) -> int:
        return self.deter + self.stoch * self.classes


@dataclass(frozen=True)
class EncoderConfig:
    depth: int = 64
    multipliers: tuple[int, ...] = (2, 3, 4, 4)
    layers: int = 3
    units: int = 1024
    activation: str = "silu"
    normalization: str = "rms"
    initializer: str = "trunc_normal_in"
    symlog: bool = True
    outer: bool = False
    kernel: int = 5
    strided: bool = True

    def __post_init__(self) -> None:
        if self.depth <= 0 or self.layers < 0 or self.units <= 0 or self.kernel <= 0:
            raise ValueError("encoder dimensions must be positive")
        if not self.multipliers or any(value <= 0 for value in self.multipliers):
            raise ValueError("encoder multipliers must be positive")

    @property
    def embedding_dim(self) -> int:
        return self.units

    @property
    def hidden_dims(self) -> tuple[int, ...]:
        return (self.units,) * self.layers


@dataclass(frozen=True)
class _LegacyEncoderConfig(EncoderConfig):
    _embedding_dim: int = field(default=64, repr=False)
    _hidden_dims: tuple[int, ...] = field(default=(128, 128), repr=False)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def hidden_dims(self) -> tuple[int, ...]:
        return self._hidden_dims


@dataclass(frozen=True)
class DecoderConfig:
    depth: int = 64
    multipliers: tuple[int, ...] = (2, 3, 4, 4)
    layers: int = 3
    units: int = 1024
    activation: str = "silu"
    normalization: str = "rms"
    output_scale: float = 1.0
    initializer: str = "trunc_normal_in"
    outer: bool = False
    kernel: int = 5
    bias_space: int = 8
    strided: bool = True
    image_output: str = "mse"

    def __post_init__(self) -> None:
        if (
            self.depth <= 0
            or self.layers < 0
            or self.units <= 0
            or self.kernel <= 0
            or self.bias_space <= 0
        ):
            raise ValueError("decoder dimensions must be positive")
        if not self.multipliers or any(value <= 0 for value in self.multipliers):
            raise ValueError("decoder multipliers must be positive")


@dataclass(frozen=True)
class HeadConfig:
    layers: int = 1
    units: int = 1024
    activation: str = "silu"
    normalization: str = "rms"
    output: str = "symexp_twohot"
    output_scale: float = 0.0
    initializer: str = "trunc_normal_in"
    bins: int | None = 255

    def __post_init__(self) -> None:
        if self.layers < 0 or self.units <= 0:
            raise ValueError("head dimensions must be positive")
        if self.output == "symexp_twohot" and (self.bins is None or self.bins <= 1):
            raise ValueError("two-hot heads require more than one bin")
        if self.output == "binary" and self.bins is not None:
            raise ValueError("binary heads do not use bins")

    @property
    def distribution(self) -> str:
        return {
            "symexp_twohot": "symlog_two_hot",
            "binary": "bernoulli",
        }.get(self.output, self.output)

    @property
    def hidden_dims(self) -> tuple[int, ...]:
        return (self.units,) * self.layers


class RewardHeadConfig(HeadConfig):
    def __init__(
        self,
        bins: int = 255,
        distribution: str = "symlog_two_hot",
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        if not hidden_dims or len(set(hidden_dims)) != 1:
            raise ValueError("legacy hidden_dims must contain one repeated width")
        output = {
            "symlog_two_hot": "symexp_twohot",
            "symexp_twohot": "symexp_twohot",
        }.get(distribution, distribution)
        super().__init__(
            layers=len(hidden_dims),
            units=hidden_dims[0],
            output=output,
            output_scale=0.0,
            bins=bins,
        )


class ContinueHeadConfig(HeadConfig):
    def __init__(
        self,
        distribution: str = "bernoulli",
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        if not hidden_dims or len(set(hidden_dims)) != 1:
            raise ValueError("legacy hidden_dims must contain one repeated width")
        output = "binary" if distribution == "bernoulli" else distribution
        super().__init__(
            layers=len(hidden_dims),
            units=hidden_dims[0],
            output=output,
            output_scale=1.0,
            bins=None,
        )


@dataclass(frozen=True)
class PolicyConfig:
    layers: int = 3
    units: int = 1024
    activation: str = "silu"
    normalization: str = "rms"
    min_std: float = 0.1
    max_std: float = 1.0
    output_scale: float = 0.01
    unimix: float = 0.01
    initializer: str = "trunc_normal_in"
    discrete: str = "categorical"
    continuous: str = "bounded_normal"

    def __post_init__(self) -> None:
        if self.layers < 0 or self.units <= 0:
            raise ValueError("policy dimensions must be positive")
        if not 0.0 < self.min_std <= self.max_std:
            raise ValueError("policy std bounds are invalid")
        if not 0.0 <= self.unimix < 1.0:
            raise ValueError("policy unimix must be in [0, 1)")


@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float = 4e-5
    agc: float = 0.3
    agc_floor: float = 1e-3
    epsilon: float = 1e-20
    beta1: float = 0.9
    beta2: float = 0.999
    momentum: bool = True
    weight_decay: float = 0.0
    schedule: str = "const"
    warmup: int = 1000
    anneal: int = 0

    def __post_init__(self) -> None:
        if self.learning_rate <= 0 or self.agc < 0 or self.agc_floor <= 0:
            raise ValueError("optimizer scales are invalid")
        if self.epsilon <= 0:
            raise ValueError("optimizer epsilon must be positive")
        if not 0.0 <= self.beta1 < 1.0 or not 0.0 <= self.beta2 < 1.0:
            raise ValueError("optimizer betas must be in [0, 1)")
        if self.weight_decay < 0 or self.warmup < 0 or self.anneal < 0:
            raise ValueError("optimizer schedule values are invalid")


@dataclass(frozen=True)
class ReplayConfig:
    capacity: int = 5_000_000
    chunk_size: int = 1024
    online: bool = True
    uniform_fraction: float = 1.0
    priority_fraction: float = 0.0
    recency_fraction: float = 0.0
    context: int = 1
    sequence_length: int = 64

    def __post_init__(self) -> None:
        if self.capacity <= 0 or self.chunk_size <= 0:
            raise ValueError("replay sizes must be positive")
        if self.context < 0 or self.sequence_length <= 0:
            raise ValueError("replay sequence values are invalid")
        fractions = (
            self.uniform_fraction,
            self.priority_fraction,
            self.recency_fraction,
        )
        if any(value < 0 for value in fractions) or sum(fractions) != 1.0:
            raise ValueError("replay selector fractions must sum to one")


@dataclass(frozen=True)
class RunConfig:
    steps: int = 1_000_000
    replay_ratio: float = 256.0
    envs: int = 16
    eval_envs: int = 4
    eval_episodes: int = 1
    log_every: int = 120
    report_every: int = 300
    save_every: int = 900
    batch_size: int = 16
    batch_length: int = 64
    report_length: int = 32
    consecutive_train: int = 1
    consecutive_report: int = 1
    replay_context: int = 1
    action_repeat: int = 1
    image_size: tuple[int, int] = (64, 64)
    camera: int = -1
    platform: str = "cuda"
    compute_dtype: str = "bfloat16"
    policy_devices: tuple[int, ...] = (0,)
    train_devices: tuple[int, ...] = (0,)
    preallocate: bool = True

    def __post_init__(self) -> None:
        positive = (
            self.steps,
            self.replay_ratio,
            self.envs,
            self.eval_envs,
            self.eval_episodes,
            self.batch_size,
            self.batch_length,
            self.report_length,
            self.consecutive_train,
            self.consecutive_report,
            self.action_repeat,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("run counts and ratios must be positive")
        if self.replay_context < 0 or any(value <= 0 for value in self.image_size):
            raise ValueError("run context and image size are invalid")

    @property
    def gradient_updates_per_transition(self) -> float:
        return self.replay_ratio / (
            self.batch_size * self.batch_length * self.action_repeat
        )


@dataclass(frozen=True)
class LossScaleConfig:
    rec: float = 1.0
    rew: float = 1.0
    con: float = 1.0
    dyn: float = 1.0
    rep: float = 0.1
    policy: float = 1.0
    value: float = 1.0
    repval: float = 0.3

    def __post_init__(self) -> None:
        if any(value < 0 for _, value in self.as_tuple()):
            raise ValueError("loss scales must be nonnegative")

    def as_tuple(self) -> tuple[tuple[str, float], ...]:
        return tuple((item.name, getattr(self, item.name)) for item in fields(self))


@dataclass(frozen=True)
class ImaginationConfig:
    length: int = 15
    horizon: int = 333
    continuation_discount: bool = True
    lambda_: float = 0.95
    actor_entropy: float = 3e-4
    slow_target: bool = False
    slow_regularizer: float = 1.0
    ac_grads: bool = False
    reward_grad: bool = True
    repval_loss: bool = True
    repval_grad: bool = True

    def __post_init__(self) -> None:
        if self.length <= 0 or self.horizon <= 0:
            raise ValueError("imagination lengths must be positive")
        if not 0.0 <= self.lambda_ <= 1.0 or self.actor_entropy < 0:
            raise ValueError("imagination objective values are invalid")


@dataclass(frozen=True)
class SlowValueConfig:
    rate: float = 0.02
    every: int = 1

    def __post_init__(self) -> None:
        if not 0.0 < self.rate <= 1.0 or self.every <= 0:
            raise ValueError("slow value update settings are invalid")


@dataclass(frozen=True)
class NormalizerConfig:
    implementation: str = "percentile"
    rate: float = 0.01
    limit: float = 1.0
    low_percentile: float = 5.0
    high_percentile: float = 95.0
    debias: bool = False

    def __post_init__(self) -> None:
        if not 0.0 < self.rate <= 1.0 or self.limit < 0:
            raise ValueError("normalizer rate and limit are invalid")
        if not 0.0 <= self.low_percentile < self.high_percentile <= 100.0:
            raise ValueError("normalizer percentiles are invalid")


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
        if self.value_bins <= 1 or self.imagination_horizon <= 0:
            raise ValueError("actor critic sizes are invalid")
        if not 0.0 <= self.discount_lambda <= 1.0 or self.entropy_scale < 0:
            raise ValueError("actor critic objective values are invalid")


@dataclass(frozen=True)
class DreamerV3Config:
    profile: DreamerProfile | str = DreamerProfile.PAPER
    observation_mode: ObservationMode | str = ObservationMode.VISION
    model_size: ModelSize | str | None = None
    network: NetworkSize | None = None
    rssm: RSSMConfig | None = None
    encoder: EncoderConfig | None = None
    decoder: DecoderConfig | None = None
    reward_head: HeadConfig | None = None
    continue_head: HeadConfig | None = None
    policy: PolicyConfig | None = None
    value_head: HeadConfig | None = None
    optimizer: OptimizerConfig | None = None
    replay: ReplayConfig | None = None
    run: RunConfig | None = None
    loss_scales: LossScaleConfig = field(default_factory=LossScaleConfig)
    imagination: ImaginationConfig = field(default_factory=ImaginationConfig)
    slow_value: SlowValueConfig = field(default_factory=SlowValueConfig)
    return_normalizer: NormalizerConfig = field(default_factory=NormalizerConfig)
    value_normalizer: NormalizerConfig = field(
        default_factory=lambda: NormalizerConfig(
            implementation="none",
            limit=1e-8,
        )
    )
    advantage_normalizer: NormalizerConfig = field(
        default_factory=lambda: NormalizerConfig(
            implementation="none",
            limit=1e-8,
        )
    )
    action_dim: int | None = None
    observation_shape: tuple[int, ...] | None = None
    action_mode: str = "discrete"
    _legacy: bool = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        profile = DreamerProfile(self.profile)
        mode = ObservationMode(self.observation_mode)
        object.__setattr__(self, "profile", profile)
        object.__setattr__(self, "observation_mode", mode)
        legacy = self.action_dim is not None or self.observation_shape is not None
        object.__setattr__(self, "_legacy", legacy)
        if legacy:
            self._fill_legacy_defaults()
        else:
            size = ModelSize(self.model_size) if self.model_size is not None else None
            size = size or _default_model_size(mode)
            object.__setattr__(self, "model_size", size)
            if any(
                value is None
                for value in (
                    self.network,
                    self.rssm,
                    self.encoder,
                    self.decoder,
                    self.reward_head,
                    self.continue_head,
                    self.policy,
                    self.value_head,
                    self.optimizer,
                    self.replay,
                    self.run,
                )
            ):
                components = (
                    _paper_components(mode, size)
                    if profile is DreamerProfile.PAPER
                    else _upstream_current_components(mode, size)
                )
                for name, value in components.items():
                    if getattr(self, name) is None:
                        object.__setattr__(self, name, value)
            object.__setattr__(self, "action_mode", "continuous")
        self.validate()

    def _fill_legacy_defaults(self) -> None:
        if self.action_dim is None or self.action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if self.observation_shape is None or not self.observation_shape:
            raise ValueError("observation_shape must contain positive dimensions")
        if any(dim <= 0 for dim in self.observation_shape):
            raise ValueError("observation_shape must contain positive dimensions")
        if self.action_mode not in {"discrete", "continuous"}:
            raise ValueError("action_mode must be 'discrete' or 'continuous'")
        defaults: dict[str, Any] = {
            "model_size": ModelSize.M1,
            "network": ModelSize.M1.resolve(),
            "rssm": RSSMConfig(
                deterministic_size=128,
                stochastic_size=16,
                discrete_classes=16,
                hidden_size=256,
            ),
            "encoder": _LegacyEncoderConfig(depth=4, layers=2, units=128),
            "decoder": DecoderConfig(depth=4, layers=2, units=128),
            "reward_head": RewardHeadConfig(),
            "continue_head": ContinueHeadConfig(),
            "policy": PolicyConfig(layers=2, units=128),
            "value_head": HeadConfig(layers=2, units=128),
            "optimizer": OptimizerConfig(beta2=0.99),
            "replay": ReplayConfig(),
            "run": RunConfig(),
        }
        for name, value in defaults.items():
            if getattr(self, name) is None:
                object.__setattr__(self, name, value)

    def validate(self) -> None:
        required = (
            self.network,
            self.rssm,
            self.encoder,
            self.decoder,
            self.reward_head,
            self.continue_head,
            self.policy,
            self.value_head,
            self.optimizer,
            self.replay,
            self.run,
        )
        if any(value is None for value in required):
            raise ValueError("DreamerV3 configuration is incomplete")
        assert self.rssm is not None
        if self._legacy:
            self.rssm.validate(canonical=False)
            return
        assert isinstance(self.model_size, ModelSize)
        assert self.network is not None
        assert self.encoder is not None
        assert self.decoder is not None
        assert self.reward_head is not None
        assert self.continue_head is not None
        assert self.policy is not None
        assert self.value_head is not None
        assert self.optimizer is not None
        assert self.replay is not None
        assert self.run is not None
        expected_size = _default_model_size(self.observation_mode)
        if self.model_size is not expected_size:
            raise ValueError(
                f"{self.observation_mode.value} requires {expected_size.value} model"
            )
        expected_network = self.model_size.resolve()
        if self.network != expected_network:
            raise ValueError("network does not match model size")
        if (
            self.rssm.deter != expected_network.deter
            or self.rssm.hidden != expected_network.model_dim
            or self.rssm.classes != expected_network.classes
        ):
            raise ValueError("RSSM does not match model size")
        self.rssm.validate(canonical=True)
        if (
            self.encoder.depth != expected_network.depth
            or self.decoder.depth != expected_network.depth
            or self.encoder.units != expected_network.model_dim
            or self.decoder.units != expected_network.model_dim
        ):
            raise ValueError("encoder or decoder does not match model size")
        if (
            any(
                head.units != expected_network.model_dim
                for head in (self.reward_head, self.continue_head, self.value_head)
            )
            or self.policy.units != expected_network.model_dim
        ):
            raise ValueError("heads do not match model size")
        paper = self.profile is DreamerProfile.PAPER
        expected_beta2 = 0.99 if paper else 0.999
        expected_strided = paper
        expected_steps = 1_000_000 if paper else 1_100_000
        if self.optimizer.beta2 != expected_beta2:
            raise ValueError("optimizer beta2 does not match profile")
        if (
            self.encoder.strided is not expected_strided
            or self.decoder.strided is not expected_strided
        ):
            raise ValueError("image stride behavior does not match profile")
        if self.run.steps != expected_steps:
            raise ValueError("run steps do not match profile")
        expected_ratio = (
            256.0 if self.observation_mode is ObservationMode.VISION else 1024.0
        )
        if self.run.replay_ratio != expected_ratio:
            raise ValueError("replay ratio does not match observation mode")
        if (
            self.replay.capacity != 5_000_000
            or self.replay.uniform_fraction != 1.0
            or self.replay.priority_fraction != 0.0
            or self.replay.recency_fraction != 0.0
        ):
            raise ValueError("replay is not canonical uniform replay")
        expected_components = (
            _paper_components(self.observation_mode, self.model_size)
            if self.profile is DreamerProfile.PAPER
            else _upstream_current_components(self.observation_mode, self.model_size)
        )
        expected_components.update(
            loss_scales=LossScaleConfig(),
            imagination=ImaginationConfig(),
            slow_value=SlowValueConfig(),
            return_normalizer=NormalizerConfig(),
            value_normalizer=NormalizerConfig(
                implementation="none",
                limit=1e-8,
            ),
            advantage_normalizer=NormalizerConfig(
                implementation="none",
                limit=1e-8,
            ),
        )
        for name, expected in expected_components.items():
            if getattr(self, name) != expected:
                raise ValueError(
                    f"{name} does not match the canonical authority snapshot"
                )
        if (
            self.action_dim is not None
            or self.observation_shape is not None
            or self.action_mode != "continuous"
        ):
            raise ValueError("runtime interface does not match the canonical profile")
        expected_hash = _CANONICAL_AUTHORITY_HASHES[
            (self.profile, self.observation_mode)
        ]
        if self.canonical_hash() != expected_hash:
            raise ValueError("configuration does not match canonical authority hash")

    @property
    def actor_critic(self) -> ActorCriticConfig:
        assert self.policy is not None
        assert self.value_head is not None
        return ActorCriticConfig(
            hidden_dims=(self.policy.units,) * self.policy.layers,
            value_bins=self.value_head.bins or 255,
            imagination_horizon=self.imagination.length,
            discount_lambda=self.imagination.lambda_,
            entropy_scale=self.imagination.actor_entropy,
        )

    @property
    def kl_free_nats(self) -> float:
        assert self.rssm is not None
        return self.rssm.free_nats

    @property
    def dynamics_kl_scale(self) -> float:
        return 0.5 if self._legacy else self.loss_scales.dyn

    @property
    def representation_kl_scale(self) -> float:
        return self.loss_scales.rep

    def to_dict(self) -> dict[str, Any]:
        names = (
            "profile",
            "observation_mode",
            "model_size",
            "network",
            "rssm",
            "encoder",
            "decoder",
            "reward_head",
            "continue_head",
            "policy",
            "value_head",
            "optimizer",
            "replay",
            "run",
            "loss_scales",
            "imagination",
            "slow_value",
            "return_normalizer",
            "value_normalizer",
            "advantage_normalizer",
        )
        payload = {name: _json_value(getattr(self, name)) for name in names}
        if self._legacy:
            payload.update(
                action_dim=self.action_dim,
                observation_shape=_json_value(self.observation_shape),
                action_mode=self.action_mode,
            )
        return payload

    def canonical_hash(self) -> str:
        encoded = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {
            item.name: _json_value(getattr(value, item.name))
            for item in fields(value)
            if not item.name.startswith("_")
        }
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value


def _default_model_size(mode: ObservationMode) -> ModelSize:
    return ModelSize.M200 if mode is ObservationMode.VISION else ModelSize.M1


def _paper_components(
    mode: ObservationMode,
    model_size: ModelSize,
) -> dict[str, Any]:
    network = model_size.resolve()
    ratio = 256.0 if mode is ObservationMode.VISION else 1024.0
    return {
        "network": network,
        "rssm": RSSMConfig(
            deter=network.deter,
            hidden=network.model_dim,
            stoch=32,
            classes=network.classes,
            blocks=8,
            free_nats=1.0,
            unimix=0.01,
            activation="silu",
            normalization="rms",
            image_layers=2,
            observation_layers=1,
            dynamics_layers=1,
            absolute=False,
            initializer="trunc_normal_in",
            output_scale=1.0,
        ),
        "encoder": EncoderConfig(
            depth=network.depth,
            multipliers=(2, 3, 4, 4),
            layers=3,
            units=network.model_dim,
            activation="silu",
            normalization="rms",
            initializer="trunc_normal_in",
            symlog=True,
            outer=False,
            kernel=5,
            strided=True,
        ),
        "decoder": DecoderConfig(
            depth=network.depth,
            multipliers=(2, 3, 4, 4),
            layers=3,
            units=network.model_dim,
            activation="silu",
            normalization="rms",
            output_scale=1.0,
            initializer="trunc_normal_in",
            outer=False,
            kernel=5,
            bias_space=8,
            strided=True,
            image_output="mse",
        ),
        "reward_head": HeadConfig(
            layers=1,
            units=network.model_dim,
            output="symexp_twohot",
            output_scale=0.0,
            bins=255,
        ),
        "continue_head": HeadConfig(
            layers=1,
            units=network.model_dim,
            output="binary",
            output_scale=1.0,
            bins=None,
        ),
        "policy": PolicyConfig(layers=3, units=network.model_dim),
        "value_head": HeadConfig(
            layers=3,
            units=network.model_dim,
            output="symexp_twohot",
            output_scale=0.0,
            bins=255,
        ),
        "optimizer": OptimizerConfig(beta2=0.99),
        "replay": ReplayConfig(),
        "run": RunConfig(steps=1_000_000, replay_ratio=ratio),
    }


def _upstream_current_components(
    mode: ObservationMode,
    model_size: ModelSize,
) -> dict[str, Any]:
    network = model_size.resolve()
    ratio = 256.0 if mode is ObservationMode.VISION else 1024.0
    return {
        "network": network,
        "rssm": RSSMConfig(
            deter=network.deter,
            hidden=network.model_dim,
            stoch=32,
            classes=network.classes,
            blocks=8,
            free_nats=1.0,
            unimix=0.01,
            activation="silu",
            normalization="rms",
            image_layers=2,
            observation_layers=1,
            dynamics_layers=1,
            absolute=False,
            initializer="trunc_normal_in",
            output_scale=1.0,
        ),
        "encoder": EncoderConfig(
            depth=network.depth,
            multipliers=(2, 3, 4, 4),
            layers=3,
            units=network.model_dim,
            activation="silu",
            normalization="rms",
            initializer="trunc_normal_in",
            symlog=True,
            outer=False,
            kernel=5,
            strided=False,
        ),
        "decoder": DecoderConfig(
            depth=network.depth,
            multipliers=(2, 3, 4, 4),
            layers=3,
            units=network.model_dim,
            activation="silu",
            normalization="rms",
            output_scale=1.0,
            initializer="trunc_normal_in",
            outer=False,
            kernel=5,
            bias_space=8,
            strided=False,
            image_output="mse",
        ),
        "reward_head": HeadConfig(
            layers=1,
            units=network.model_dim,
            output="symexp_twohot",
            output_scale=0.0,
            bins=255,
        ),
        "continue_head": HeadConfig(
            layers=1,
            units=network.model_dim,
            output="binary",
            output_scale=1.0,
            bins=None,
        ),
        "policy": PolicyConfig(layers=3, units=network.model_dim),
        "value_head": HeadConfig(
            layers=3,
            units=network.model_dim,
            output="symexp_twohot",
            output_scale=0.0,
            bins=255,
        ),
        "optimizer": OptimizerConfig(beta2=0.999),
        "replay": ReplayConfig(),
        "run": RunConfig(steps=1_100_000, replay_ratio=ratio),
    }


def resolve_dreamer_config(
    profile: DreamerProfile | str = DreamerProfile.PAPER,
    observation_mode: ObservationMode | str = ObservationMode.VISION,
    model_size: ModelSize | str | None = None,
) -> DreamerV3Config:
    resolved_profile = DreamerProfile(profile)
    resolved_mode = ObservationMode(observation_mode)
    expected_size = _default_model_size(resolved_mode)
    resolved_size = ModelSize(model_size) if model_size is not None else expected_size
    if resolved_size is not expected_size:
        raise ValueError(
            f"{resolved_mode.value} requires {expected_size.value} model, "
            f"not {resolved_size.value}"
        )
    components = (
        _paper_components(resolved_mode, resolved_size)
        if resolved_profile is DreamerProfile.PAPER
        else _upstream_current_components(resolved_mode, resolved_size)
    )
    return DreamerV3Config(
        profile=resolved_profile,
        observation_mode=resolved_mode,
        model_size=resolved_size,
        action_mode="continuous",
        **components,
    )


__all__ = [
    "ActorCriticConfig",
    "ContinueHeadConfig",
    "DecoderConfig",
    "DreamerProfile",
    "DreamerV3Config",
    "EncoderConfig",
    "HeadConfig",
    "ImaginationConfig",
    "LossScaleConfig",
    "ModelSize",
    "NetworkSize",
    "NormalizerConfig",
    "ObservationMode",
    "OptimizerConfig",
    "PolicyConfig",
    "RSSMConfig",
    "ReplayConfig",
    "RewardHeadConfig",
    "RunConfig",
    "SlowValueConfig",
    "resolve_dreamer_config",
]
