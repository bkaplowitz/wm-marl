from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, fields, replace
from enum import Enum, EnumType
from typing import Any, cast


def _reject_state_container_aliases(
    state: dict[object, object], record_name: str
) -> None:
    pending: list[object] = [state]
    seen: set[int] = set()
    while pending:
        value = pending.pop()
        if type(value) is dict:
            mapping = cast(dict[object, object], value)
            identity = id(mapping)
            if identity in seen:
                raise TypeError(f"{record_name} state contains a container alias")
            seen.add(identity)
            pending.extend(mapping.values())
        elif type(value) is list:
            sequence = cast(list[object], value)
            identity = id(sequence)
            if identity in seen:
                raise TypeError(f"{record_name} state contains a container alias")
            seen.add(identity)
            pending.extend(sequence)


def _require_state(
    state: object,
    expected_keys: tuple[str, ...],
    record_name: str,
) -> dict[str, Any]:
    if type(state) is not dict:
        raise TypeError(f"{record_name} state must be a plain dict")
    raw_state = cast(dict[object, object], state)
    _reject_state_container_aliases(raw_state, record_name)
    if any(type(key) is not str for key in raw_state):
        raise TypeError(f"{record_name} state keys must be built-in strings")
    if len(state) != len(expected_keys) or set(state) != set(expected_keys):
        raise ValueError(
            f"{record_name} state keys must be exactly {expected_keys}, got {tuple(state)}"
        )
    return state


def _state_int(value: object, name: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be a Python integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _state_float(
    value: object,
    name: str,
    *,
    minimum: float | None = None,
) -> float:
    if type(value) is not float:
        raise TypeError(f"{name} must be a Python float")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _state_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a Python bool")
    return value


def _state_str(value: object, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    return value


def _state_optional_int(value: object, name: str) -> int | None:
    return None if value is None else _state_int(value, name)


def _state_int_list(
    value: object,
    name: str,
    *,
    minimum: int = 1,
) -> tuple[int, ...]:
    if type(value) is not list:
        raise TypeError(f"{name} must be a plain list")
    return tuple(_state_int(item, f"{name} item", minimum=minimum) for item in value)


class _ExactStringEnumType(EnumType):
    def __call__(cls, value: object, *args: Any, **kwargs: Any) -> Any:
        if type(value) is not str and not isinstance(value, cls):
            raise TypeError(f"{cls.__name__} value must be an exact string or member")
        return super().__call__(value, *args, **kwargs)


class DreamerProfile(str, Enum, metaclass=_ExactStringEnumType):
    PAPER = "paper"
    UPSTREAM_CURRENT = "upstream-current"

    def state_dict(self) -> dict[str, str]:
        return {"value": self.value}

    @classmethod
    def from_state(cls, state: object) -> DreamerProfile:
        record = _require_state(state, ("value",), "DreamerProfile")
        return cls(_state_str(record["value"], "DreamerProfile value"))


class ObservationMode(str, Enum, metaclass=_ExactStringEnumType):
    VISION = "vision"
    PROPRIO = "proprio"

    def state_dict(self) -> dict[str, str]:
        return {"value": self.value}

    @classmethod
    def from_state(cls, state: object) -> ObservationMode:
        record = _require_state(state, ("value",), "ObservationMode")
        return cls(_state_str(record["value"], "ObservationMode value"))


class ModelSize(str, Enum, metaclass=_ExactStringEnumType):
    M1 = "1m"
    M12 = "12m"
    M25 = "25m"
    M50 = "50m"
    M100 = "100m"
    M200 = "200m"
    M400 = "400m"

    @classmethod
    def _missing_(cls, value: object) -> ModelSize | None:
        if type(value) is str:
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

    def state_dict(self) -> dict[str, str]:
        return {"value": self.value}

    @classmethod
    def from_state(cls, state: object) -> ModelSize:
        record = _require_state(state, ("value",), "ModelSize")
        return cls(_state_str(record["value"], "ModelSize value"))


@dataclass(frozen=True)
class NetworkSize:
    model_dim: int
    deter: int
    depth: int
    classes: int

    def __post_init__(self) -> None:
        for name in ("model_dim", "deter", "depth", "classes"):
            _state_int(getattr(self, name), name, minimum=1)

    @property
    def hidden(self) -> int:
        return self.model_dim

    @property
    def units(self) -> int:
        return self.model_dim

    def state_dict(self) -> dict[str, int]:
        return {
            "model_dim": self.model_dim,
            "deter": self.deter,
            "depth": self.depth,
            "classes": self.classes,
        }

    @classmethod
    def from_state(cls, state: object) -> NetworkSize:
        record = _require_state(
            state,
            ("model_dim", "deter", "depth", "classes"),
            "NetworkSize",
        )
        return cls(
            model_dim=_state_int(record["model_dim"], "model_dim", minimum=1),
            deter=_state_int(record["deter"], "deter", minimum=1),
            depth=_state_int(record["depth"], "depth", minimum=1),
            classes=_state_int(record["classes"], "classes", minimum=1),
        )


@dataclass(frozen=True)
class RSSMConfig:
    deter: int = 8192
    hidden: int = 1024
    stoch: int = 32
    classes: int = 64
    blocks: int = 8
    free_nats: float = 1.0
    unimix: float = 0.01
    activation: str = "silu"
    normalization: str = "rms"
    image_layers: int = 2
    observation_layers: int = 1
    dynamics_layers: int = 1
    absolute: bool = False
    initializer: str = "trunc_normal_in"
    output_scale: float = 1.0

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _state_int(self.deter, "deter", minimum=1)
        _state_int(self.hidden, "hidden", minimum=1)
        _state_int(self.stoch, "stoch", minimum=1)
        _state_int(self.classes, "classes", minimum=1)
        _state_int(self.blocks, "blocks", minimum=1)
        _state_float(self.free_nats, "free_nats", minimum=0.0)
        _state_float(self.unimix, "unimix", minimum=0.0)
        _state_str(self.activation, "activation")
        _state_str(self.normalization, "normalization")
        _state_int(self.image_layers, "image_layers", minimum=0)
        _state_int(self.observation_layers, "observation_layers", minimum=0)
        _state_int(self.dynamics_layers, "dynamics_layers", minimum=0)
        _state_bool(self.absolute, "absolute")
        _state_str(self.initializer, "initializer")
        _state_float(self.output_scale, "output_scale")
        if self.deter <= 0 or self.deter % 8:
            raise ValueError("deter must be positive and divisible by eight")
        if self.hidden <= 0:
            raise ValueError("hidden must be positive")
        if self.stoch <= 0:
            raise ValueError("stoch must be positive")
        if self.classes <= 0:
            raise ValueError("classes must be positive")
        if self.blocks <= 0:
            raise ValueError("blocks must be positive")
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

    def state_dict(self) -> dict[str, Any]:
        return {
            "deter": self.deter,
            "hidden": self.hidden,
            "stoch": self.stoch,
            "classes": self.classes,
            "blocks": self.blocks,
            "free_nats": self.free_nats,
            "unimix": self.unimix,
            "activation": self.activation,
            "normalization": self.normalization,
            "image_layers": self.image_layers,
            "observation_layers": self.observation_layers,
            "dynamics_layers": self.dynamics_layers,
            "absolute": self.absolute,
            "initializer": self.initializer,
            "output_scale": self.output_scale,
        }

    @classmethod
    def from_state(cls, state: object) -> RSSMConfig:
        keys = (
            "deter",
            "hidden",
            "stoch",
            "classes",
            "blocks",
            "free_nats",
            "unimix",
            "activation",
            "normalization",
            "image_layers",
            "observation_layers",
            "dynamics_layers",
            "absolute",
            "initializer",
            "output_scale",
        )
        record = _require_state(state, keys, "RSSMConfig")
        return cls(
            deter=_state_int(record["deter"], "deter", minimum=1),
            hidden=_state_int(record["hidden"], "hidden", minimum=1),
            stoch=_state_int(record["stoch"], "stoch", minimum=1),
            classes=_state_int(record["classes"], "classes", minimum=1),
            blocks=_state_int(record["blocks"], "blocks", minimum=1),
            free_nats=_state_float(record["free_nats"], "free_nats", minimum=0.0),
            unimix=_state_float(record["unimix"], "unimix", minimum=0.0),
            activation=_state_str(record["activation"], "activation"),
            normalization=_state_str(record["normalization"], "normalization"),
            image_layers=_state_int(record["image_layers"], "image_layers", minimum=0),
            observation_layers=_state_int(
                record["observation_layers"], "observation_layers", minimum=0
            ),
            dynamics_layers=_state_int(
                record["dynamics_layers"], "dynamics_layers", minimum=0
            ),
            absolute=_state_bool(record["absolute"], "absolute"),
            initializer=_state_str(record["initializer"], "initializer"),
            output_scale=_state_float(record["output_scale"], "output_scale"),
        )


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
        _state_int(self.depth, "depth", minimum=1)
        if type(self.multipliers) is not tuple:
            raise TypeError("multipliers must be a tuple")
        for value in self.multipliers:
            _state_int(value, "multipliers item", minimum=1)
        _state_int(self.layers, "layers", minimum=0)
        _state_int(self.units, "units", minimum=1)
        _state_str(self.activation, "activation")
        _state_str(self.normalization, "normalization")
        _state_str(self.initializer, "initializer")
        _state_bool(self.symlog, "symlog")
        _state_bool(self.outer, "outer")
        _state_int(self.kernel, "kernel", minimum=1)
        _state_bool(self.strided, "strided")
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

    def state_dict(self) -> dict[str, Any]:
        return {
            "depth": self.depth,
            "multipliers": list(self.multipliers),
            "layers": self.layers,
            "units": self.units,
            "activation": self.activation,
            "normalization": self.normalization,
            "initializer": self.initializer,
            "symlog": self.symlog,
            "outer": self.outer,
            "kernel": self.kernel,
            "strided": self.strided,
        }

    @classmethod
    def from_state(cls, state: object) -> EncoderConfig:
        keys = (
            "depth",
            "multipliers",
            "layers",
            "units",
            "activation",
            "normalization",
            "initializer",
            "symlog",
            "outer",
            "kernel",
            "strided",
        )
        record = _require_state(state, keys, "EncoderConfig")
        return cls(
            depth=_state_int(record["depth"], "depth", minimum=1),
            multipliers=_state_int_list(record["multipliers"], "multipliers"),
            layers=_state_int(record["layers"], "layers", minimum=0),
            units=_state_int(record["units"], "units", minimum=1),
            activation=_state_str(record["activation"], "activation"),
            normalization=_state_str(record["normalization"], "normalization"),
            initializer=_state_str(record["initializer"], "initializer"),
            symlog=_state_bool(record["symlog"], "symlog"),
            outer=_state_bool(record["outer"], "outer"),
            kernel=_state_int(record["kernel"], "kernel", minimum=1),
            strided=_state_bool(record["strided"], "strided"),
        )


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
        _state_int(self.depth, "depth", minimum=1)
        if type(self.multipliers) is not tuple:
            raise TypeError("multipliers must be a tuple")
        for value in self.multipliers:
            _state_int(value, "multipliers item", minimum=1)
        _state_int(self.layers, "layers", minimum=0)
        _state_int(self.units, "units", minimum=1)
        _state_str(self.activation, "activation")
        _state_str(self.normalization, "normalization")
        _state_float(self.output_scale, "output_scale")
        _state_str(self.initializer, "initializer")
        _state_bool(self.outer, "outer")
        _state_int(self.kernel, "kernel", minimum=1)
        _state_int(self.bias_space, "bias_space", minimum=1)
        _state_bool(self.strided, "strided")
        _state_str(self.image_output, "image_output")
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

    def state_dict(self) -> dict[str, Any]:
        return {
            "depth": self.depth,
            "multipliers": list(self.multipliers),
            "layers": self.layers,
            "units": self.units,
            "activation": self.activation,
            "normalization": self.normalization,
            "output_scale": self.output_scale,
            "initializer": self.initializer,
            "outer": self.outer,
            "kernel": self.kernel,
            "bias_space": self.bias_space,
            "strided": self.strided,
            "image_output": self.image_output,
        }

    @classmethod
    def from_state(cls, state: object) -> DecoderConfig:
        keys = (
            "depth",
            "multipliers",
            "layers",
            "units",
            "activation",
            "normalization",
            "output_scale",
            "initializer",
            "outer",
            "kernel",
            "bias_space",
            "strided",
            "image_output",
        )
        record = _require_state(state, keys, "DecoderConfig")
        return cls(
            depth=_state_int(record["depth"], "depth", minimum=1),
            multipliers=_state_int_list(record["multipliers"], "multipliers"),
            layers=_state_int(record["layers"], "layers", minimum=0),
            units=_state_int(record["units"], "units", minimum=1),
            activation=_state_str(record["activation"], "activation"),
            normalization=_state_str(record["normalization"], "normalization"),
            output_scale=_state_float(record["output_scale"], "output_scale"),
            initializer=_state_str(record["initializer"], "initializer"),
            outer=_state_bool(record["outer"], "outer"),
            kernel=_state_int(record["kernel"], "kernel", minimum=1),
            bias_space=_state_int(record["bias_space"], "bias_space", minimum=1),
            strided=_state_bool(record["strided"], "strided"),
            image_output=_state_str(record["image_output"], "image_output"),
        )


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
        _state_int(self.layers, "layers", minimum=0)
        _state_int(self.units, "units", minimum=1)
        _state_str(self.activation, "activation")
        _state_str(self.normalization, "normalization")
        _state_str(self.output, "output")
        _state_float(self.output_scale, "output_scale")
        _state_str(self.initializer, "initializer")
        if self.bins is not None:
            _state_int(self.bins, "bins")
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

    def state_dict(self) -> dict[str, Any]:
        return {
            "layers": self.layers,
            "units": self.units,
            "activation": self.activation,
            "normalization": self.normalization,
            "output": self.output,
            "output_scale": self.output_scale,
            "initializer": self.initializer,
            "bins": self.bins,
        }

    @classmethod
    def from_state(cls, state: object) -> HeadConfig:
        keys = (
            "layers",
            "units",
            "activation",
            "normalization",
            "output",
            "output_scale",
            "initializer",
            "bins",
        )
        record = _require_state(state, keys, cls.__name__)
        return cls(
            layers=_state_int(record["layers"], "layers", minimum=0),
            units=_state_int(record["units"], "units", minimum=1),
            activation=_state_str(record["activation"], "activation"),
            normalization=_state_str(record["normalization"], "normalization"),
            output=_state_str(record["output"], "output"),
            output_scale=_state_float(record["output_scale"], "output_scale"),
            initializer=_state_str(record["initializer"], "initializer"),
            bins=_state_optional_int(record["bins"], "bins"),
        )


@dataclass(frozen=True)
class RewardHeadConfig(HeadConfig):
    pass


@dataclass(frozen=True)
class ContinueHeadConfig(HeadConfig):
    output: str = "binary"
    output_scale: float = 1.0
    bins: int | None = None


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
        _state_int(self.layers, "layers", minimum=0)
        _state_int(self.units, "units", minimum=1)
        _state_str(self.activation, "activation")
        _state_str(self.normalization, "normalization")
        _state_float(self.min_std, "min_std", minimum=0.0)
        _state_float(self.max_std, "max_std", minimum=0.0)
        _state_float(self.output_scale, "output_scale")
        _state_float(self.unimix, "unimix", minimum=0.0)
        _state_str(self.initializer, "initializer")
        _state_str(self.discrete, "discrete")
        _state_str(self.continuous, "continuous")
        if self.layers < 0 or self.units <= 0:
            raise ValueError("policy dimensions must be positive")
        if not 0.0 < self.min_std <= self.max_std:
            raise ValueError("policy std bounds are invalid")
        if not 0.0 <= self.unimix < 1.0:
            raise ValueError("policy unimix must be in [0, 1)")

    def state_dict(self) -> dict[str, Any]:
        return {
            "layers": self.layers,
            "units": self.units,
            "activation": self.activation,
            "normalization": self.normalization,
            "min_std": self.min_std,
            "max_std": self.max_std,
            "output_scale": self.output_scale,
            "unimix": self.unimix,
            "initializer": self.initializer,
            "discrete": self.discrete,
            "continuous": self.continuous,
        }

    @classmethod
    def from_state(cls, state: object) -> PolicyConfig:
        keys = (
            "layers",
            "units",
            "activation",
            "normalization",
            "min_std",
            "max_std",
            "output_scale",
            "unimix",
            "initializer",
            "discrete",
            "continuous",
        )
        record = _require_state(state, keys, "PolicyConfig")
        return cls(
            layers=_state_int(record["layers"], "layers", minimum=0),
            units=_state_int(record["units"], "units", minimum=1),
            activation=_state_str(record["activation"], "activation"),
            normalization=_state_str(record["normalization"], "normalization"),
            min_std=_state_float(record["min_std"], "min_std", minimum=0.0),
            max_std=_state_float(record["max_std"], "max_std", minimum=0.0),
            output_scale=_state_float(record["output_scale"], "output_scale"),
            unimix=_state_float(record["unimix"], "unimix", minimum=0.0),
            initializer=_state_str(record["initializer"], "initializer"),
            discrete=_state_str(record["discrete"], "discrete"),
            continuous=_state_str(record["continuous"], "continuous"),
        )


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
        _state_float(self.learning_rate, "learning_rate", minimum=0.0)
        _state_float(self.agc, "agc", minimum=0.0)
        _state_float(self.agc_floor, "agc_floor", minimum=0.0)
        _state_float(self.epsilon, "epsilon", minimum=0.0)
        _state_float(self.beta1, "beta1", minimum=0.0)
        _state_float(self.beta2, "beta2", minimum=0.0)
        _state_bool(self.momentum, "momentum")
        _state_float(self.weight_decay, "weight_decay", minimum=0.0)
        _state_str(self.schedule, "schedule")
        _state_int(self.warmup, "warmup", minimum=0)
        _state_int(self.anneal, "anneal", minimum=0)
        if self.learning_rate <= 0 or self.agc < 0 or self.agc_floor <= 0:
            raise ValueError("optimizer scales are invalid")
        if self.epsilon <= 0:
            raise ValueError("optimizer epsilon must be positive")
        if not 0.0 <= self.beta1 < 1.0 or not 0.0 <= self.beta2 < 1.0:
            raise ValueError("optimizer betas must be in [0, 1)")
        if self.weight_decay < 0 or self.warmup < 0 or self.anneal < 0:
            raise ValueError("optimizer schedule values are invalid")

    def state_dict(self) -> dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "agc": self.agc,
            "agc_floor": self.agc_floor,
            "epsilon": self.epsilon,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "schedule": self.schedule,
            "warmup": self.warmup,
            "anneal": self.anneal,
        }

    @classmethod
    def from_state(cls, state: object) -> OptimizerConfig:
        keys = (
            "learning_rate",
            "agc",
            "agc_floor",
            "epsilon",
            "beta1",
            "beta2",
            "momentum",
            "weight_decay",
            "schedule",
            "warmup",
            "anneal",
        )
        record = _require_state(state, keys, "OptimizerConfig")
        return cls(
            learning_rate=_state_float(
                record["learning_rate"], "learning_rate", minimum=0.0
            ),
            agc=_state_float(record["agc"], "agc", minimum=0.0),
            agc_floor=_state_float(record["agc_floor"], "agc_floor", minimum=0.0),
            epsilon=_state_float(record["epsilon"], "epsilon", minimum=0.0),
            beta1=_state_float(record["beta1"], "beta1", minimum=0.0),
            beta2=_state_float(record["beta2"], "beta2", minimum=0.0),
            momentum=_state_bool(record["momentum"], "momentum"),
            weight_decay=_state_float(
                record["weight_decay"], "weight_decay", minimum=0.0
            ),
            schedule=_state_str(record["schedule"], "schedule"),
            warmup=_state_int(record["warmup"], "warmup", minimum=0),
            anneal=_state_int(record["anneal"], "anneal", minimum=0),
        )


@dataclass(frozen=True)
class SequenceShapeConfig:
    batch_size: int = 16
    sequence_length: int = 64
    context: int = 1
    consecutive: int = 1
    report_length: int = 32
    report_consecutive: int = 1

    def __post_init__(self) -> None:
        _state_int(self.batch_size, "batch_size", minimum=1)
        _state_int(self.sequence_length, "sequence_length", minimum=1)
        _state_int(self.context, "context", minimum=0)
        _state_int(self.consecutive, "consecutive", minimum=1)
        _state_int(self.report_length, "report_length", minimum=1)
        _state_int(self.report_consecutive, "report_consecutive", minimum=1)

    @property
    def raw_length(self) -> int:
        return self.context + self.sequence_length * self.consecutive

    @property
    def report_raw_length(self) -> int:
        return self.context + self.report_length * self.report_consecutive

    def state_dict(self) -> dict[str, int]:
        return {
            "batch_size": self.batch_size,
            "sequence_length": self.sequence_length,
            "context": self.context,
            "consecutive": self.consecutive,
            "report_length": self.report_length,
            "report_consecutive": self.report_consecutive,
        }

    @classmethod
    def from_state(cls, state: object) -> SequenceShapeConfig:
        record = _require_state(
            state,
            (
                "batch_size",
                "sequence_length",
                "context",
                "consecutive",
                "report_length",
                "report_consecutive",
            ),
            "SequenceShapeConfig",
        )
        return cls(
            batch_size=_state_int(record["batch_size"], "batch_size", minimum=1),
            sequence_length=_state_int(
                record["sequence_length"], "sequence_length", minimum=1
            ),
            context=_state_int(record["context"], "context", minimum=0),
            consecutive=_state_int(record["consecutive"], "consecutive", minimum=1),
            report_length=_state_int(
                record["report_length"], "report_length", minimum=1
            ),
            report_consecutive=_state_int(
                record["report_consecutive"], "report_consecutive", minimum=1
            ),
        )


@dataclass(frozen=True)
class ReplayConfig:
    capacity: int = 5_000_000
    chunk_size: int = 1024
    online_queue_size: int = 16

    def __post_init__(self) -> None:
        _state_int(self.capacity, "capacity", minimum=1)
        _state_int(self.chunk_size, "chunk_size", minimum=1)
        _state_int(self.online_queue_size, "online_queue_size", minimum=1)
        if self.capacity <= 0 or self.chunk_size <= 0 or self.online_queue_size <= 0:
            raise ValueError("replay sizes must be positive")

    def state_dict(self) -> dict[str, int]:
        return {
            "capacity": self.capacity,
            "chunk_size": self.chunk_size,
            "online_queue_size": self.online_queue_size,
        }

    @classmethod
    def from_state(cls, state: object) -> ReplayConfig:
        keys = ("capacity", "chunk_size", "online_queue_size")
        record = _require_state(state, keys, "ReplayConfig")
        return cls(
            capacity=_state_int(record["capacity"], "capacity", minimum=1),
            chunk_size=_state_int(record["chunk_size"], "chunk_size", minimum=1),
            online_queue_size=_state_int(
                record["online_queue_size"], "online_queue_size", minimum=1
            ),
        )


@dataclass(frozen=True)
class RunConfig:
    env_steps: int = 1_000_000
    num_envs: int = 16
    eval_envs: int = 4
    train_ratio: float = 256.0
    eval_every: int = 100_000
    eval_episodes: int = 1
    report_every: int = 10_000
    log_every: int = 1_000
    checkpoint_every: int = 100_000
    report_batches: int = 1

    def __post_init__(self) -> None:
        for name in (
            "env_steps",
            "num_envs",
            "eval_envs",
            "eval_every",
            "eval_episodes",
            "report_every",
            "log_every",
            "checkpoint_every",
            "report_batches",
        ):
            _state_int(getattr(self, name), name, minimum=1)
        _state_float(self.train_ratio, "train_ratio", minimum=0.0)
        if self.train_ratio == 0.0:
            raise ValueError("train_ratio must be positive")

    def state_dict(self) -> dict[str, Any]:
        return {
            "env_steps": self.env_steps,
            "num_envs": self.num_envs,
            "eval_envs": self.eval_envs,
            "train_ratio": self.train_ratio,
            "eval_every": self.eval_every,
            "eval_episodes": self.eval_episodes,
            "report_every": self.report_every,
            "log_every": self.log_every,
            "checkpoint_every": self.checkpoint_every,
            "report_batches": self.report_batches,
        }

    @classmethod
    def from_state(cls, state: object) -> RunConfig:
        keys = (
            "env_steps",
            "num_envs",
            "eval_envs",
            "train_ratio",
            "eval_every",
            "eval_episodes",
            "report_every",
            "log_every",
            "checkpoint_every",
            "report_batches",
        )
        record = _require_state(state, keys, "RunConfig")
        return cls(
            env_steps=_state_int(record["env_steps"], "env_steps", minimum=1),
            num_envs=_state_int(record["num_envs"], "num_envs", minimum=1),
            eval_envs=_state_int(record["eval_envs"], "eval_envs", minimum=1),
            train_ratio=_state_float(record["train_ratio"], "train_ratio", minimum=0.0),
            eval_every=_state_int(record["eval_every"], "eval_every", minimum=1),
            eval_episodes=_state_int(
                record["eval_episodes"], "eval_episodes", minimum=1
            ),
            report_every=_state_int(record["report_every"], "report_every", minimum=1),
            log_every=_state_int(record["log_every"], "log_every", minimum=1),
            checkpoint_every=_state_int(
                record["checkpoint_every"], "checkpoint_every", minimum=1
            ),
            report_batches=_state_int(
                record["report_batches"], "report_batches", minimum=1
            ),
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
        for name, value in self.as_tuple():
            _state_float(value, name, minimum=0.0)
        if any(value < 0 for _, value in self.as_tuple()):
            raise ValueError("loss scales must be nonnegative")

    def as_tuple(self) -> tuple[tuple[str, float], ...]:
        return tuple((item.name, getattr(self, item.name)) for item in fields(self))

    def state_dict(self) -> dict[str, float]:
        return {name: value for name, value in self.as_tuple()}

    @classmethod
    def from_state(cls, state: object) -> LossScaleConfig:
        keys = ("rec", "rew", "con", "dyn", "rep", "policy", "value", "repval")
        record = _require_state(state, keys, "LossScaleConfig")
        return cls(**{key: _state_float(record[key], key, minimum=0.0) for key in keys})


@dataclass(frozen=True)
class ImaginationConfig:
    length: int = 15
    last: int = 0
    horizon: int = 333
    continuation_discount: bool = True
    lambda_: float = 0.95
    actor_entropy: float = 3e-4
    imagination_slow_target: bool = False
    replay_slow_target: bool = False
    slow_regularizer: float = 1.0
    ac_grads: bool = False
    reward_grad: bool = True
    repval_loss: bool = True
    repval_grad: bool = True

    def __post_init__(self) -> None:
        _state_int(self.length, "length", minimum=1)
        _state_int(self.last, "last", minimum=0)
        _state_int(self.horizon, "horizon", minimum=1)
        _state_bool(self.continuation_discount, "continuation_discount")
        _state_float(self.lambda_, "lambda_", minimum=0.0)
        _state_float(self.actor_entropy, "actor_entropy", minimum=0.0)
        _state_bool(self.imagination_slow_target, "imagination_slow_target")
        _state_bool(self.replay_slow_target, "replay_slow_target")
        _state_float(self.slow_regularizer, "slow_regularizer", minimum=0.0)
        _state_bool(self.ac_grads, "ac_grads")
        _state_bool(self.reward_grad, "reward_grad")
        _state_bool(self.repval_loss, "repval_loss")
        _state_bool(self.repval_grad, "repval_grad")
        if self.length <= 0 or self.last < 0 or self.horizon <= 0:
            raise ValueError("imagination lengths must be positive")
        if not 0.0 <= self.lambda_ <= 1.0 or self.actor_entropy < 0:
            raise ValueError("imagination objective values are invalid")

    def state_dict(self) -> dict[str, Any]:
        return {
            "length": self.length,
            "last": self.last,
            "horizon": self.horizon,
            "continuation_discount": self.continuation_discount,
            "lambda_": self.lambda_,
            "actor_entropy": self.actor_entropy,
            "imagination_slow_target": self.imagination_slow_target,
            "replay_slow_target": self.replay_slow_target,
            "slow_regularizer": self.slow_regularizer,
            "ac_grads": self.ac_grads,
            "reward_grad": self.reward_grad,
            "repval_loss": self.repval_loss,
            "repval_grad": self.repval_grad,
        }

    @classmethod
    def from_state(cls, state: object) -> ImaginationConfig:
        keys = (
            "length",
            "last",
            "horizon",
            "continuation_discount",
            "lambda_",
            "actor_entropy",
            "imagination_slow_target",
            "replay_slow_target",
            "slow_regularizer",
            "ac_grads",
            "reward_grad",
            "repval_loss",
            "repval_grad",
        )
        record = _require_state(state, keys, "ImaginationConfig")
        return cls(
            length=_state_int(record["length"], "length", minimum=1),
            last=_state_int(record["last"], "last", minimum=0),
            horizon=_state_int(record["horizon"], "horizon", minimum=1),
            continuation_discount=_state_bool(
                record["continuation_discount"], "continuation_discount"
            ),
            lambda_=_state_float(record["lambda_"], "lambda_", minimum=0.0),
            actor_entropy=_state_float(
                record["actor_entropy"], "actor_entropy", minimum=0.0
            ),
            imagination_slow_target=_state_bool(
                record["imagination_slow_target"], "imagination_slow_target"
            ),
            replay_slow_target=_state_bool(
                record["replay_slow_target"], "replay_slow_target"
            ),
            slow_regularizer=_state_float(
                record["slow_regularizer"], "slow_regularizer", minimum=0.0
            ),
            ac_grads=_state_bool(record["ac_grads"], "ac_grads"),
            reward_grad=_state_bool(record["reward_grad"], "reward_grad"),
            repval_loss=_state_bool(record["repval_loss"], "repval_loss"),
            repval_grad=_state_bool(record["repval_grad"], "repval_grad"),
        )


@dataclass(frozen=True)
class SlowValueConfig:
    rate: float = 0.02
    every: int = 1

    def __post_init__(self) -> None:
        _state_float(self.rate, "rate", minimum=0.0)
        _state_int(self.every, "every", minimum=1)
        if not 0.0 < self.rate <= 1.0 or self.every <= 0:
            raise ValueError("slow value update settings are invalid")

    def state_dict(self) -> dict[str, float | int]:
        return {"rate": self.rate, "every": self.every}

    @classmethod
    def from_state(cls, state: object) -> SlowValueConfig:
        record = _require_state(state, ("rate", "every"), "SlowValueConfig")
        return cls(
            rate=_state_float(record["rate"], "rate", minimum=0.0),
            every=_state_int(record["every"], "every", minimum=1),
        )


@dataclass(frozen=True)
class NormalizerConfig:
    implementation: str = "percentile"
    rate: float = 0.01
    limit: float = 1.0
    low_percentile: float = 5.0
    high_percentile: float = 95.0
    debias: bool = True

    def __post_init__(self) -> None:
        _state_str(self.implementation, "implementation")
        _state_float(self.rate, "rate", minimum=0.0)
        _state_float(self.limit, "limit", minimum=0.0)
        _state_float(self.low_percentile, "low_percentile", minimum=0.0)
        _state_float(self.high_percentile, "high_percentile", minimum=0.0)
        _state_bool(self.debias, "debias")
        if not 0.0 < self.rate <= 1.0 or self.limit < 0:
            raise ValueError("normalizer rate and limit are invalid")
        if not 0.0 <= self.low_percentile < self.high_percentile <= 100.0:
            raise ValueError("normalizer percentiles are invalid")

    def state_dict(self) -> dict[str, Any]:
        return {
            "implementation": self.implementation,
            "rate": self.rate,
            "limit": self.limit,
            "low_percentile": self.low_percentile,
            "high_percentile": self.high_percentile,
            "debias": self.debias,
        }

    @classmethod
    def from_state(cls, state: object) -> NormalizerConfig:
        keys = (
            "implementation",
            "rate",
            "limit",
            "low_percentile",
            "high_percentile",
            "debias",
        )
        record = _require_state(state, keys, "NormalizerConfig")
        return cls(
            implementation=_state_str(record["implementation"], "implementation"),
            rate=_state_float(record["rate"], "rate", minimum=0.0),
            limit=_state_float(record["limit"], "limit", minimum=0.0),
            low_percentile=_state_float(
                record["low_percentile"], "low_percentile", minimum=0.0
            ),
            high_percentile=_state_float(
                record["high_percentile"], "high_percentile", minimum=0.0
            ),
            debias=_state_bool(record["debias"], "debias"),
        )


@dataclass(frozen=True)
class RuntimeOverrides:
    env_steps: int | None = None
    num_envs: int | None = None
    batch_size: int | None = None
    batch_length: int | None = None
    train_ratio: float | None = None
    eval_every: int | None = None
    eval_episodes: int | None = None
    report_every: int | None = None
    checkpoint_every: int | None = None
    camera: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "env_steps",
            "num_envs",
            "batch_size",
            "batch_length",
            "eval_every",
            "eval_episodes",
            "report_every",
            "checkpoint_every",
        ):
            value = getattr(self, name)
            if value is not None:
                _state_int(value, name, minimum=1)
        if self.train_ratio is not None:
            _state_float(self.train_ratio, "train_ratio", minimum=0.0)
            if self.train_ratio == 0.0:
                raise ValueError("train_ratio must be positive")
        if self.camera is not None:
            _state_int(self.camera, "camera")

    def state_dict(self) -> dict[str, int | float | None]:
        return {
            "env_steps": self.env_steps,
            "num_envs": self.num_envs,
            "batch_size": self.batch_size,
            "batch_length": self.batch_length,
            "train_ratio": self.train_ratio,
            "eval_every": self.eval_every,
            "eval_episodes": self.eval_episodes,
            "report_every": self.report_every,
            "checkpoint_every": self.checkpoint_every,
            "camera": self.camera,
        }

    @classmethod
    def from_state(cls, state: object) -> RuntimeOverrides:
        keys = (
            "env_steps",
            "num_envs",
            "batch_size",
            "batch_length",
            "train_ratio",
            "eval_every",
            "eval_episodes",
            "report_every",
            "checkpoint_every",
            "camera",
        )
        record = _require_state(state, keys, "RuntimeOverrides")
        return cls(
            env_steps=_state_optional_int(record["env_steps"], "env_steps"),
            num_envs=_state_optional_int(record["num_envs"], "num_envs"),
            batch_size=_state_optional_int(record["batch_size"], "batch_size"),
            batch_length=_state_optional_int(record["batch_length"], "batch_length"),
            train_ratio=(
                None
                if record["train_ratio"] is None
                else _state_float(record["train_ratio"], "train_ratio", minimum=0.0)
            ),
            eval_every=_state_optional_int(record["eval_every"], "eval_every"),
            eval_episodes=_state_optional_int(record["eval_episodes"], "eval_episodes"),
            report_every=_state_optional_int(record["report_every"], "report_every"),
            checkpoint_every=_state_optional_int(
                record["checkpoint_every"], "checkpoint_every"
            ),
            camera=_state_optional_int(record["camera"], "camera"),
        )

    def algorithm_state(self) -> dict[str, int | float]:
        return {
            name: value
            for name, value in sorted(self.state_dict().items())
            if name != "camera" and value is not None
        }

    def environment_state(self) -> dict[str, int]:
        return {} if self.camera is None else {"camera": self.camera}


def _debug_local_snapshot_state() -> dict[str, Any]:
    return {
        "name": "debug-local-v1",
        "model": "debug-local-v1",
        "mlp_layers": 1,
        "mlp_units": 32,
        "rssm_deter": 32,
        "rssm_stoch": 4,
        "rssm_classes": 4,
        "vision_depths": [8, 16, 32, 64],
        "sequence": {
            "batch_size": 1,
            "sequence_length": 4,
            "context": 0,
            "consecutive": 1,
            "report_length": 4,
            "report_consecutive": 1,
        },
        "replay": {
            "capacity": 256,
            "chunk_size": 32,
            "online_queue_size": 16,
        },
        "run": {
            "env_steps": 48,
            "num_envs": 1,
            "eval_envs": 1,
            "train_ratio": 4.0,
            "eval_every": 16,
            "eval_episodes": 1,
            "report_every": 16,
            "log_every": 16,
            "checkpoint_every": 16,
            "report_batches": 1,
        },
        "imagination_horizon": 5,
        "platform": "cpu",
        "preallocate": False,
    }


@dataclass(frozen=True)
class DebugSnapshot:
    name: str
    model: str
    mlp_layers: int
    mlp_units: int
    rssm_deter: int
    rssm_stoch: int
    rssm_classes: int
    vision_depths: tuple[int, ...]
    sequence: SequenceShapeConfig
    replay: ReplayConfig
    run: RunConfig
    imagination_horizon: int
    platform: str
    preallocate: bool

    def __post_init__(self) -> None:
        _state_str(self.name, "name")
        _state_str(self.model, "model")
        if self.name != "debug-local-v1" or self.model != "debug-local-v1":
            raise ValueError("debug snapshot must be debug-local-v1")
        for name in (
            "mlp_layers",
            "mlp_units",
            "rssm_deter",
            "rssm_stoch",
            "rssm_classes",
            "imagination_horizon",
        ):
            _state_int(getattr(self, name), name, minimum=1)
        if type(self.vision_depths) is not tuple:
            raise TypeError("vision_depths must be a tuple")
        for value in self.vision_depths:
            _state_int(value, "vision_depths item", minimum=1)
        if type(self.sequence) is not SequenceShapeConfig:
            raise TypeError("sequence must be SequenceShapeConfig")
        if type(self.replay) is not ReplayConfig:
            raise TypeError("replay must be ReplayConfig")
        if type(self.run) is not RunConfig:
            raise TypeError("run must be RunConfig")
        if self.vision_depths != (8, 16, 32, 64):
            raise ValueError("debug vision depths must be (8, 16, 32, 64)")
        _state_str(self.platform, "platform")
        _state_bool(self.preallocate, "preallocate")
        expected = _debug_local_snapshot_state()
        if self.state_dict() != expected:
            raise ValueError("debug snapshot does not match debug-local-v1")

    def state_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model,
            "mlp_layers": self.mlp_layers,
            "mlp_units": self.mlp_units,
            "rssm_deter": self.rssm_deter,
            "rssm_stoch": self.rssm_stoch,
            "rssm_classes": self.rssm_classes,
            "vision_depths": list(self.vision_depths),
            "sequence": self.sequence.state_dict(),
            "replay": self.replay.state_dict(),
            "run": self.run.state_dict(),
            "imagination_horizon": self.imagination_horizon,
            "platform": self.platform,
            "preallocate": self.preallocate,
        }

    @classmethod
    def from_state(cls, state: object) -> DebugSnapshot:
        keys = (
            "name",
            "model",
            "mlp_layers",
            "mlp_units",
            "rssm_deter",
            "rssm_stoch",
            "rssm_classes",
            "vision_depths",
            "sequence",
            "replay",
            "run",
            "imagination_horizon",
            "platform",
            "preallocate",
        )
        record = _require_state(state, keys, "DebugSnapshot")
        return cls(
            name=_state_str(record["name"], "name"),
            model=_state_str(record["model"], "model"),
            mlp_layers=_state_int(record["mlp_layers"], "mlp_layers", minimum=1),
            mlp_units=_state_int(record["mlp_units"], "mlp_units", minimum=1),
            rssm_deter=_state_int(record["rssm_deter"], "rssm_deter", minimum=1),
            rssm_stoch=_state_int(record["rssm_stoch"], "rssm_stoch", minimum=1),
            rssm_classes=_state_int(record["rssm_classes"], "rssm_classes", minimum=1),
            vision_depths=_state_int_list(record["vision_depths"], "vision_depths"),
            sequence=SequenceShapeConfig.from_state(record["sequence"]),
            replay=ReplayConfig.from_state(record["replay"]),
            run=RunConfig.from_state(record["run"]),
            imagination_horizon=_state_int(
                record["imagination_horizon"], "imagination_horizon", minimum=1
            ),
            platform=_state_str(record["platform"], "platform"),
            preallocate=_state_bool(record["preallocate"], "preallocate"),
        )


@dataclass(frozen=True)
class DreamerV3Config:
    profile: DreamerProfile
    observation_mode: ObservationMode
    task: str
    seed: int
    model: str
    network: NetworkSize
    rssm: RSSMConfig
    encoder: EncoderConfig
    decoder: DecoderConfig
    reward_head: HeadConfig
    continue_head: HeadConfig
    policy: PolicyConfig
    value_head: HeadConfig
    optimizer: OptimizerConfig
    sequence: SequenceShapeConfig
    replay: ReplayConfig
    run: RunConfig
    loss_scales: LossScaleConfig
    imagination: ImaginationConfig
    slow_value: SlowValueConfig
    return_normalizer: NormalizerConfig
    value_normalizer: NormalizerConfig
    advantage_normalizer: NormalizerConfig
    action_repeat: int = 1
    image_size: tuple[int, int] = (64, 64)
    platform: str = "cuda"
    compute_dtype: str = "bfloat16"
    preallocate: bool = True

    def __post_init__(self) -> None:
        profile_value = self.profile
        if type(profile_value) is DreamerProfile:
            profile = profile_value
        elif type(profile_value) is str:
            profile = DreamerProfile(profile_value)
        else:
            raise TypeError("profile must be DreamerProfile or string")
        mode_value = self.observation_mode
        if type(mode_value) is ObservationMode:
            mode = mode_value
        elif type(mode_value) is str:
            mode = ObservationMode(mode_value)
        else:
            raise TypeError("observation_mode must be ObservationMode or string")
        object.__setattr__(self, "profile", profile)
        object.__setattr__(self, "observation_mode", mode)
        if type(self.task) is not str:
            raise TypeError("task must be a string")
        if not self.task:
            raise ValueError("task must be a nonempty string")
        _state_int(self.seed, "seed", minimum=0)
        if self.seed > 2**32 - 1 - 10_000:
            raise ValueError("seed exceeds the supported public seed range")
        _state_str(self.model, "model")
        expected_types = (
            ("network", NetworkSize),
            ("rssm", RSSMConfig),
            ("encoder", EncoderConfig),
            ("decoder", DecoderConfig),
            ("reward_head", HeadConfig),
            ("continue_head", HeadConfig),
            ("policy", PolicyConfig),
            ("value_head", HeadConfig),
            ("optimizer", OptimizerConfig),
            ("sequence", SequenceShapeConfig),
            ("replay", ReplayConfig),
            ("run", RunConfig),
            ("loss_scales", LossScaleConfig),
            ("imagination", ImaginationConfig),
            ("slow_value", SlowValueConfig),
            ("return_normalizer", NormalizerConfig),
            ("value_normalizer", NormalizerConfig),
            ("advantage_normalizer", NormalizerConfig),
        )
        for name, expected_type in expected_types:
            if type(getattr(self, name)) is not expected_type:
                raise TypeError(f"{name} must be {expected_type.__name__}")
        _state_int(self.action_repeat, "action_repeat", minimum=1)
        if type(self.image_size) is not tuple:
            raise TypeError("image_size must be a tuple")
        if len(self.image_size) != 2:
            raise ValueError("image_size must contain two dimensions")
        for value in self.image_size:
            _state_int(value, "image_size item", minimum=1)
        _state_str(self.platform, "platform")
        _state_str(self.compute_dtype, "compute_dtype")
        _state_bool(self.preallocate, "preallocate")
        self.validate()

    def validate(self) -> None:
        paper = self.profile is DreamerProfile.PAPER
        expected_beta2 = 0.99 if paper else 0.999
        expected_strided = paper
        if self.optimizer.beta2 != expected_beta2:
            raise ValueError("optimizer beta2 does not match profile")
        if (
            self.encoder.strided is not expected_strided
            or self.decoder.strided is not expected_strided
        ):
            raise ValueError("image stride behavior does not match profile")
        if self.action_repeat != 1 or self.image_size != (64, 64):
            raise ValueError("action repeat or image size does not match DreamerV3")
        if self.compute_dtype != "bfloat16":
            raise ValueError("compute dtype does not match DreamerV3")
        if self.loss_scales != LossScaleConfig():
            raise ValueError("loss scales do not match DreamerV3")
        if self.slow_value != SlowValueConfig():
            raise ValueError("slow value settings do not match DreamerV3")
        if self.return_normalizer != NormalizerConfig(debias=False):
            raise ValueError("return normalizer does not match DreamerV3")
        identity_normalizer = NormalizerConfig(implementation="none", limit=1e-8)
        if (
            self.value_normalizer != identity_normalizer
            or self.advantage_normalizer != identity_normalizer
        ):
            raise ValueError("value normalizers do not match DreamerV3")
        if self.model == "debug-local-v1":
            self._validate_debug()
        else:
            self._validate_profile()

    def _validate_profile(self) -> None:
        expected_size = _default_model_size(self.profile, self.observation_mode)
        expected_model = f"size{expected_size.value}"
        if self.model != expected_model:
            raise ValueError(f"model must be {expected_model}")
        expected = (
            _paper_components(self.observation_mode, expected_size)
            if self.profile is DreamerProfile.PAPER
            else _upstream_current_components(self.observation_mode, expected_size)
        )
        for name in (
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
        ):
            if getattr(self, name) != expected[name]:
                raise ValueError(f"{name} does not match profile")
        expected_sequence = SequenceShapeConfig()
        if (
            self.sequence.context,
            self.sequence.consecutive,
            self.sequence.report_length,
            self.sequence.report_consecutive,
        ) != (
            expected_sequence.context,
            expected_sequence.consecutive,
            expected_sequence.report_length,
            expected_sequence.report_consecutive,
        ):
            raise ValueError("sequence boundaries do not match profile")
        expected_run = expected["run"]
        if (
            self.run.eval_envs,
            self.run.log_every,
            self.run.report_batches,
        ) != (
            expected_run.eval_envs,
            expected_run.log_every,
            expected_run.report_batches,
        ):
            raise ValueError("run logging does not match profile")
        if self.imagination != ImaginationConfig():
            raise ValueError("imagination does not match profile")
        if self.platform != "cuda" or self.preallocate is not True:
            raise ValueError("platform does not match profile")

    def _validate_debug(self) -> None:
        size = _default_model_size(self.profile, self.observation_mode)
        base = (
            _paper_components(self.observation_mode, size)
            if self.profile is DreamerProfile.PAPER
            else _upstream_current_components(self.observation_mode, size)
        )
        expected = {
            "network": NetworkSize(32, 32, 8, 4),
            "rssm": replace(base["rssm"], deter=32, hidden=32, stoch=4, classes=4),
            "encoder": replace(
                base["encoder"],
                depth=8,
                multipliers=(1, 2, 4, 8),
                layers=1,
                units=32,
            ),
            "decoder": replace(
                base["decoder"],
                depth=8,
                multipliers=(1, 2, 4, 8),
                layers=1,
                units=32,
            ),
            "reward_head": replace(base["reward_head"], layers=1, units=32),
            "continue_head": replace(base["continue_head"], layers=1, units=32),
            "policy": replace(base["policy"], layers=1, units=32),
            "value_head": replace(base["value_head"], layers=1, units=32),
            "optimizer": base["optimizer"],
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"debug {name} does not match debug-local-v1")
        debug_snapshot = _debug_local_snapshot()
        if (
            self.sequence.context,
            self.sequence.consecutive,
            self.sequence.report_length,
            self.sequence.report_consecutive,
        ) != (
            debug_snapshot.sequence.context,
            debug_snapshot.sequence.consecutive,
            debug_snapshot.sequence.report_length,
            debug_snapshot.sequence.report_consecutive,
        ):
            raise ValueError("debug sequence does not match debug-local-v1")
        if self.replay != debug_snapshot.replay:
            raise ValueError("debug replay does not match debug-local-v1")
        if (
            self.run.eval_envs,
            self.run.log_every,
            self.run.report_batches,
        ) != (
            debug_snapshot.run.eval_envs,
            debug_snapshot.run.log_every,
            debug_snapshot.run.report_batches,
        ):
            raise ValueError("debug run does not match debug-local-v1")
        if self.imagination != ImaginationConfig(length=5):
            raise ValueError("debug imagination does not match debug-local-v1")
        if self.platform != "cpu" or self.preallocate is not False:
            raise ValueError("debug platform does not match debug-local-v1")

    @property
    def kl_free_nats(self) -> float:
        return self.rssm.free_nats

    @property
    def dynamics_kl_scale(self) -> float:
        return self.loss_scales.dyn

    @property
    def representation_kl_scale(self) -> float:
        return self.loss_scales.rep

    def state_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile.value,
            "observation_mode": self.observation_mode.value,
            "task": self.task,
            "seed": self.seed,
            "model": self.model,
            "network": self.network.state_dict(),
            "rssm": self.rssm.state_dict(),
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "reward_head": self.reward_head.state_dict(),
            "continue_head": self.continue_head.state_dict(),
            "policy": self.policy.state_dict(),
            "value_head": self.value_head.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "replay": self.replay.state_dict(),
            "run": self.run.state_dict(),
            "sequence": self.sequence.state_dict(),
            "loss_scales": self.loss_scales.state_dict(),
            "imagination": self.imagination.state_dict(),
            "slow_value": self.slow_value.state_dict(),
            "return_normalizer": self.return_normalizer.state_dict(),
            "value_normalizer": self.value_normalizer.state_dict(),
            "advantage_normalizer": self.advantage_normalizer.state_dict(),
            "action_repeat": self.action_repeat,
            "image_size": list(self.image_size),
            "platform": self.platform,
            "compute_dtype": self.compute_dtype,
            "preallocate": self.preallocate,
        }

    @classmethod
    def from_state(cls, state: object) -> DreamerV3Config:
        keys = (
            "profile",
            "observation_mode",
            "task",
            "seed",
            "model",
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
            "sequence",
            "loss_scales",
            "imagination",
            "slow_value",
            "return_normalizer",
            "value_normalizer",
            "advantage_normalizer",
            "action_repeat",
            "image_size",
            "platform",
            "compute_dtype",
            "preallocate",
        )
        record = _require_state(state, keys, "DreamerV3Config")
        image_size = _state_int_list(record["image_size"], "image_size")
        if len(image_size) != 2:
            raise ValueError("image_size must contain two dimensions")
        return cls(
            profile=DreamerProfile(_state_str(record["profile"], "profile")),
            observation_mode=ObservationMode(
                _state_str(record["observation_mode"], "observation_mode")
            ),
            task=_state_str(record["task"], "task"),
            seed=_state_int(record["seed"], "seed", minimum=0),
            model=_state_str(record["model"], "model"),
            network=NetworkSize.from_state(record["network"]),
            rssm=RSSMConfig.from_state(record["rssm"]),
            encoder=EncoderConfig.from_state(record["encoder"]),
            decoder=DecoderConfig.from_state(record["decoder"]),
            reward_head=HeadConfig.from_state(record["reward_head"]),
            continue_head=HeadConfig.from_state(record["continue_head"]),
            policy=PolicyConfig.from_state(record["policy"]),
            value_head=HeadConfig.from_state(record["value_head"]),
            optimizer=OptimizerConfig.from_state(record["optimizer"]),
            replay=ReplayConfig.from_state(record["replay"]),
            run=RunConfig.from_state(record["run"]),
            sequence=SequenceShapeConfig.from_state(record["sequence"]),
            loss_scales=LossScaleConfig.from_state(record["loss_scales"]),
            imagination=ImaginationConfig.from_state(record["imagination"]),
            slow_value=SlowValueConfig.from_state(record["slow_value"]),
            return_normalizer=NormalizerConfig.from_state(record["return_normalizer"]),
            value_normalizer=NormalizerConfig.from_state(record["value_normalizer"]),
            advantage_normalizer=NormalizerConfig.from_state(
                record["advantage_normalizer"]
            ),
            action_repeat=_state_int(
                record["action_repeat"], "action_repeat", minimum=1
            ),
            image_size=image_size,
            platform=_state_str(record["platform"], "platform"),
            compute_dtype=_state_str(record["compute_dtype"], "compute_dtype"),
            preallocate=_state_bool(record["preallocate"], "preallocate"),
        )

    def canonical_json(self) -> str:
        return (
            json.dumps(
                self.state_dict(),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        )

    def canonical_hash(self) -> str:
        encoded = self.canonical_json().encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ResolvedDreamerRun:
    config: DreamerV3Config
    explicit_overrides: RuntimeOverrides
    debug_snapshot: DebugSnapshot | None
    authority_revision: str
    canonical_json: str
    config_sha256: str

    def __post_init__(self) -> None:
        if type(self.config) is not DreamerV3Config:
            raise TypeError("config must be DreamerV3Config")
        if type(self.explicit_overrides) is not RuntimeOverrides:
            raise TypeError("explicit_overrides must be RuntimeOverrides")
        if (
            self.debug_snapshot is not None
            and type(self.debug_snapshot) is not DebugSnapshot
        ):
            raise TypeError("debug_snapshot must be DebugSnapshot or None")
        _state_str(self.authority_revision, "authority_revision")
        _state_str(self.canonical_json, "canonical_json")
        _state_str(self.config_sha256, "config_sha256")
        expected_revision = (
            "bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01"
            if self.config.profile is DreamerProfile.PAPER
            else "e3f02248693a79dc8b0ebd62c93683888ddaccfe"
        )
        if self.authority_revision != expected_revision:
            raise ValueError("authority revision does not match profile")
        if self.canonical_json != self.config.canonical_json():
            raise ValueError("canonical JSON does not match resolved config")
        expected_hash = hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()
        if self.config_sha256 != expected_hash:
            raise ValueError("config SHA-256 does not match canonical JSON")
        expected_debug = (
            _debug_local_snapshot() if self.debug_snapshot is not None else None
        )
        if self.debug_snapshot != expected_debug:
            raise ValueError("debug snapshot does not match debug-local-v1")
        expected_config = _resolve_profile_config(
            profile=self.config.profile,
            mode=self.config.observation_mode,
            task=self.config.task,
            seed=self.config.seed,
            model=None,
        )
        if expected_debug is not None:
            expected_config = _apply_debug_snapshot(expected_config, expected_debug)
        expected_config = _apply_runtime_overrides(
            expected_config, self.explicit_overrides
        )
        if self.config != expected_config:
            raise ValueError("resolved config does not match its merge coordinates")

    @property
    def algorithm_overrides(self) -> dict[str, int | float]:
        return self.explicit_overrides.algorithm_state()

    @property
    def environment_overrides(self) -> dict[str, int]:
        return self.explicit_overrides.environment_state()

    def identity_state(self) -> dict[str, Any]:
        return {
            "canonical_config": self.config.state_dict(),
            "config_sha256": self.config_sha256,
            "authority_revision": self.authority_revision,
            "debug_snapshot": (
                None
                if self.debug_snapshot is None
                else self.debug_snapshot.state_dict()
            ),
            "runtime_overrides": {
                "algorithm": self.algorithm_overrides,
                "environment": self.environment_overrides,
            },
        }

    def state_dict(self) -> dict[str, Any]:
        return self.identity_state()

    @classmethod
    def from_state(cls, state: object) -> ResolvedDreamerRun:
        keys = (
            "canonical_config",
            "config_sha256",
            "authority_revision",
            "debug_snapshot",
            "runtime_overrides",
        )
        record = _require_state(state, keys, "ResolvedDreamerRun")
        overrides_state = _require_state(
            record["runtime_overrides"],
            ("algorithm", "environment"),
            "resolved runtime overrides",
        )
        algorithm = overrides_state["algorithm"]
        environment = overrides_state["environment"]
        if type(algorithm) is not dict or type(environment) is not dict:
            raise TypeError("resolved override maps must be plain dicts")
        algorithm_keys = cast(dict[object, object], algorithm)
        environment_keys = cast(dict[object, object], environment)
        if any(type(key) is not str for key in algorithm_keys):
            raise TypeError("resolved algorithm override keys must be built-in strings")
        if any(type(key) is not str for key in environment_keys):
            raise TypeError(
                "resolved environment override keys must be built-in strings"
            )
        allowed_algorithm = {
            "env_steps",
            "num_envs",
            "batch_size",
            "batch_length",
            "train_ratio",
            "eval_every",
            "eval_episodes",
            "report_every",
            "checkpoint_every",
        }
        if set(algorithm) - allowed_algorithm or set(environment) - {"camera"}:
            raise ValueError("resolved override map contains unknown fields")
        if tuple(algorithm) != tuple(sorted(algorithm)):
            raise ValueError("algorithm override keys must be sorted")
        runtime_overrides = RuntimeOverrides(
            env_steps=(
                _state_int(algorithm["env_steps"], "env_steps", minimum=1)
                if "env_steps" in algorithm
                else None
            ),
            num_envs=(
                _state_int(algorithm["num_envs"], "num_envs", minimum=1)
                if "num_envs" in algorithm
                else None
            ),
            batch_size=(
                _state_int(algorithm["batch_size"], "batch_size", minimum=1)
                if "batch_size" in algorithm
                else None
            ),
            batch_length=(
                _state_int(algorithm["batch_length"], "batch_length", minimum=1)
                if "batch_length" in algorithm
                else None
            ),
            train_ratio=(
                _state_float(algorithm["train_ratio"], "train_ratio", minimum=0.0)
                if "train_ratio" in algorithm
                else None
            ),
            eval_every=(
                _state_int(algorithm["eval_every"], "eval_every", minimum=1)
                if "eval_every" in algorithm
                else None
            ),
            eval_episodes=(
                _state_int(algorithm["eval_episodes"], "eval_episodes", minimum=1)
                if "eval_episodes" in algorithm
                else None
            ),
            report_every=(
                _state_int(algorithm["report_every"], "report_every", minimum=1)
                if "report_every" in algorithm
                else None
            ),
            checkpoint_every=(
                _state_int(
                    algorithm["checkpoint_every"],
                    "checkpoint_every",
                    minimum=1,
                )
                if "checkpoint_every" in algorithm
                else None
            ),
            camera=(
                _state_int(environment["camera"], "camera")
                if "camera" in environment
                else None
            ),
        )
        if (
            runtime_overrides.algorithm_state() != algorithm
            or runtime_overrides.environment_state() != environment
        ):
            raise ValueError("resolved override maps must be canonical")
        config = DreamerV3Config.from_state(record["canonical_config"])
        debug_state = record["debug_snapshot"]
        debug_snapshot = (
            None if debug_state is None else DebugSnapshot.from_state(debug_state)
        )
        canonical_json = config.canonical_json()
        return cls(
            config=config,
            explicit_overrides=runtime_overrides,
            debug_snapshot=debug_snapshot,
            authority_revision=_state_str(
                record["authority_revision"], "authority_revision"
            ),
            canonical_json=canonical_json,
            config_sha256=_state_str(record["config_sha256"], "config_sha256"),
        )


def _default_model_size(
    profile: DreamerProfile,
    mode: ObservationMode,
) -> ModelSize:
    if profile is DreamerProfile.PAPER:
        return ModelSize.M200
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
        "run": RunConfig(env_steps=1_000_000, train_ratio=ratio),
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
        "run": RunConfig(env_steps=1_100_000, train_ratio=ratio),
    }


def _resolve_profile_config(
    *,
    profile: DreamerProfile,
    mode: ObservationMode,
    task: str,
    seed: int,
    model: ModelSize | str | None,
) -> DreamerV3Config:
    if type(task) is not str:
        raise TypeError("task must be a string")
    if not task:
        raise ValueError("task must be a nonempty string")
    expected_size = _default_model_size(profile, mode)
    resolved_size = ModelSize(model) if model is not None else expected_size
    if resolved_size is not expected_size:
        raise ValueError(
            f"{mode.value} model must be {expected_size.value}, not {resolved_size.value}"
        )
    components = (
        _paper_components(mode, resolved_size)
        if profile is DreamerProfile.PAPER
        else _upstream_current_components(mode, resolved_size)
    )
    return DreamerV3Config(
        profile=profile,
        observation_mode=mode,
        task=task,
        seed=seed,
        model=f"size{resolved_size.value}",
        sequence=SequenceShapeConfig(),
        loss_scales=LossScaleConfig(),
        imagination=ImaginationConfig(),
        slow_value=SlowValueConfig(),
        return_normalizer=NormalizerConfig(debias=False),
        value_normalizer=NormalizerConfig(implementation="none", limit=1e-8),
        advantage_normalizer=NormalizerConfig(implementation="none", limit=1e-8),
        **components,
    )


def _debug_local_snapshot() -> DebugSnapshot:
    run = RunConfig(
        env_steps=48,
        num_envs=1,
        eval_envs=1,
        train_ratio=4.0,
        eval_every=16,
        eval_episodes=1,
        report_every=16,
        log_every=16,
        checkpoint_every=16,
        report_batches=1,
    )
    return DebugSnapshot(
        name="debug-local-v1",
        model="debug-local-v1",
        mlp_layers=1,
        mlp_units=32,
        rssm_deter=32,
        rssm_stoch=4,
        rssm_classes=4,
        vision_depths=(8, 16, 32, 64),
        sequence=SequenceShapeConfig(
            batch_size=1,
            sequence_length=4,
            context=0,
            consecutive=1,
            report_length=4,
            report_consecutive=1,
        ),
        replay=ReplayConfig(capacity=256, chunk_size=32, online_queue_size=16),
        run=run,
        imagination_horizon=5,
        platform="cpu",
        preallocate=False,
    )


def _apply_debug_snapshot(
    config: DreamerV3Config,
    snapshot: DebugSnapshot,
) -> DreamerV3Config:
    return replace(
        config,
        model=snapshot.model,
        network=NetworkSize(32, 32, 8, 4),
        rssm=replace(
            config.rssm,
            deter=snapshot.rssm_deter,
            hidden=snapshot.mlp_units,
            stoch=snapshot.rssm_stoch,
            classes=snapshot.rssm_classes,
        ),
        encoder=replace(
            config.encoder,
            depth=8,
            multipliers=(1, 2, 4, 8),
            layers=snapshot.mlp_layers,
            units=snapshot.mlp_units,
        ),
        decoder=replace(
            config.decoder,
            depth=8,
            multipliers=(1, 2, 4, 8),
            layers=snapshot.mlp_layers,
            units=snapshot.mlp_units,
        ),
        reward_head=replace(
            config.reward_head,
            layers=snapshot.mlp_layers,
            units=snapshot.mlp_units,
        ),
        continue_head=replace(
            config.continue_head,
            layers=snapshot.mlp_layers,
            units=snapshot.mlp_units,
        ),
        policy=replace(
            config.policy,
            layers=snapshot.mlp_layers,
            units=snapshot.mlp_units,
        ),
        value_head=replace(
            config.value_head,
            layers=snapshot.mlp_layers,
            units=snapshot.mlp_units,
        ),
        sequence=snapshot.sequence,
        replay=snapshot.replay,
        run=snapshot.run,
        imagination=replace(config.imagination, length=snapshot.imagination_horizon),
        platform=snapshot.platform,
        preallocate=snapshot.preallocate,
    )


def _apply_runtime_overrides(
    config: DreamerV3Config,
    overrides: RuntimeOverrides,
) -> DreamerV3Config:
    sequence = config.sequence
    run = config.run
    if overrides.batch_size is not None:
        sequence = replace(sequence, batch_size=overrides.batch_size)
    if overrides.batch_length is not None:
        sequence = replace(sequence, sequence_length=overrides.batch_length)
    changes: dict[str, int | float] = {}
    for override_name, run_name in (
        ("env_steps", "env_steps"),
        ("num_envs", "num_envs"),
        ("train_ratio", "train_ratio"),
        ("eval_every", "eval_every"),
        ("eval_episodes", "eval_episodes"),
        ("report_every", "report_every"),
        ("checkpoint_every", "checkpoint_every"),
    ):
        value = getattr(overrides, override_name)
        if value is not None:
            changes[run_name] = value
    run = replace(run, **changes)
    result = replace(config, sequence=sequence, run=run)
    if result.sequence.batch_size * result.sequence.raw_length > result.replay.capacity:
        raise ValueError("replay capacity is smaller than one raw batch")
    return result


def resolve_dreamer_run(
    *,
    mode: ObservationMode | str,
    task: str,
    profile: DreamerProfile | str = DreamerProfile.PAPER,
    seed: int = 0,
    model: ModelSize | str | None = None,
    debug_local: bool = False,
    overrides: RuntimeOverrides = RuntimeOverrides(),
) -> ResolvedDreamerRun:
    resolved_profile = DreamerProfile(profile)
    resolved_mode = ObservationMode(mode)
    _state_int(seed, "seed", minimum=0)
    if seed > 2**32 - 1 - 10_000:
        raise ValueError("seed exceeds the supported public seed range")
    if type(debug_local) is not bool:
        raise TypeError("debug_local must be a bool")
    if type(overrides) is not RuntimeOverrides:
        raise TypeError("overrides must be RuntimeOverrides")
    if debug_local and model is not None:
        raise ValueError("debug-local resolution does not accept a model override")
    config = _resolve_profile_config(
        profile=resolved_profile,
        mode=resolved_mode,
        task=task,
        seed=seed,
        model=model,
    )
    debug_snapshot = _debug_local_snapshot() if debug_local else None
    if debug_snapshot is not None:
        config = _apply_debug_snapshot(config, debug_snapshot)
    config = _apply_runtime_overrides(config, overrides)
    canonical_json = config.canonical_json()
    revision = (
        "bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01"
        if resolved_profile is DreamerProfile.PAPER
        else "e3f02248693a79dc8b0ebd62c93683888ddaccfe"
    )
    return ResolvedDreamerRun(
        config=config,
        explicit_overrides=overrides,
        debug_snapshot=debug_snapshot,
        authority_revision=revision,
        canonical_json=canonical_json,
        config_sha256=hashlib.sha256(canonical_json.encode("utf-8")).hexdigest(),
    )


def resolve_dreamer_config(
    *,
    mode: ObservationMode | str,
    task: str,
    profile: DreamerProfile | str = DreamerProfile.PAPER,
    seed: int = 0,
    model: ModelSize | str | None = None,
    debug_local: bool = False,
) -> DreamerV3Config:
    return resolve_dreamer_run(
        mode=mode,
        task=task,
        profile=profile,
        seed=seed,
        model=model,
        debug_local=debug_local,
        overrides=RuntimeOverrides(),
    ).config


__all__ = [
    "ContinueHeadConfig",
    "DebugSnapshot",
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
    "ResolvedDreamerRun",
    "RewardHeadConfig",
    "RuntimeOverrides",
    "RunConfig",
    "SequenceShapeConfig",
    "SlowValueConfig",
    "resolve_dreamer_config",
    "resolve_dreamer_run",
]
