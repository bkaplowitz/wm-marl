from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from world_marl.dreamer_v3_baseline.config import (
    DecoderConfig,
    EncoderConfig,
    HeadConfig,
    PolicyConfig,
    RSSMConfig,
)
from world_marl.dreamer_v3_baseline.distributions import (
    AggregateOutput,
    BinaryOutput,
    CategoricalOutput,
    MSEOutput,
    NormalOutput,
    OneHotOutput,
    TwoHotOutput,
    symlog,
)


Array = jax.Array
_f32 = jnp.float32
_DEFAULT_COMPUTE_DTYPE = jnp.bfloat16
_METADATA_KEYS = frozenset({"action", "is_first", "is_last", "is_terminal", "reward"})


def _require_compute_dtype(value: Array, compute_dtype: Any, name: str) -> None:
    expected = jnp.dtype(compute_dtype)
    if value.dtype != expected:
        raise TypeError(
            f"{name} requires {expected.name} compute input, got {value.dtype}"
        )


def _space_dtype(space: TensorSpace) -> np.dtype:
    return np.dtype(space.dtype)


def _uniform_classes(space: TensorSpace, family: str) -> int:
    if not space.discrete:
        raise ValueError(f"{family} head requires a discrete TensorSpace")
    classes = space.class_values.reshape(-1)
    if not len(classes) or not np.all(classes == classes[0]):
        raise ValueError(f"{family} head requires uniform discrete classes")
    return int(classes[0])


def _activation(name: str, value: Array) -> Array:
    if name == "none":
        return value
    if name == "mish":
        return value * jnp.tanh(jax.nn.softplus(value))
    if name == "relu2":
        return jnp.square(jax.nn.relu(value))
    if name == "swiglu":
        left, right = jnp.split(value, 2, -1)
        return jax.nn.silu(left) * right
    try:
        function = getattr(jax.nn, name)
    except AttributeError as error:
        raise ValueError(f"unknown activation: {name}") from error
    return function(value)


@dataclass(frozen=True)
class Initializer:
    name: str = "trunc_normal_in"
    scale: float = 1.0

    def __post_init__(self) -> None:
        dist, fan = self.parts
        if dist not in {"zeros", "uniform", "normal", "trunc_normal", "normed"}:
            raise ValueError(f"unknown initializer distribution: {dist}")
        if fan not in {"in", "out", "avg", "none"}:
            raise ValueError(f"unknown initializer fan: {fan}")

    @property
    def parts(self) -> tuple[str, str]:
        for suffix in ("_in", "_out", "_avg", "_none"):
            if self.name.endswith(suffix):
                return self.name[: -len(suffix)], suffix[1:]
        return self.name, "in"

    def __call__(
        self,
        key: Array,
        shape: tuple[int, ...],
        dtype: Any = _f32,
        *,
        fshape: tuple[int, ...] | None = None,
    ) -> Array:
        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        if not all(isinstance(size, int) and size > 0 for size in shape):
            raise ValueError(
                f"initializer shape must contain positive integers: {shape}"
            )
        dist, fan_name = self.parts
        fanin, fanout = self.compute_fans(fshape or shape)
        fan = {
            "avg": (fanin + fanout) / 2,
            "in": fanin,
            "none": 1,
            "out": fanout,
        }[fan_name]
        if dist == "zeros":
            value = jnp.zeros(shape, dtype)
        elif dist == "uniform":
            limit = np.sqrt(1 / fan)
            value = jax.random.uniform(key, shape, dtype, -limit, limit)
        elif dist == "normal":
            value = jax.random.normal(key, shape) * np.sqrt(1 / fan)
        elif dist == "trunc_normal":
            value = jax.random.truncated_normal(key, -2, 2, shape)
            value = value * (1.1368 * np.sqrt(1 / fan))
        else:
            value = jax.random.uniform(key, shape, dtype, -1, 1)
            flattened = value.reshape((-1, shape[-1]))
            value = value * (1 / jnp.linalg.norm(flattened, 2, 0))
        return (value * self.scale).astype(dtype)

    @staticmethod
    def compute_fans(shape: tuple[int, ...]) -> tuple[float, float]:
        if len(shape) == 0:
            return (1, 1)
        if len(shape) == 1:
            return (1, shape[0])
        if len(shape) == 2:
            return shape
        space = math.prod(shape[:-2])
        return (shape[-2] * space, shape[-1] * space)


class RMSNorm(nn.Module):
    eps: float = 1e-4
    use_scale: bool = True
    use_bias: bool = False
    param_dtype: Any = _f32
    compute_dtype: Any = _DEFAULT_COMPUTE_DTYPE

    @nn.compact
    def __call__(self, value: Array) -> Array:
        _require_compute_dtype(value, self.compute_dtype, "RMSNorm")
        dtype = value.dtype
        value = _f32(value)
        mean_square = jnp.square(value).mean(-1, keepdims=True)
        if self.use_scale:
            scale = self.param(
                "scale",
                lambda key, shape, dtype: jnp.ones(shape, dtype),
                (value.shape[-1],),
                self.param_dtype,
            ).astype(value.dtype)
        else:
            scale = jnp.ones((value.shape[-1],), value.dtype)
        value = value * (jax.lax.rsqrt(mean_square + self.eps) * scale)
        if self.use_bias:
            bias = self.param(
                "bias",
                lambda key, shape, dtype: jnp.zeros(shape, dtype),
                (value.shape[-1],),
                self.param_dtype,
            ).astype(value.dtype)
            value = value + bias
        return value.astype(dtype)


class Linear(nn.Module):
    units: int | tuple[int, ...]
    bias: bool = True
    initializer: str = "trunc_normal_in"
    bias_initializer: str = "zeros"
    output_scale: float = 1.0
    normalization: str = "none"
    activation: str = "none"
    param_dtype: Any = _f32
    compute_dtype: Any = _DEFAULT_COMPUTE_DTYPE

    @nn.compact
    def __call__(self, value: Array) -> Array:
        _require_compute_dtype(value, self.compute_dtype, "Linear")
        units = (self.units,) if isinstance(self.units, int) else tuple(self.units)
        size = math.prod(units)
        kernel = self.param(
            "kernel",
            Initializer(self.initializer, self.output_scale),
            (value.shape[-1], size),
            self.param_dtype,
        ).astype(value.dtype)
        value = value @ kernel
        if self.bias:
            bias = self.param(
                "bias",
                Initializer(self.bias_initializer),
                (size,),
                self.param_dtype,
            ).astype(value.dtype)
            value = value + bias
        value = value.reshape((*value.shape[:-1], *units))
        if self.normalization == "rms":
            value = RMSNorm(
                param_dtype=self.param_dtype,
                compute_dtype=self.compute_dtype,
                name="norm",
            )(value)
        elif self.normalization != "none":
            raise ValueError(f"unsupported linear normalization: {self.normalization}")
        return _activation(self.activation, value)


class BlockLinear(nn.Module):
    units: int
    blocks: int
    bias: bool = True
    initializer: str = "trunc_normal_in"
    bias_initializer: str = "zeros"
    output_scale: float = 1.0
    normalization: str = "none"
    activation: str = "none"
    param_dtype: Any = _f32
    compute_dtype: Any = _DEFAULT_COMPUTE_DTYPE

    @nn.compact
    def __call__(self, value: Array) -> Array:
        _require_compute_dtype(value, self.compute_dtype, "BlockLinear")
        if self.blocks > self.units or self.units % self.blocks:
            raise ValueError("BlockLinear output units must be divisible by blocks")
        if value.shape[-1] % self.blocks:
            raise ValueError("BlockLinear input units must be divisible by blocks")
        input_size = value.shape[-1]
        shape = (
            self.blocks,
            input_size // self.blocks,
            self.units // self.blocks,
        )
        kernel = self.param(
            "kernel",
            Initializer(self.initializer, self.output_scale),
            shape,
            self.param_dtype,
        ).astype(value.dtype)
        grouped = value.reshape(
            (*value.shape[:-1], self.blocks, input_size // self.blocks)
        )
        value = jnp.einsum("...ki,kio->...ko", grouped, kernel)
        value = value.reshape((*value.shape[:-2], self.units))
        if self.bias:
            bias = self.param(
                "bias",
                Initializer(self.bias_initializer),
                (self.units,),
                self.param_dtype,
            ).astype(value.dtype)
            value = value + bias
        if self.normalization == "rms":
            value = RMSNorm(
                param_dtype=self.param_dtype,
                compute_dtype=self.compute_dtype,
                name="norm",
            )(value)
        elif self.normalization != "none":
            raise ValueError(
                f"unsupported block linear normalization: {self.normalization}"
            )
        return _activation(self.activation, value)


class Conv2D(nn.Module):
    depth: int
    kernel: int | tuple[int, int]
    stride: int = 1
    transposed: bool = False
    groups: int = 1
    padding: str = "same"
    bias: bool = True
    initializer: str = "trunc_normal_in"
    bias_initializer: str = "zeros"
    output_scale: float = 1.0
    param_dtype: Any = _f32
    compute_dtype: Any = _DEFAULT_COMPUTE_DTYPE

    @nn.compact
    def __call__(self, value: Array) -> Array:
        _require_compute_dtype(value, self.compute_dtype, "Conv2D")
        kernel_size = (
            (self.kernel, self.kernel)
            if isinstance(self.kernel, int)
            else tuple(self.kernel)
        )
        if value.shape[-1] % self.groups:
            raise ValueError("Conv2D input channels must be divisible by groups")
        shape = (*kernel_size, value.shape[-1] // self.groups, self.depth)
        kernel = self.param(
            "kernel",
            Initializer(self.initializer, self.output_scale),
            shape,
            self.param_dtype,
        ).astype(value.dtype)
        if self.transposed:
            if self.padding.lower() != "same":
                raise ValueError("manual transposed Conv2D requires same padding")
            value = value.repeat(self.stride, -2).repeat(self.stride, -3)
            mask_height = ((jnp.arange(value.shape[-3]) - 1) % self.stride == 0)[
                :, None
            ]
            mask_width = ((jnp.arange(value.shape[-2]) - 1) % self.stride == 0)[None, :]
            value = value * (mask_height * mask_width)[:, :, None]
            strides = (1, 1)
        else:
            strides = (self.stride, self.stride)
        value = jax.lax.conv_general_dilated(
            value,
            kernel,
            strides,
            self.padding.upper(),
            feature_group_count=self.groups,
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )
        if self.bias:
            bias = self.param(
                "bias",
                Initializer(self.bias_initializer),
                (self.depth,),
                self.param_dtype,
            ).astype(value.dtype)
            value = value + bias
        return value


class MLP(nn.Module):
    layers: int = 5
    units: int = 1024
    activation: str = "silu"
    normalization: str = "rms"
    bias: bool = True
    initializer: str = "trunc_normal_in"
    bias_initializer: str = "zeros"
    param_dtype: Any = _f32
    compute_dtype: Any = _DEFAULT_COMPUTE_DTYPE

    @nn.compact
    def __call__(self, value: Array) -> Array:
        shape = value.shape[:-1]
        value = jnp.asarray(value).astype(self.compute_dtype)
        value = value.reshape((-1, value.shape[-1]))
        for index in range(self.layers):
            value = Linear(
                self.units,
                bias=self.bias,
                initializer=self.initializer,
                bias_initializer=self.bias_initializer,
                param_dtype=self.param_dtype,
                compute_dtype=self.compute_dtype,
                name=f"linear{index}",
            )(value)
            if self.normalization == "rms":
                value = RMSNorm(
                    param_dtype=self.param_dtype,
                    compute_dtype=self.compute_dtype,
                    name=f"norm{index}",
                )(value)
            elif self.normalization != "none":
                raise ValueError(f"unsupported MLP normalization: {self.normalization}")
            value = _activation(self.activation, value)
        return value.reshape((*shape, value.shape[-1]))


class BlockGRU(nn.Module):
    config: RSSMConfig
    action_dim: int
    param_dtype: Any = _f32
    compute_dtype: Any = _DEFAULT_COMPUTE_DTYPE

    @nn.compact
    def __call__(
        self,
        deter: Array,
        stoch: Array,
        action: Array,
        is_first: Array | None = None,
    ) -> Array:
        config = self.config
        if action.shape[-1] != self.action_dim:
            raise ValueError("BlockGRU action width does not match action_dim")
        if deter.shape[-1] != config.deter:
            raise ValueError("BlockGRU deter width does not match configuration")
        if stoch.shape[-2:] != (config.stoch, config.classes):
            raise ValueError("BlockGRU stochastic shape does not match configuration")
        for name, value in (("deter", deter), ("stoch", stoch), ("action", action)):
            if not jnp.issubdtype(value.dtype, jnp.floating):
                raise TypeError(f"BlockGRU {name} must be floating")
        deter = deter.astype(self.compute_dtype)
        stoch = stoch.astype(self.compute_dtype)
        action = action.astype(self.compute_dtype)
        if is_first is not None:
            reset = jnp.asarray(is_first, bool)
            expanded = reset[(...,) + (None,) * (deter.ndim - reset.ndim)]
            deter = jnp.where(expanded, jnp.zeros_like(deter), deter)
            stoch_expanded = reset[(...,) + (None,) * (stoch.ndim - reset.ndim)]
            stoch = jnp.where(stoch_expanded, jnp.zeros_like(stoch), stoch)
            action_expanded = reset[(...,) + (None,) * (action.ndim - reset.ndim)]
            action = jnp.where(action_expanded, jnp.zeros_like(action), action)
        stoch = stoch.reshape((*stoch.shape[:-2], -1))
        action = action / jax.lax.stop_gradient(jnp.maximum(1, jnp.abs(action)))
        shared = dict(
            bias=True,
            initializer=config.initializer,
            param_dtype=self.param_dtype,
            compute_dtype=self.compute_dtype,
        )

        def project(name: str, value: Array) -> Array:
            value = Linear(config.hidden, name=name, **shared)(value)
            value = RMSNorm(
                param_dtype=self.param_dtype,
                compute_dtype=self.compute_dtype,
                name=f"{name}norm",
            )(value)
            return _activation(config.activation, value)

        x0 = project("dynin0", deter)
        x1 = project("dynin1", stoch)
        x2 = project("dynin2", action)
        groups = config.blocks
        repeated = jnp.concatenate([x0, x1, x2], -1)[..., None, :]
        repeated = jnp.repeat(repeated, groups, -2)
        grouped_deter = deter.reshape((*deter.shape[:-1], groups, -1))
        value = jnp.concatenate([grouped_deter, repeated], -1)
        value = value.reshape((*value.shape[:-2], -1))
        for index in range(config.dynamics_layers):
            value = BlockLinear(
                config.deter,
                groups,
                name=f"dynhid{index}",
                **shared,
            )(value)
            value = RMSNorm(
                param_dtype=self.param_dtype,
                compute_dtype=self.compute_dtype,
                name=f"dynhid{index}norm",
            )(value)
            value = _activation(config.activation, value)
        value = BlockLinear(
            3 * config.deter,
            groups,
            name="dyngru",
            **shared,
        )(value)
        grouped = value.reshape((*value.shape[:-1], groups, -1))
        reset, candidate, update = jnp.split(grouped, 3, -1)
        reset = reset.reshape((*reset.shape[:-2], -1))
        candidate = candidate.reshape((*candidate.shape[:-2], -1))
        update = update.reshape((*update.shape[:-2], -1))
        reset = jax.nn.sigmoid(reset)
        candidate = jnp.tanh(reset * candidate)
        update = jax.nn.sigmoid(update - 1)
        return update * candidate + (1 - update) * deter


@dataclass(frozen=True)
class TensorSpace:
    shape: tuple[int, ...]
    dtype: str
    classes: int | tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if any(size <= 0 for size in self.shape):
            raise ValueError("TensorSpace dimensions must be positive")
        try:
            dtype = np.dtype(self.dtype)
        except TypeError as error:
            raise ValueError(f"invalid TensorSpace dtype: {self.dtype}") from error
        object.__setattr__(self, "dtype", dtype.name)
        if self.classes is not None:
            classes = np.asarray(self.classes, np.int32)
            if classes.shape == ():
                classes = np.full(self.shape or (), classes.item(), np.int32)
            elif classes.shape != self.shape:
                raise ValueError("TensorSpace classes must be scalar or match shape")
            if np.any(classes <= 1):
                raise ValueError("discrete TensorSpace requires at least two classes")
            if classes.shape:
                object.__setattr__(self, "classes", tuple(classes.reshape(-1).tolist()))
            else:
                object.__setattr__(self, "classes", int(classes.item()))

    @property
    def discrete(self) -> bool:
        return self.classes is not None

    @property
    def class_values(self) -> np.ndarray:
        if self.classes is None:
            raise ValueError("continuous TensorSpace has no classes")
        values = np.asarray(self.classes, np.int32)
        if values.shape == ():
            return np.full(self.shape or (), values.item(), np.int32)
        return values.reshape(self.shape)

    @property
    def image(self) -> bool:
        return len(self.shape) == 3


class DictEncoder(nn.Module):
    spaces: Mapping[str, TensorSpace]
    config: EncoderConfig
    param_dtype: Any = _f32
    compute_dtype: Any = _DEFAULT_COMPUTE_DTYPE

    def setup(self) -> None:
        if not self.spaces:
            raise ValueError("DictEncoder requires declared observation spaces")
        metadata = _METADATA_KEYS.intersection(self.spaces)
        if metadata:
            raise ValueError(
                f"DictEncoder metadata keys are forbidden: {sorted(metadata)}"
            )
        if any(len(space.shape) > 3 for space in self.spaces.values()):
            raise ValueError("DictEncoder supports observation ranks up to three")
        invalid_images = [
            key
            for key, space in self.spaces.items()
            if len(space.shape) == 3 and _space_dtype(space) != np.dtype(np.uint8)
        ]
        if invalid_images:
            raise TypeError(
                "DictEncoder rank-three image spaces must declare uint8: "
                + ", ".join(invalid_images)
            )

    @nn.compact
    def __call__(self, observations: Mapping[str, Array]) -> Array:
        if set(observations) != set(self.spaces):
            raise ValueError(
                "DictEncoder observation keys do not match declared spaces"
            )
        vector_keys = sorted(
            key for key, space in self.spaces.items() if not space.image
        )
        image_keys = sorted(key for key, space in self.spaces.items() if space.image)
        outputs: list[Array] = []
        leading_shape: tuple[int, ...] | None = None
        if vector_keys:
            vectors = []
            for key in vector_keys:
                space = self.spaces[key]
                value = observations[key]
                if np.dtype(value.dtype) != _space_dtype(space):
                    raise TypeError(
                        f"DictEncoder {key!r} runtime dtype {value.dtype} does not "
                        f"match declared dtype {space.dtype}"
                    )
                current_leading = value.shape[: value.ndim - len(space.shape)]
                leading_shape = leading_shape or current_leading
                if (
                    current_leading != leading_shape
                    or value.shape[len(leading_shape) :] != space.shape
                ):
                    raise ValueError(f"DictEncoder vector shape mismatch: {key}")
                if space.discrete:
                    classes = _uniform_classes(space, "encoder discrete")
                    value = jax.nn.one_hot(
                        value.astype(jnp.int32),
                        classes,
                        dtype=self.compute_dtype,
                    )
                else:
                    if self.config.symlog:
                        value = symlog(value)
                    value = value.astype(self.compute_dtype)
                vectors.append(value.reshape((*leading_shape, -1)))
            value = jnp.concatenate(vectors, -1).reshape(
                (-1, sum(x.shape[-1] for x in vectors))
            )
            for index in range(self.config.layers):
                value = Linear(
                    self.config.units,
                    initializer=self.config.initializer,
                    param_dtype=self.param_dtype,
                    compute_dtype=self.compute_dtype,
                    name=f"mlp{index}",
                )(value)
                value = RMSNorm(
                    param_dtype=self.param_dtype,
                    compute_dtype=self.compute_dtype,
                    name=f"mlp{index}norm",
                )(value)
                value = _activation(self.config.activation, value)
            outputs.append(value)
        if image_keys:
            images = []
            for key in image_keys:
                space = self.spaces[key]
                value = observations[key]
                if np.dtype(value.dtype) != _space_dtype(space):
                    raise TypeError(
                        f"DictEncoder image {key!r} runtime dtype {value.dtype} does "
                        f"not match declared dtype {space.dtype}"
                    )
                if value.dtype != jnp.uint8:
                    raise TypeError(f"DictEncoder image {key!r} must use uint8")
                current_leading = value.shape[: value.ndim - len(space.shape)]
                leading_shape = leading_shape or current_leading
                if (
                    current_leading != leading_shape
                    or value.shape[len(leading_shape) :] != space.shape
                ):
                    raise ValueError(f"DictEncoder image shape mismatch: {key}")
                images.append(value)
            value = jnp.concatenate(images, -1).astype(self.compute_dtype) / 255 - 0.5
            value = value.reshape((-1, *value.shape[len(leading_shape) :]))
            depths = tuple(
                self.config.depth * multiplier for multiplier in self.config.multipliers
            )
            for index, depth in enumerate(depths):
                if self.config.outer and index == 0:
                    value = Conv2D(
                        depth,
                        self.config.kernel,
                        initializer=self.config.initializer,
                        param_dtype=self.param_dtype,
                        compute_dtype=self.compute_dtype,
                        name=f"cnn{index}",
                    )(value)
                elif self.config.strided:
                    value = Conv2D(
                        depth,
                        self.config.kernel,
                        stride=2,
                        initializer=self.config.initializer,
                        param_dtype=self.param_dtype,
                        compute_dtype=self.compute_dtype,
                        name=f"cnn{index}",
                    )(value)
                else:
                    value = Conv2D(
                        depth,
                        self.config.kernel,
                        initializer=self.config.initializer,
                        param_dtype=self.param_dtype,
                        compute_dtype=self.compute_dtype,
                        name=f"cnn{index}",
                    )(value)
                    batch, height, width, channels = value.shape
                    if height % 2 or width % 2:
                        raise ValueError("DictEncoder pooling requires even resolution")
                    value = value.reshape(
                        (batch, height // 2, 2, width // 2, 2, channels)
                    ).max((2, 4))
                value = RMSNorm(
                    param_dtype=self.param_dtype,
                    compute_dtype=self.compute_dtype,
                    name=f"cnn{index}norm",
                )(value)
                value = _activation(self.config.activation, value)
            if not (3 <= value.shape[-3] <= 16 and 3 <= value.shape[-2] <= 16):
                raise ValueError(
                    "DictEncoder final image resolution must be in [3, 16]"
                )
            outputs.append(value.reshape((value.shape[0], -1)))
        assert leading_shape is not None
        value = jnp.concatenate(outputs, -1)
        return value.reshape((*leading_shape, *value.shape[1:]))


class _OutputHead(nn.Module):
    space: TensorSpace
    output: str
    initializer: str
    output_scale: float
    bins: int = 255
    min_std: float = 0.1
    max_std: float = 1.0
    unimix: float = 0.0
    param_dtype: Any = _f32
    compute_dtype: Any = _DEFAULT_COMPUTE_DTYPE

    @nn.compact
    def __call__(self, value: Array):
        _require_compute_dtype(value, self.compute_dtype, "output head")
        shared = dict(
            initializer=self.initializer,
            output_scale=self.output_scale,
            param_dtype=self.param_dtype,
            compute_dtype=self.compute_dtype,
        )
        event_shape = self.space.shape
        if self.output == "binary":
            classes = _uniform_classes(self.space, "binary")
            if classes != 2:
                raise ValueError("binary head requires exactly two classes")
            result = BinaryOutput(
                Linear(self.space.shape, name="logit", **shared)(value)
            )
        elif self.output == "categorical":
            classes = _uniform_classes(self.space, "categorical")
            shape = (*self.space.shape, classes)
            logits = Linear(shape, name="logits", **shared)(value)
            result = CategoricalOutput(logits)
            result.minent = 0
            result.maxent = np.log(classes)
        elif self.output == "onehot":
            classes = _uniform_classes(self.space, "onehot")
            event_shape = (*self.space.shape, classes)
            logits = Linear(event_shape, name="logits", **shared)(value)
            result = OneHotOutput(logits, self.unimix)
        elif self.output == "mse":
            if self.space.discrete:
                raise ValueError("mse head requires a continuous TensorSpace")
            result = MSEOutput(Linear(self.space.shape, name="pred", **shared)(value))
        elif self.output == "symlog_mse":
            if self.space.discrete:
                raise ValueError("symlog_mse head requires a continuous TensorSpace")
            result = MSEOutput(
                Linear(self.space.shape, name="pred", **shared)(value),
                symlog,
            )
        elif self.output == "symexp_twohot":
            if self.space.discrete:
                raise ValueError("symexp_twohot head requires a continuous TensorSpace")
            shape = (*self.space.shape, self.bins)
            result = TwoHotOutput(
                Linear(shape, name="logits", **shared)(value), self.bins
            )
        elif self.output == "bounded_normal":
            if self.space.discrete:
                raise ValueError(
                    "bounded_normal head requires a continuous TensorSpace"
                )
            mean = Linear(self.space.shape, name="mean", **shared)(value)
            stddev = Linear(self.space.shape, name="stddev", **shared)(value)
            result = NormalOutput.bounded(mean, stddev, self.min_std, self.max_std)
            result.minent = NormalOutput(jnp.zeros_like(mean), self.min_std).entropy()
            result.maxent = NormalOutput(jnp.zeros_like(mean), self.max_std).entropy()
        else:
            raise ValueError(f"unsupported output family: {self.output}")
        if event_shape:
            return AggregateOutput(result, len(event_shape), jnp.sum)
        return result


class _DictOutputHead(nn.Module):
    spaces: Mapping[str, TensorSpace]
    outputs: Mapping[str, str]
    initializer: str
    output_scale: float
    param_dtype: Any = _f32
    compute_dtype: Any = _DEFAULT_COMPUTE_DTYPE

    @nn.compact
    def __call__(self, value: Array) -> dict[str, Any]:
        return {
            key: _OutputHead(
                self.spaces[key],
                self.outputs[key],
                self.initializer,
                self.output_scale,
                param_dtype=self.param_dtype,
                compute_dtype=self.compute_dtype,
                name=key,
            )(value)
            for key in self.spaces
        }


class DictDecoder(nn.Module):
    spaces: Mapping[str, TensorSpace]
    config: DecoderConfig
    param_dtype: Any = _f32
    compute_dtype: Any = _DEFAULT_COMPUTE_DTYPE

    def setup(self) -> None:
        if not self.spaces:
            raise ValueError("DictDecoder requires declared observation spaces")
        metadata = _METADATA_KEYS.intersection(self.spaces)
        if metadata:
            raise ValueError(
                f"DictDecoder metadata keys are forbidden: {sorted(metadata)}"
            )
        if any(len(space.shape) > 3 for space in self.spaces.values()):
            raise ValueError("DictDecoder supports observation ranks up to three")
        invalid_images = [
            key
            for key, space in self.spaces.items()
            if len(space.shape) == 3 and _space_dtype(space) != np.dtype(np.uint8)
        ]
        if invalid_images:
            raise TypeError(
                "DictDecoder rank-three image spaces must declare uint8: "
                + ", ".join(invalid_images)
            )

    @nn.compact
    def __call__(self, features: Mapping[str, Array]) -> dict[str, Any]:
        if set(features) != {"deter", "stoch"}:
            raise ValueError("DictDecoder features must contain deter and stoch")
        deter = jnp.asarray(features["deter"])
        stoch = jnp.asarray(features["stoch"])
        if not jnp.issubdtype(deter.dtype, jnp.floating) or not jnp.issubdtype(
            stoch.dtype, jnp.floating
        ):
            raise TypeError("DictDecoder features must use floating dtypes")
        deter = deter.astype(self.compute_dtype)
        stoch = stoch.astype(self.compute_dtype)
        if deter.shape[:-1] != stoch.shape[:-2]:
            raise ValueError("DictDecoder feature leading shapes must match")
        leading = deter.shape[:-1]
        flat_deter = deter.reshape((-1, deter.shape[-1]))
        flat_stoch = stoch.reshape((-1, math.prod(stoch.shape[-2:])))
        combined = jnp.concatenate([flat_stoch, flat_deter], -1)
        vector_spaces = {
            key: space for key, space in self.spaces.items() if not space.image
        }
        image_keys = [key for key, space in self.spaces.items() if space.image]
        reconstructions: dict[str, Any] = {}
        if vector_spaces:
            value = MLP(
                self.config.layers,
                self.config.units,
                activation=self.config.activation,
                normalization=self.config.normalization,
                initializer=self.config.initializer,
                param_dtype=self.param_dtype,
                compute_dtype=self.compute_dtype,
                name="mlp",
            )(combined)
            value = value.reshape((*leading, value.shape[-1]))
            output_families = {
                key: "categorical" if space.discrete else "symlog_mse"
                for key, space in vector_spaces.items()
            }
            reconstructions.update(
                _DictOutputHead(
                    vector_spaces,
                    output_families,
                    self.config.initializer,
                    self.config.output_scale,
                    param_dtype=self.param_dtype,
                    compute_dtype=self.compute_dtype,
                    name="vec",
                )(value)
            )
        if image_keys:
            image_resolution = self.spaces[image_keys[0]].shape[:-1]
            if any(
                self.spaces[key].shape[:-1] != image_resolution for key in image_keys
            ):
                raise ValueError("DictDecoder image resolutions must match")
            image_depth = sum(self.spaces[key].shape[-1] for key in image_keys)
            depths = tuple(
                self.config.depth * multiplier for multiplier in self.config.multipliers
            )
            factor = 2 ** (len(depths) - int(bool(self.config.outer)))
            minimum = tuple(int(size // factor) for size in image_resolution)
            if not all(3 <= size <= 16 for size in minimum):
                raise ValueError(
                    "DictDecoder minimum image resolution must be in [3, 16]"
                )
            shape = (*minimum, depths[-1])
            groups = self.config.bias_space
            if deter.shape[-1] % groups:
                raise ValueError("DictDecoder deter width must divide bias_space")
            spatial_size = math.prod(shape)
            x0 = BlockLinear(
                spatial_size,
                groups,
                initializer=self.config.initializer,
                param_dtype=self.param_dtype,
                compute_dtype=self.compute_dtype,
                name="sp0",
            )(flat_deter)
            channels_per_group = shape[-1] // groups
            x0 = x0.reshape((-1, groups, *minimum, channels_per_group))
            x0 = jnp.transpose(x0, (0, 2, 3, 1, 4))
            x0 = x0.reshape((-1, *minimum, shape[-1]))
            x1 = Linear(
                2 * self.config.units,
                initializer=self.config.initializer,
                param_dtype=self.param_dtype,
                compute_dtype=self.compute_dtype,
                name="sp1",
            )(flat_stoch)
            x1 = RMSNorm(
                param_dtype=self.param_dtype,
                compute_dtype=self.compute_dtype,
                name="sp1norm",
            )(x1)
            x1 = _activation(self.config.activation, x1)
            x1 = Linear(
                shape,
                initializer=self.config.initializer,
                param_dtype=self.param_dtype,
                compute_dtype=self.compute_dtype,
                name="sp2",
            )(x1)
            value = RMSNorm(
                param_dtype=self.param_dtype,
                compute_dtype=self.compute_dtype,
                name="spnorm",
            )(x0 + x1)
            value = _activation(self.config.activation, value)
            for index, depth in reversed(tuple(enumerate(depths[:-1]))):
                if self.config.strided:
                    value = Conv2D(
                        depth,
                        self.config.kernel,
                        stride=2,
                        transposed=True,
                        initializer=self.config.initializer,
                        param_dtype=self.param_dtype,
                        compute_dtype=self.compute_dtype,
                        name=f"conv{index}",
                    )(value)
                else:
                    value = value.repeat(2, -2).repeat(2, -3)
                    value = Conv2D(
                        depth,
                        self.config.kernel,
                        initializer=self.config.initializer,
                        param_dtype=self.param_dtype,
                        compute_dtype=self.compute_dtype,
                        name=f"conv{index}",
                    )(value)
                value = RMSNorm(
                    param_dtype=self.param_dtype,
                    compute_dtype=self.compute_dtype,
                    name=f"conv{index}norm",
                )(value)
                value = _activation(self.config.activation, value)
            if self.config.outer:
                value = Conv2D(
                    image_depth,
                    self.config.kernel,
                    initializer=self.config.initializer,
                    output_scale=self.config.output_scale,
                    param_dtype=self.param_dtype,
                    compute_dtype=self.compute_dtype,
                    name="imgout",
                )(value)
            elif self.config.strided:
                value = Conv2D(
                    image_depth,
                    self.config.kernel,
                    stride=2,
                    transposed=True,
                    initializer=self.config.initializer,
                    output_scale=self.config.output_scale,
                    param_dtype=self.param_dtype,
                    compute_dtype=self.compute_dtype,
                    name="imgout",
                )(value)
            else:
                value = value.repeat(2, -2).repeat(2, -3)
                value = Conv2D(
                    image_depth,
                    self.config.kernel,
                    initializer=self.config.initializer,
                    output_scale=self.config.output_scale,
                    param_dtype=self.param_dtype,
                    compute_dtype=self.compute_dtype,
                    name="imgout",
                )(value)
            value = jax.nn.sigmoid(value)
            value = value.reshape((*leading, *value.shape[1:]))
            split = np.cumsum([self.spaces[key].shape[-1] for key in image_keys][:-1])
            for key, output in zip(image_keys, jnp.split(value, split, -1)):
                reconstructions[key] = AggregateOutput(MSEOutput(output), 3, jnp.sum)
        return reconstructions


class MLPHead(nn.Module):
    space: TensorSpace
    config: HeadConfig | PolicyConfig
    param_dtype: Any = _f32
    compute_dtype: Any = _DEFAULT_COMPUTE_DTYPE

    @nn.compact
    def __call__(self, features: Array):
        value = features.reshape((*features.shape[:-1], -1))
        value = MLP(
            self.config.layers,
            self.config.units,
            activation=self.config.activation,
            normalization=self.config.normalization,
            initializer=self.config.initializer,
            param_dtype=self.param_dtype,
            compute_dtype=self.compute_dtype,
            name="mlp",
        )(value)
        if isinstance(self.config, PolicyConfig):
            output = (
                self.config.discrete if self.space.discrete else self.config.continuous
            )
            return _OutputHead(
                self.space,
                output,
                self.config.initializer,
                self.config.output_scale,
                min_std=self.config.min_std,
                max_std=self.config.max_std,
                unimix=self.config.unimix,
                param_dtype=self.param_dtype,
                compute_dtype=self.compute_dtype,
                name="head",
            )(value)
        return _OutputHead(
            self.space,
            self.config.output,
            self.config.initializer,
            self.config.output_scale,
            bins=self.config.bins or 255,
            param_dtype=self.param_dtype,
            compute_dtype=self.compute_dtype,
            name="head",
        )(value)


__all__ = [
    "BlockGRU",
    "BlockLinear",
    "Conv2D",
    "DictDecoder",
    "DictEncoder",
    "Initializer",
    "Linear",
    "MLP",
    "MLPHead",
    "RMSNorm",
    "TensorSpace",
]
