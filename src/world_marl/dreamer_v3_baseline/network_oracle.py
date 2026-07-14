from __future__ import annotations

import ast
import functools
import json
import math
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Callable, Iterator, Mapping, Sequence

import jax
import jax.ad_checkpoint as adc
import jax.numpy as jnp
import numpy as np

if __package__:
    from .config import DreamerProfile, ObservationMode
    from .distributions import (
        AggregateOutput,
        BinaryOutput,
        CategoricalOutput,
        MSEOutput,
        NormalOutput,
        OneHotOutput,
        TwoHotOutput,
    )
    from .oracle import (
        PAPER_REVISION,
        UPSTREAM_CURRENT_REVISION,
        OracleHarness,
        OracleSourceSpec,
        _canonical_json,
        _git_show,
        _sha256_bytes,
        official_revision,
        profile_overrides,
        register_oracle_source_spec,
    )
else:  # pragma: no cover - subprocess boundary.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from world_marl.dreamer_v3_baseline.config import (
        DreamerProfile,
        ObservationMode,
    )
    from world_marl.dreamer_v3_baseline.distributions import (
        AggregateOutput,
        BinaryOutput,
        CategoricalOutput,
        MSEOutput,
        NormalOutput,
        OneHotOutput,
        TwoHotOutput,
    )
    from world_marl.dreamer_v3_baseline.oracle import (
        PAPER_REVISION,
        UPSTREAM_CURRENT_REVISION,
        OracleHarness,
        OracleSourceSpec,
        _canonical_json,
        _git_show,
        _sha256_bytes,
        official_revision,
        profile_overrides,
        register_oracle_source_spec,
    )


_SOURCE_HASHES = {
    "dreamerv3/rssm.py": (
        "d6d50166914e94fb8bd17a5d5dbda9d42cdd37b85819bb1e9fff3a64d4ad2eb6"
    ),
    "embodied/jax/heads.py": (
        "437641cde21e7f9e3f69b88ad8f6b7e7c22e54eec8c5b19eef6127afde1a9b3f"
    ),
    "embodied/jax/nets.py": (
        "9a1c0c71ad7d3596572a44416e78434f777d8f4dbcbe8ca0dd6b86bb8246392c"
    ),
}

NETWORKS_SOURCE_SPEC = OracleSourceSpec(
    name="networks",
    revision_hashes={
        PAPER_REVISION: _SOURCE_HASHES,
        UPSTREAM_CURRENT_REVISION: _SOURCE_HASHES,
    },
    execution_dtypes=("bfloat16", "float32"),
)
register_oracle_source_spec(NETWORKS_SOURCE_SPEC)


class _ModuleMeta(type):
    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        name = kwargs.pop("name", None)
        overrides: dict[str, Any] = {}
        for key in tuple(kwargs):
            if any(key in base.__dict__ for base in cls.__mro__):
                overrides[key] = kwargs.pop(key)
        instance = cls.__new__(cls)
        instance._fields = {}
        instance._declared_name = name
        instance._context = None
        instance._prefix = ""
        for key, value in overrides.items():
            setattr(instance, key, value)
        cls.__init__(instance, *args, **kwargs)
        return instance


class _Module(metaclass=_ModuleMeta):
    def _bind(self, context: _ParameterContext, prefix: str) -> None:
        self._context = context
        self._prefix = prefix
        for value in vars(self).values():
            if isinstance(value, _Module):
                child_name = value._declared_name
                if child_name is None:
                    raise ValueError(
                        "official oracle submodule is missing a source name"
                    )
                value._bind(context, _join_path(prefix, child_name))

    def value(self, name: str, initializer: Callable[..., Any], *args: Any) -> Any:
        if self._context is None:
            raise ValueError("official oracle module is not bound")
        return self._context.value(_join_path(self._prefix, name), initializer, *args)

    def sub(
        self,
        name: str,
        module_type: type[_Module],
        *args: Any,
        **kwargs: Any,
    ) -> _Module:
        if self._context is None:
            raise ValueError("official oracle module is not bound")
        child = module_type(*args, name=name, **kwargs)
        child._bind(self._context, _join_path(self._prefix, name))
        return child


class _ParameterContext:
    def __init__(
        self,
        seed: int,
        parameters: Mapping[str, ArrayLike] | None = None,
        fixed_seed: jax.Array | None = None,
    ) -> None:
        self.base_seed = jax.random.PRNGKey(seed)
        self.parameters = {
            key: jnp.asarray(value) for key, value in (parameters or {}).items()
        }
        self.fixed_seed = fixed_seed
        self.counter = 0

    def seed(self) -> jax.Array:
        if self.fixed_seed is not None:
            key = self.fixed_seed
            self.fixed_seed = None
            return key
        key = jax.random.fold_in(self.base_seed, self.counter)
        self.counter += 1
        return key

    def value(
        self,
        path: str,
        initializer: Callable[..., Any],
        *args: Any,
    ) -> jax.Array:
        if path not in self.parameters:
            self.parameters[path] = jnp.asarray(initializer(*args))
        return self.parameters[path]


ArrayLike = np.ndarray | jax.Array
_ACTIVE_CONTEXT: _ParameterContext | None = None


def _seed(count: int | None = None, optional: bool = False) -> jax.Array:
    del optional
    if _ACTIVE_CONTEXT is None:
        raise ValueError("official oracle seed requested outside execution context")
    if count is None:
        return _ACTIVE_CONTEXT.seed()
    return jnp.stack([_ACTIVE_CONTEXT.seed() for _ in range(count)])


@contextmanager
def _activate(context: _ParameterContext) -> Iterator[None]:
    global _ACTIVE_CONTEXT
    if _ACTIVE_CONTEXT is not None:
        raise ValueError("official oracle parameter contexts cannot be nested")
    _ACTIVE_CONTEXT = context
    try:
        yield
    finally:
        _ACTIVE_CONTEXT = None


class _Einops:
    @staticmethod
    def rearrange(value: jax.Array, pattern: str, **axes: int) -> jax.Array:
        if pattern == "... (g h) -> ... g h":
            groups = axes["g"]
            return value.reshape((*value.shape[:-1], groups, value.shape[-1] // groups))
        if pattern == "... g h -> ... (g h)":
            return value.reshape((*value.shape[:-2], value.shape[-2] * value.shape[-1]))
        if pattern == "... (g h w c) -> ... h w (g c)":
            groups, height, width = axes["g"], axes["h"], axes["w"]
            channels = value.shape[-1] // (groups * height * width)
            value = value.reshape((*value.shape[:-1], groups, height, width, channels))
            order = (
                *range(value.ndim - 4),
                value.ndim - 3,
                value.ndim - 2,
                value.ndim - 4,
                value.ndim - 1,
            )
            value = jnp.transpose(value, order)
            return value.reshape((*value.shape[:-2], groups * channels))
        raise ValueError(f"unsupported exact official rearrange pattern: {pattern}")

    @staticmethod
    def einsum(*args: Any, **kwargs: Any) -> jax.Array:
        raise ValueError(f"unsupported exact official einsum call: {args}, {kwargs}")


class _Space:
    def __init__(
        self,
        dtype: Any,
        shape: Sequence[int] = (),
        low: Any = None,
        high: Any = None,
        *,
        classes: int | Sequence[int] | np.ndarray | None = None,
    ) -> None:
        del low, high
        self.dtype = jnp.dtype(dtype)
        self.shape = tuple(shape)
        self.discrete = classes is not None
        if classes is None:
            self.classes = None
        else:
            values = np.asarray(classes, np.int32)
            if values.shape == ():
                values = np.full(self.shape or (), values.item(), np.int32)
            elif values.shape != self.shape:
                raise ValueError("official oracle class metadata shape mismatch")
            self.classes = values


def _join_path(prefix: str, name: str) -> str:
    return f"{prefix}/{name}" if prefix else name


def _exec_source(
    source: bytes,
    filename: str,
    namespace: Mapping[str, Any],
) -> ModuleType:
    tree = ast.parse(source, filename)
    tree.body = [
        node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    ast.fix_missing_locations(tree)
    module = ModuleType(filename)
    module.__dict__.update(namespace)
    exec(compile(tree, filename, "exec"), module.__dict__)
    return module


def _load_official_modules(
    sources: Mapping[str, bytes],
    revision: str,
    compute_dtype: Any,
) -> tuple[ModuleType, ModuleType, ModuleType, ModuleType]:
    fake_ninjax = SimpleNamespace(Module=_Module, seed=_seed)
    official_nets = _exec_source(
        sources["embodied/jax/nets.py"],
        f"{revision}:embodied/jax/nets.py",
        {
            "Callable": Callable,
            "adc": adc,
            "einops": _Einops,
            "functools": functools,
            "jax": jax,
            "jnp": jnp,
            "math": math,
            "nj": fake_ninjax,
            "np": np,
        },
    )
    official_nets.COMPUTE_DTYPE = {
        "bfloat16": jnp.bfloat16,
        "float32": jnp.float32,
    }[jnp.dtype(compute_dtype).name]
    official_outs = SimpleNamespace(
        Agg=AggregateOutput,
        Binary=BinaryOutput,
        Categorical=CategoricalOutput,
        MSE=MSEOutput,
        Normal=NormalOutput,
        OneHot=OneHotOutput,
        TwoHot=TwoHotOutput,
    )
    official_heads = _exec_source(
        sources["embodied/jax/heads.py"],
        f"{revision}:embodied/jax/heads.py",
        {
            "Callable": Callable,
            "elements": SimpleNamespace(Space=_Space),
            "jax": jax,
            "jnp": jnp,
            "nets": official_nets,
            "nj": fake_ninjax,
            "np": np,
            "outs": official_outs,
        },
    )
    embodied = SimpleNamespace(
        jax=SimpleNamespace(
            DictHead=official_heads.DictHead,
            outs=official_outs,
        )
    )
    official_rssm = _exec_source(
        sources["dreamerv3/rssm.py"],
        f"{revision}:dreamerv3/rssm.py",
        {
            "einops": _Einops,
            "elements": SimpleNamespace(Space=_Space),
            "embodied": embodied,
            "jax": jax,
            "jnp": jnp,
            "math": math,
            "nj": fake_ninjax,
            "nn": official_nets,
            "np": np,
        },
    )
    return official_nets, official_heads, official_rssm, official_outs


def _bind(module: _Module, context: _ParameterContext) -> _Module:
    module._bind(context, "")
    return module


def _collect_case(
    arrays: dict[str, np.ndarray],
    prefix: str,
    context: _ParameterContext,
    output: Any,
) -> None:
    arrays[f"{prefix}.output"] = _host(output)
    for path, value in sorted(context.parameters.items()):
        arrays[f"{prefix}.param.{path.replace('/', '__')}"] = _host(value)


def _host(value: Any) -> np.ndarray:
    return np.asarray(jax.device_get(value))


def _input_grid(shape: tuple[int, ...], low: float, high: float) -> jax.Array:
    return jnp.linspace(low, high, math.prod(shape), dtype=jnp.float32).reshape(shape)


def _official_network_arrays(
    nets: ModuleType,
    heads: ModuleType,
    rssm: ModuleType,
    profile: DreamerProfile,
    seed: int,
    compute_dtype: Any,
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        "execution.compute_dtype": np.frombuffer(
            jnp.dtype(compute_dtype).name.encode(),
            np.uint8,
        )
    }
    case_index = 0

    def context() -> _ParameterContext:
        nonlocal case_index
        result = _ParameterContext(seed + 1009 * case_index)
        case_index += 1
        return result

    initializer_cases = {
        "zeros": ("zeros", 1.0, (3, 4), ()),
        "uniform_in": ("uniform_in", 0.75, (3, 4), ()),
        "normal_out": ("normal_out", 1.25, (3, 4), ()),
        "trunc_normal_avg": ("trunc_normal_avg", 0.5, (3, 4), (2, 3, 4)),
        "normed_none": ("normed_none", 1.0, (3, 4), ()),
    }
    for index, (case, (name, scale, shape, fshape)) in enumerate(
        initializer_cases.items()
    ):
        key = jax.random.PRNGKey(seed + index + 1)
        dist, fan = (
            name.rsplit("_", 1)
            if name.endswith(("_in", "_out", "_avg", "_none"))
            else (name, "in")
        )
        initializer = nets.Initializer(dist, fan, scale)
        init_context = _ParameterContext(seed, fixed_seed=key)
        with _activate(init_context):
            output = initializer(shape, jnp.float32, fshape or None)
        arrays[f"initializer.{case}.fshape"] = np.asarray(fshape, np.int32)
        arrays[f"initializer.{case}.name"] = np.frombuffer(name.encode(), np.uint8)
        arrays[f"initializer.{case}.output"] = _host(output)
        arrays[f"initializer.{case}.scale"] = np.asarray(scale, np.float32)
        arrays[f"initializer.{case}.seed"] = _host(key)
        arrays[f"initializer.{case}.shape"] = np.asarray(shape, np.int32)

    compute_dtype = {
        "bfloat16": jnp.bfloat16,
        "float32": jnp.float32,
    }[jnp.dtype(compute_dtype).name]
    rms_input = _input_grid((2, 3), -2.0, 3.0).astype(compute_dtype)
    rms_context = context()
    rms = _bind(nets.Norm("rms"), rms_context)
    with _activate(rms_context):
        rms_output = rms(rms_input)
    arrays["rms.input"] = _host(rms_input)
    _collect_case(arrays, "rms", rms_context, rms_output)

    linear_input = _input_grid((2, 3), -1.0, 2.0).astype(compute_dtype)
    linear_context = context()
    linear = _bind(nets.Linear(5, winit="trunc_normal_in"), linear_context)
    with _activate(linear_context):
        linear_output = linear(linear_input)
    arrays["linear.input"] = _host(linear_input)
    _collect_case(arrays, "linear", linear_context, linear_output)

    block_input = _input_grid((2, 8), -2.0, 1.0).astype(compute_dtype)
    block_context = context()
    block = _bind(nets.BlockLinear(8, 4, winit="trunc_normal_in"), block_context)
    with _activate(block_context):
        block_output = block(block_input)
    arrays["blocklinear.input"] = _host(block_input)
    _collect_case(arrays, "blocklinear", block_context, block_output)

    conv_input = _input_grid((1, 6, 6, 2), -1.0, 1.0).astype(compute_dtype)
    for prefix, transposed in (("conv", False), ("transposed_conv", True)):
        conv_context = context()
        conv = _bind(
            nets.Conv2D(
                3,
                3,
                2,
                transp=transposed,
                winit="trunc_normal_in",
            ),
            conv_context,
        )
        with _activate(conv_context):
            conv_output = conv(conv_input)
        arrays[f"{prefix}.input"] = _host(conv_input)
        _collect_case(arrays, prefix, conv_context, conv_output)

    mlp_input = _input_grid((2, 4), -2.0, 2.0)
    mlp_context = context()
    mlp = _bind(
        nets.MLP(
            2,
            6,
            act="silu",
            norm="rms",
            winit="trunc_normal_in",
        ),
        mlp_context,
    )
    with _activate(mlp_context):
        mlp_output = mlp(mlp_input)
    arrays["mlp.input"] = _host(mlp_input)
    _collect_case(arrays, "mlp", mlp_context, mlp_output)

    deter = _input_grid((2, 16), -1.5, 2.0).astype(compute_dtype)
    stoch = _input_grid((2, 2, 3), -1.0, 1.0).astype(compute_dtype)
    action = jnp.asarray([[3.0, -0.5, 1.5], [-2.0, 0.25, 4.0]], compute_dtype)
    is_first = jnp.asarray([True, False])
    blockgru_context = context()
    blockgru = _bind(
        rssm.RSSM(
            {},
            deter=16,
            hidden=8,
            stoch=2,
            classes=3,
            blocks=8,
            dynlayers=1,
            act="silu",
            norm="rms",
            winit="trunc_normal_in",
        ),
        blockgru_context,
    )

    def blockgru_call(candidate_action: jax.Array) -> jax.Array:
        masked_deter, masked_stoch, masked_action = nets.mask(
            (deter, stoch, candidate_action),
            ~is_first,
        )
        return blockgru._core(masked_deter, masked_stoch, masked_action)

    with _activate(blockgru_context):
        blockgru_output = blockgru_call(action)
        blockgru_grad = jax.grad(lambda value: blockgru_call(value).sum())(action)
    arrays.update(
        {
            "blockgru.action": _host(action),
            "blockgru.deter": _host(deter),
            "blockgru.grad_action": _host(blockgru_grad),
            "blockgru.is_first": _host(is_first),
            "blockgru.stoch": _host(stoch),
        }
    )
    _collect_case(arrays, "blockgru", blockgru_context, blockgru_output)

    spaces = {
        "a_image": _Space(jnp.float32, (32, 32, 1)),
        "m_vector": _Space(jnp.float16, (2,)),
        "z_image": _Space(jnp.float32, (32, 32, 2)),
        "z_vector": _Space(jnp.float32, (1,)),
    }
    image_values = np.arange(2 * 2 * 32 * 32 * 3, dtype=np.uint32)
    image_values = (image_values % 256).astype(np.uint8).reshape(2, 2, 32, 32, 3)
    observations = {
        "a_image": jnp.asarray(image_values[..., :1]),
        "m_vector": _input_grid((2, 2, 2), -3.0, 5.0),
        "z_image": jnp.asarray(image_values[..., 1:]),
        "z_vector": _input_grid((2, 2, 1), -10.0, 10.0),
    }
    reset = jnp.zeros((2, 2), bool)
    observation_metadata = {
        "is_first": reset,
        "is_last": jnp.asarray([[False, False], [False, True]]),
        "is_terminal": jnp.asarray([[False, False], [True, False]]),
        "reward": _input_grid((2, 2), -1.0, 1.0),
    }
    encoder_kwargs = dict(
        units=6,
        norm="rms",
        act="silu",
        depth=2,
        mults=(1, 2, 2),
        layers=2,
        kernel=3,
        symlog=True,
        outer=False,
        strided=profile is DreamerProfile.PAPER,
        winit="trunc_normal_in",
    )
    for prefix, selected_spaces in (
        ("encoder", spaces),
        (
            "encoder_image",
            {key: value for key, value in spaces.items() if "image" in key},
        ),
        (
            "encoder_vector",
            {key: value for key, value in spaces.items() if "vector" in key},
        ),
    ):
        selected_obs = {key: observations[key] for key in selected_spaces}
        source_obs = (
            {**selected_obs, **observation_metadata}
            if prefix == "encoder"
            else selected_obs
        )
        encoder_context = context()
        encoder = _bind(
            rssm.Encoder(selected_spaces, **encoder_kwargs), encoder_context
        )
        with _activate(encoder_context):
            _, _, encoder_output = encoder({}, source_obs, reset, False)
        for key, value in source_obs.items():
            if prefix == "encoder":
                arrays[f"encoder.input.{key}"] = _host(value)
        _collect_case(arrays, prefix, encoder_context, encoder_output)
        if prefix == "encoder":
            with _activate(encoder_context):
                _, _, subset_output = encoder({}, selected_obs, reset, False)
            arrays["encoder.subset_output"] = _host(subset_output)
            for missing_key in selected_spaces:
                missing_obs = {
                    key: value
                    for key, value in source_obs.items()
                    if key != missing_key
                }
                try:
                    with _activate(encoder_context):
                        encoder({}, missing_obs, reset, False)
                except KeyError:
                    arrays[f"encoder.missing.{missing_key}.rejected"] = np.asarray(
                        1, np.uint8
                    )
                else:
                    raise AssertionError(
                        f"official encoder accepted missing key: {missing_key}"
                    )

    invalid_image_context = context()
    invalid_image_encoder = _bind(
        rssm.Encoder(
            {"image": _Space(jnp.float32, (32, 32, 1))},
            **encoder_kwargs,
        ),
        invalid_image_context,
    )
    try:
        with _activate(invalid_image_context):
            invalid_image_encoder(
                {},
                {"image": jnp.zeros((2, 2, 32, 32, 1), jnp.float32)},
                reset,
                False,
            )
    except Exception:
        arrays["partition.encoder_rank3_float_rejected"] = np.asarray(1, np.uint8)
    else:
        raise AssertionError("official rank-three float encoder case was not rejected")

    decoder_spaces = {
        "a_image": _Space(jnp.float32, (32, 32, 1)),
        "m_cont": _Space(jnp.float32, (2,)),
        "z_disc": _Space(jnp.int32, (1,), classes=3),
        "z_image": _Space(jnp.float32, (32, 32, 2)),
    }
    features = {
        "deter": _input_grid((2, 2, 16), -1.0, 1.5),
        "logit": _input_grid((2, 2, 2, 3), -2.0, 2.5),
        "stoch": _input_grid((2, 2, 2, 3), -0.75, 1.25),
    }
    decoder_kwargs = dict(
        units=6,
        norm="rms",
        act="silu",
        outscale=1.0,
        depth=2,
        mults=(1, 2, 2),
        layers=2,
        kernel=3,
        symlog=True,
        bspace=2,
        outer=False,
        strided=profile is DreamerProfile.PAPER,
        winit="trunc_normal_in",
    )
    decoder_context = context()
    decoder = _bind(rssm.Decoder(decoder_spaces, **decoder_kwargs), decoder_context)
    with _activate(decoder_context):
        _, _, decoder_outputs = decoder({}, features, reset, False)
    arrays["decoder.deter"] = _host(features["deter"])
    arrays["decoder.logit"] = _host(features["logit"])
    arrays["decoder.stoch"] = _host(features["stoch"])
    decoder_targets = {
        "a_image": jnp.asarray(image_values[..., :1], jnp.float32) / 255,
        "m_cont": _input_grid((2, 2, 2), -1.5, 2.0),
        "z_disc": jnp.asarray([[[0], [1]], [[2], [0]]], jnp.int32),
        "z_image": jnp.asarray(image_values[..., 1:], jnp.float32) / 255,
    }
    for key, output in decoder_outputs.items():
        arrays[f"decoder.target.{key}"] = _host(decoder_targets[key])
        arrays[f"decoder.loss.{key}"] = _host(output.loss(decoder_targets[key]))
        arrays[f"decoder.pred.{key}"] = _host(output.pred())
    for path, value in sorted(decoder_context.parameters.items()):
        arrays[f"decoder.param.{path.replace('/', '__')}"] = _host(value)

    subset_features = {key: features[key] for key in ("deter", "stoch")}
    with _activate(decoder_context):
        _, _, decoder_subset_outputs = decoder({}, subset_features, reset, False)
    for key, output in decoder_subset_outputs.items():
        arrays[f"decoder.subset.loss.{key}"] = _host(output.loss(decoder_targets[key]))
        arrays[f"decoder.subset.pred.{key}"] = _host(output.pred())
    for missing_key in ("deter", "stoch"):
        missing_features = {
            key: value for key, value in features.items() if key != missing_key
        }
        try:
            with _activate(decoder_context):
                decoder({}, missing_features, reset, False)
        except KeyError:
            arrays[f"decoder.missing.{missing_key}.rejected"] = np.asarray(1, np.uint8)
        else:
            raise AssertionError(
                f"official decoder accepted missing feature: {missing_key}"
            )

    vector_decoder_spaces = {key: decoder_spaces[key] for key in ("m_cont", "z_disc")}
    vector_decoder_context = context()
    vector_decoder = _bind(
        rssm.Decoder(vector_decoder_spaces, **decoder_kwargs),
        vector_decoder_context,
    )
    with _activate(vector_decoder_context):
        _, _, vector_decoder_outputs = vector_decoder({}, features, reset, False)
    for key, output in vector_decoder_outputs.items():
        arrays[f"decoder_vector.pred.{key}"] = _host(output.pred())
    for path, value in sorted(vector_decoder_context.parameters.items()):
        arrays[f"decoder_vector.param.{path.replace('/', '__')}"] = _host(value)

    image_decoder_context = context()
    image_decoder = _bind(
        rssm.Decoder({"image": _Space(jnp.float32, (32, 32, 3))}, **decoder_kwargs),
        image_decoder_context,
    )
    with _activate(image_decoder_context):
        _, _, image_decoder_output = image_decoder({}, features, reset, False)
    _collect_case(
        arrays,
        "decoder_image",
        image_decoder_context,
        image_decoder_output["image"].pred(),
    )

    rank3_float_decoder_context = context()
    rank3_float_decoder = _bind(
        rssm.Decoder(
            {"image": _Space(jnp.float32, (32, 32, 1))},
            **decoder_kwargs,
        ),
        rank3_float_decoder_context,
    )
    with _activate(rank3_float_decoder_context):
        _, _, rank3_float_output = rank3_float_decoder({}, features, reset, False)
    rank3_float_shape = rank3_float_output["image"].pred().shape
    arrays["partition.decoder_rank3_float_is_image"] = np.asarray(
        rank3_float_shape[-3:] == (32, 32, 1),
        np.uint8,
    )

    ordered_spaces = {
        "z_image": _Space(jnp.uint8, (32, 32, 2)),
        "a_image": _Space(jnp.uint8, (32, 32, 1)),
    }
    ordered_context = context()
    ordered_decoder = _bind(
        rssm.Decoder(ordered_spaces, **decoder_kwargs),
        ordered_context,
    )
    with _activate(ordered_context):
        _, _, ordered_outputs = ordered_decoder({}, features, reset, False)
    arrays["decoder_order.deter"] = _host(features["deter"])
    arrays["decoder_order.stoch"] = _host(features["stoch"])
    ordered_targets = {
        "z_image": jnp.linspace(
            0.0,
            1.0,
            2 * 2 * 32 * 32 * 2,
            dtype=jnp.float32,
        ).reshape((2, 2, 32, 32, 2)),
        "a_image": jnp.linspace(
            1.0,
            0.0,
            2 * 2 * 32 * 32,
            dtype=jnp.float32,
        ).reshape((2, 2, 32, 32, 1)),
    }
    for key, output in ordered_outputs.items():
        arrays[f"decoder_order.pred.{key}"] = _host(output.pred())
        arrays[f"decoder_order.target.{key}"] = _host(ordered_targets[key])
        arrays[f"decoder_order.loss.{key}"] = _host(output.loss(ordered_targets[key]))
    for path, value in sorted(ordered_context.parameters.items()):
        arrays[f"decoder_order.param.{path.replace('/', '__')}"] = _host(value)

    head_cases = {
        "reward": (
            _Space(jnp.float32, ()),
            "symexp_twohot",
            1,
            dict(outscale=0.0, bins=255),
        ),
        "continue": (
            _Space(jnp.bool_, (), classes=2),
            "binary",
            1,
            dict(outscale=1.0),
        ),
        "policy": (
            _Space(jnp.float32, (3,)),
            "bounded_normal",
            2,
            dict(outscale=0.01, minstd=0.1, maxstd=1.0),
        ),
        "categorical": (
            _Space(jnp.int32, (1,), classes=4),
            "categorical",
            1,
            dict(outscale=1.0),
        ),
    }
    for name, (space, output_family, layers, output_kwargs) in head_cases.items():
        head_input = _input_grid((2, 4), -1.0, 2.0)
        head_context = context()
        head = _bind(
            heads.MLPHead(
                space,
                output_family,
                units=6,
                layers=layers,
                act="silu",
                norm="rms",
                winit="trunc_normal_in",
                **output_kwargs,
            ),
            head_context,
        )
        with _activate(head_context):
            head_output = head(head_input, 1)
        arrays[f"head.{name}.input"] = _host(head_input)
        _collect_case(arrays, f"head.{name}", head_context, head_output.pred())

    family_cases = {
        "binary_scalar": (_Space(jnp.bool_, (), classes=2), "binary", {}),
        "binary_vector": (_Space(jnp.bool_, (2,), classes=2), "binary", {}),
        "categorical_scalar": (
            _Space(jnp.int32, (), classes=4),
            "categorical",
            {},
        ),
        "categorical_vector": (
            _Space(jnp.int32, (2,), classes=3),
            "categorical",
            {},
        ),
        "onehot_scalar": (
            _Space(jnp.int32, (), classes=3),
            "onehot",
            {"unimix": 0.01},
        ),
        "onehot_vector": (
            _Space(jnp.int32, (2,), classes=3),
            "onehot",
            {"unimix": 0.01},
        ),
        "mse_scalar": (_Space(jnp.float32, ()), "mse", {}),
        "mse_vector": (_Space(jnp.float32, (2,)), "mse", {}),
        "symlog_mse_scalar": (_Space(jnp.float32, ()), "symlog_mse", {}),
        "symlog_mse_vector": (_Space(jnp.float32, (2,)), "symlog_mse", {}),
        "symexp_twohot_scalar": (
            _Space(jnp.float32, ()),
            "symexp_twohot",
            {"bins": 7},
        ),
        "symexp_twohot_vector": (
            _Space(jnp.float32, (2,)),
            "symexp_twohot",
            {"bins": 7},
        ),
        "bounded_normal_scalar": (
            _Space(jnp.float32, ()),
            "bounded_normal",
            {"minstd": 0.1, "maxstd": 1.0},
        ),
        "bounded_normal_vector": (
            _Space(jnp.float32, (2,)),
            "bounded_normal",
            {"minstd": 0.1, "maxstd": 1.0},
        ),
    }
    for name, (space, output_family, output_kwargs) in family_cases.items():
        head_input = _input_grid((2, 4), -1.0, 2.0)
        head_context = context()
        head = _bind(
            heads.MLPHead(
                space,
                output_family,
                units=6,
                layers=1,
                act="silu",
                norm="rms",
                winit="trunc_normal_in",
                **output_kwargs,
            ),
            head_context,
        )
        with _activate(head_context):
            head_output = head(head_input, 1)
        prefix = f"head_family.{name}"
        arrays[f"{prefix}.input"] = _host(head_input)
        _collect_case(arrays, prefix, head_context, head_output.pred())
        raw_output = (
            head_output.output if hasattr(head_output, "output") else head_output
        )
        if hasattr(raw_output, "minent"):
            arrays[f"{prefix}.minent"] = _host(raw_output.minent)
            arrays[f"{prefix}.maxent"] = _host(raw_output.maxent)

    invalid_head_cases = {
        "binary": (_Space(jnp.float32, ()), "binary"),
        "categorical": (_Space(jnp.float32, (2,)), "categorical"),
        "onehot": (_Space(jnp.float32, (2,)), "onehot"),
        "mse": (_Space(jnp.int32, (2,), classes=3), "mse"),
        "symlog_mse": (_Space(jnp.int32, (2,), classes=3), "symlog_mse"),
        "symexp_twohot": (
            _Space(jnp.int32, (2,), classes=3),
            "symexp_twohot",
        ),
        "bounded_normal": (
            _Space(jnp.int32, (2,), classes=3),
            "bounded_normal",
        ),
        "categorical_nonuniform": (
            _Space(jnp.int32, (2,), classes=(2, 3)),
            "categorical",
        ),
        "onehot_nonuniform": (
            _Space(jnp.int32, (2,), classes=(2, 3)),
            "onehot",
        ),
    }
    for name, (space, output_family) in invalid_head_cases.items():
        invalid_context = context()
        try:
            invalid_head = _bind(
                heads.MLPHead(
                    space,
                    output_family,
                    units=6,
                    layers=1,
                    act="silu",
                    norm="rms",
                    winit="trunc_normal_in",
                    bins=7,
                ),
                invalid_context,
            )
            with _activate(invalid_context):
                invalid_head(_input_grid((2, 4), -1.0, 2.0), 1)
        except Exception:
            arrays[f"head_invalid.{name}.rejected"] = np.asarray(1, np.uint8)
        else:
            raise AssertionError(f"official invalid head case was accepted: {name}")

    return {name: np.asarray(value) for name, value in sorted(arrays.items())}


def _networks_worker(request: Mapping[str, Any]) -> dict[str, Any]:
    checkout = Path(request["official_checkout"]).resolve()
    revision = str(request["official_commit"])
    profile = DreamerProfile(request["profile"])
    compute_dtype = jnp.dtype(request["compute_dtype"]).name
    if compute_dtype not in NETWORKS_SOURCE_SPEC.execution_dtypes:
        raise ValueError("network worker compute dtype is not source-spec authorized")
    ObservationMode(request["observation_mode"])
    if revision != official_revision(profile):
        raise ValueError("network worker revision does not match requested profile")
    if dict(request["overrides"]) != dict(profile_overrides(profile)):
        raise ValueError("network worker override map does not match requested profile")
    if request["source_spec"] != NETWORKS_SOURCE_SPEC.name:
        raise ValueError("network worker requires the registered networks source spec")
    sources = {}
    for path, digest in NETWORKS_SOURCE_SPEC.hashes_for(revision).items():
        source = _git_show(checkout, revision, path)
        if _sha256_bytes(source) != digest:
            raise ValueError(f"official network source hash mismatch: {path}")
        sources[path] = source
    nets, heads, rssm, _ = _load_official_modules(
        sources,
        revision,
        compute_dtype,
    )
    arrays = _official_network_arrays(
        nets,
        heads,
        rssm,
        profile,
        int(request["seed"]),
        compute_dtype,
    )
    return {
        "arrays": {
            name: {"dtype": value.dtype.name, "values": value.tolist()}
            for name, value in arrays.items()
        },
        "compute_dtype": compute_dtype,
        "worker_pid": os.getpid(),
    }


def run_networks_case(
    harness: OracleHarness,
    profile: DreamerProfile | str,
    observation_mode: ObservationMode | str,
    *,
    case_name: str | None = None,
    seed: int = 0,
    compute_dtype: str = "bfloat16",
) -> tuple[Path, Path]:
    resolved_profile = DreamerProfile(profile)
    resolved_mode = ObservationMode(observation_mode)
    resolved_dtype = jnp.dtype(compute_dtype).name
    if resolved_dtype not in NETWORKS_SOURCE_SPEC.execution_dtypes:
        raise ValueError("network oracle compute dtype is not source-spec authorized")
    expected_case_name = (
        "networks" if resolved_dtype == "bfloat16" else "networks-float32"
    )
    resolved_case_name = case_name or expected_case_name
    if resolved_case_name != expected_case_name:
        raise ValueError("network oracle case name does not identify execution dtype")
    request = {
        "compute_dtype": resolved_dtype,
        "official_checkout": str(harness.official_checkout),
        "official_commit": official_revision(resolved_profile),
        "observation_mode": resolved_mode.value,
        "overrides": dict(profile_overrides(resolved_profile)),
        "profile": resolved_profile.value,
        "seed": seed,
        "source_spec": NETWORKS_SOURCE_SPEC.name,
    }
    command = (
        harness.python_executable,
        str(Path(__file__).resolve()),
        "_networks_worker",
    )
    completed = subprocess.run(
        command,
        cwd=harness.official_checkout,
        input=_canonical_json(request).decode(),
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    if payload.get("compute_dtype") != resolved_dtype:
        raise ValueError("oracle network worker reported a different compute dtype")
    worker_pid = int(payload["worker_pid"])
    if worker_pid <= 0 or worker_pid == os.getpid():
        raise ValueError("oracle network worker did not cross a process boundary")
    harness._last_worker_pid = worker_pid
    arrays = {}
    for name, spec in payload["arrays"].items():
        value = np.asarray(spec["values"], dtype=spec["dtype"])
        if value.dtype.name == "bfloat16":
            value = value.astype(np.float32)
        arrays[name] = value
    return harness.write_fixture(
        case_name=resolved_case_name,
        profile=resolved_profile,
        observation_mode=resolved_mode,
        arrays=arrays,
        seed=seed,
        generator_command=command,
        generator_request=request,
        source_spec=NETWORKS_SOURCE_SPEC,
        dtype=resolved_dtype,
    )


def _main(argv: Sequence[str]) -> int:
    if tuple(argv) != ("_networks_worker",):
        raise SystemExit("network_oracle.py is an internal fixture worker")
    request = json.loads(sys.stdin.read())
    sys.stdout.write(json.dumps(_networks_worker(request), sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - subprocess boundary.
    raise SystemExit(_main(sys.argv[1:]))


__all__ = ["NETWORKS_SOURCE_SPEC", "run_networks_case"]
