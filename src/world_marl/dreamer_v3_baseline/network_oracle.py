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
        classes: int | None = None,
    ) -> None:
        del low, high
        self.dtype = jnp.dtype(dtype)
        self.shape = tuple(shape)
        self.discrete = classes is not None
        self.classes = (
            np.full(self.shape or (), classes, np.int32)
            if classes is not None
            else None
        )


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
    official_nets.COMPUTE_DTYPE = jnp.float32
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
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
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

    rms_input = _input_grid((2, 3), -2.0, 3.0)
    rms_context = context()
    rms = _bind(nets.Norm("rms"), rms_context)
    with _activate(rms_context):
        rms_output = rms(rms_input)
    arrays["rms.input"] = _host(rms_input)
    _collect_case(arrays, "rms", rms_context, rms_output)

    linear_input = _input_grid((2, 3), -1.0, 2.0)
    linear_context = context()
    linear = _bind(nets.Linear(5, winit="trunc_normal_in"), linear_context)
    with _activate(linear_context):
        linear_output = linear(linear_input)
    arrays["linear.input"] = _host(linear_input)
    _collect_case(arrays, "linear", linear_context, linear_output)

    block_input = _input_grid((2, 8), -2.0, 1.0)
    block_context = context()
    block = _bind(nets.BlockLinear(8, 4, winit="trunc_normal_in"), block_context)
    with _activate(block_context):
        block_output = block(block_input)
    arrays["blocklinear.input"] = _host(block_input)
    _collect_case(arrays, "blocklinear", block_context, block_output)

    conv_input = _input_grid((1, 6, 6, 2), -1.0, 1.0)
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

    deter = _input_grid((2, 16), -1.5, 2.0)
    stoch = _input_grid((2, 2, 3), -1.0, 1.0)
    action = jnp.asarray([[3.0, -0.5, 1.5], [-2.0, 0.25, 4.0]], jnp.float32)
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
        "a_image": _Space(jnp.uint8, (32, 32, 1)),
        "m_vector": _Space(jnp.float32, (2,)),
        "z_image": _Space(jnp.uint8, (32, 32, 2)),
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
        encoder_context = context()
        encoder = _bind(
            rssm.Encoder(selected_spaces, **encoder_kwargs), encoder_context
        )
        with _activate(encoder_context):
            _, _, encoder_output = encoder({}, selected_obs, reset, False)
        for key, value in selected_obs.items():
            if prefix == "encoder":
                arrays[f"encoder.input.{key}"] = _host(value)
        _collect_case(arrays, prefix, encoder_context, encoder_output)

    decoder_spaces = {
        "a_image": _Space(jnp.uint8, (32, 32, 1)),
        "m_cont": _Space(jnp.float32, (2,)),
        "z_disc": _Space(jnp.int32, (1,), classes=3),
        "z_image": _Space(jnp.uint8, (32, 32, 2)),
    }
    features = {
        "deter": _input_grid((2, 2, 16), -1.0, 1.5),
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
    arrays["decoder.stoch"] = _host(features["stoch"])
    image_target = jnp.asarray(image_values[..., :1], jnp.float32) / 255
    arrays["decoder.target.a_image"] = _host(image_target)
    arrays["decoder.loss.a_image"] = _host(
        decoder_outputs["a_image"].loss(image_target)
    )
    for key, output in decoder_outputs.items():
        arrays[f"decoder.pred.{key}"] = _host(output.pred())
    for path, value in sorted(decoder_context.parameters.items()):
        arrays[f"decoder.param.{path.replace('/', '__')}"] = _host(value)

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
        rssm.Decoder({"image": _Space(jnp.uint8, (32, 32, 3))}, **decoder_kwargs),
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

    return {name: np.asarray(value) for name, value in sorted(arrays.items())}


def _networks_worker(request: Mapping[str, Any]) -> dict[str, Any]:
    checkout = Path(request["official_checkout"]).resolve()
    revision = str(request["official_commit"])
    profile = DreamerProfile(request["profile"])
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
    nets, heads, rssm, _ = _load_official_modules(sources, revision)
    arrays = _official_network_arrays(nets, heads, rssm, profile, int(request["seed"]))
    return {
        "arrays": {
            name: {"dtype": value.dtype.name, "values": value.tolist()}
            for name, value in arrays.items()
        },
        "worker_pid": os.getpid(),
    }


def run_networks_case(
    harness: OracleHarness,
    profile: DreamerProfile | str,
    observation_mode: ObservationMode | str,
    *,
    case_name: str = "networks",
    seed: int = 0,
) -> tuple[Path, Path]:
    resolved_profile = DreamerProfile(profile)
    resolved_mode = ObservationMode(observation_mode)
    request = {
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
    worker_pid = int(payload["worker_pid"])
    if worker_pid <= 0 or worker_pid == os.getpid():
        raise ValueError("oracle network worker did not cross a process boundary")
    harness._last_worker_pid = worker_pid
    arrays = {
        name: np.asarray(spec["values"], dtype=spec["dtype"])
        for name, spec in payload["arrays"].items()
    }
    return harness.write_fixture(
        case_name=case_name,
        profile=resolved_profile,
        observation_mode=resolved_mode,
        arrays=arrays,
        seed=seed,
        generator_command=command,
        generator_request=request,
        source_spec=NETWORKS_SOURCE_SPEC,
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
