from __future__ import annotations

import ast
import json
import math
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import yaml

if __package__:
    from . import network_oracle as _network_oracle
    from .config import DreamerProfile, ObservationMode, resolve_dreamer_config
    from .network_oracle import (
        _ParameterContext,
        _Space,
        _activate,
        _bind,
        _exec_source,
        _load_official_modules,
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
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import world_marl.dreamer_v3_baseline.network_oracle as _network_oracle
    from world_marl.dreamer_v3_baseline.config import (
        DreamerProfile,
        ObservationMode,
        resolve_dreamer_config,
    )
    from world_marl.dreamer_v3_baseline.network_oracle import (
        _ParameterContext,
        _Space,
        _activate,
        _bind,
        _exec_source,
        _load_official_modules,
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
    "dreamerv3/agent.py": (
        "adce8e4274bc098c218bf9a20fd3327545f0ad7d850b5fe328597382e91b5269"
    ),
    "dreamerv3/configs.yaml": (
        "9dff9c7062e3e33951cb54c6dd4b598aaf7e56e18e2cff39c812eaa797bcfcfc"
    ),
    "dreamerv3/rssm.py": (
        "d6d50166914e94fb8bd17a5d5dbda9d42cdd37b85819bb1e9fff3a64d4ad2eb6"
    ),
    "embodied/jax/heads.py": (
        "437641cde21e7f9e3f69b88ad8f6b7e7c22e54eec8c5b19eef6127afde1a9b3f"
    ),
    "embodied/jax/nets.py": (
        "9a1c0c71ad7d3596572a44416e78434f777d8f4dbcbe8ca0dd6b86bb8246392c"
    ),
    "embodied/jax/outs.py": (
        "7e80691f175c71be614f089023cce3a809e0d026c6d5ce89bf566d5f11eb3ed0"
    ),
}

RSSM_SOURCE_SPEC = OracleSourceSpec(
    name="rssm",
    revision_hashes={
        PAPER_REVISION: _SOURCE_HASHES,
        UPSTREAM_CURRENT_REVISION: _SOURCE_HASHES,
    },
    execution_dtypes=("bfloat16", "float32"),
)
register_oracle_source_spec(RSSM_SOURCE_SPEC)


def _host(value: Any) -> np.ndarray:
    return np.asarray(jax.device_get(value))


def _grid(shape: tuple[int, ...], low: float, high: float) -> jax.Array:
    return jnp.linspace(low, high, math.prod(shape), dtype=jnp.float32).reshape(shape)


def _scan_keys(root: jax.Array, length: int) -> jax.Array:
    outer = jax.random.split(root, length + 1)[1:]
    return jax.vmap(lambda key: jax.random.split(key, 16)[1])(outer)


_ACTIVE_SCAN_KEYS: tuple[jax.Array, ...] | None = None


@contextmanager
def _scan_key_scope(keys: jax.Array):
    global _ACTIVE_SCAN_KEYS
    if _ACTIVE_SCAN_KEYS is not None:
        raise ValueError("source scan key scopes cannot be nested")
    _ACTIVE_SCAN_KEYS = tuple(keys)
    try:
        yield
    finally:
        _ACTIVE_SCAN_KEYS = None


def _source_scan(fn, carry, inputs, length=None, unroll=1, axis=1):
    del unroll
    if axis != 1:
        raise ValueError("RSSM source adapter only authorizes batch-major scans")
    if inputs == ():
        if length is None:
            raise ValueError("empty source scan requires length")
        steps = [None] * length
    else:
        leaves = jax.tree.leaves(inputs)
        resolved = leaves[0].shape[axis]
        if length is not None and length != resolved:
            raise ValueError("source scan length disagrees with inputs")
        length = resolved
        time_inputs = jax.tree.map(lambda value: jnp.moveaxis(value, axis, 0), inputs)
        steps = [
            jax.tree.map(lambda value, i=i: value[i], time_inputs)
            for i in range(length)
        ]
    outputs = []
    for index, step in enumerate(steps):
        if _ACTIVE_SCAN_KEYS is not None:
            if len(_ACTIVE_SCAN_KEYS) != length:
                raise ValueError("source scan key count mismatch")
            context = _network_oracle._ACTIVE_CONTEXT
            if context is None:
                raise ValueError("source scan ran outside parameter context")
            context.fixed_seed = _ACTIVE_SCAN_KEYS[index]
        carry, output = fn(carry, step)
        outputs.append(output)
    stacked = jax.tree.map(lambda *values: jnp.stack(values, axis), *outputs)
    return carry, stacked


class _OrderedCategoricalNoise:
    def __init__(self, noises: jax.Array):
        self.noises = tuple(jnp.asarray(noise, jnp.float32) for noise in noises)
        self.logits: list[jax.Array] = []
        self.seeds: list[jax.Array] = []

    def __call__(self, seed, logits, axis=-1, shape=None):
        if axis != -1:
            raise ValueError("RSSM categorical axis changed")
        if len(self.logits) >= len(self.noises):
            raise ValueError("RSSM consumed extra supplied categorical noise")
        noise = self.noises[len(self.logits)]
        expected_shape = tuple(logits.shape[:-1])
        if shape is not None and tuple(shape) != expected_shape:
            raise ValueError("RSSM categorical output shape changed")
        if tuple(noise.shape) != tuple(logits.shape):
            raise ValueError("RSSM supplied categorical noise shape mismatch")
        self.logits.append(logits)
        self.seeds.append(seed)
        return jnp.argmax(noise + logits, axis=-1)


class _RSSMSpace(_Space):
    def __init__(self, dtype, shape=(), *args, **kwargs):
        if isinstance(shape, (int, np.integer)):
            shape = (int(shape),)
        super().__init__(dtype, shape, *args, **kwargs)


@contextmanager
def _ordered_noise_scope(noises: jax.Array):
    injected = _OrderedCategoricalNoise(noises)
    original = jax.random.categorical
    jax.random.categorical = injected
    try:
        yield injected
        if len(injected.logits) != len(injected.noises):
            raise ValueError("RSSM left supplied categorical noise unused")
    finally:
        jax.random.categorical = original


@contextmanager
def _categorical_forbidden(label: str):
    original = jax.random.categorical
    evidence = {"calls": 0}

    def forbidden(*args, **kwargs):
        del args, kwargs
        evidence["calls"] += 1
        raise ValueError(f"categorical draw is forbidden during {label}")

    jax.random.categorical = forbidden
    try:
        yield evidence
    finally:
        jax.random.categorical = original


def _gumbel_noise(shape: tuple[int, ...], index: int) -> jax.Array:
    uniform = jnp.linspace(0.05, 0.95, math.prod(shape), dtype=jnp.float32)
    uniform = jnp.roll(uniform, index % uniform.size)
    return (-jnp.log(-jnp.log(uniform))).reshape(shape)


def _load_feat2tensor(source: bytes, revision: str, nets: Any):
    tree = ast.parse(source, f"{revision}:dreamerv3/agent.py")
    matches = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute) and target.attr == "feat2tensor"
            for target in node.targets
        )
        and isinstance(node.value, ast.Lambda)
    ]
    if len(matches) != 1:
        raise ValueError("official Agent feat2tensor lambda is not unique")
    expression = ast.Expression(matches[0])
    ast.fix_missing_locations(expression)
    return eval(
        compile(expression, f"{revision}:dreamerv3/agent.py", "eval"),
        {"jnp": jnp, "nn": nets},
    )


def _source_model_dimensions(source: bytes) -> dict[str, dict[str, int]]:
    config = yaml.safe_load(source)
    base = dict(config["defaults"]["agent"]["dyn"]["rssm"])
    result = {}
    for name in ("size1m", "size200m"):
        values = dict(base)
        values.update(config[name][r".*\.rssm"])
        result[name] = {
            key: int(values[key]) for key in ("deter", "hidden", "stoch", "classes")
        }
    return result


class _InitializationDistribution:
    def __init__(self, logits):
        self.logits = logits

    def sample(self, seed):
        del seed
        return jnp.zeros_like(self.logits, jnp.float32)


def _initialize_parameters(
    module, context, initial, tokens, actions, resets, *, include_posterior
):
    original_dist = module._dist
    module._dist = lambda logits: _InitializationDistribution(logits)
    try:
        with _categorical_forbidden("parameter initialization"):
            with _activate(context):
                if include_posterior:
                    module.observe(
                        initial,
                        tokens[:, 0],
                        {name: value[:, 0] for name, value in actions.items()},
                        resets[:, 0],
                        False,
                        single=True,
                    )
                    module._prior(initial["deter"])
                else:
                    module.imagine(
                        initial,
                        {name: value[:, 0] for name, value in actions.items()},
                        1,
                        False,
                        single=True,
                    )
    finally:
        module._dist = original_dist


def _loss_from_exact_source(module, initial, tokens, actions, resets, post, prior):
    original_observe = module.observe
    original_prior = module._prior
    entries = {
        "deter": jnp.zeros((*post.shape[:2], module.deter), post.dtype),
        "stoch": jnp.zeros_like(post),
    }
    try:
        module.observe = lambda *args, **kwargs: (
            initial,
            entries,
            {"deter": entries["deter"], "logit": post},
        )
        module._prior = lambda deter: prior
        return module.loss(initial, tokens, actions, resets, False)[2]
    finally:
        module.observe = original_observe
        module._prior = original_prior


def _official_arrays_exact(
    nets: Any,
    outs: Any,
    rssm_source: Any,
    feat2tensor: Any,
    seed: int,
    compute_dtype: Any,
    dimensions: Mapping[str, int],
    source_model_dimensions: Mapping[str, Mapping[str, int]],
) -> dict[str, np.ndarray]:
    dtype = {"bfloat16": jnp.bfloat16, "float32": jnp.float32}[
        jnp.dtype(compute_dtype).name
    ]
    batch = dimensions["batch"]
    time = dimensions["time"]
    deter = dimensions["deter"]
    hidden = dimensions["hidden"]
    stoch = dimensions["stoch"]
    classes = dimensions["classes"]
    action_dim = dimensions["action"]
    token_dim = dimensions["token"]
    action_space = {"action": _Space(jnp.float32, (action_dim,))}
    context = _ParameterContext(seed)
    module = _bind(
        rssm_source.RSSM(
            action_space,
            deter=deter,
            hidden=hidden,
            stoch=stoch,
            classes=classes,
            blocks=8,
            free_nats=1.0,
            unimix=0.01,
            act="silu",
            norm="rms",
            imglayers=2,
            obslayers=1,
            dynlayers=1,
            absolute=False,
            outscale=1.0,
            winit="trunc_normal_in",
        ),
        context,
    )
    rssm_source.nj.scan = _source_scan
    with _categorical_forbidden("initial") as initial_draws:
        with _activate(context):
            initial = module.initial(batch)

    actions = {"action": _grid((batch, time, action_dim), -2.0, 2.5)}
    tokens = _grid((batch, time, token_dim), -1.5, 1.75)
    resets = jnp.asarray(
        [[False, False, True, False], [False, True, False, False]], bool
    )
    if resets.shape != (batch, time):
        raise ValueError("RSSM exact case dimensions require batch=2 and time=4")
    _initialize_parameters(
        module, context, initial, tokens, actions, resets, include_posterior=True
    )
    sample_keys = _scan_keys(jax.random.PRNGKey(seed + 97), time)
    observe_noise = jnp.stack(
        [_gumbel_noise((batch, stoch, classes), index) for index in range(time)]
    )
    with _scan_key_scope(sample_keys), _ordered_noise_scope(observe_noise) as draws:
        with _activate(context):
            observe_final, posterior, observe_feat = module.observe(
                initial, tokens, actions, resets, False
            )
    observe_effective = jnp.stack(draws.logits)
    observe_keys = jnp.stack(draws.seeds)
    if not np.array_equal(_host(observe_keys), _host(sample_keys)):
        raise ValueError(
            "source observe categorical seeds do not match Ninjax scan keys"
        )
    if len({tuple(value) for value in _host(observe_keys)}) != time:
        raise ValueError("source observe categorical seeds are not unique")
    with _categorical_forbidden("vectorized prior logits"):
        with _activate(context):
            prior_logits = module._prior(observe_feat["deter"])

    imagine_noise = jnp.stack(
        [_gumbel_noise((batch, stoch, classes), 100 + index) for index in range(time)]
    )
    with _scan_key_scope(sample_keys), _ordered_noise_scope(imagine_noise) as draws:
        with _activate(context):
            imagine_final, imagine_feat, imagine_actions = module.imagine(
                initial, actions, time, False
            )
    imagine_effective = jnp.stack(draws.logits)
    imagine_keys = jnp.stack(draws.seeds)
    if not np.array_equal(_host(imagine_keys), _host(sample_keys)):
        raise ValueError(
            "source imagine categorical seeds do not match Ninjax scan keys"
        )

    with _ordered_noise_scope(observe_noise[:1]) as single_draw:
        context.fixed_seed = sample_keys[0]
        with _activate(context):
            single_post_false, _, single_post_false_feat = module.observe(
                initial,
                tokens[:, 0],
                {"action": actions["action"][:, 0]},
                resets[:, 0],
                False,
                single=True,
            )
    single_post_false_effective = single_draw.logits[0]
    with _ordered_noise_scope(observe_noise[1:2]) as single_draw:
        context.fixed_seed = sample_keys[1]
        previous = {name: value[:, 0] for name, value in posterior.items()}
        with _activate(context):
            single_post_true, _, single_post_true_feat = module.observe(
                previous,
                tokens[:, 1],
                {"action": actions["action"][:, 1]},
                resets[:, 1],
                False,
                single=True,
            )
    single_post_true_effective = single_draw.logits[0]
    with _ordered_noise_scope(imagine_noise[:1]) as single_draw:
        context.fixed_seed = sample_keys[0]
        with _activate(context):
            single_prior, (single_prior_feat, _) = module.imagine(
                initial,
                {"action": actions["action"][:, 0]},
                1,
                False,
                single=True,
            )
    single_prior_effective = single_draw.logits[0]

    changed_tokens = tokens.at[0, :2].set(jnp.float32(999))
    changed_tokens = changed_tokens.at[1, :1].set(jnp.float32(999))
    changed_action = actions["action"].at[0, :2].set(jnp.float32(-999))
    changed_action = changed_action.at[1, :1].set(jnp.float32(-999))
    changed_actions = {"action": changed_action}
    with _scan_key_scope(sample_keys), _ordered_noise_scope(observe_noise):
        with _activate(context):
            _, _, changed_feat = module.observe(
                initial, changed_tokens, changed_actions, resets, False
            )

    post_kl = jnp.asarray([[[[3.0, -1.0, 0.5], [-2.0, 1.0, 2.0]]]], jnp.float32)
    prior_kl = -post_kl

    def dyn_value(post, prior):
        return _loss_from_exact_source(
            module, initial, tokens, actions, resets, post, prior
        )["dyn"]

    def rep_value(post, prior):
        return _loss_from_exact_source(
            module, initial, tokens, actions, resets, post, prior
        )["rep"]

    with _categorical_forbidden("KL and free nats"):
        dyn = dyn_value(post_kl, prior_kl)
        rep = rep_value(post_kl, prior_kl)
        dyn_grad = jax.grad(lambda p, q: dyn_value(p, q).sum(), argnums=(0, 1))(
            post_kl, prior_kl
        )
        rep_grad = jax.grad(lambda p, q: rep_value(p, q).sum(), argnums=(0, 1))(
            post_kl, prior_kl
        )
        equal = jnp.zeros_like(post_kl)
        dyn_free = dyn_value(equal, equal)
        rep_free = rep_value(equal, equal)
        dyn_below_grad = jax.grad(lambda p, q: dyn_value(p, q).sum(), argnums=(0, 1))(
            equal, equal
        )
        rep_below_grad = jax.grad(lambda p, q: rep_value(p, q).sum(), argnums=(0, 1))(
            equal, equal
        )

    first_action = actions["action"][:, 0].astype(dtype)

    def img_objective(state_deter, action):
        with _activate(context):
            next_deter = module._core(state_deter, initial["stoch"], action)
            return module._prior(next_deter).astype(jnp.float32).sum()

    img_deter_grad, img_action_grad = jax.grad(img_objective, argnums=(0, 1))(
        initial["deter"], first_action
    )

    with _categorical_forbidden("reset masking"):
        reset_state = nets.mask(
            (posterior["deter"][:, 1], posterior["stoch"][:, 1]), ~resets[:, 2]
        )
    combined_spaces = {
        "a_cont": _Space(jnp.float32, (2,)),
        "z_disc": _Space(jnp.int32, (1,), classes=3),
    }
    combined_actions = {
        "a_cont": jnp.asarray([[-jnp.inf, 1.0], [3.0, 4.0]], jnp.float32),
        "z_disc": jnp.asarray([[-1], [2]], jnp.int32),
    }
    combined_encoded = nets.DictConcat(combined_spaces, 1)(combined_actions)
    combined_reset = nets.mask(combined_encoded, ~jnp.asarray([True, False]))

    replay_entries = {
        "deter": jnp.arange(batch * time * deter, dtype=jnp.float32).reshape(
            batch, time, deter
        ),
        "stoch": jnp.arange(batch * time * stoch * classes, dtype=jnp.float32).reshape(
            batch, time, stoch, classes
        ),
    }
    replay_carry = module.truncate(replay_entries)
    replay_starts = module.starts(replay_entries, replay_carry, 2)
    entry_space = module.entry_space

    imagine_only_context = _ParameterContext(seed)
    imagine_only = _bind(
        rssm_source.RSSM(
            action_space,
            deter=deter,
            hidden=hidden,
            stoch=stoch,
            classes=classes,
            blocks=8,
            free_nats=1.0,
            unimix=0.01,
            act="silu",
            norm="rms",
            imglayers=2,
            obslayers=1,
            dynlayers=1,
            absolute=False,
            outscale=1.0,
            winit="trunc_normal_in",
        ),
        imagine_only_context,
    )
    with _categorical_forbidden("imagine-only initial"):
        with _activate(imagine_only_context):
            imagine_only_initial = imagine_only.initial(batch)
    _initialize_parameters(
        imagine_only,
        imagine_only_context,
        imagine_only_initial,
        tokens,
        actions,
        resets,
        include_posterior=False,
    )
    with _scan_key_scope(sample_keys), _ordered_noise_scope(imagine_noise):
        with _activate(imagine_only_context):
            imagine_only.imagine(imagine_only_initial, actions, time, False)
    imagine_only_obs_count = sum(
        path.startswith("obs") for path in imagine_only_context.parameters
    )
    if imagine_only_obs_count:
        raise ValueError("source imagine-only path created posterior parameters")

    scan_root = jax.random.PRNGKey(seed + 97)
    scan_split = jax.random.split(scan_root, time + 1)

    arrays: dict[str, Any] = {
        "execution.compute_dtype": np.frombuffer(
            jnp.dtype(dtype).name.encode(), np.uint8
        ),
        **{
            f"dimensions.{name}": np.asarray(value, np.int32)
            for name, value in dimensions.items()
        },
        **{
            f"source_config.{model}.{name}": np.asarray(value, np.int32)
            for model, values in source_model_dimensions.items()
            for name, value in values.items()
        },
        "initial.deter": initial["deter"],
        "initial.stoch": initial["stoch"],
        "observe.action": actions["action"],
        "observe.tokens": tokens,
        "observe.reset": resets,
        "observe.gumbel": observe_noise,
        "observe.keys": observe_keys,
        "observe.effective_logits": jnp.moveaxis(observe_effective, 0, 1),
        "observe.posterior.deter": posterior["deter"],
        "observe.posterior.stoch": posterior["stoch"],
        "observe.posterior_logits": observe_feat["logit"],
        "observe.prior_logits": prior_logits,
        "observe.features": feat2tensor(observe_feat),
        "observe.final.deter": observe_final["deter"],
        "observe.final.stoch": observe_final["stoch"],
        "observe.changed_suffix.deter": changed_feat["deter"][:, 2:],
        "observe.changed_suffix.stoch": changed_feat["stoch"][:, 2:],
        "observe.draw_count": np.asarray(len(observe_keys), np.int32),
        "single.posterior_false.deter": single_post_false["deter"],
        "single.posterior_false.stoch": single_post_false["stoch"],
        "single.posterior_false.logits": single_post_false_feat["logit"],
        "single.posterior_false.effective_logits": single_post_false_effective,
        "single.posterior_false.draw_count": np.asarray(1, np.int32),
        "single.posterior_true.deter": single_post_true["deter"],
        "single.posterior_true.stoch": single_post_true["stoch"],
        "single.posterior_true.logits": single_post_true_feat["logit"],
        "single.posterior_true.effective_logits": single_post_true_effective,
        "single.posterior_true.draw_count": np.asarray(1, np.int32),
        "single.prior.deter": single_prior["deter"],
        "single.prior.stoch": single_prior["stoch"],
        "single.prior.logits": single_prior_feat["logit"],
        "single.prior.effective_logits": single_prior_effective,
        "single.prior.draw_count": np.asarray(1, np.int32),
        "reset.masked_deter": reset_state[0],
        "reset.masked_stoch": reset_state[1],
        "action.combined_cont": combined_actions["a_cont"],
        "action.combined_disc": combined_actions["z_disc"],
        "action.combined_encoded": combined_encoded,
        "action.combined_reset": combined_reset,
        "imagine.gumbel": imagine_noise,
        "imagine.keys": imagine_keys,
        "imagine.effective_logits": jnp.moveaxis(imagine_effective, 0, 1),
        "imagine.prior.deter": imagine_feat["deter"],
        "imagine.prior.stoch": imagine_feat["stoch"],
        "imagine.prior_logits": imagine_feat["logit"],
        "imagine.features": feat2tensor(imagine_feat),
        "imagine.final.deter": imagine_final["deter"],
        "imagine.final.stoch": imagine_final["stoch"],
        "imagine.draw_count": np.asarray(len(imagine_keys), np.int32),
        "imagine_only.obs_param_count": np.asarray(imagine_only_obs_count, np.int32),
        "imagine_only.param_count": np.asarray(
            len(imagine_only_context.parameters), np.int32
        ),
        "gradient.img_action_logits": img_action_grad,
        "gradient.img_deter_logits": img_deter_grad,
        "kl.post": post_kl,
        "kl.prior": prior_kl,
        "kl.dyn": dyn,
        "kl.rep": rep,
        "kl.dyn_free": dyn_free,
        "kl.rep_free": rep_free,
        "kl.dyn_grad_post": dyn_grad[0],
        "kl.dyn_grad_prior": dyn_grad[1],
        "kl.rep_grad_post": rep_grad[0],
        "kl.rep_grad_prior": rep_grad[1],
        "kl.dyn_below_grad_post": dyn_below_grad[0],
        "kl.dyn_below_grad_prior": dyn_below_grad[1],
        "kl.rep_below_grad_post": rep_below_grad[0],
        "kl.rep_below_grad_prior": rep_below_grad[1],
        "replay.entries_deter": replay_entries["deter"],
        "replay.entries_stoch": replay_entries["stoch"],
        "replay.truncate_deter": replay_carry["deter"],
        "replay.truncate_stoch": replay_carry["stoch"],
        "replay.starts_deter": replay_starts["deter"],
        "replay.starts_stoch": replay_starts["stoch"],
        "entry_space.deter_shape": np.asarray(entry_space["deter"].shape, np.int32),
        "entry_space.stoch_shape": np.asarray(entry_space["stoch"].shape, np.int32),
        "entry_space.deter_dtype": np.frombuffer(
            np.dtype(entry_space["deter"].dtype).name.encode(), np.uint8
        ),
        "entry_space.stoch_dtype": np.frombuffer(
            np.dtype(entry_space["stoch"].dtype).name.encode(), np.uint8
        ),
        "draws.initial": np.asarray(initial_draws["calls"], np.int32),
        "draws.reset": np.asarray(0, np.int32),
        "draws.prior_logits": np.asarray(0, np.int32),
        "draws.kl": np.asarray(0, np.int32),
        "scan.root": scan_root,
        "scan.next_root": scan_split[0],
        "source_dtype.state": np.frombuffer(jnp.dtype(dtype).name.encode(), np.uint8),
        "source_dtype.logits": np.frombuffer(jnp.dtype(dtype).name.encode(), np.uint8),
        "source_dtype.params": np.frombuffer(b"float32", np.uint8),
        "source_dtype.kl": np.frombuffer(b"float32", np.uint8),
    }
    for path, value in sorted(context.parameters.items()):
        arrays[f"param.{path.replace('/', '__')}"] = value
    return {name: _host(value) for name, value in sorted(arrays.items())}


def _worker(request: Mapping[str, Any]) -> dict[str, Any]:
    checkout = Path(request["official_checkout"]).resolve()
    revision = str(request["official_commit"])
    profile = DreamerProfile(request["profile"])
    mode = ObservationMode(request["observation_mode"])
    if mode is not ObservationMode.PROPRIO:
        raise ValueError("RSSM oracle only authorizes proprio fixtures")
    dtype = jnp.dtype(request["compute_dtype"]).name
    if dtype not in RSSM_SOURCE_SPEC.execution_dtypes:
        raise ValueError("RSSM worker compute dtype is not authorized")
    if revision != official_revision(profile):
        raise ValueError("RSSM worker revision does not match profile")
    if dict(request["overrides"]) != dict(profile_overrides(profile)):
        raise ValueError("RSSM worker override map does not match profile")
    if request["source_spec"] != RSSM_SOURCE_SPEC.name:
        raise ValueError("RSSM worker source spec mismatch")
    dimensions = {
        name: int(value) for name, value in request["case_dimensions"].items()
    }
    expected_dimensions = {
        "action": 3,
        "batch": 2,
        "classes": 3,
        "deter": 16,
        "hidden": 8,
        "stoch": 2,
        "time": 4,
        "token": 5,
    }
    if dimensions != expected_dimensions:
        raise ValueError("RSSM worker case dimensions are not authorized")
    sources = {}
    for path, digest in RSSM_SOURCE_SPEC.hashes_for(revision).items():
        source = _git_show(checkout, revision, path)
        if _sha256_bytes(source) != digest:
            raise ValueError(f"official RSSM source hash mismatch: {path}")
        sources[path] = source
    nets, _, rssm_source, _ = _load_official_modules(sources, revision, dtype)
    outs = _exec_source(
        sources["embodied/jax/outs.py"],
        f"{revision}:embodied/jax/outs.py",
        {"functools": __import__("functools"), "jax": jax, "jnp": jnp},
    )
    rssm_source.embodied.jax.outs = outs
    rssm_source.elements.Space = _RSSMSpace
    if rssm_source.embodied.jax.outs.OneHot is not outs.OneHot:
        raise ValueError("RSSM source did not retain exact OneHot global")
    if rssm_source.nn is not nets:
        raise ValueError("RSSM source did not retain exact nets global")
    source_dimensions = _source_model_dimensions(sources["dreamerv3/configs.yaml"])
    if request["canonical_dimensions"] != source_dimensions["size1m"]:
        raise ValueError("RSSM worker canonical dimensions do not match source size1m")
    feat2tensor = _load_feat2tensor(sources["dreamerv3/agent.py"], revision, nets)
    arrays = _official_arrays_exact(
        nets,
        outs,
        rssm_source,
        feat2tensor,
        int(request["seed"]),
        dtype,
        dimensions,
        source_dimensions,
    )
    return {
        "arrays": {
            name: {"dtype": value.dtype.name, "values": value.tolist()}
            for name, value in arrays.items()
        },
        "compute_dtype": dtype,
        "worker_pid": os.getpid(),
    }


def run_rssm_case(
    harness: OracleHarness,
    profile: DreamerProfile | str,
    observation_mode: ObservationMode | str = ObservationMode.PROPRIO,
    *,
    case_name: str | None = None,
    seed: int = 0,
    compute_dtype: str = "bfloat16",
) -> tuple[Path, Path]:
    profile = DreamerProfile(profile)
    mode = ObservationMode(observation_mode)
    if mode is not ObservationMode.PROPRIO:
        raise ValueError("RSSM oracle only authorizes proprio fixtures")
    dtype = jnp.dtype(compute_dtype).name
    if dtype not in RSSM_SOURCE_SPEC.execution_dtypes:
        raise ValueError("RSSM oracle compute dtype is not authorized")
    expected_name = "rssm" if dtype == "bfloat16" else "rssm-float32"
    if case_name is not None and case_name != expected_name:
        raise ValueError("RSSM oracle case name does not identify dtype")
    canonical = resolve_dreamer_config(profile, mode)
    assert canonical.rssm is not None
    request = {
        "canonical_dimensions": {
            "classes": canonical.rssm.classes,
            "deter": canonical.rssm.deter,
            "hidden": canonical.rssm.hidden,
            "stoch": canonical.rssm.stoch,
        },
        "case_dimensions": {
            "action": 3,
            "batch": 2,
            "classes": 3,
            "deter": 16,
            "hidden": 8,
            "stoch": 2,
            "time": 4,
            "token": 5,
        },
        "compute_dtype": dtype,
        "official_checkout": str(harness.official_checkout),
        "official_commit": official_revision(profile),
        "observation_mode": mode.value,
        "overrides": dict(profile_overrides(profile)),
        "profile": profile.value,
        "seed": seed,
        "source_spec": RSSM_SOURCE_SPEC.name,
    }
    command = (
        harness.python_executable,
        str(Path(__file__).resolve()),
        "_worker",
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
    if payload["compute_dtype"] != dtype:
        raise ValueError("RSSM worker reported a different dtype")
    worker_pid = int(payload["worker_pid"])
    if worker_pid <= 0 or worker_pid == os.getpid():
        raise ValueError("RSSM oracle worker did not cross a process boundary")
    harness._last_worker_pid = worker_pid
    arrays = {}
    for name, spec in payload["arrays"].items():
        value = np.asarray(spec["values"], dtype=spec["dtype"])
        if value.dtype.name == "bfloat16":
            value = value.astype(np.float32)
        arrays[name] = value
    return harness.write_fixture(
        case_name=case_name or expected_name,
        profile=profile,
        observation_mode=mode,
        arrays=arrays,
        seed=seed,
        generator_command=command,
        generator_request=request,
        source_spec=RSSM_SOURCE_SPEC,
        dtype=dtype,
    )


def _main(argv: Sequence[str]) -> int:
    if tuple(argv) != ("_worker",):
        raise SystemExit("rssm_oracle.py is an internal fixture worker")
    request = json.loads(sys.stdin.read())
    sys.stdout.write(json.dumps(_worker(request), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))


__all__ = ["RSSM_SOURCE_SPEC", "run_rssm_case"]
