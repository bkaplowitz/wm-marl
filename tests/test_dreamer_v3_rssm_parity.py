from __future__ import annotations

import json
import inspect
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.core import freeze, unfreeze
from flax.traverse_util import unflatten_dict

import world_marl.dreamer_v3_baseline as dreamer_package
import world_marl.dreamer_v3_baseline.rssm as rssm_module
from world_marl.dreamer_v3_baseline.config import (
    DreamerProfile,
    RSSMConfig,
)
from world_marl.dreamer_v3_baseline.networks import TensorSpace
from world_marl.dreamer_v3_baseline.oracle import (
    OracleManifest,
    ParameterTranslator,
    official_revision,
    profile_overrides,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "dreamer_v3"
OFFICIAL_CHECKOUT = Path(
    os.environ.get(
        "DREAMERV3_ORACLE_CHECKOUT",
        "/private/tmp/danijar-dreamerv3-20260713",
    )
)
SOURCE_HASHES = {
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


def _api():
    return (
        getattr(rssm_module, "RSSM"),
        getattr(rssm_module, "RSSMState"),
        getattr(rssm_module, "RSSMTrajectory"),
    )


def _config(free_nats: float = 1.0) -> RSSMConfig:
    return RSSMConfig(
        deter=16,
        hidden=8,
        stoch=2,
        classes=3,
        blocks=8,
        free_nats=free_nats,
        unimix=0.01,
        activation="silu",
        normalization="rms",
        image_layers=2,
        observation_layers=1,
        dynamics_layers=1,
        absolute=False,
        initializer="trunc_normal_in",
        output_scale=1.0,
        _legacy=True,
    )


def _module(*, dtype=jnp.float32, free_nats: float = 1.0):
    RSSM, _, _ = _api()
    return RSSM(
        _config(free_nats),
        {"action": TensorSpace((3,), "float32")},
        compute_dtype=dtype,
    )


def _inputs(dtype=jnp.float32, *, batch=2, time=4):
    state = rssm_module.RSSMState(
        deter=jnp.linspace(-1.0, 1.0, batch * 16, dtype=dtype).reshape(batch, 16),
        stoch=jnp.linspace(-0.5, 0.75, batch * 2 * 3, dtype=dtype).reshape(batch, 2, 3),
    )
    actions = {
        "action": jnp.linspace(-2.0, 2.0, batch * time * 3, dtype=jnp.float32).reshape(
            batch, time, 3
        )
    }
    tokens = jnp.linspace(-1.5, 1.5, batch * time * 5, dtype=jnp.float32).reshape(
        batch, time, 5
    )
    resets = jnp.asarray([[False, False, True, False], [False, True, False, False]])
    keys = jax.random.split(jax.random.PRNGKey(19), time)
    return state, actions, tokens, resets, keys


def _flat_params(variables) -> dict[str, np.ndarray]:
    flat = {}

    def visit(prefix, value):
        if isinstance(value, dict):
            for key, child in value.items():
                visit((*prefix, key), child)
        else:
            flat["/".join(prefix)] = np.asarray(value)

    visit((), unfreeze(variables.get("params", {})))
    return flat


def _tolerance(manifest: OracleManifest) -> float:
    return 2e-2 if manifest.dtype == "bfloat16" else 1e-5


def _assert_source(manifest: OracleManifest, actual, expected, *, exact=False):
    if exact:
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    else:
        np.testing.assert_allclose(
            np.asarray(actual, np.float32),
            np.asarray(expected, np.float32),
            rtol=_tolerance(manifest),
            atol=_tolerance(manifest),
        )


def _source_params(arrays) -> dict[str, np.ndarray]:
    return {
        name.removeprefix("param.").replace("__", "/"): value
        for name, value in arrays.items()
        if name.startswith("param.")
    }


def _translator(source: dict[str, np.ndarray]) -> ParameterTranslator:
    result = ParameterTranslator()
    for path in source:
        result.register(path, f"core/{path}" if path.startswith("dyn") else path)
    return result


def _variables_from_source(module, arrays):
    initial = module.apply({}, 2, method=module.initial)
    actions = {"action": jnp.asarray(arrays["observe.action"])}
    tokens = jnp.asarray(arrays["observe.tokens"])
    resets = jnp.asarray(arrays["observe.reset"])
    keys = jnp.asarray(arrays["observe.keys"])
    initialized = module.init(
        jax.random.PRNGKey(0),
        initial,
        tokens,
        actions,
        resets,
        keys,
        method=module.observe,
    )
    destination = _flat_params(initialized)
    source = _source_params(arrays)
    translator = _translator(source)
    translated = translator.translate(source, destination)
    translator.assert_fully_consumed()
    nested = unflatten_dict(
        {tuple(path.split("/")): value for path, value in translated.items()}
    )
    return freeze({"params": nested}), translated, destination


def _native_case(official_case):
    manifest, arrays = official_case
    dtype = jnp.dtype(manifest.dtype)
    module = _module(dtype=dtype)
    variables, translated, initialized = _variables_from_source(module, arrays)
    initial = module.apply({}, 2, method=module.initial)
    actions = {"action": jnp.asarray(arrays["observe.action"])}
    tokens = jnp.asarray(arrays["observe.tokens"])
    resets = jnp.asarray(arrays["observe.reset"])
    keys = jnp.asarray(arrays["observe.keys"])
    return (
        manifest,
        arrays,
        module,
        variables,
        translated,
        initialized,
        initial,
        actions,
        tokens,
        resets,
        keys,
    )


class _NativeCategoricalGumbels:
    def __init__(self, keys, gumbels):
        self.keys = np.asarray(keys, np.uint32)
        self.gumbels = jnp.asarray(gumbels, jnp.float32)
        self.seen = []
        if self.keys.ndim != 2 or self.keys.shape[-1] != 2:
            raise ValueError("supplied Gumbel keys must have shape [T,2]")
        if self.gumbels.ndim != 4 or len(self.gumbels) != len(self.keys):
            raise ValueError("supplied Gumbel count must equal sample key count")
        if len({tuple(value) for value in self.keys}) != len(self.keys):
            raise ValueError("supplied Gumbel sample keys must be unique")

    def categorical(self, seed, logits, axis=-1, shape=None):
        if axis != -1 or tuple(logits.shape) != tuple(self.gumbels.shape[1:]):
            raise ValueError("supplied Gumbel categorical shape changed")
        if shape is not None and tuple(shape) != tuple(logits.shape[:-1]):
            raise ValueError("supplied Gumbel output shape changed")
        keys = jnp.asarray(self.keys)
        matches = jnp.all(keys == seed, axis=-1)
        index = jnp.argmax(matches.astype(jnp.int32))

        def record(value):
            host = np.asarray(value, np.uint32)
            found = np.flatnonzero(np.all(self.keys == host, axis=-1))
            if len(found) != 1:
                raise ValueError(
                    "native categorical received an unauthorized sample key"
                )
            self.seen.append(int(found[0]))

        jax.debug.callback(record, seed)
        return jnp.argmax(jnp.take(self.gumbels, index, axis=0) + logits, axis=-1)

    def assert_fully_consumed(self):
        if self.seen != list(range(len(self.keys))):
            raise ValueError(
                f"supplied Gumbels were not consumed in order: {self.seen}"
            )


@contextmanager
def _native_gumbel_scope(keys, gumbels):
    injected = _NativeCategoricalGumbels(keys, gumbels)
    original = jax.random.categorical
    jax.random.categorical = injected.categorical
    try:
        yield injected
    finally:
        jax.random.categorical = original
    injected.assert_fully_consumed()


def _native_effective_logits(module, logits):
    return module.apply(
        {},
        logits,
        method=lambda instance, value: instance._dist(value).output.dist.logits,
    )


@pytest.fixture(
    params=(
        (DreamerProfile.PAPER, "bfloat16"),
        (DreamerProfile.PAPER, "float32"),
        (DreamerProfile.UPSTREAM_CURRENT, "bfloat16"),
        (DreamerProfile.UPSTREAM_CURRENT, "float32"),
    ),
    ids=lambda item: f"{item[0].value}-{item[1]}",
)
def official_case(request):
    profile, dtype = request.param
    suffix = "" if dtype == "bfloat16" else "-float32"
    stem = f"{profile.value}-proprio-rssm{suffix}"
    fixture_path = FIXTURE_DIR / f"{stem}.npz"
    manifest_path = FIXTURE_DIR / f"{stem}.manifest.json"
    manifest = OracleManifest.load(manifest_path, fixture_path=fixture_path)
    with np.load(fixture_path, allow_pickle=False) as fixture:
        arrays = {name: fixture[name] for name in fixture.files}
    return manifest, arrays


def test_rssm_oracle_authority_dtype_and_source_parameter_consumption(official_case):
    manifest, arrays = official_case
    source_spec = getattr(rssm_module, "RSSM_SOURCE_SPEC")
    request = json.loads(manifest.generator_request)
    assert manifest.source_spec == source_spec.name == "rssm"
    assert dict(manifest.official_file_hashes) == SOURCE_HASHES
    assert request["official_commit"] == official_revision(manifest.profile)
    assert request["profile"] == manifest.profile.value
    assert request["observation_mode"] == manifest.observation_mode.value
    assert request["source_spec"] == "rssm"
    assert request["case_dimensions"] == {
        "action": 3,
        "batch": 2,
        "classes": 3,
        "deter": 16,
        "hidden": 8,
        "stoch": 2,
        "time": 4,
        "token": 5,
    }
    assert request["overrides"] == profile_overrides(manifest.profile)
    assert request["compute_dtype"] == manifest.dtype
    assert request["canonical_dimensions"] == {
        "classes": int(arrays["source_config.size1m.classes"]),
        "deter": int(arrays["source_config.size1m.deter"]),
        "hidden": int(arrays["source_config.size1m.hidden"]),
        "stoch": int(arrays["source_config.size1m.stoch"]),
    }
    assert int(arrays["source_config.size1m.deter"]) == 512
    assert int(arrays["source_config.size200m.deter"]) == 8192
    assert arrays["execution.compute_dtype"].tobytes().decode() == manifest.dtype
    assert int(arrays["observe.draw_count"]) == int(arrays["dimensions.time"])
    for operation in ("initial", "reset", "prior_logits", "kl"):
        assert int(arrays[f"draws.{operation}"]) == 0
    module = _module(dtype=jnp.dtype(manifest.dtype))
    _variables, translated, destination = _variables_from_source(module, arrays)
    source = _source_params(arrays)
    expected_destinations = {
        f"core/{path}" if path.startswith("dyn") else path for path in source
    }
    assert set(destination) == expected_destinations == set(translated)
    for path, value in source.items():
        destination_path = f"core/{path}" if path.startswith("dyn") else path
        np.testing.assert_array_equal(translated[destination_path], value)
        assert translated[destination_path].shape == destination[destination_path].shape
        assert value.dtype == translated[destination_path].dtype == np.float32


def test_rssm_oracle_replays_exact_command_stdin_and_is_deterministic(official_case):
    if not (OFFICIAL_CHECKOUT / ".git").exists():
        pytest.skip("explicit DreamerV3 oracle checkout is unavailable")
    manifest, arrays = official_case
    payloads = []
    for _ in range(2):
        replayed = subprocess.run(
            manifest.generator_command,
            cwd=OFFICIAL_CHECKOUT,
            input=manifest.generator_request,
            check=True,
            capture_output=True,
            text=True,
        )
        payloads.append(json.loads(replayed.stdout))
    assert all(int(payload["worker_pid"]) != os.getpid() for payload in payloads)
    assert all(payload["compute_dtype"] == manifest.dtype for payload in payloads)
    assert payloads[0]["arrays"] == payloads[1]["arrays"]
    assert tuple(sorted(payloads[0]["arrays"])) == tuple(arrays)
    representative_dtypes = {
        "initial.deter": manifest.dtype,
        "observe.reset": "bool",
        "observe.keys": "uint32",
        "draws.initial": "int32",
        "source_dtype.state": "uint8",
        "param.dyngru__kernel": "float32",
    }
    for payload in payloads:
        for name, dtype in representative_dtypes.items():
            assert payload["arrays"][name]["dtype"] == dtype
    for name, spec in payloads[0]["arrays"].items():
        np.testing.assert_array_equal(
            arrays[name], np.asarray(spec["values"], dtype=spec["dtype"])
        )


def test_oracle_control_fields_have_explicit_operational_meaning(official_case):
    manifest, arrays = official_case
    dimensions = {
        name.removeprefix("dimensions."): int(value)
        for name, value in arrays.items()
        if name.startswith("dimensions.")
    }
    assert dimensions == {
        "action": arrays["observe.action"].shape[-1],
        "batch": arrays["observe.tokens"].shape[0],
        "classes": arrays["observe.posterior.stoch"].shape[-1],
        "deter": arrays["observe.posterior.deter"].shape[-1],
        "hidden": 8,
        "stoch": arrays["observe.posterior.stoch"].shape[-2],
        "time": arrays["observe.tokens"].shape[1],
        "token": arrays["observe.tokens"].shape[-1],
    }
    size1m = {
        name.removeprefix("source_config.size1m."): int(value)
        for name, value in arrays.items()
        if name.startswith("source_config.size1m.")
    }
    assert size1m == {"classes": 4, "deter": 512, "hidden": 64, "stoch": 32}
    size200m = {
        name.removeprefix("source_config.size200m."): int(value)
        for name, value in arrays.items()
        if name.startswith("source_config.size200m.")
    }
    assert size200m == {
        "classes": 64,
        "deter": 8192,
        "hidden": 1024,
        "stoch": 32,
    }
    for operation in (
        "single.prior",
        "single.posterior_false",
        "single.posterior_true",
    ):
        assert int(arrays[f"{operation}.draw_count"]) == 1
    source = _source_params(arrays)
    assert int(arrays["imagine_only.param_count"]) == sum(
        not path.startswith("obs") for path in source
    )
    expected_source_dtypes = {
        "state": manifest.dtype,
        "logits": manifest.dtype,
        "params": "float32",
        "kl": "float32",
    }
    for name, expected in expected_source_dtypes.items():
        assert arrays[f"source_dtype.{name}"].tobytes().decode() == expected


def test_parameter_translation_rejects_missing_extra_and_duplicate_use():
    translator = ParameterTranslator()
    translator.register("a", "x")
    with pytest.raises(ValueError, match="unregistered source"):
        translator.translate({"a": np.zeros(1), "b": np.zeros(1)}, {"x": (1,)})
    with pytest.raises(ValueError, match="unregistered destination"):
        translator.translate({"a": np.zeros(1)}, {"x": (1,), "y": (1,)})
    with pytest.raises(ValueError, match="unconsumed registered"):
        translator.translate({}, {})
    translated = translator.translate({"a": np.zeros(1)}, {"x": (1,)})
    assert set(translated) == {"x"}
    with pytest.raises(ValueError, match="more than once"):
        translator.consume("a", "x", np.zeros(1), (1,))


def test_initial_is_exact_zero_not_onehot_and_getfeat_orders_deter_then_stoch(
    official_case,
):
    manifest, arrays = official_case
    module = _module(dtype=jnp.dtype(manifest.dtype))
    state = module.apply({}, 2, method=module.initial)
    _assert_source(manifest, state.deter, arrays["initial.deter"], exact=True)
    _assert_source(manifest, state.stoch, arrays["initial.stoch"], exact=True)
    source_state = rssm_module.RSSMState(
        jnp.asarray(arrays["observe.posterior.deter"][:, 0]),
        jnp.asarray(arrays["observe.posterior.stoch"][:, 0]),
    )
    feature = module.apply({}, source_state, method=module.getfeat)
    _assert_source(manifest, feature, arrays["observe.features"][:, 0])
    assert feature.shape == (2, 22)


def test_reset_zeros_selected_rows_only_and_rejects_wrong_shape(official_case):
    manifest, arrays = official_case
    module = _module(dtype=jnp.dtype(manifest.dtype))
    state = rssm_module.RSSMState(
        jnp.asarray(arrays["observe.posterior.deter"][:, 1]),
        jnp.asarray(arrays["observe.posterior.stoch"][:, 1]),
    )
    reset = module.apply(
        {}, state, jnp.asarray(arrays["observe.reset"][:, 2]), method=module.reset
    )
    _assert_source(manifest, reset.deter, arrays["reset.masked_deter"])
    _assert_source(manifest, reset.stoch, arrays["reset.masked_stoch"])
    with pytest.raises(ValueError):
        module.apply({}, state, jnp.ones((2, 1), bool), method=module.reset)
    with pytest.raises(TypeError):
        module.apply({}, state, jnp.ones((2,), jnp.int32), method=module.reset)


def test_action_embedding_matches_dictconcat_and_post_onehot_reset_mask(official_case):
    manifest, arrays = official_case
    RSSM, State, _ = _api()
    module = RSSM(
        _config(),
        {
            "a_cont": TensorSpace((2,), "float32"),
            "z_disc": TensorSpace((1,), "int32", classes=3),
        },
        compute_dtype=jnp.dtype(manifest.dtype),
    )
    action = {
        "z_disc": jnp.asarray(arrays["action.combined_disc"]),
        "a_cont": jnp.asarray(arrays["action.combined_cont"]),
    }
    embedded = module.apply({}, action, method=module._flatten_action)
    _assert_source(manifest, embedded, arrays["action.combined_encoded"], exact=True)
    masked = module.apply(
        {},
        embedded,
        ~jnp.asarray([True, False]),
        method=lambda instance, value, available: instance._mask(value, available),
    )
    _assert_source(manifest, masked, arrays["action.combined_reset"], exact=True)
    assert embedded.dtype == jnp.dtype(manifest.dtype)
    np.testing.assert_array_equal(masked[0], 0)
    np.testing.assert_array_equal(masked[1], embedded[1])
    with pytest.raises(ValueError, match="keys"):
        module.apply({}, {"a_cont": action["a_cont"]}, method=module._flatten_action)
    with pytest.raises(ValueError, match="keys"):
        module.apply(
            {}, {**action, "extra": jnp.zeros((2, 1))}, method=module._flatten_action
        )
    with pytest.raises(ValueError, match="shape"):
        module.apply(
            {}, {**action, "a_cont": jnp.zeros((2, 3))}, method=module._flatten_action
        )


def test_parameter_tree_uses_blockgru_and_exact_named_prior_posterior_layers(
    official_case,
):
    manifest, arrays = official_case
    module = _module(dtype=jnp.dtype(manifest.dtype))
    _variables, _translated, paths = _variables_from_source(module, arrays)
    expected = {
        f"core/{path}" if path.startswith("dyn") else path
        for path in _source_params(arrays)
    }
    assert set(paths) == expected
    assert "core/dyngru/kernel" in paths
    assert paths["core/dyngru/kernel"].shape == (8, 2, 6)
    assert "obs0/kernel" in paths and "obslogit/kernel" in paths
    assert not any(value.shape == (16, 48) for value in paths.values())
    assert all(value.dtype.name == "float32" for value in paths.values())
    source = inspect.getsource(rssm_module)
    assert "nn.GRUCell" not in source
    assert "categorical_straight_through" not in source


def test_img_step_matches_source_prior_logits_gumbel_sample_and_gradients(
    official_case,
):
    (
        manifest,
        arrays,
        module,
        variables,
        *_rest,
        initial,
        actions,
        _tokens,
        _resets,
        keys,
    ) = _native_case(official_case)
    action = {name: value[:, 0] for name, value in actions.items()}
    gumbel = jnp.asarray(arrays["imagine.gumbel"][0])
    with _native_gumbel_scope(keys[:1], gumbel[None]):
        prior, logits = module.apply(
            variables,
            initial,
            action,
            keys[0],
            method=module.img_step,
        )
        jax.block_until_ready(prior.deter)
    _assert_source(manifest, prior.deter, arrays["single.prior.deter"])
    _assert_source(manifest, prior.stoch, arrays["single.prior.stoch"], exact=True)
    _assert_source(manifest, logits, arrays["single.prior.logits"])
    effective = _native_effective_logits(module, logits)
    _assert_source(manifest, effective, arrays["single.prior.effective_logits"])
    assert int(arrays["single.prior.draw_count"]) == 1

    def objective(deter, action_value):
        def prior_logits(instance, state_deter, value):
            embedded = instance._flatten_action({"action": value})
            next_deter = instance.core(state_deter, initial.stoch, embedded)
            return instance._prior_logits(next_deter)

        value = module.apply(
            variables,
            deter,
            action_value,
            method=prior_logits,
        )
        return value.astype(jnp.float32).sum()

    grad_deter, grad_action = jax.grad(objective, argnums=(0, 1))(
        initial.deter, action["action"]
    )
    _assert_source(manifest, grad_deter, arrays["gradient.img_deter_logits"])
    _assert_source(manifest, grad_action, arrays["gradient.img_action_logits"])


def test_obs_step_matches_source_for_reset_false_and_true_with_one_draw_each(
    official_case,
):
    (
        manifest,
        arrays,
        module,
        variables,
        *_rest,
        initial,
        actions,
        tokens,
        resets,
        keys,
    ) = _native_case(official_case)
    with _native_gumbel_scope(keys[:1], arrays["observe.gumbel"][:1]):
        false_state, false_logits = module.apply(
            variables,
            initial,
            {"action": actions["action"][:, 0]},
            tokens[:, 0],
            resets[:, 0],
            keys[0],
            method=module.obs_step,
        )
        jax.block_until_ready(false_state.deter)
    for field, actual in (("deter", false_state.deter), ("stoch", false_state.stoch)):
        _assert_source(
            manifest,
            actual,
            arrays[f"single.posterior_false.{field}"],
            exact=field == "stoch",
        )
    _assert_source(manifest, false_logits, arrays["single.posterior_false.logits"])
    false_effective = _native_effective_logits(module, false_logits)
    _assert_source(
        manifest, false_effective, arrays["single.posterior_false.effective_logits"]
    )
    previous = rssm_module.RSSMState(
        jnp.asarray(arrays["observe.posterior.deter"][:, 0]),
        jnp.asarray(arrays["observe.posterior.stoch"][:, 0]),
    )
    with _native_gumbel_scope(keys[1:2], arrays["observe.gumbel"][1:2]):
        true_state, true_logits = module.apply(
            variables,
            previous,
            {"action": actions["action"][:, 1]},
            tokens[:, 1],
            resets[:, 1],
            keys[1],
            method=module.obs_step,
        )
        jax.block_until_ready(true_state.deter)
    for field, actual in (("deter", true_state.deter), ("stoch", true_state.stoch)):
        _assert_source(
            manifest,
            actual,
            arrays[f"single.posterior_true.{field}"],
            exact=field == "stoch",
        )
    _assert_source(manifest, true_logits, arrays["single.posterior_true.logits"])
    true_effective = _native_effective_logits(module, true_logits)
    _assert_source(
        manifest, true_effective, arrays["single.posterior_true.effective_logits"]
    )
    assert int(arrays["single.posterior_false.draw_count"]) == 1
    assert int(arrays["single.posterior_true.draw_count"]) == 1


def test_observe_matches_every_source_trajectory_value_and_consumes_ordered_gumbels(
    official_case,
):
    (
        manifest,
        arrays,
        module,
        variables,
        *_rest,
        initial,
        actions,
        tokens,
        resets,
        keys,
    ) = _native_case(official_case)
    gumbels = jnp.asarray(arrays["observe.gumbel"])
    with _native_gumbel_scope(keys, gumbels):
        trajectory = module.apply(
            variables,
            initial,
            tokens,
            actions,
            resets,
            keys,
            method=module.observe,
        )
        jax.block_until_ready(trajectory.final_state.deter)
    assert trajectory.mode == "observe" and trajectory.prior is None
    for field in ("deter", "stoch"):
        _assert_source(
            manifest,
            getattr(trajectory.posterior, field),
            arrays[f"observe.posterior.{field}"],
            exact=field == "stoch",
        )
        _assert_source(
            manifest,
            getattr(trajectory.final_state, field),
            arrays[f"observe.final.{field}"],
            exact=field == "stoch",
        )
    for field, actual in (
        ("posterior_logits", trajectory.posterior_logits),
        ("prior_logits", trajectory.prior_logits),
        ("features", trajectory.features),
    ):
        _assert_source(manifest, actual, arrays[f"observe.{field}"])
    effective = _native_effective_logits(module, trajectory.posterior_logits)
    _assert_source(manifest, effective, arrays["observe.effective_logits"])
    sampled_indices = np.asarray(trajectory.posterior.stoch).argmax(-1)
    raw_argmax = np.asarray(trajectory.posterior_logits).argmax(-1)
    assert np.any(sampled_indices != raw_argmax)
    assert int(arrays["observe.draw_count"]) == gumbels.shape[0] == keys.shape[0]
    with pytest.raises(ValueError, match="Gumbel"):
        with _native_gumbel_scope(keys, gumbels[:-1]):
            pass
    with pytest.raises(ValueError, match="Gumbel"):
        with _native_gumbel_scope(keys, jnp.concatenate([gumbels, gumbels[:1]])):
            pass


def test_observe_and_imagine_reject_zero_length_sequences(official_case):
    (
        _manifest,
        _arrays,
        module,
        variables,
        *_rest,
        state,
        actions,
        tokens,
        resets,
        keys,
    ) = _native_case(official_case)
    empty_actions = jax.tree.map(lambda value: value[:, :0], actions)
    with pytest.raises(ValueError, match="positive"):
        module.apply(
            variables,
            state,
            tokens[:, :0],
            empty_actions,
            resets[:, :0],
            keys[:0],
            method=module.observe,
        )
    with pytest.raises(ValueError, match="positive"):
        module.apply(
            variables,
            state,
            empty_actions,
            keys[:0],
            method=module.imagine,
        )


def test_midsequence_reset_makes_suffix_independent_of_prefix(official_case):
    (
        manifest,
        arrays,
        module,
        variables,
        *_rest,
        state,
        actions,
        tokens,
        resets,
        keys,
    ) = _native_case(official_case)
    gumbels = jnp.asarray(arrays["observe.gumbel"])
    changed_tokens = tokens.at[0, :2].set(999)
    changed_tokens = changed_tokens.at[1, :1].set(999)
    changed_action = actions["action"].at[0, :2].set(-999)
    changed_action = changed_action.at[1, :1].set(-999)
    changed_actions = {"action": changed_action}
    with _native_gumbel_scope(keys, gumbels):
        left = module.apply(
            variables,
            state,
            tokens,
            actions,
            resets,
            keys,
            method=module.observe,
        )
        jax.block_until_ready(left.final_state.deter)
    with _native_gumbel_scope(keys, gumbels):
        right = module.apply(
            variables,
            state,
            changed_tokens,
            changed_actions,
            resets,
            keys,
            method=module.observe,
        )
        jax.block_until_ready(right.final_state.deter)
    _assert_source(
        manifest, right.posterior.deter[:, 2:], arrays["observe.changed_suffix.deter"]
    )
    _assert_source(
        manifest,
        right.posterior.stoch[:, 2:],
        arrays["observe.changed_suffix.stoch"],
        exact=True,
    )
    _assert_source(manifest, left.posterior.deter[:, 2:], right.posterior.deter[:, 2:])
    _assert_source(
        manifest, left.posterior.stoch[:, 2:], right.posterior.stoch[:, 2:], exact=True
    )


def test_imagine_matches_source_open_loop_trajectory_and_has_no_obs_parameters(
    official_case,
):
    (
        manifest,
        arrays,
        module,
        variables,
        *_rest,
        state,
        actions,
        _tokens,
        _resets,
        keys,
    ) = _native_case(official_case)
    with _native_gumbel_scope(keys, arrays["imagine.gumbel"]):
        trajectory = module.apply(
            variables,
            state,
            actions,
            keys,
            method=module.imagine,
        )
        jax.block_until_ready(trajectory.final_state.deter)
    assert trajectory.mode == "imagine" and trajectory.posterior is None
    for field in ("deter", "stoch"):
        _assert_source(
            manifest,
            getattr(trajectory.prior, field),
            arrays[f"imagine.prior.{field}"],
            exact=field == "stoch",
        )
        _assert_source(
            manifest,
            getattr(trajectory.final_state, field),
            arrays[f"imagine.final.{field}"],
            exact=field == "stoch",
        )
    _assert_source(manifest, trajectory.prior_logits, arrays["imagine.prior_logits"])
    _assert_source(manifest, trajectory.features, arrays["imagine.features"])
    effective = _native_effective_logits(module, trajectory.prior_logits)
    _assert_source(manifest, effective, arrays["imagine.effective_logits"])
    assert int(arrays["imagine.draw_count"]) == len(keys)
    assert int(arrays["imagine_only.obs_param_count"]) == 0
    with _native_gumbel_scope(keys, arrays["imagine.gumbel"]):
        imagine_output, imagine_only = module.init_with_output(
            jax.random.PRNGKey(0),
            state,
            actions,
            keys,
            method=module.imagine,
        )
        jax.block_until_ready(imagine_output.final_state.deter)
    assert not any(path.startswith("obs") for path in _flat_params(imagine_only))


def test_dyn_and_rep_kl_match_source_unimix_latent_sum_and_pointwise_free_nats(
    official_case,
):
    manifest, arrays = official_case
    module = _module(dtype=jnp.dtype(manifest.dtype), free_nats=1.0)
    post = jnp.asarray(arrays["kl.post"])
    prior = jnp.asarray(arrays["kl.prior"])
    dyn = module.apply({}, post, prior, method=module.dyn_loss)
    rep = module.apply({}, post, prior, method=module.rep_loss)
    _assert_source(manifest, dyn, arrays["kl.dyn"])
    _assert_source(manifest, rep, arrays["kl.rep"])
    _assert_source(
        manifest,
        module.apply(
            {}, jnp.zeros_like(post), jnp.zeros_like(prior), method=module.dyn_loss
        ),
        arrays["kl.dyn_free"],
        exact=True,
    )
    _assert_source(
        manifest,
        module.apply(
            {}, jnp.zeros_like(post), jnp.zeros_like(prior), method=module.rep_loss
        ),
        arrays["kl.rep_free"],
        exact=True,
    )
    assert dyn.shape == rep.shape == post.shape[:-2]
    assert float(dyn[0, 0]) > 1.0


def test_kl_stop_gradient_boundaries_and_positive_free_nats_gradients_match_source(
    official_case,
):
    manifest, arrays = official_case
    module = _module(dtype=jnp.dtype(manifest.dtype), free_nats=1.0)
    post = jnp.asarray(arrays["kl.post"])
    prior = jnp.asarray(arrays["kl.prior"])
    dyn_post, dyn_prior = jax.grad(
        lambda a, b: module.apply({}, a, b, method=module.dyn_loss).sum(),
        argnums=(0, 1),
    )(post, prior)
    rep_post, rep_prior = jax.grad(
        lambda a, b: module.apply({}, a, b, method=module.rep_loss).sum(),
        argnums=(0, 1),
    )(post, prior)
    for name, actual in (
        ("dyn_grad_post", dyn_post),
        ("dyn_grad_prior", dyn_prior),
        ("rep_grad_post", rep_post),
        ("rep_grad_prior", rep_prior),
    ):
        _assert_source(manifest, actual, arrays[f"kl.{name}"])
    equal = jnp.zeros_like(post)
    dyn_below = jax.grad(
        lambda a, b: module.apply({}, a, b, method=module.dyn_loss).sum(),
        argnums=(0, 1),
    )(equal, equal)
    rep_below = jax.grad(
        lambda a, b: module.apply({}, a, b, method=module.rep_loss).sum(),
        argnums=(0, 1),
    )(equal, equal)
    for side, actual in zip(("post", "prior"), dyn_below, strict=True):
        _assert_source(
            manifest, actual, arrays[f"kl.dyn_below_grad_{side}"], exact=True
        )
    for side, actual in zip(("post", "prior"), rep_below, strict=True):
        _assert_source(
            manifest, actual, arrays[f"kl.rep_below_grad_{side}"], exact=True
        )


def test_source_compute_dtype_params_probabilities_and_kl_contract(official_case):
    (
        manifest,
        arrays,
        module,
        variables,
        *_rest,
        state,
        actions,
        _tokens,
        _resets,
        keys,
    ) = _native_case(official_case)
    action = {key: value[:, 0] for key, value in actions.items()}
    with _native_gumbel_scope(keys[:1], arrays["imagine.gumbel"][:1]):
        next_state, logits = module.apply(
            variables,
            state,
            action,
            keys[0],
            method=module.img_step,
        )
        jax.block_until_ready(next_state.deter)
    expected = jnp.dtype(manifest.dtype)
    assert next_state.deter.dtype == next_state.stoch.dtype == logits.dtype == expected
    assert all(
        value.dtype.name == "float32" for value in _flat_params(variables).values()
    )
    assert module.apply({}, logits, logits, method=module.dyn_loss).dtype == jnp.float32
    assert _native_effective_logits(module, logits).dtype == jnp.float32
    assert arrays["source_dtype.state"].tobytes().decode() == manifest.dtype
    assert arrays["source_dtype.logits"].tobytes().decode() == manifest.dtype
    assert arrays["source_dtype.params"].tobytes().decode() == "float32"
    assert arrays["source_dtype.kl"].tobytes().decode() == "float32"


def test_trajectory_constructor_rejects_invalid_mode_shapes_and_presence():
    _, State, Trajectory = _api()
    seq = State(jnp.zeros((2, 4, 16)), jnp.zeros((2, 4, 2, 3)))
    final = State(jnp.zeros((2, 16)), jnp.zeros((2, 2, 3)))
    logits = jnp.zeros((2, 4, 2, 3))
    features = jnp.zeros((2, 4, 22))
    valid = Trajectory(seq, None, logits, logits, features, final, "observe")
    assert valid.prior is None
    with pytest.raises(ValueError):
        Trajectory(seq, seq, logits, logits, features, final, "observe")
    with pytest.raises(ValueError):
        Trajectory(seq, None, logits[:, :3], logits, features, final, "observe")
    with pytest.raises(ValueError):
        Trajectory(seq, None, logits, logits, features, final, "unknown")
    with pytest.raises(ValueError):
        Trajectory(None, None, None, logits, features, final, "observe")
    with pytest.raises(ValueError):
        Trajectory(seq, None, logits, logits, features, final, "imagine")
    with pytest.raises(ValueError):
        Trajectory(None, seq, logits, logits, features, final, "imagine")
    with pytest.raises(ValueError):
        Trajectory(
            seq.replace(deter=jnp.zeros((2, 4, 15))),
            None,
            logits,
            logits,
            features,
            final,
            "observe",
        )
    with pytest.raises(ValueError):
        Trajectory(
            seq.replace(stoch=jnp.zeros((2, 4, 3, 3))),
            None,
            logits,
            logits,
            features,
            final,
            "observe",
        )
    with pytest.raises(ValueError):
        Trajectory(seq, None, logits[..., :2], logits, features, final, "observe")
    with pytest.raises(ValueError):
        Trajectory(seq, None, logits, logits, features[..., :-1], final, "observe")
    with pytest.raises(ValueError):
        Trajectory(
            seq,
            None,
            logits,
            logits,
            features,
            final.replace(deter=jnp.zeros((3, 16))),
            "observe",
        )


def test_truncate_starts_and_entry_space_match_source_batch_major_order(official_case):
    manifest, arrays = official_case
    module = _module(dtype=jnp.dtype(manifest.dtype))
    entries = rssm_module.RSSMState(
        jnp.asarray(arrays["replay.entries_deter"]),
        jnp.asarray(arrays["replay.entries_stoch"]),
    )
    truncated = module.apply({}, entries, method=module.truncate)
    _assert_source(
        manifest, truncated.deter, arrays["replay.truncate_deter"], exact=True
    )
    _assert_source(
        manifest, truncated.stoch, arrays["replay.truncate_stoch"], exact=True
    )
    starts = module.apply({}, entries, truncated, 2, method=module.starts)
    _assert_source(manifest, starts.deter, arrays["replay.starts_deter"], exact=True)
    _assert_source(manifest, starts.stoch, arrays["replay.starts_stoch"], exact=True)
    spaces = module.apply({}, method=lambda instance: instance.entry_space)
    assert spaces["deter"].shape == tuple(arrays["entry_space.deter_shape"])
    assert spaces["stoch"].shape == tuple(arrays["entry_space.stoch_shape"])
    assert (
        np.dtype(spaces["deter"].dtype).name
        == arrays["entry_space.deter_dtype"].tobytes().decode()
    )
    assert (
        np.dtype(spaces["stoch"].dtype).name
        == arrays["entry_space.stoch_dtype"].tobytes().decode()
    )
    wrong_carry = rssm_module.RSSMState(truncated.deter[:1], truncated.stoch[:1])
    with pytest.raises(ValueError):
        module.apply({}, entries, wrong_carry, 2, method=module.starts)


def test_ninjax_scan_keys_match_captured_source_draws_and_public_exports(official_case):
    _manifest, arrays = official_case
    root = jnp.asarray(arrays["scan.root"])
    actual = rssm_module.ninjax_scan_sample_keys(root, 4)
    np.testing.assert_array_equal(actual, arrays["observe.keys"])
    np.testing.assert_array_equal(actual, arrays["imagine.keys"])
    assert len({tuple(value) for value in np.asarray(actual)}) == 4
    np.testing.assert_array_equal(
        jax.random.split(root, 5)[0], arrays["scan.next_root"]
    )
    assert not hasattr(rssm_module, "categorical_straight_through")
    assert not hasattr(dreamer_package, "categorical_straight_through")
    for name in ("RSSM", "RSSMState", "RSSMTrajectory", "RSSM_SOURCE_SPEC"):
        assert getattr(dreamer_package, name) is getattr(rssm_module, name)
    for name in ("flatten_rssm_state", "initial_rssm_state", "reset_rssm_state"):
        assert name not in rssm_module.__all__
        assert not hasattr(dreamer_package, name)
