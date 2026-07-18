from __future__ import annotations

import copy
import hashlib
import inspect
import json
from dataclasses import FrozenInstanceError, fields, replace
from enum import Enum
from typing import Any

import numpy as np
import pytest
from flax import serialization
from flax.core import FrozenDict

import world_marl.dreamer_v3_baseline as dreamer_v3
import world_marl.dreamer_v3_baseline.config as config_module


PAPER_REVISION = "bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01"
CURRENT_REVISION = "e3f02248693a79dc8b0ebd62c93683888ddaccfe"
PUBLIC_SEED_MAX = 2**32 - 1 - 10_000


class _StringSubclass(str):
    pass


class _StringEnumKey(str, Enum):
    MODEL_DIM = "model_dim"
    ENV_STEPS = "env_steps"
    CAMERA = "camera"


def _symbol(name: str) -> Any:
    return getattr(config_module, name)


def _resolve(**kwargs: Any) -> Any:
    defaults = {"mode": "vision", "task": "walker_walk"}
    return _symbol("resolve_dreamer_run")(**(defaults | kwargs))


def _config(**kwargs: Any) -> Any:
    defaults = {"mode": "vision", "task": "walker_walk"}
    return _symbol("resolve_dreamer_config")(**(defaults | kwargs))


def test_public_resolver_signatures_and_package_exports_are_frozen() -> None:
    required = {
        "DebugSnapshot",
        "ResolvedDreamerRun",
        "RuntimeOverrides",
        "SequenceShapeConfig",
        "resolve_dreamer_run",
    }
    assert required <= set(config_module.__all__)
    profile = _symbol("DreamerProfile")
    overrides = _symbol("RuntimeOverrides")
    expected_run = {
        "mode": inspect.Parameter.KEYWORD_ONLY,
        "task": inspect.Parameter.KEYWORD_ONLY,
        "profile": inspect.Parameter.KEYWORD_ONLY,
        "seed": inspect.Parameter.KEYWORD_ONLY,
        "model": inspect.Parameter.KEYWORD_ONLY,
        "debug_local": inspect.Parameter.KEYWORD_ONLY,
        "overrides": inspect.Parameter.KEYWORD_ONLY,
    }
    run_signature = inspect.signature(_symbol("resolve_dreamer_run"))
    config_signature = inspect.signature(_symbol("resolve_dreamer_config"))

    assert {
        name: value.kind for name, value in run_signature.parameters.items()
    } == expected_run
    assert run_signature.parameters["mode"].default is inspect.Parameter.empty
    assert run_signature.parameters["task"].default is inspect.Parameter.empty
    assert run_signature.parameters["profile"].default is profile.PAPER
    assert run_signature.parameters["seed"].default == 0
    assert run_signature.parameters["model"].default is None
    assert run_signature.parameters["debug_local"].default is False
    assert run_signature.parameters["overrides"].default == overrides()
    assert list(config_signature.parameters) == [
        "mode",
        "task",
        "profile",
        "seed",
        "model",
        "debug_local",
    ]
    assert all(
        value.kind is inspect.Parameter.KEYWORD_ONLY
        for value in config_signature.parameters.values()
    )
    assert config_signature.parameters["profile"].default is profile.PAPER
    assert config_signature.parameters["seed"].default == 0
    assert config_signature.parameters["model"].default is None
    assert config_signature.parameters["debug_local"].default is False

    with pytest.raises(TypeError):
        _symbol("resolve_dreamer_run")("vision", "walker_walk")
    with pytest.raises(TypeError):
        _symbol("resolve_dreamer_config")("vision", "walker_walk")
    for name in (
        "DebugSnapshot",
        "DreamerProfile",
        "DreamerV3Config",
        "ObservationMode",
        "ResolvedDreamerRun",
        "RuntimeOverrides",
        "SequenceShapeConfig",
        "resolve_dreamer_config",
        "resolve_dreamer_run",
    ):
        assert name in dreamer_v3.__all__
        assert getattr(dreamer_v3, name) is getattr(config_module, name)
    assert "ActorCriticConfig" not in config_module.__all__
    assert "ActorCriticConfig" not in dreamer_v3.__all__
    assert not hasattr(config_module, "ActorCriticConfig")
    assert not hasattr(config_module, "_LegacyEncoderConfig")


@pytest.mark.parametrize(
    (
        "profile_name",
        "mode_name",
        "model",
        "model_dim",
        "deter",
        "depth",
        "classes",
        "env_steps",
        "train_ratio",
        "beta2",
        "strided",
        "revision",
    ),
    [
        (
            "paper",
            "vision",
            "size200m",
            1024,
            8192,
            64,
            64,
            1_000_000,
            256.0,
            0.99,
            True,
            PAPER_REVISION,
        ),
        (
            "paper",
            "proprio",
            "size200m",
            1024,
            8192,
            64,
            64,
            1_000_000,
            1024.0,
            0.99,
            True,
            PAPER_REVISION,
        ),
        (
            "upstream-current",
            "vision",
            "size200m",
            1024,
            8192,
            64,
            64,
            1_100_000,
            256.0,
            0.999,
            False,
            CURRENT_REVISION,
        ),
        (
            "upstream-current",
            "proprio",
            "size1m",
            64,
            512,
            4,
            4,
            1_100_000,
            1024.0,
            0.999,
            False,
            CURRENT_REVISION,
        ),
    ],
)
def test_all_four_profile_mode_snapshots_match_the_pinned_authorities(
    profile_name: str,
    mode_name: str,
    model: str,
    model_dim: int,
    deter: int,
    depth: int,
    classes: int,
    env_steps: int,
    train_ratio: float,
    beta2: float,
    strided: bool,
    revision: str,
) -> None:
    resolved = _resolve(profile=profile_name, mode=mode_name)
    config = resolved.config

    assert config.profile.value == profile_name
    assert config.observation_mode.value == mode_name
    assert config.task == "walker_walk"
    assert config.model == model
    assert config.network.state_dict() == {
        "model_dim": model_dim,
        "deter": deter,
        "depth": depth,
        "classes": classes,
    }
    assert config.rssm.state_dict() == {
        "deter": deter,
        "hidden": model_dim,
        "stoch": 32,
        "classes": classes,
        "blocks": 8,
        "free_nats": 1.0,
        "unimix": 0.01,
        "activation": "silu",
        "normalization": "rms",
        "image_layers": 2,
        "observation_layers": 1,
        "dynamics_layers": 1,
        "absolute": False,
        "initializer": "trunc_normal_in",
        "output_scale": 1.0,
    }
    assert config.encoder.state_dict() == {
        "depth": depth,
        "multipliers": [2, 3, 4, 4],
        "layers": 3,
        "units": model_dim,
        "activation": "silu",
        "normalization": "rms",
        "initializer": "trunc_normal_in",
        "symlog": True,
        "outer": False,
        "kernel": 5,
        "strided": strided,
    }
    assert config.decoder.state_dict() == {
        "depth": depth,
        "multipliers": [2, 3, 4, 4],
        "layers": 3,
        "units": model_dim,
        "activation": "silu",
        "normalization": "rms",
        "output_scale": 1.0,
        "initializer": "trunc_normal_in",
        "outer": False,
        "kernel": 5,
        "bias_space": 8,
        "strided": strided,
        "image_output": "mse",
    }
    assert config.reward_head.state_dict() == {
        "layers": 1,
        "units": model_dim,
        "activation": "silu",
        "normalization": "rms",
        "output": "symexp_twohot",
        "output_scale": 0.0,
        "initializer": "trunc_normal_in",
        "bins": 255,
    }
    assert config.continue_head.state_dict() == {
        "layers": 1,
        "units": model_dim,
        "activation": "silu",
        "normalization": "rms",
        "output": "binary",
        "output_scale": 1.0,
        "initializer": "trunc_normal_in",
        "bins": None,
    }
    assert config.policy.state_dict() == {
        "layers": 3,
        "units": model_dim,
        "activation": "silu",
        "normalization": "rms",
        "min_std": 0.1,
        "max_std": 1.0,
        "output_scale": 0.01,
        "unimix": 0.01,
        "initializer": "trunc_normal_in",
        "discrete": "categorical",
        "continuous": "bounded_normal",
    }
    assert config.value_head.state_dict() == {
        "layers": 3,
        "units": model_dim,
        "activation": "silu",
        "normalization": "rms",
        "output": "symexp_twohot",
        "output_scale": 0.0,
        "initializer": "trunc_normal_in",
        "bins": 255,
    }
    assert config.optimizer.state_dict() == {
        "learning_rate": 4e-5,
        "agc": 0.3,
        "agc_floor": 1e-3,
        "epsilon": 1e-20,
        "beta1": 0.9,
        "beta2": beta2,
        "momentum": True,
        "weight_decay": 0.0,
        "schedule": "const",
        "warmup": 1000,
        "anneal": 0,
    }
    assert config.sequence.state_dict() == {
        "batch_size": 16,
        "sequence_length": 64,
        "context": 1,
        "consecutive": 1,
        "report_length": 32,
        "report_consecutive": 1,
    }
    assert config.replay.state_dict() == {
        "capacity": 5_000_000,
        "chunk_size": 1024,
        "online_queue_size": 16,
    }
    assert config.run.state_dict() == {
        "env_steps": env_steps,
        "num_envs": 16,
        "eval_envs": 4,
        "train_ratio": train_ratio,
        "eval_every": 100_000,
        "eval_episodes": 1,
        "report_every": 10_000,
        "log_every": 1_000,
        "checkpoint_every": 100_000,
        "report_batches": 1,
    }
    assert config.loss_scales.state_dict() == {
        "rec": 1.0,
        "rew": 1.0,
        "con": 1.0,
        "dyn": 1.0,
        "rep": 0.1,
        "policy": 1.0,
        "value": 1.0,
        "repval": 0.3,
    }
    assert config.imagination.state_dict() == {
        "length": 15,
        "last": 0,
        "horizon": 333,
        "continuation_discount": True,
        "lambda_": 0.95,
        "actor_entropy": 3e-4,
        "imagination_slow_target": False,
        "replay_slow_target": False,
        "slow_regularizer": 1.0,
        "ac_grads": False,
        "reward_grad": True,
        "repval_loss": True,
        "repval_grad": True,
    }
    assert config.slow_value.state_dict() == {"rate": 0.02, "every": 1}
    assert config.return_normalizer.state_dict() == {
        "implementation": "percentile",
        "rate": 0.01,
        "limit": 1.0,
        "low_percentile": 5.0,
        "high_percentile": 95.0,
        "debias": False,
    }
    identity_normalizer = {
        "implementation": "none",
        "rate": 0.01,
        "limit": 1e-8,
        "low_percentile": 5.0,
        "high_percentile": 95.0,
        "debias": True,
    }
    assert config.value_normalizer.state_dict() == identity_normalizer
    assert config.advantage_normalizer.state_dict() == identity_normalizer
    assert config.seed == 0
    assert config.action_repeat == 1
    assert config.image_size == (64, 64)
    assert config.platform == "cuda"
    assert config.compute_dtype == "bfloat16"
    assert config.preallocate is True
    assert resolved.authority_revision == revision
    assert resolved.debug_snapshot is None


def test_omitted_profile_is_exactly_paper_and_current_requires_explicit_selection() -> (
    None
):
    profile = _symbol("DreamerProfile")
    omitted = _resolve()
    paper = _resolve(profile=profile.PAPER)
    current = _resolve(profile=profile.UPSTREAM_CURRENT)

    assert omitted == paper
    assert omitted.canonical_json == paper.canonical_json
    assert omitted.config_sha256 == paper.config_sha256
    assert current.config.profile is profile.UPSTREAM_CURRENT
    assert current.canonical_json != paper.canonical_json
    assert current.config_sha256 != paper.config_sha256
    assert _config() == paper.config
    assert _config(profile=profile.UPSTREAM_CURRENT) == current.config


def test_canonical_json_hash_and_identity_are_stable_and_seed_owning() -> None:
    first = _resolve(seed=7)
    second = _resolve(seed=7)
    other = _resolve(seed=11)

    assert first.canonical_json == second.canonical_json
    assert first.canonical_json.endswith("\n")
    assert first.canonical_json.count("\n") == 1
    assert json.loads(first.canonical_json) == first.config.state_dict()
    assert (
        first.config_sha256
        == hashlib.sha256(first.canonical_json.encode("utf-8")).hexdigest()
    )
    assert first.config_sha256 == second.config_sha256
    assert other.config_sha256 != first.config_sha256
    assert other.canonical_json != first.canonical_json
    assert first.identity_state() == {
        "canonical_config": first.config.state_dict(),
        "config_sha256": first.config_sha256,
        "authority_revision": PAPER_REVISION,
        "debug_snapshot": None,
        "runtime_overrides": {"algorithm": {}, "environment": {}},
    }
    assert first.identity_state()["canonical_config"]["seed"] == 7
    assert "seed" not in first.identity_state()
    assert "public_seed" not in first.identity_state()
    assert "replay_seed" not in first.identity_state()


def test_public_seed_controls_official_roots_and_dmc_identity_but_not_replay_rng() -> (
    None
):
    replay_config = _symbol("ReplayConfig")
    assert "seed" not in {item.name for item in fields(replay_config)}
    assert "seed" not in inspect.signature(replay_config).parameters

    first = _resolve(seed=7).config
    second = _resolve(seed=11).config
    assert not np.array_equal(
        np.array([first.seed, 0], np.uint32),
        np.array([second.seed, 0], np.uint32),
    )
    first_counter = np.random.default_rng([first.seed, 17]).integers(
        0, np.iinfo(np.uint32).max, (2,), np.uint32
    )
    second_counter = np.random.default_rng([second.seed, 17]).integers(
        0, np.iinfo(np.uint32).max, (2,), np.uint32
    )
    assert not np.array_equal(first_counter, second_counter)
    first_child = np.random.SeedSequence([first.seed, 2]).generate_state(1, np.uint32)[
        0
    ]
    second_child = np.random.SeedSequence([second.seed, 2]).generate_state(
        1, np.uint32
    )[0]
    assert first_child != second_child

    from world_marl.dreamer_v3_baseline.replay import UniformSelector

    selectors = [UniformSelector(0), UniformSelector(0)]
    for selector in selectors:
        for item_id in range(10):
            selector.insert(item_id)
    assert selectors[0].state_dict() == selectors[1].state_dict()
    assert [selectors[0].sample() for _ in range(8)] == [
        selectors[1].sample() for _ in range(8)
    ]
    advanced = selectors[0].state_dict()
    expected_next = selectors[0].sample()
    restored = UniformSelector.from_state_dict(advanced)
    assert restored.sample() == expected_next


@pytest.mark.parametrize("seed", [True, False, np.int64(1), -1, 2**32 - 10_000])
def test_invalid_public_seed_is_rejected_before_resolution(seed: object) -> None:
    with pytest.raises((TypeError, ValueError), match="seed"):
        _resolve(seed=seed)


def test_sequence_shape_is_the_sole_batch_time_context_owner() -> None:
    config = _resolve().config
    shape_fields = {
        "batch_size",
        "sequence_length",
        "context",
        "consecutive",
        "report_length",
        "report_consecutive",
    }

    assert shape_fields == {
        item.name for item in fields(_symbol("SequenceShapeConfig"))
    }
    assert not shape_fields & {item.name for item in fields(_symbol("ReplayConfig"))}
    assert not shape_fields & {item.name for item in fields(_symbol("RunConfig"))}
    assert config.sequence.raw_length == 65
    assert config.sequence.report_raw_length == 33


@pytest.mark.parametrize(
    ("override_name", "override_value", "owner", "field_name"),
    [
        ("env_steps", 222, "run", "env_steps"),
        ("num_envs", 3, "run", "num_envs"),
        ("batch_size", 2, "sequence", "batch_size"),
        ("batch_length", 8, "sequence", "sequence_length"),
        ("train_ratio", 7.5, "run", "train_ratio"),
        ("eval_every", 33, "run", "eval_every"),
        ("eval_episodes", 4, "run", "eval_episodes"),
        ("report_every", 44, "run", "report_every"),
        ("checkpoint_every", 55, "run", "checkpoint_every"),
    ],
)
def test_each_algorithm_runtime_override_has_one_typed_owner(
    override_name: str,
    override_value: int | float,
    owner: str,
    field_name: str,
) -> None:
    overrides = _symbol("RuntimeOverrides")(**{override_name: override_value})
    base = _resolve()
    resolved = _resolve(overrides=overrides)

    assert getattr(getattr(resolved.config, owner), field_name) == override_value
    assert resolved.algorithm_overrides == {override_name: override_value}
    assert resolved.environment_overrides == {}
    assert list(resolved.algorithm_overrides) == sorted(resolved.algorithm_overrides)
    assert resolved.config_sha256 != base.config_sha256


def test_camera_is_environment_only_and_all_overrides_follow_the_merge_order() -> None:
    runtime_overrides = _symbol("RuntimeOverrides")
    base = _resolve(debug_local=True)
    camera = _resolve(overrides=runtime_overrides(camera=0))
    resolved = _resolve(
        debug_local=True,
        overrides=runtime_overrides(
            env_steps=96,
            num_envs=2,
            batch_size=2,
            batch_length=8,
            train_ratio=8.0,
            eval_every=32,
            eval_episodes=2,
            report_every=24,
            checkpoint_every=40,
            camera=2,
        ),
    )

    assert camera.config == _resolve().config
    assert camera.config_sha256 == _resolve().config_sha256
    assert camera.algorithm_overrides == {}
    assert camera.environment_overrides == {"camera": 0}
    assert "camera" not in camera.canonical_json
    assert resolved.config.run.state_dict() == {
        "env_steps": 96,
        "num_envs": 2,
        "eval_envs": 1,
        "train_ratio": 8.0,
        "eval_every": 32,
        "eval_episodes": 2,
        "report_every": 24,
        "log_every": 16,
        "checkpoint_every": 40,
        "report_batches": 1,
    }
    assert resolved.config.sequence.state_dict() == {
        "batch_size": 2,
        "sequence_length": 8,
        "context": 0,
        "consecutive": 1,
        "report_length": 4,
        "report_consecutive": 1,
    }
    assert resolved.algorithm_overrides == {
        "batch_length": 8,
        "batch_size": 2,
        "checkpoint_every": 40,
        "env_steps": 96,
        "eval_episodes": 2,
        "eval_every": 32,
        "num_envs": 2,
        "report_every": 24,
        "train_ratio": 8.0,
    }
    assert resolved.environment_overrides == {"camera": 2}
    assert resolved.config_sha256 != base.config_sha256


def test_runtime_override_surface_is_closed_and_revalidated() -> None:
    runtime_overrides = _symbol("RuntimeOverrides")
    names = {item.name for item in fields(runtime_overrides)}
    assert names == {
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
    }
    assert not names & {
        "out_dir",
        "resume",
        "dry_run",
        "dry_run_matrix",
        "stop_after_env_steps",
        "log_every",
        "seed",
    }
    with pytest.raises(TypeError):
        runtime_overrides(unknown=1)
    for kwargs in (
        {"env_steps": 0},
        {"num_envs": True},
        {"batch_size": -1},
        {"batch_length": np.int64(4)},
        {"train_ratio": float("nan")},
        {"eval_every": 0},
        {"eval_episodes": 0},
        {"report_every": 0},
        {"checkpoint_every": 0},
        {"camera": True},
    ):
        with pytest.raises((TypeError, ValueError)):
            runtime_overrides(**kwargs)
    with pytest.raises(ValueError, match="replay capacity"):
        _resolve(overrides=runtime_overrides(batch_size=1000, batch_length=10_000))


def test_debug_local_v1_is_complete_deterministic_and_noncanonical() -> None:
    resolved = _resolve(debug_local=True)
    config = resolved.config
    debug = resolved.debug_snapshot

    assert debug is not None
    assert debug.state_dict() == {
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
        "replay": {"capacity": 256, "chunk_size": 32, "online_queue_size": 16},
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
    assert config.model == "debug-local-v1"
    assert config.network.state_dict() == {
        "model_dim": 32,
        "deter": 32,
        "depth": 8,
        "classes": 4,
    }
    assert config.encoder.multipliers == (1, 2, 4, 8)
    assert config.decoder.multipliers == (1, 2, 4, 8)
    assert config.sequence == debug.sequence
    assert config.replay == debug.replay
    assert config.run == debug.run
    assert config.imagination.length == 5
    assert config.platform == "cpu"
    assert config.preallocate is False
    assert config.loss_scales == _resolve().config.loss_scales
    assert config.optimizer == _resolve().config.optimizer
    assert config.action_repeat == 1
    assert config.image_size == (64, 64)
    assert resolved.identity_state()["debug_snapshot"] == debug.state_dict()


def test_all_config_runtime_and_debug_records_roundtrip_through_public_flax() -> None:
    resolved = _resolve(debug_local=True, seed=19)
    config = resolved.config
    records = [
        config.profile,
        config.observation_mode,
        _symbol("ModelSize").M200,
        config.network,
        config.rssm,
        config.encoder,
        config.decoder,
        config.reward_head,
        config.continue_head,
        config.policy,
        config.value_head,
        config.optimizer,
        config.sequence,
        config.replay,
        config.run,
        config.loss_scales,
        config.imagination,
        config.slow_value,
        config.return_normalizer,
        config.value_normalizer,
        config.advantage_normalizer,
        config,
        resolved.explicit_overrides,
        resolved.debug_snapshot,
        resolved,
    ]
    for record in records:
        assert record is not None
        first = record.state_dict()
        second = record.state_dict()
        assert first == second
        assert first is not second
        restored_state = serialization.msgpack_restore(
            serialization.msgpack_serialize(first)
        )
        restored = type(record).from_state(restored_state)
        assert restored == record
        assert restored.state_dict() == first


def test_state_records_are_closed_fresh_and_reject_nonprimitive_leaves() -> None:
    resolved = _resolve(debug_local=True)
    state = resolved.config.state_dict()
    other = resolved.config.state_dict()
    assert state is not other
    assert state["encoder"] is not other["encoder"]
    assert state["encoder"]["multipliers"] is not other["encoder"]["multipliers"]
    state["encoder"]["multipliers"][0] = 999
    assert resolved.config.encoder.multipliers == (1, 2, 4, 8)

    invalid_states = []
    missing = resolved.config.state_dict()
    missing.pop("seed")
    invalid_states.append(missing)
    extra = resolved.config.state_dict()
    extra["extra"] = 1
    invalid_states.append(extra)
    tuple_leaf = resolved.config.state_dict()
    tuple_leaf["image_size"] = (64, 64)
    invalid_states.append(tuple_leaf)
    dataclass_leaf = resolved.config.state_dict()
    dataclass_leaf["network"] = resolved.config.network
    invalid_states.append(dataclass_leaf)
    enum_leaf = resolved.config.state_dict()
    enum_leaf["profile"] = resolved.config.profile
    invalid_states.append(enum_leaf)
    frozen_leaf = resolved.config.state_dict()
    frozen_leaf["network"] = FrozenDict(frozen_leaf["network"])
    invalid_states.append(frozen_leaf)
    for invalid in invalid_states:
        with pytest.raises((TypeError, ValueError)):
            _symbol("DreamerV3Config").from_state(invalid)
    with pytest.raises((TypeError, ValueError)):
        _symbol("DreamerV3Config").from_state(
            tuple(resolved.config.state_dict().items())
        )
    with pytest.raises((TypeError, ValueError)):
        _symbol("DreamerV3Config").from_state(FrozenDict(resolved.config.state_dict()))


def test_config_state_rejects_shared_nested_record_mapping() -> None:
    state = _resolve().config.state_dict()
    state["advantage_normalizer"] = state["value_normalizer"]

    with pytest.raises(TypeError, match="alias"):
        _symbol("DreamerV3Config").from_state(state)


def test_config_state_rejects_shared_schema_compatible_list() -> None:
    state = _resolve().config.state_dict()
    state["decoder"]["multipliers"] = state["encoder"]["multipliers"]

    with pytest.raises(TypeError, match="alias"):
        _symbol("DreamerV3Config").from_state(state)


def test_resolved_state_rejects_cross_config_debug_alias() -> None:
    state = _resolve(debug_local=True).state_dict()
    state["debug_snapshot"]["sequence"] = state["canonical_config"]["sequence"]

    with pytest.raises(TypeError, match="alias"):
        _symbol("ResolvedDreamerRun").from_state(state)


def test_resolved_state_rejects_shared_sparse_override_maps() -> None:
    state = _resolve().state_dict()
    state["runtime_overrides"]["environment"] = state["runtime_overrides"]["algorithm"]

    with pytest.raises(TypeError, match="alias"):
        _symbol("ResolvedDreamerRun").from_state(state)


@pytest.mark.parametrize(
    "noncanonical_key",
    [
        _StringSubclass("model_dim"),
        np.str_("model_dim"),
        _StringEnumKey.MODEL_DIM,
    ],
)
def test_fixed_record_states_reject_non_builtin_string_keys(
    noncanonical_key: object,
) -> None:
    network = _symbol("NetworkSize")(32, 64, 8, 4)
    state = network.state_dict()
    value = state.pop("model_dim")
    state[noncanonical_key] = value

    with pytest.raises(TypeError, match="keys"):
        type(network).from_state(state)


def test_frozen_records_reject_legacy_construction_and_profile_patching() -> None:
    config = _resolve().config
    with pytest.raises(FrozenInstanceError):
        config.seed = 3
    with pytest.raises(TypeError):
        _symbol("DreamerV3Config")(action_dim=4, observation_shape=(8, 8, 3))
    with pytest.raises(ValueError, match="beta2"):
        replace(config, optimizer=replace(config.optimizer, beta2=0.999))
    with pytest.raises(ValueError, match="model"):
        _resolve(model="size1m")
    with pytest.raises(ValueError, match="debug"):
        _resolve(debug_local=True, model="size200m")


def test_model_size_has_one_canonical_public_member_per_official_size() -> None:
    model_size = _symbol("ModelSize")
    assert tuple(model_size.__members__) == (
        "M1",
        "M12",
        "M25",
        "M50",
        "M100",
        "M200",
        "M400",
    )


@pytest.mark.parametrize(
    ("enum_name", "member_name", "canonical_value", "invalid_value"),
    [
        ("DreamerProfile", "PAPER", "paper", np.str_("paper")),
        ("DreamerProfile", "PAPER", "paper", _StringSubclass("paper")),
        ("DreamerProfile", "PAPER", "paper", 1),
        ("ObservationMode", "VISION", "vision", np.str_("vision")),
        ("ObservationMode", "VISION", "vision", _StringSubclass("vision")),
        ("ObservationMode", "VISION", "vision", 1),
        ("ModelSize", "M200", "200m", np.str_("200m")),
        ("ModelSize", "M200", "200m", _StringSubclass("200m")),
        ("ModelSize", "M200", "200m", 200),
    ],
)
def test_public_string_enum_construction_requires_exact_builtin_strings(
    enum_name: str,
    member_name: str,
    canonical_value: str,
    invalid_value: object,
) -> None:
    enum_type = _symbol(enum_name)
    member = getattr(enum_type, member_name)

    assert enum_type(canonical_value) is member
    assert enum_type(member) is member
    with pytest.raises(TypeError):
        enum_type(invalid_value)


def test_model_size_preserves_exact_official_size_prefix_normalization() -> None:
    model_size = _symbol("ModelSize")
    assert model_size("size200m") is model_size.M200
    assert model_size("SIZE200M") is model_size.M200


@pytest.mark.parametrize(
    "resolver_name", ["resolve_dreamer_run", "resolve_dreamer_config"]
)
@pytest.mark.parametrize(
    ("coordinate", "plain_value", "member_name", "expected_field", "expected_value"),
    [
        ("profile", "paper", "PAPER", "profile", "paper"),
        ("mode", "vision", "VISION", "observation_mode", "vision"),
        ("model", "size200m", "M200", "model", "size200m"),
    ],
)
def test_public_resolvers_accept_exact_strings_and_enum_members(
    resolver_name: str,
    coordinate: str,
    plain_value: str,
    member_name: str,
    expected_field: str,
    expected_value: str,
) -> None:
    enum_name = {
        "profile": "DreamerProfile",
        "mode": "ObservationMode",
        "model": "ModelSize",
    }[coordinate]
    enum_type = _symbol(enum_name)
    resolver = _symbol(resolver_name)
    member = getattr(enum_type, member_name)

    for value in (plain_value, member):
        kwargs: dict[str, object] = {"mode": "vision", "task": "walker_walk"}
        kwargs[coordinate] = value
        result = resolver(**kwargs)
        config = result.config if resolver_name == "resolve_dreamer_run" else result
        actual = getattr(config, expected_field)
        assert getattr(actual, "value", actual) == expected_value


@pytest.mark.parametrize(
    "resolver_name", ["resolve_dreamer_run", "resolve_dreamer_config"]
)
@pytest.mark.parametrize(
    ("coordinate", "invalid_value"),
    [
        ("profile", np.str_("paper")),
        ("profile", _StringSubclass("paper")),
        ("profile", 1),
        ("mode", np.str_("vision")),
        ("mode", _StringSubclass("vision")),
        ("mode", 1),
        ("model", np.str_("size200m")),
        ("model", _StringSubclass("size200m")),
        ("model", 200),
    ],
)
def test_public_resolvers_reject_nonexact_string_coordinates(
    resolver_name: str,
    coordinate: str,
    invalid_value: object,
) -> None:
    kwargs: dict[str, object] = {"mode": "vision", "task": "walker_walk"}
    kwargs[coordinate] = invalid_value
    with pytest.raises(TypeError):
        _symbol(resolver_name)(**kwargs)


@pytest.mark.parametrize(
    ("record_name", "kwargs"),
    [
        ("NetworkSize", {"model_dim": True, "deter": 8, "depth": 1, "classes": 1}),
        ("RSSMConfig", {"hidden": True}),
        ("RSSMConfig", {"free_nats": 1}),
        ("RSSMConfig", {"unimix": np.float64(0.01)}),
        ("RSSMConfig", {"absolute": 0}),
        ("RSSMConfig", {"activation": 1}),
        ("EncoderConfig", {"units": True}),
        ("EncoderConfig", {"multipliers": [2, 3, 4, 4]}),
        ("EncoderConfig", {"multipliers": (2, 3, np.int64(4), 4)}),
        ("EncoderConfig", {"symlog": 1}),
        ("DecoderConfig", {"output_scale": 1}),
        ("DecoderConfig", {"outer": 0}),
        ("HeadConfig", {"output_scale": 0}),
        ("HeadConfig", {"bins": True}),
        ("RewardHeadConfig", {"output_scale": 0}),
        ("ContinueHeadConfig", {"output_scale": 1}),
        ("PolicyConfig", {"layers": True}),
        ("PolicyConfig", {"min_std": 0}),
        ("PolicyConfig", {"discrete": 1}),
        ("OptimizerConfig", {"learning_rate": 1}),
        ("OptimizerConfig", {"momentum": 1}),
        ("OptimizerConfig", {"warmup": True}),
        ("SequenceShapeConfig", {"batch_size": True}),
        ("SequenceShapeConfig", {"report_length": np.int64(32)}),
        ("ReplayConfig", {"capacity": True}),
        ("RunConfig", {"train_ratio": 256}),
        ("RunConfig", {"eval_envs": True}),
        ("RunConfig", {"report_batches": np.int64(1)}),
        ("LossScaleConfig", {"rec": True}),
        ("ImaginationConfig", {"length": True}),
        ("ImaginationConfig", {"lambda_": 1}),
        ("ImaginationConfig", {"continuation_discount": 1}),
        ("SlowValueConfig", {"rate": 1}),
        ("SlowValueConfig", {"every": True}),
        ("NormalizerConfig", {"rate": 1}),
        ("NormalizerConfig", {"debias": 0}),
        ("NormalizerConfig", {"implementation": 1}),
        ("RuntimeOverrides", {"train_ratio": 1}),
        ("RuntimeOverrides", {"env_steps": np.int64(1)}),
    ],
)
def test_public_record_constructors_reject_noncanonical_primitive_types(
    record_name: str,
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(TypeError):
        _symbol(record_name)(**kwargs)


def test_nested_public_record_constructors_reject_wrong_record_types() -> None:
    resolved = _resolve(debug_local=True)
    debug = resolved.debug_snapshot
    assert debug is not None
    with pytest.raises(TypeError):
        replace(debug, sequence=debug.sequence.state_dict())
    with pytest.raises(TypeError):
        replace(resolved.config, optimizer=resolved.config.optimizer.state_dict())
    with pytest.raises(TypeError):
        replace(resolved, explicit_overrides={})
    with pytest.raises(TypeError):
        replace(resolved.config, task=1)
    with pytest.raises(ValueError):
        replace(resolved.config, task="")


def test_every_named_record_has_an_exact_constructor_state_inverse() -> None:
    resolved = _resolve(debug_local=True, seed=19)
    debug = resolved.debug_snapshot
    assert debug is not None
    records = (
        _symbol("DreamerProfile").PAPER,
        _symbol("ObservationMode").VISION,
        _symbol("ModelSize").M200,
        _symbol("NetworkSize")(32, 64, 8, 4),
        _symbol("RSSMConfig")(),
        _symbol("EncoderConfig")(),
        _symbol("DecoderConfig")(),
        _symbol("HeadConfig")(),
        _symbol("RewardHeadConfig")(),
        _symbol("ContinueHeadConfig")(),
        _symbol("PolicyConfig")(),
        _symbol("OptimizerConfig")(),
        _symbol("SequenceShapeConfig")(),
        _symbol("ReplayConfig")(),
        _symbol("RunConfig")(),
        _symbol("LossScaleConfig")(),
        _symbol("ImaginationConfig")(),
        _symbol("SlowValueConfig")(),
        _symbol("NormalizerConfig")(),
        _symbol("RuntimeOverrides")(env_steps=96, train_ratio=8.0, camera=0),
        debug,
        resolved.config,
        resolved,
    )
    for record in records:
        assert type(record).from_state(record.state_dict()) == record


@pytest.mark.parametrize(
    ("family", "mutation"),
    [
        (
            "network",
            lambda config: replace(
                config, network=replace(config.network, model_dim=512)
            ),
        ),
        (
            "rssm",
            lambda config: replace(config, rssm=replace(config.rssm, output_scale=0.5)),
        ),
        (
            "encoder",
            lambda config: replace(
                config, encoder=replace(config.encoder, activation="relu")
            ),
        ),
        (
            "decoder",
            lambda config: replace(
                config, decoder=replace(config.decoder, image_output="normal")
            ),
        ),
        (
            "reward_head",
            lambda config: replace(
                config, reward_head=replace(config.reward_head, initializer="uniform")
            ),
        ),
        (
            "continue_head",
            lambda config: replace(
                config, continue_head=replace(config.continue_head, activation="relu")
            ),
        ),
        (
            "policy",
            lambda config: replace(
                config, policy=replace(config.policy, continuous="bogus")
            ),
        ),
        (
            "value_head",
            lambda config: replace(
                config, value_head=replace(config.value_head, bins=254)
            ),
        ),
        (
            "optimizer",
            lambda config: replace(
                config, optimizer=replace(config.optimizer, learning_rate=0.5)
            ),
        ),
        (
            "sequence",
            lambda config: replace(
                config, sequence=replace(config.sequence, report_length=31)
            ),
        ),
        (
            "replay",
            lambda config: replace(
                config, replay=replace(config.replay, capacity=4_000_000)
            ),
        ),
        (
            "run",
            lambda config: replace(config, run=replace(config.run, eval_envs=3)),
        ),
        (
            "loss_scales",
            lambda config: replace(
                config, loss_scales=replace(config.loss_scales, rec=0.5)
            ),
        ),
        (
            "imagination",
            lambda config: replace(
                config, imagination=replace(config.imagination, horizon=100)
            ),
        ),
        (
            "slow_value",
            lambda config: replace(
                config, slow_value=replace(config.slow_value, rate=0.03)
            ),
        ),
        (
            "return_normalizer",
            lambda config: replace(
                config,
                return_normalizer=replace(config.return_normalizer, rate=0.02),
            ),
        ),
        (
            "value_normalizer",
            lambda config: replace(
                config,
                value_normalizer=replace(config.value_normalizer, debias=False),
            ),
        ),
        (
            "advantage_normalizer",
            lambda config: replace(
                config,
                advantage_normalizer=replace(config.advantage_normalizer, debias=False),
            ),
        ),
        ("action_repeat", lambda config: replace(config, action_repeat=2)),
        ("image_size", lambda config: replace(config, image_size=(32, 32))),
        ("platform", lambda config: replace(config, platform="tpu")),
        ("compute_dtype", lambda config: replace(config, compute_dtype="float32")),
        ("preallocate", lambda config: replace(config, preallocate=False)),
    ],
)
def test_locked_profile_rejects_each_component_family_patch(
    family: str,
    mutation: Any,
) -> None:
    del family
    with pytest.raises(ValueError):
        mutation(_resolve().config)


def test_debug_config_preserves_locked_optimizer_snapshot() -> None:
    config = _resolve(debug_local=True).config
    with pytest.raises(ValueError):
        replace(
            config,
            optimizer=replace(config.optimizer, learning_rate=0.5),
        )


def _resolved_with(
    config: Any,
    *,
    overrides: Any | None = None,
    debug: Any | None = None,
) -> Any:
    canonical_json = config.canonical_json()
    return _symbol("ResolvedDreamerRun")(
        config=config,
        explicit_overrides=(
            _symbol("RuntimeOverrides")() if overrides is None else overrides
        ),
        debug_snapshot=debug,
        authority_revision=(
            PAPER_REVISION
            if config.profile is _symbol("DreamerProfile").PAPER
            else CURRENT_REVISION
        ),
        canonical_json=canonical_json,
        config_sha256=hashlib.sha256(canonical_json.encode("utf-8")).hexdigest(),
    )


def test_resolved_run_reconstructs_and_checks_exact_merge_coordinates() -> None:
    base = _resolve()
    debug = _resolve(debug_local=True)
    assert debug.debug_snapshot is not None
    changed_steps = replace(base.config, run=replace(base.config.run, env_steps=123))
    with pytest.raises(ValueError):
        _resolved_with(changed_steps)
    with pytest.raises(ValueError):
        _resolved_with(base.config, overrides=_symbol("RuntimeOverrides")(env_steps=7))
    with pytest.raises(ValueError):
        _resolved_with(debug.config)
    with pytest.raises(ValueError):
        _resolved_with(base.config, debug=debug.debug_snapshot)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("mlp_layers",), 2),
        (("mlp_units",), 33),
        (("rssm_deter",), 40),
        (("rssm_stoch",), 5),
        (("rssm_classes",), 5),
        (("sequence", "batch_size"), 2),
        (("sequence", "report_length"), 5),
        (("replay", "capacity"), 512),
        (("run", "env_steps"), 64),
        (("run", "eval_envs"), 2),
        (("run", "report_batches"), 2),
        (("imagination_horizon",), 6),
        (("platform",), "tpu"),
        (("preallocate",), True),
    ],
)
def test_debug_local_snapshot_rejects_every_fixed_value_patch(
    path: tuple[str, ...],
    value: object,
) -> None:
    debug = _resolve(debug_local=True).debug_snapshot
    assert debug is not None
    state = debug.state_dict()
    target = state
    for name in path[:-1]:
        target = target[name]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        _symbol("DebugSnapshot").from_state(state)


def _canonical_hash(state: dict[str, object]) -> str:
    text = (
        json.dumps(state, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@pytest.mark.parametrize("case", ["missing", "extra", "disagreeing"])
def test_resolved_state_rejects_incoherent_explicit_override_maps(case: str) -> None:
    runtime_overrides = _symbol("RuntimeOverrides")
    if case == "missing":
        state = _resolve(overrides=runtime_overrides(env_steps=96)).state_dict()
        state["runtime_overrides"]["algorithm"] = {}
    elif case == "extra":
        state = _resolve().state_dict()
        state["runtime_overrides"]["algorithm"] = {"env_steps": 96}
    else:
        state = _resolve(overrides=runtime_overrides(env_steps=96)).state_dict()
        state["runtime_overrides"]["algorithm"] = {"env_steps": 97}
    with pytest.raises(ValueError):
        _symbol("ResolvedDreamerRun").from_state(state)


@pytest.mark.parametrize(
    ("section", "key"),
    [
        ("algorithm", "batch_length"),
        ("algorithm", "batch_size"),
        ("algorithm", "checkpoint_every"),
        ("algorithm", "env_steps"),
        ("algorithm", "eval_episodes"),
        ("algorithm", "eval_every"),
        ("algorithm", "num_envs"),
        ("algorithm", "report_every"),
        ("algorithm", "train_ratio"),
        ("environment", "camera"),
    ],
)
def test_resolved_state_rejects_explicit_none_override_values(
    section: str,
    key: str,
) -> None:
    state = _resolve().state_dict()
    state["runtime_overrides"][section][key] = None

    with pytest.raises(TypeError):
        _symbol("ResolvedDreamerRun").from_state(state)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("algorithm", "batch_length", 16),
        ("algorithm", "batch_size", 4),
        ("algorithm", "checkpoint_every", 100),
        ("algorithm", "env_steps", 96),
        ("algorithm", "eval_episodes", 2),
        ("algorithm", "eval_every", 32),
        ("algorithm", "num_envs", 2),
        ("algorithm", "report_every", 16),
        ("algorithm", "train_ratio", 8.0),
        ("environment", "camera", 1),
    ],
)
def test_resolved_state_reemits_each_accepted_sparse_override_tree_exactly(
    section: str,
    key: str,
    value: int | float,
) -> None:
    runtime_overrides = _symbol("RuntimeOverrides")(**{key: value})
    state = _resolve(overrides=runtime_overrides).state_dict()
    assert state["runtime_overrides"][section] == {key: value}

    restored = _symbol("ResolvedDreamerRun").from_state(copy.deepcopy(state))
    assert restored.state_dict() == state


@pytest.mark.parametrize(
    ("section", "canonical_key", "noncanonical_key"),
    [
        ("algorithm", "env_steps", _StringSubclass("env_steps")),
        ("algorithm", "env_steps", np.str_("env_steps")),
        ("algorithm", "env_steps", _StringEnumKey.ENV_STEPS),
        ("environment", "camera", _StringSubclass("camera")),
        ("environment", "camera", np.str_("camera")),
        ("environment", "camera", _StringEnumKey.CAMERA),
    ],
)
def test_resolved_sparse_override_maps_reject_non_builtin_string_keys(
    section: str,
    canonical_key: str,
    noncanonical_key: object,
) -> None:
    overrides = _symbol("RuntimeOverrides")(env_steps=96, camera=2)
    state = _resolve(overrides=overrides).state_dict()
    override_map = state["runtime_overrides"][section]
    value = override_map.pop(canonical_key)
    override_map[noncanonical_key] = value

    with pytest.raises(TypeError, match="keys"):
        _symbol("ResolvedDreamerRun").from_state(state)


def test_builtin_string_state_keys_roundtrip_without_canonicalization() -> None:
    network = _symbol("NetworkSize")(32, 64, 8, 4)
    network_state = network.state_dict()
    restored_network = type(network).from_state(copy.deepcopy(network_state))
    assert restored_network.state_dict() == network_state
    assert all(type(key) is str for key in restored_network.state_dict())

    overrides = _symbol("RuntimeOverrides")(env_steps=96, camera=2)
    resolved_state = _resolve(overrides=overrides).state_dict()
    restored_run = _symbol("ResolvedDreamerRun").from_state(
        copy.deepcopy(resolved_state)
    )
    assert restored_run.state_dict() == resolved_state
    assert all(
        type(key) is str
        for override_map in restored_run.state_dict()["runtime_overrides"].values()
        for key in override_map
    )


def test_resolved_state_rejects_debug_presence_and_locked_component_patches() -> None:
    debug_state = _resolve(debug_local=True).state_dict()
    debug_state["debug_snapshot"] = None
    with pytest.raises(ValueError):
        _symbol("ResolvedDreamerRun").from_state(debug_state)

    production_state = _resolve().state_dict()
    production_state["debug_snapshot"] = _resolve(
        debug_local=True
    ).debug_snapshot.state_dict()
    with pytest.raises(ValueError):
        _symbol("ResolvedDreamerRun").from_state(production_state)

    patched = copy.deepcopy(_resolve().state_dict())
    patched["canonical_config"]["optimizer"]["learning_rate"] = 0.5
    patched["config_sha256"] = _canonical_hash(patched["canonical_config"])
    with pytest.raises(ValueError):
        _symbol("ResolvedDreamerRun").from_state(patched)


def test_normalizer_debias_defaults_are_source_derived() -> None:
    config = _resolve().config
    assert _symbol("NormalizerConfig")().debias is True
    assert config.return_normalizer.debias is False
    assert config.value_normalizer.debias is True
    assert config.advantage_normalizer.debias is True
