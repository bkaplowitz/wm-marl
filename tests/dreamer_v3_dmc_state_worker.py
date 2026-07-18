from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

if os.environ.get("MUJOCO_GL") != "off":
    raise RuntimeError("the DMC state worker requires MUJOCO_GL=off")

import dm_env
import mujoco
import numpy as np
from dm_control import suite


TASKS = (
    ("acrobot_swingup", "acrobot", "swingup"),
    ("ball_in_cup_catch", "ball_in_cup", "catch"),
    ("cartpole_balance", "cartpole", "balance"),
    ("cartpole_balance_sparse", "cartpole", "balance_sparse"),
    ("cartpole_swingup", "cartpole", "swingup"),
    ("cartpole_swingup_sparse", "cartpole", "swingup_sparse"),
    ("cheetah_run", "cheetah", "run"),
    ("finger_spin", "finger", "spin"),
    ("finger_turn_easy", "finger", "turn_easy"),
    ("finger_turn_hard", "finger", "turn_hard"),
    ("hopper_hop", "hopper", "hop"),
    ("hopper_stand", "hopper", "stand"),
    ("pendulum_swingup", "pendulum", "swingup"),
    ("quadruped_run", "quadruped", "run"),
    ("quadruped_walk", "quadruped", "walk"),
    ("reacher_easy", "reacher", "easy"),
    ("reacher_hard", "reacher", "hard"),
    ("walker_run", "walker", "run"),
    ("walker_stand", "walker", "stand"),
    ("walker_walk", "walker", "walk"),
)
TASK_MAP = {task_id: (domain, task) for task_id, domain, task in TASKS}
TASK_IDS = tuple(task_id for task_id, _, _ in TASKS)
INTEGRATION_SPEC = 8191
SENSOR_NOISE_BIT = int(mujoco.mjtEnableBit.mjENBL_SENSORNOISE)
INTEGRATION_COMPONENTS = (
    ("time", "mjSTATE_TIME", 1),
    ("qpos", "mjSTATE_QPOS", 2),
    ("qvel", "mjSTATE_QVEL", 4),
    ("act", "mjSTATE_ACT", 8),
    ("qacc_warmstart", "mjSTATE_WARMSTART", 16),
    ("ctrl", "mjSTATE_CTRL", 32),
    ("qfrc_applied", "mjSTATE_QFRC_APPLIED", 64),
    ("xfrc_applied", "mjSTATE_XFRC_APPLIED", 128),
    ("eq_active", "mjSTATE_EQ_ACTIVE", 256),
    ("mocap_pos", "mjSTATE_MOCAP_POS", 512),
    ("mocap_quat", "mjSTATE_MOCAP_QUAT", 1024),
    ("userdata", "mjSTATE_USERDATA", 2048),
    ("plugin_state", "mjSTATE_PLUGIN", 4096),
)
RESTORE_ORDER = (
    "validate_closed_candidate",
    "construct_locked_task",
    "copy_complete_model_arrays",
    "mj_setState(INTEGRATION)",
    "mj_step1(legacy_step=True)",
    "restore_task_rng_and_mutable_task_fields",
    "restore_environment_counters_and_adapter_current_time_step",
    "clear_only_enumerated_derived_caches",
)
EXPECTED_TASK_FIELDS = {
    "acrobot_swingup": {"_sparse": "static"},
    "ball_in_cup_catch": {},
    "cartpole_balance": {"_sparse": "static", "_swing_up": "static"},
    "cartpole_balance_sparse": {"_sparse": "static", "_swing_up": "static"},
    "cartpole_swingup": {"_sparse": "static", "_swing_up": "static"},
    "cartpole_swingup_sparse": {"_sparse": "static", "_swing_up": "static"},
    "cheetah_run": {},
    "finger_spin": {},
    "finger_turn_easy": {"_target_radius": "static"},
    "finger_turn_hard": {"_target_radius": "static"},
    "hopper_hop": {"_hopping": "static"},
    "hopper_stand": {"_hopping": "static"},
    "pendulum_swingup": {},
    "quadruped_run": {"_desired_speed": "static"},
    "quadruped_walk": {"_desired_speed": "static"},
    "reacher_easy": {"_target_size": "static"},
    "reacher_hard": {"_target_size": "static"},
    "walker_run": {"_move_speed": "static"},
    "walker_stand": {"_move_speed": "static"},
    "walker_walk": {"_move_speed": "static"},
}
RESET_ONLY_TASK_FIELDS = {
    "cheetah_run": {"_timeout_progress"},
    "hopper_hop": {"_timeout_progress"},
    "hopper_stand": {"_timeout_progress"},
}
MODEL_ARRAY_FIELDS = {
    "finger_spin": {
        "dof_damping": "hinge",
        "site_rgba": "target/tip alpha",
    },
    "finger_turn_easy": {
        "site_pos": "target x/z",
        "site_size": "target radius",
    },
    "finger_turn_hard": {
        "site_pos": "target x/z",
        "site_size": "target radius",
    },
    "reacher_easy": {
        "geom_pos": "target x/y",
        "geom_size": "target radius",
    },
    "reacher_hard": {
        "geom_pos": "target x/y",
        "geom_size": "target radius",
    },
}
CORRUPTION_FAMILIES = (
    "closed_top_level",
    "state_format",
    "dmc_spec_mode",
    "dmc_spec_image_dimension_float",
    "dmc_spec_image_container",
    "dmc_spec_camera_bool",
    "dmc_spec_camera_out_of_range",
    "backend_identity",
    "compiled_model_identity",
    "legacy_step",
    "environment_static",
    "task_static_finger_turn",
    "action_spec",
    "observation_spec",
    "sensor_gate",
    "derived_cache_schema_quadruped",
    "environment_counter_dtype",
    "environment_reset_flag_dtype",
    "step_count_negative",
    "step_count_above_limit",
    "first_reward_nonnull",
    "first_discount_nonnull",
    "first_step_count_nonzero",
    "first_reset_pending",
    "mid_reward_null",
    "mid_discount_not_one",
    "mid_step_count_zero",
    "mid_step_count_at_limit",
    "mid_reset_pending",
    "last_reward_null",
    "last_discount_null",
    "last_step_count_below_limit",
    "last_reset_not_pending",
    "integration_dtype",
    "model_array_finger",
    "rng_algorithm",
    "rng_fractional_position_with_changed_integration",
    "rng_flags",
    "rng_keys",
    "time_step_enum",
    "time_step_observation",
)


def _load(domain: str, task: str, seed: int) -> Any:
    return suite.load(
        domain,
        task,
        task_kwargs={"random": seed},
        visualize_reward=False,
    )


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")


def _array_schema(value: Any) -> dict[str, Any]:
    array = np.asarray(value)
    return {"dtype": array.dtype.str, "shape": list(array.shape)}


def _encoded_array(value: Any) -> dict[str, Any]:
    array = np.asarray(value)
    return {
        **_array_schema(array),
        "data_hex": array.tobytes(order="C").hex(),
        "encoding": "c_order_lower_hex",
    }


def _spec_schema(spec: Any) -> dict[str, Any]:
    result = {
        "dtype": np.dtype(spec.dtype).str,
        "name": spec.name,
        "shape": list(spec.shape),
    }
    if hasattr(spec, "minimum"):
        result["minimum"] = _encoded_array(
            np.broadcast_to(spec.minimum, spec.shape).astype(spec.dtype, copy=False)
        )
        result["maximum"] = _encoded_array(
            np.broadcast_to(spec.maximum, spec.shape).astype(spec.dtype, copy=False)
        )
    return result


def _runtime_scalar_type(value: Any) -> dict[str, str]:
    if isinstance(value, (bool, np.bool_)):
        return {"kind": "scalar", "module": "builtins", "name": "bool"}
    if isinstance(value, (int, np.integer)):
        return {"kind": "scalar", "module": "builtins", "name": "int"}
    if isinstance(value, (float, np.floating)):
        return {"kind": "scalar", "module": "builtins", "name": "float"}
    raise AssertionError(type(value).__name__)


def _serialized_scalar_schema(kind: str) -> dict[str, Any]:
    dtype = {"bool": "|b1", "int": "<i8", "float": "<f8"}[kind]
    return {
        "boolean_allowed_as_integer": False,
        "dtype": dtype,
        "kind": "ndarray",
        "shape": [],
    }


def _scalar_field(value: Any, role: str) -> dict[str, Any]:
    runtime = _runtime_scalar_type(value)
    kind = runtime["name"]
    result = {
        "role": role,
        "runtime": runtime,
        "serialized": _serialized_scalar_schema(kind),
    }
    if role == "static_compatibility":
        dtype = np.dtype(result["serialized"]["dtype"])
        result["value"] = _encoded_array(np.asarray(value, dtype=dtype))
    return result


def _integration_component_sizes(model: Any) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for key, enum_name, flag in INTEGRATION_COMPONENTS:
        enum_value = int(getattr(mujoco.mjtState, enum_name))
        assert enum_value == flag
        sizes[key] = mujoco.mj_stateSize(model.ptr, enum_value)
    return sizes


def _derived_cache_fields(task_id: str) -> list[dict[str, Any]]:
    if not task_id.startswith("quadruped_"):
        return []
    return [
        {
            "name": "_hinge_names",
            "runtime": {
                "container": "list",
                "item": {"kind": "string"},
                "kind": "sequence",
            },
            "restore": "clear_then_lazy_rebuild",
        },
        {
            "name": "_sensor_types_to_names",
            "runtime": {
                "key": {"item": {"kind": "integer"}, "kind": "tuple"},
                "kind": "mapping",
                "value": {
                    "container": "list",
                    "item": {"kind": "string"},
                    "kind": "sequence",
                },
            },
            "restore": "clear_then_lazy_rebuild",
        },
    ]


def _task_schema(task_id: str, domain: str, task_name: str) -> dict[str, Any]:
    environment = _load(domain, task_name, seed=0)
    physics = environment.physics
    model = physics.model
    compiled_model_sha256 = hashlib.sha256(model.to_bytes()).hexdigest()
    model_before_reset = {
        name: np.array(value, copy=True)
        for name in dir(model)
        if not name.startswith("_")
        and isinstance((value := getattr(model, name, None)), np.ndarray)
    }
    environment.reset()
    observed_reset_mutations = sorted(
        name
        for name, before in model_before_reset.items()
        if not np.array_equal(before, getattr(model, name), equal_nan=True)
    )
    assert set(observed_reset_mutations) <= set(MODEL_ARRAY_FIELDS.get(task_id, {}))
    actual_task_fields = (
        set(environment.task.__dict__)
        - {
            "_random",
            "_visualize_reward",
        }
        - RESET_ONLY_TASK_FIELDS.get(task_id, set())
    )
    assert actual_task_fields == set(EXPECTED_TASK_FIELDS[task_id])
    task_fields = []
    for name, role in sorted(EXPECTED_TASK_FIELDS[task_id].items()):
        field_role = "static_compatibility" if role == "static" else "mutable_state"
        task_fields.append(
            {"name": name, **_scalar_field(getattr(environment.task, name), field_role)}
        )
    task_fields.append(
        {
            "name": "_visualize_reward",
            **_scalar_field(environment.task._visualize_reward, "static_compatibility"),
        }
    )
    model_arrays = [
        {
            "dtype": np.asarray(getattr(model, name)).dtype.str,
            "name": name,
            "restore": "np.copyto_complete_array",
            "shape": list(np.asarray(getattr(model, name)).shape),
            "source_mutation": source_mutation,
        }
        for name, source_mutation in sorted(MODEL_ARRAY_FIELDS.get(task_id, {}).items())
    ]
    component_sizes = _integration_component_sizes(model)
    integration_size = mujoco.mj_stateSize(model.ptr, INTEGRATION_SPEC)
    assert sum(component_sizes.values()) == integration_size
    sensor_noise = np.asarray(model.sensor_noise)
    sensor_enabled = bool(int(model.opt.enableflags) & SENSOR_NOISE_BIT)
    assert not sensor_enabled
    assert np.count_nonzero(sensor_noise) == 0
    if task_id.startswith("quadruped_"):
        assert set(physics.__dict__) >= {"_sensor_types_to_names", "_hinge_names"}
    return {
        "action_spec": _spec_schema(environment.action_spec()),
        "canonical_id": task_id,
        "compiled_model_sha256": compiled_model_sha256,
        "derived_caches": _derived_cache_fields(task_id),
        "domain": domain,
        "environment_fields": {
            "_flat_observation": _scalar_field(
                environment._flat_observation, "static_compatibility"
            ),
            "_n_sub_steps": _scalar_field(
                environment._n_sub_steps, "static_compatibility"
            ),
            "_reset_next_step": {
                "role": "mutable_state",
                "runtime": {"kind": "scalar", "module": "builtins", "name": "bool"},
                "serialized": _serialized_scalar_schema("bool"),
            },
            "_step_count": {
                "minimum": 0,
                "maximum_from": "_step_limit",
                "role": "mutable_state",
                "runtime": {"kind": "scalar", "module": "builtins", "name": "int"},
                "serialized": _serialized_scalar_schema("int"),
            },
            "_step_limit": _scalar_field(
                environment._step_limit, "static_compatibility"
            ),
        },
        "integration_profile": f"I{integration_size}",
        "integration_state": {
            "components": component_sizes,
            "dtype": "<f8",
            "shape": [integration_size],
            "spec": INTEGRATION_SPEC,
        },
        "legacy_step": {
            "required_value": True,
            "runtime": {"kind": "scalar", "module": "builtins", "name": "bool"},
        },
        "model_arrays": model_arrays,
        "observation_spec": {
            key: _spec_schema(spec)
            for key, spec in environment.observation_spec().items()
        },
        "observed_reset_model_mutations": observed_reset_mutations,
        "sensor_noise": {
            "all_zero": True,
            "dtype": sensor_noise.dtype.str,
            "enable_bit": SENSOR_NOISE_BIT,
            "enabled": sensor_enabled,
            "restore": "validate_only",
            "shape": list(sensor_noise.shape),
        },
        "suite_task": task_name,
        "task_fields": task_fields,
        "task_rng": {
            "algorithm": {"kind": "string", "required_value": "MT19937"},
            "cached_gaussian": {
                **_serialized_scalar_schema("float"),
                "finite": True,
            },
            "has_gauss": {
                **_serialized_scalar_schema("int"),
                "allowed": [0, 1],
            },
            "keys": {"dtype": "<u4", "kind": "ndarray", "shape": [624]},
            "position": {
                **_serialized_scalar_schema("int"),
                "maximum": 624,
                "minimum": 0,
            },
        },
        "time_step": {
            "discount": {
                "kind": "nullable",
                "value": {
                    **_serialized_scalar_schema("float"),
                    "finite": True,
                },
            },
            "observation": {
                "kind": "mapping",
                "keys": {
                    key: {"dtype": spec["dtype"], "shape": spec["shape"]}
                    for key, spec in {
                        key: _spec_schema(value)
                        for key, value in environment.observation_spec().items()
                    }.items()
                },
            },
            "reward": {
                "kind": "nullable",
                "value": {
                    **_serialized_scalar_schema("float"),
                    "finite": True,
                },
            },
            "step_type": {
                "allowed": [0, 1, 2],
                "boolean_allowed_as_integer": False,
                "dtype": "|i1",
                "kind": "ndarray",
                "runtime": {
                    "kind": "enum",
                    "module": "dm_env",
                    "name": "StepType",
                },
                "shape": [],
            },
        },
    }


def _dmc_spec_schema() -> dict[str, Any]:
    public_maximum = 2**32 - 1 - 10_000
    return {
        "closed_keys": [
            "action_repeat",
            "backend",
            "base_seed",
            "camera_override",
            "canonical_task",
            "child_index",
            "child_seed",
            "domain",
            "effective_camera",
            "image_size",
            "mode",
            "profile",
            "public_seed",
            "suite_task",
            "vector_role",
        ],
        "fields": {
            "action_repeat": {"kind": "integer", "required_value": 1},
            "backend": {
                "kind": "mapping",
                "required_value": {
                    "dm_control": "1.0.17",
                    "legacy_step": True,
                    "mujoco": "3.1.3",
                    "schema_format": "world_marl.dreamer_v3.dmc_state",
                },
            },
            "base_seed": {"kind": "integer", "maximum": 2**32 - 1, "minimum": 0},
            "camera_override": {
                "kind": "nullable",
                "value": {
                    "kind": "integer",
                    "maximum_by_task": {
                        task_id: (3 if domain == "quadruped" else 1)
                        for task_id, domain, _ in TASKS
                    },
                    "minimum": 0,
                },
            },
            "canonical_task": {
                "allowed": list(TASK_IDS),
                "kind": "string",
                "mapping": {
                    task_id: {"domain": domain, "suite_task": task}
                    for task_id, domain, task in TASKS
                },
            },
            "child_index": {"kind": "integer", "maximum": 2**32 - 1, "minimum": 0},
            "child_seed": {"kind": "integer", "maximum": 2**32 - 1, "minimum": 0},
            "domain": {"kind": "string"},
            "effective_camera": {
                "default_by_task": {
                    task_id: (2 if domain == "quadruped" else 0)
                    for task_id, domain, _ in TASKS
                },
                "kind": "integer",
                "maximum_by_task": {
                    task_id: (3 if domain == "quadruped" else 1)
                    for task_id, domain, _ in TASKS
                },
                "minimum": 0,
                "rule": "camera_override_when_present_else_task_default",
            },
            "image_size": {
                "items": [
                    {"kind": "integer", "required_value": 64},
                    {"kind": "integer", "required_value": 64},
                ],
                "kind": "sequence",
                "length": 2,
            },
            "mode": {"allowed": ["proprio", "vision"], "kind": "string"},
            "profile": {
                "allowed": ["paper", "upstream-current"],
                "kind": "string",
            },
            "public_seed": {"kind": "integer", "maximum": public_maximum, "minimum": 0},
            "suite_task": {"kind": "string"},
            "vector_role": {"allowed": ["evaluation", "train"], "kind": "string"},
        },
        "seed_derivation": {
            "child_seed": "SeedSequence([uint32(base_seed),uint32(child_index)]).generate_state(1,uint32)[0]",
            "evaluation_base_seed": "public_seed_plus_10000_checked_before_uint32",
            "evaluation_offset": 10_000,
            "train_base_seed": "public_seed",
        },
    }


def _build_schema() -> dict[str, Any]:
    assert importlib.metadata.version("dm-control") == "1.0.17"
    assert mujoco.__version__ == "3.1.3"
    assert int(mujoco.mjtState.mjSTATE_INTEGRATION) == INTEGRATION_SPEC
    assert len(TASK_IDS) == len(set(TASK_IDS)) == 20
    tasks = {
        task_id: _task_schema(task_id, domain, task) for task_id, domain, task in TASKS
    }
    profiles: dict[str, dict[str, Any]] = {}
    for row in tasks.values():
        name = row["integration_profile"]
        candidate = {
            "components": [
                {
                    "flag": flag,
                    "key": key,
                    "mujoco_enum": enum_name,
                    "size": row["integration_state"]["components"][key],
                }
                for key, enum_name, flag in INTEGRATION_COMPONENTS
            ],
            "dtype": "<f8",
            "shape": row["integration_state"]["shape"],
        }
        if name in profiles:
            assert profiles[name] == candidate
        else:
            profiles[name] = candidate
    return {
        "backend": {
            "dm_control_distribution": "dm-control",
            "dm_control_version": "1.0.17",
            "integration_spec_name": "mjSTATE_INTEGRATION",
            "integration_spec_value": INTEGRATION_SPEC,
            "legacy_step": True,
            "mujoco_version": "3.1.3",
        },
        "canonical_task_order": list(TASK_IDS),
        "dmc_spec_schema": _dmc_spec_schema(),
        "format": "world_marl.dreamer_v3.dmc_state_schema",
        "format_version": 3,
        "integration_components": [
            {"flag": flag, "key": key, "mujoco_enum": enum_name}
            for key, enum_name, flag in INTEGRATION_COMPONENTS
        ],
        "integration_profiles": dict(sorted(profiles.items())),
        "restore_order": list(RESTORE_ORDER),
        "state_record_schema": {
            "closed_keys": [
                "compatibility",
                "dmc_spec",
                "format",
                "format_version",
                "mutable",
            ],
            "format": "world_marl.dreamer_v3.dmc_state",
            "format_version": 1,
            "runtime_vs_serialized": {
                "configuration": "strict_json_scalars_with_bool_excluded_from_integer",
                "mutable_scalars": "zero_dimensional_numpy_arrays_with_exact_dtype",
                "mutable_trees": "closed_mappings_and_exact_numpy_arrays",
            },
        },
        "tasks": tasks,
    }


def _strict_equal(actual: Any, expected: Any, path: str = "root") -> None:
    assert type(actual) is type(expected), (path, type(actual), type(expected))
    if isinstance(expected, dict):
        assert set(actual) == set(expected), (path, set(actual), set(expected))
        for key in expected:
            _strict_equal(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        assert len(actual) == len(expected), (path, len(actual), len(expected))
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            _strict_equal(left, right, f"{path}[{index}]")
    else:
        assert actual == expected, (path, actual, expected)


def _expect_array(
    value: Any,
    dtype: str,
    shape: tuple[int, ...],
    *,
    finite: bool = False,
) -> np.ndarray:
    assert type(value) is np.ndarray
    assert value.dtype.str == dtype, (value.dtype.str, dtype)
    assert value.shape == shape, (value.shape, shape)
    if finite:
        assert np.all(np.isfinite(value))
    return value


def _child_seed(base_seed: int, child_index: int) -> int:
    return int(
        np.random.SeedSequence(
            [np.uint32(base_seed), np.uint32(child_index)]
        ).generate_state(1, dtype=np.uint32)[0]
    )


def _make_dmc_spec(
    task_id: str,
    public_seed: int = 7,
    *,
    profile: str = "paper",
    mode: str = "proprio",
    vector_role: str = "train",
    child_index: int = 0,
    camera_override: int | None = None,
) -> dict[str, Any]:
    domain, suite_task = TASK_MAP[task_id]
    base_seed = public_seed if vector_role == "train" else public_seed + 10_000
    default_camera = 2 if domain == "quadruped" else 0
    return {
        "action_repeat": 1,
        "backend": {
            "dm_control": "1.0.17",
            "legacy_step": True,
            "mujoco": "3.1.3",
            "schema_format": "world_marl.dreamer_v3.dmc_state",
        },
        "base_seed": base_seed,
        "camera_override": camera_override,
        "canonical_task": task_id,
        "child_index": child_index,
        "child_seed": _child_seed(base_seed, child_index),
        "domain": domain,
        "effective_camera": (
            camera_override if camera_override is not None else default_camera
        ),
        "image_size": [64, 64],
        "mode": mode,
        "profile": profile,
        "public_seed": public_seed,
        "suite_task": suite_task,
        "vector_role": vector_role,
    }


def _validate_integer(value: Any, minimum: int, maximum: int | None = None) -> None:
    assert type(value) is int
    assert value >= minimum
    if maximum is not None:
        assert value <= maximum


def _validate_dmc_spec(spec: Any, schema: Mapping[str, Any]) -> None:
    assert type(spec) is dict
    assert list(spec) == schema["closed_keys"]
    fields = schema["fields"]
    for name in (
        "action_repeat",
        "base_seed",
        "child_index",
        "child_seed",
        "effective_camera",
        "public_seed",
    ):
        rule = fields[name]
        _validate_integer(spec[name], rule.get("minimum", 0), rule.get("maximum"))
        if "required_value" in rule:
            assert spec[name] == rule["required_value"]
    assert (
        type(spec["camera_override"]) is type(None)
        or type(spec["camera_override"]) is int
    )
    for name in (
        "canonical_task",
        "domain",
        "mode",
        "profile",
        "suite_task",
        "vector_role",
    ):
        assert type(spec[name]) is str
    assert spec["canonical_task"] in fields["canonical_task"]["allowed"]
    assert spec["mode"] in fields["mode"]["allowed"]
    assert spec["profile"] in fields["profile"]["allowed"]
    assert spec["vector_role"] in fields["vector_role"]["allowed"]
    mapping = fields["canonical_task"]["mapping"][spec["canonical_task"]]
    assert spec["domain"] == mapping["domain"]
    assert spec["suite_task"] == mapping["suite_task"]
    if spec["camera_override"] is not None:
        camera_rule = fields["camera_override"]["value"]
        _validate_integer(
            spec["camera_override"],
            camera_rule["minimum"],
            camera_rule["maximum_by_task"][spec["canonical_task"]],
        )
    assert type(spec["image_size"]) is list and len(spec["image_size"]) == 2
    assert all(type(value) is int for value in spec["image_size"])
    assert spec["image_size"] == [64, 64]
    _strict_equal(spec["backend"], fields["backend"]["required_value"])
    assert spec["public_seed"] <= 2**32 - 1 - 10_000
    expected_base = (
        spec["public_seed"]
        if spec["vector_role"] == "train"
        else spec["public_seed"] + 10_000
    )
    assert spec["base_seed"] == expected_base
    assert spec["child_seed"] == _child_seed(spec["base_seed"], spec["child_index"])
    default_camera = fields["effective_camera"]["default_by_task"][
        spec["canonical_task"]
    ]
    assert (
        spec["effective_camera"]
        <= fields["effective_camera"]["maximum_by_task"][spec["canonical_task"]]
    )
    assert spec["effective_camera"] == (
        spec["camera_override"]
        if spec["camera_override"] is not None
        else default_camera
    )


def _verify_dmc_spec_matrix(
    schema: Mapping[str, Any],
) -> tuple[list[tuple[Any, ...]], list[tuple[str, str]]]:
    public_seed_max = 2**32 - 1 - 10_000
    uint32_max = 2**32 - 1
    valid_cases = (
        (
            "paper-proprio-train-min-child0-nonquad-default",
            "acrobot_swingup",
            "paper",
            "proprio",
            "train",
            0,
            0,
            None,
        ),
        (
            "upstream-vision-evaluation-max-childmax-quad-default",
            "quadruped_run",
            "upstream-current",
            "vision",
            "evaluation",
            public_seed_max,
            uint32_max,
            None,
        ),
        (
            "paper-vision-evaluation-min-child1-nonquad-override",
            "cartpole_balance",
            "paper",
            "vision",
            "evaluation",
            0,
            1,
            1,
        ),
        (
            "upstream-proprio-train-max-child1-quad-override",
            "quadruped_walk",
            "upstream-current",
            "proprio",
            "train",
            public_seed_max,
            1,
            3,
        ),
    )
    for (
        _,
        task_id,
        profile,
        mode,
        vector_role,
        public_seed,
        child_index,
        camera_override,
    ) in valid_cases:
        candidate = _make_dmc_spec(
            task_id,
            public_seed,
            profile=profile,
            mode=mode,
            vector_role=vector_role,
            child_index=child_index,
            camera_override=camera_override,
        )
        _validate_dmc_spec(candidate, schema)

    invalid_cases: list[tuple[str, str, Callable[[dict[str, Any]], Any] | None]] = []

    def add(
        name: str,
        category: str,
        mutation: Callable[[dict[str, Any]], Any],
    ) -> None:
        invalid_cases.append((name, category, mutation))

    invalid_cases.append(("not_mapping", "type", None))
    add(
        "closed_key_order",
        "closed-schema",
        lambda value: dict(reversed(tuple(value.items()))),
    )
    add(
        "action_repeat_bool",
        "type",
        lambda value: value.__setitem__("action_repeat", True),
    )
    add(
        "action_repeat_range",
        "range",
        lambda value: value.__setitem__("action_repeat", 0),
    )
    add(
        "backend_identity",
        "identity",
        lambda value: value["backend"].__setitem__("mujoco", "9"),
    )
    add(
        "canonical_task_type",
        "type",
        lambda value: value.__setitem__("canonical_task", 0),
    )
    add(
        "canonical_task_unknown",
        "mapping",
        lambda value: value.__setitem__("canonical_task", "unknown"),
    )
    add("domain_type", "type", lambda value: value.__setitem__("domain", 0))
    add(
        "domain_mapping_mismatch",
        "mapping",
        lambda value: value.__setitem__("domain", "cartpole"),
    )
    add(
        "suite_task_type",
        "type",
        lambda value: value.__setitem__("suite_task", 0),
    )
    add(
        "suite_task_mapping_mismatch",
        "mapping",
        lambda value: value.__setitem__("suite_task", "balance"),
    )
    add("profile_type", "type", lambda value: value.__setitem__("profile", 0))
    add(
        "profile_unknown",
        "enum",
        lambda value: value.__setitem__("profile", "unknown"),
    )
    add("mode_type", "type", lambda value: value.__setitem__("mode", 0))
    add(
        "mode_unknown",
        "enum",
        lambda value: value.__setitem__("mode", "pixels"),
    )
    add(
        "vector_role_type",
        "type",
        lambda value: value.__setitem__("vector_role", 0),
    )
    add(
        "vector_role_unknown",
        "role",
        lambda value: value.__setitem__("vector_role", "report"),
    )
    add(
        "public_seed_bool",
        "type",
        lambda value: value.__setitem__("public_seed", True),
    )
    add(
        "public_seed_negative",
        "range",
        lambda value: value.__setitem__("public_seed", -1),
    )
    add(
        "public_seed_overflow",
        "range",
        lambda value: value.__setitem__("public_seed", public_seed_max + 1),
    )
    add(
        "base_seed_bool",
        "type",
        lambda value: value.__setitem__("base_seed", True),
    )
    add(
        "base_seed_overflow",
        "range",
        lambda value: value.__setitem__("base_seed", uint32_max + 1),
    )
    add(
        "train_base_seed_mismatch",
        "derived",
        lambda value: value.__setitem__("base_seed", value["base_seed"] + 1),
    )

    def evaluation_offset_mismatch(value: dict[str, Any]) -> None:
        value["vector_role"] = "evaluation"
        value["base_seed"] = value["public_seed"]
        value["child_seed"] = _child_seed(value["base_seed"], value["child_index"])

    add("evaluation_offset_mismatch", "derived", evaluation_offset_mismatch)
    add(
        "child_index_bool",
        "type",
        lambda value: value.__setitem__("child_index", True),
    )
    add(
        "child_index_negative",
        "range",
        lambda value: value.__setitem__("child_index", -1),
    )
    add(
        "child_index_overflow",
        "range",
        lambda value: value.__setitem__("child_index", uint32_max + 1),
    )
    add(
        "child_seed_bool",
        "type",
        lambda value: value.__setitem__("child_seed", True),
    )
    add(
        "child_seed_mismatch",
        "derived",
        lambda value: value.__setitem__("child_seed", value["child_seed"] ^ 1),
    )
    add(
        "camera_override_bool",
        "type",
        lambda value: value.__setitem__("camera_override", True),
    )
    add(
        "camera_override_negative",
        "range",
        lambda value: value.__setitem__("camera_override", -1),
    )

    def camera_override_out_of_range(value: dict[str, Any]) -> None:
        value["camera_override"] = 2
        value["effective_camera"] = 2

    add("camera_override_out_of_range", "range", camera_override_out_of_range)
    add(
        "effective_camera_bool",
        "type",
        lambda value: value.__setitem__("effective_camera", True),
    )
    add(
        "effective_camera_out_of_range",
        "range",
        lambda value: value.__setitem__("effective_camera", 2),
    )
    add(
        "effective_camera_mismatch",
        "derived",
        lambda value: value.__setitem__("effective_camera", 1),
    )
    add(
        "image_size_container",
        "type",
        lambda value: value.__setitem__("image_size", (64, 64)),
    )
    add(
        "image_size_element_type",
        "type",
        lambda value: value.__setitem__("image_size", [64.0, 64]),
    )
    add(
        "image_size_value",
        "identity",
        lambda value: value.__setitem__("image_size", [32, 64]),
    )

    rejected: list[tuple[str, str]] = []
    for name, category, mutation in invalid_cases:
        candidate: Any = _make_dmc_spec("acrobot_swingup")
        candidate = (
            [candidate] if mutation is None else mutation(candidate) or candidate
        )
        try:
            _validate_dmc_spec(candidate, schema)
        except (AssertionError, KeyError):
            rejected.append((name, category))
        else:
            raise AssertionError(f"invalid DMCSpec accepted: {name}")
    assert len(rejected) == len(invalid_cases)
    return list(valid_cases), rejected


def _compatibility(schema: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "action_spec": row["action_spec"],
        "backend": schema["backend"],
        "compiled_model_sha256": row["compiled_model_sha256"],
        "derived_caches": row["derived_caches"],
        "environment_static": {
            name: field
            for name, field in row["environment_fields"].items()
            if field["role"] == "static_compatibility"
        },
        "integration_profile": row["integration_profile"],
        "legacy_step": True,
        "observation_spec": row["observation_spec"],
        "sensor_noise": row["sensor_noise"],
        "task_static": [
            field
            for field in row["task_fields"]
            if field["role"] == "static_compatibility"
        ],
    }


def _scalar_array(value: Any, kind: str) -> np.ndarray:
    dtype = {"bool": np.bool_, "int": np.int64, "float": np.float64}[kind]
    return np.asarray(value, dtype=dtype)


def _static_scalar_value(field: Mapping[str, Any]) -> Any:
    encoded = field["value"]
    assert encoded["encoding"] == "c_order_lower_hex"
    assert encoded["dtype"] == field["serialized"]["dtype"]
    assert encoded["shape"] == field["serialized"]["shape"] == []
    array = np.frombuffer(
        bytes.fromhex(encoded["data_hex"]), dtype=np.dtype(encoded["dtype"])
    )
    assert array.size == 1
    return array.reshape(()).item()


def _capture_time_step(time_step: dm_env.TimeStep) -> dict[str, Any]:
    return {
        "discount": (
            None
            if time_step.discount is None
            else _scalar_array(time_step.discount, "float")
        ),
        "observation": {
            key: np.array(value, copy=True)
            for key, value in time_step.observation.items()
        },
        "reward": (
            None
            if time_step.reward is None
            else _scalar_array(time_step.reward, "float")
        ),
        "step_type": np.asarray(int(time_step.step_type), dtype=np.int8),
    }


def _capture_state(
    environment: Any,
    task_id: str,
    time_step: dm_env.TimeStep,
    schema: Mapping[str, Any],
    dmc_spec: Mapping[str, Any],
) -> dict[str, Any]:
    row = schema["tasks"][task_id]
    model = environment.physics.model
    size = row["integration_state"]["shape"][0]
    integration = np.empty(size, dtype=np.float64)
    mujoco.mj_getState(
        model.ptr,
        environment.physics.data.ptr,
        integration,
        INTEGRATION_SPEC,
    )
    rng = environment.task.random.get_state(legacy=True)
    return {
        "compatibility": _compatibility(schema, row),
        "dmc_spec": copy.deepcopy(dmc_spec),
        "format": "world_marl.dreamer_v3.dmc_state",
        "format_version": 1,
        "mutable": {
            "environment": {
                "_reset_next_step": _scalar_array(environment._reset_next_step, "bool"),
                "_step_count": _scalar_array(environment._step_count, "int"),
            },
            "integration": integration,
            "model_arrays": {
                field["name"]: np.array(getattr(model, field["name"]), copy=True)
                for field in row["model_arrays"]
            },
            "task_fields": {
                field["name"]: _scalar_array(
                    getattr(environment.task, field["name"]),
                    field["serialized"]["dtype"] == "<i8" and "int" or "float",
                )
                for field in row["task_fields"]
                if field["role"] == "mutable_state"
            },
            "task_rng": {
                "algorithm": rng[0],
                "cached_gaussian": _scalar_array(rng[4], "float"),
                "has_gauss": _scalar_array(rng[3], "int"),
                "keys": np.array(rng[1], dtype=np.uint32, copy=True),
                "position": _scalar_array(rng[2], "int"),
            },
            "time_step": _capture_time_step(time_step),
        },
    }


def _validate_state(
    state: Any,
    schema: Mapping[str, Any],
) -> tuple[str, Mapping[str, Any]]:
    assert type(state) is dict
    record = schema["state_record_schema"]
    assert list(state) == record["closed_keys"]
    assert type(state["format"]) is str and state["format"] == record["format"]
    assert type(state["format_version"]) is int
    assert state["format_version"] == record["format_version"]
    _validate_dmc_spec(state["dmc_spec"], schema["dmc_spec_schema"])
    task_id = state["dmc_spec"]["canonical_task"]
    row = schema["tasks"][task_id]
    _strict_equal(state["compatibility"], _compatibility(schema, row))
    mutable = state["mutable"]
    assert type(mutable) is dict
    assert list(mutable) == [
        "environment",
        "integration",
        "model_arrays",
        "task_fields",
        "task_rng",
        "time_step",
    ]
    environment = mutable["environment"]
    assert type(environment) is dict
    assert list(environment) == ["_reset_next_step", "_step_count"]
    reset_next_step = _expect_array(environment["_reset_next_step"], "|b1", ())
    step_count = _expect_array(environment["_step_count"], "<i8", ())
    step_limit = _static_scalar_value(row["environment_fields"]["_step_limit"])
    assert type(step_limit) is float and step_limit.is_integer()
    assert 0 <= int(step_count) <= int(step_limit)
    integration_schema = row["integration_state"]
    _expect_array(
        mutable["integration"],
        integration_schema["dtype"],
        tuple(integration_schema["shape"]),
        finite=True,
    )
    model_arrays = mutable["model_arrays"]
    assert type(model_arrays) is dict
    expected_model_names = [field["name"] for field in row["model_arrays"]]
    assert list(model_arrays) == expected_model_names
    for field in row["model_arrays"]:
        _expect_array(
            model_arrays[field["name"]],
            field["dtype"],
            tuple(field["shape"]),
            finite=True,
        )
    task_fields = mutable["task_fields"]
    assert type(task_fields) is dict
    expected_task_fields = [
        field for field in row["task_fields"] if field["role"] == "mutable_state"
    ]
    assert list(task_fields) == [field["name"] for field in expected_task_fields]
    for field in expected_task_fields:
        _expect_array(
            task_fields[field["name"]],
            field["serialized"]["dtype"],
            (),
            finite=True,
        )
    rng = mutable["task_rng"]
    assert type(rng) is dict
    assert list(rng) == [
        "algorithm",
        "cached_gaussian",
        "has_gauss",
        "keys",
        "position",
    ]
    assert type(rng["algorithm"]) is str and rng["algorithm"] == "MT19937"
    _expect_array(rng["keys"], "<u4", (624,))
    position = _expect_array(rng["position"], "<i8", ())
    assert 0 <= int(position) <= 624
    has_gauss = _expect_array(rng["has_gauss"], "<i8", ())
    assert int(has_gauss) in (0, 1)
    _expect_array(rng["cached_gaussian"], "<f8", (), finite=True)
    time_step = mutable["time_step"]
    assert type(time_step) is dict
    assert list(time_step) == ["discount", "observation", "reward", "step_type"]
    step_type = _expect_array(time_step["step_type"], "|i1", ())
    assert int(step_type) in (0, 1, 2)
    for name in ("reward", "discount"):
        if time_step[name] is not None:
            _expect_array(time_step[name], "<f8", (), finite=True)
    observation = time_step["observation"]
    assert type(observation) is dict
    assert set(observation) == set(row["observation_spec"])
    for name, spec in row["observation_spec"].items():
        _expect_array(
            observation[name],
            spec["dtype"],
            tuple(spec["shape"]),
            finite=True,
        )
    count = int(step_count)
    pending = bool(reset_next_step)
    kind = int(step_type)
    reward = time_step["reward"]
    discount = time_step["discount"]
    if kind == int(dm_env.StepType.FIRST):
        assert count == 0
        assert pending is False
        assert reward is None
        assert discount is None
    elif kind == int(dm_env.StepType.MID):
        assert 1 <= count < int(step_limit)
        assert pending is False
        assert reward is not None
        assert discount is not None and float(discount) == 1.0
    else:
        assert count == int(step_limit)
        assert pending is True
        assert reward is not None
        assert discount is not None and float(discount) == 1.0
    return task_id, row


def _validate_constructed(
    environment: Any,
    row: Mapping[str, Any],
    compiled_model_sha256: str,
) -> None:
    assert compiled_model_sha256 == row["compiled_model_sha256"]
    assert type(environment.physics.legacy_step) is bool
    assert environment.physics.legacy_step is True
    assert _spec_schema(environment.action_spec()) == row["action_spec"]
    assert {
        key: _spec_schema(spec) for key, spec in environment.observation_spec().items()
    } == row["observation_spec"]
    for name, field in row["environment_fields"].items():
        if field["role"] == "static_compatibility":
            actual = _scalar_field(getattr(environment, name), "static_compatibility")
            _strict_equal(actual, field)
    actual_task_fields = (
        set(environment.task.__dict__)
        - {
            "_random",
            "_visualize_reward",
        }
        - RESET_ONLY_TASK_FIELDS.get(row["canonical_id"], set())
    )
    assert actual_task_fields == set(EXPECTED_TASK_FIELDS[row["canonical_id"]])
    for field in row["task_fields"]:
        if field["role"] == "static_compatibility":
            actual = {
                "name": field["name"],
                **_scalar_field(
                    getattr(environment.task, field["name"]),
                    "static_compatibility",
                ),
            }
            _strict_equal(actual, field)
    sensor_noise = np.asarray(environment.physics.model.sensor_noise)
    assert sensor_noise.dtype.str == row["sensor_noise"]["dtype"]
    assert list(sensor_noise.shape) == row["sensor_noise"]["shape"]
    assert np.count_nonzero(sensor_noise) == 0
    assert not (int(environment.physics.model.opt.enableflags) & SENSOR_NOISE_BIT)


def _restore_state(
    state: Mapping[str, Any],
    schema: Mapping[str, Any],
    *,
    loader: Callable[[str, str, int], Any] = _load,
) -> tuple[Any, dm_env.TimeStep, tuple[str, ...]]:
    task_id, row = _validate_state(state, schema)
    spec = state["dmc_spec"]
    trace = [RESTORE_ORDER[0]]
    environment = loader(spec["domain"], spec["suite_task"], spec["child_seed"])
    compiled_model_sha256 = hashlib.sha256(
        environment.physics.model.to_bytes()
    ).hexdigest()
    environment.reset()
    _validate_constructed(environment, row, compiled_model_sha256)
    trace.append(RESTORE_ORDER[1])
    mutable = state["mutable"]
    model = environment.physics.model
    data = environment.physics.data
    for field in row["model_arrays"]:
        np.copyto(
            getattr(model, field["name"]),
            mutable["model_arrays"][field["name"]],
        )
    trace.append(RESTORE_ORDER[2])
    mujoco.mj_setState(model.ptr, data.ptr, mutable["integration"], INTEGRATION_SPEC)
    trace.append(RESTORE_ORDER[3])
    mujoco.mj_step1(model.ptr, data.ptr)
    trace.append(RESTORE_ORDER[4])
    rng = mutable["task_rng"]
    environment.task.random.set_state(
        (
            rng["algorithm"],
            rng["keys"],
            int(rng["position"]),
            int(rng["has_gauss"]),
            float(rng["cached_gaussian"]),
        )
    )
    for name, value in mutable["task_fields"].items():
        setattr(environment.task, name, int(value))
    trace.append(RESTORE_ORDER[5])
    environment._step_count = int(mutable["environment"]["_step_count"])
    environment._reset_next_step = bool(mutable["environment"]["_reset_next_step"])
    time_step_state = mutable["time_step"]
    time_step = dm_env.TimeStep(
        step_type=dm_env.StepType(int(time_step_state["step_type"])),
        reward=(
            None
            if time_step_state["reward"] is None
            else float(time_step_state["reward"])
        ),
        discount=(
            None
            if time_step_state["discount"] is None
            else float(time_step_state["discount"])
        ),
        observation={
            key: np.array(value, copy=True)
            for key, value in time_step_state["observation"].items()
        },
    )
    trace.append(RESTORE_ORDER[6])
    for cache in row["derived_caches"]:
        getattr(environment.physics, cache["name"]).clear()
    trace.append(RESTORE_ORDER[7])
    assert tuple(trace) == RESTORE_ORDER
    return environment, time_step, tuple(trace)


def _action(environment: Any, sign: float) -> np.ndarray:
    spec = environment.action_spec()
    return (
        np.linspace(-0.25, 0.25, int(np.prod(spec.shape)), dtype=spec.dtype)
        .reshape(spec.shape)
        .__mul__(sign)
    )


def _assert_time_step_equal(left: dm_env.TimeStep, right: dm_env.TimeStep) -> None:
    assert left.step_type == right.step_type
    assert left.reward == right.reward
    assert left.discount == right.discount
    assert list(left.observation) == list(right.observation)
    for key in left.observation:
        np.testing.assert_array_equal(left.observation[key], right.observation[key])


def _state_digest(value: Any) -> str:
    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        digest.update(type(item).__module__.encode())
        digest.update(type(item).__name__.encode())
        if isinstance(item, dict):
            for key, child in item.items():
                visit(key)
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)
        elif isinstance(item, np.ndarray):
            digest.update(item.dtype.str.encode())
            digest.update(repr(item.shape).encode())
            digest.update(np.ascontiguousarray(item).tobytes())
        elif item is None:
            digest.update(b"none")
        else:
            digest.update(repr(item).encode())

    visit(value)
    return digest.hexdigest()


def _assert_state_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> None:
    assert _state_digest(left) == _state_digest(right)


def _step_to_last(environment: Any, current: dm_env.TimeStep) -> dm_env.TimeStep:
    assert not current.last()
    while not current.last():
        current = environment.step(_action(environment, 1.0))
    assert current.last()
    assert environment._step_count == environment._step_limit
    assert environment._reset_next_step is True
    return current


def _verify_roundtrips(
    schema: Mapping[str, Any],
) -> tuple[dict[str, dict[str, bool]], tuple[str, ...]]:
    results: dict[str, dict[str, bool]] = {}
    last_trace: tuple[str, ...] = ()
    for task_id, domain, task_name in TASKS:
        spec = _make_dmc_spec(task_id)
        source = _load(domain, task_name, spec["child_seed"])
        source.reset()
        current = source.step(_action(source, 1.0))
        state = _capture_state(source, task_id, current, schema, spec)
        candidate, candidate_current, trace = _restore_state(state, schema)
        _assert_time_step_equal(current, candidate_current)
        _assert_state_equal(
            state,
            _capture_state(
                candidate,
                task_id,
                candidate_current,
                schema,
                spec,
            ),
        )
        source_next = source.step(_action(source, -1.0))
        candidate_next = candidate.step(_action(candidate, -1.0))
        _assert_time_step_equal(source_next, candidate_next)
        _assert_state_equal(
            _capture_state(source, task_id, source_next, schema, spec),
            _capture_state(
                candidate,
                task_id,
                candidate_next,
                schema,
                spec,
            ),
        )
        last_current = _step_to_last(source, source_next)
        last_state = _capture_state(
            source,
            task_id,
            last_current,
            schema,
            spec,
        )
        last_candidate, last_candidate_current, last_trace = _restore_state(
            last_state, schema
        )
        _assert_time_step_equal(last_current, last_candidate_current)
        _assert_state_equal(
            last_state,
            _capture_state(
                last_candidate,
                task_id,
                last_candidate_current,
                schema,
                spec,
            ),
        )
        source_first = source.step(_action(source, -1.0))
        candidate_first = last_candidate.step(_action(last_candidate, -1.0))
        assert source_first.first() and candidate_first.first()
        _assert_time_step_equal(source_first, candidate_first)
        _assert_state_equal(
            _capture_state(source, task_id, source_first, schema, spec),
            _capture_state(
                last_candidate,
                task_id,
                candidate_first,
                schema,
                spec,
            ),
        )
        assert trace == last_trace == RESTORE_ORDER
        last_trace = trace
        results[task_id] = {
            "following_first": True,
            "last_episode": True,
            "mid_episode": True,
        }
    return results, last_trace


def _verify_corruptions(
    schema: Mapping[str, Any],
) -> tuple[list[str], dict[str, bool]]:
    contexts: dict[
        str, tuple[Any, str, dm_env.TimeStep, dict[str, Any], dict[str, Any]]
    ] = {}

    def context(name: str, task_id: str, phase: str) -> None:
        domain, task_name = TASK_MAP[task_id]
        spec = _make_dmc_spec(task_id)
        source = _load(domain, task_name, spec["child_seed"])
        current = source.reset()
        if phase == "mid":
            current = source.step(_action(source, 1.0))
            assert current.mid()
        elif phase == "last":
            current = _step_to_last(source, current)
        else:
            assert phase == "first" and current.first()
        original = _capture_state(source, task_id, current, schema, spec)
        contexts[name] = (source, task_id, current, spec, original)

    context("finger_first", "finger_spin", "first")
    context("finger_mid", "finger_spin", "mid")
    context("finger_last", "finger_spin", "last")
    context("turn_first", "finger_turn_easy", "first")
    context("quadruped_first", "quadruped_run", "first")

    cases: list[tuple[str, str, Callable[[dict[str, Any]], None]]] = []

    def add(
        name: str,
        mutation: Callable[[dict[str, Any]], None],
        base: str = "finger_first",
    ) -> None:
        cases.append((name, base, mutation))

    add("closed_top_level", lambda value: value.__setitem__("unknown", None))
    add("state_format", lambda value: value.__setitem__("format", "wrong"))
    add(
        "dmc_spec_mode",
        lambda value: value["dmc_spec"].__setitem__("mode", "pixels"),
    )
    add(
        "dmc_spec_image_dimension_float",
        lambda value: value["dmc_spec"].__setitem__("image_size", [64.0, 64]),
    )
    add(
        "dmc_spec_image_container",
        lambda value: value["dmc_spec"].__setitem__("image_size", (64, 64)),
    )
    add(
        "dmc_spec_camera_bool",
        lambda value: value["dmc_spec"].__setitem__("camera_override", True),
    )

    def camera_out_of_range(value: dict[str, Any]) -> None:
        value["dmc_spec"]["camera_override"] = 2
        value["dmc_spec"]["effective_camera"] = 2

    add("dmc_spec_camera_out_of_range", camera_out_of_range)
    add(
        "backend_identity",
        lambda value: value["compatibility"]["backend"].__setitem__(
            "mujoco_version", "9"
        ),
    )
    add(
        "compiled_model_identity",
        lambda value: value["compatibility"].__setitem__(
            "compiled_model_sha256", "0" * 64
        ),
    )
    add(
        "legacy_step",
        lambda value: value["compatibility"].__setitem__("legacy_step", False),
    )
    add(
        "environment_static",
        lambda value: value["compatibility"]["environment_static"]["_n_sub_steps"][
            "value"
        ].__setitem__("data_hex", "00"),
    )

    def mutate_task_static(value: dict[str, Any]) -> None:
        field = next(
            item
            for item in value["compatibility"]["task_static"]
            if item["name"] == "_target_radius"
        )
        field["value"]["data_hex"] = "00"

    add("task_static_finger_turn", mutate_task_static, "turn_first")
    add(
        "action_spec",
        lambda value: value["compatibility"]["action_spec"].__setitem__("dtype", "<f4"),
    )
    add(
        "observation_spec",
        lambda value: next(
            iter(value["compatibility"]["observation_spec"].values())
        ).__setitem__("shape", [999]),
    )
    add(
        "sensor_gate",
        lambda value: value["compatibility"]["sensor_noise"].__setitem__(
            "enabled", True
        ),
    )
    add(
        "derived_cache_schema_quadruped",
        lambda value: value["compatibility"].__setitem__("derived_caches", [None]),
        "quadruped_first",
    )
    add(
        "environment_counter_dtype",
        lambda value: value["mutable"]["environment"].__setitem__(
            "_step_count", np.asarray(True, dtype=np.bool_)
        ),
    )
    add(
        "environment_reset_flag_dtype",
        lambda value: value["mutable"]["environment"].__setitem__(
            "_reset_next_step", np.asarray(0, dtype=np.int64)
        ),
    )
    add(
        "step_count_negative",
        lambda value: value["mutable"]["environment"].__setitem__(
            "_step_count", np.asarray(-1, dtype=np.int64)
        ),
    )
    add(
        "step_count_above_limit",
        lambda value: value["mutable"]["environment"].__setitem__(
            "_step_count", np.asarray(1001, dtype=np.int64)
        ),
        "finger_last",
    )
    add(
        "first_reward_nonnull",
        lambda value: value["mutable"]["time_step"].__setitem__(
            "reward", np.asarray(1.0, dtype=np.float64)
        ),
    )
    add(
        "first_discount_nonnull",
        lambda value: value["mutable"]["time_step"].__setitem__(
            "discount", np.asarray(7.0, dtype=np.float64)
        ),
    )
    add(
        "first_step_count_nonzero",
        lambda value: value["mutable"]["environment"].__setitem__(
            "_step_count", np.asarray(1, dtype=np.int64)
        ),
    )
    add(
        "first_reset_pending",
        lambda value: value["mutable"]["environment"].__setitem__(
            "_reset_next_step", np.asarray(True, dtype=np.bool_)
        ),
    )
    add(
        "mid_reward_null",
        lambda value: value["mutable"]["time_step"].__setitem__("reward", None),
        "finger_mid",
    )
    add(
        "mid_discount_not_one",
        lambda value: value["mutable"]["time_step"].__setitem__(
            "discount", np.asarray(0.5, dtype=np.float64)
        ),
        "finger_mid",
    )
    add(
        "mid_step_count_zero",
        lambda value: value["mutable"]["environment"].__setitem__(
            "_step_count", np.asarray(0, dtype=np.int64)
        ),
        "finger_mid",
    )
    add(
        "mid_step_count_at_limit",
        lambda value: value["mutable"]["environment"].__setitem__(
            "_step_count", np.asarray(1000, dtype=np.int64)
        ),
        "finger_mid",
    )
    add(
        "mid_reset_pending",
        lambda value: value["mutable"]["environment"].__setitem__(
            "_reset_next_step", np.asarray(True, dtype=np.bool_)
        ),
        "finger_mid",
    )
    add(
        "last_reward_null",
        lambda value: value["mutable"]["time_step"].__setitem__("reward", None),
        "finger_last",
    )
    add(
        "last_discount_null",
        lambda value: value["mutable"]["time_step"].__setitem__("discount", None),
        "finger_last",
    )
    add(
        "last_step_count_below_limit",
        lambda value: value["mutable"]["environment"].__setitem__(
            "_step_count", np.asarray(999, dtype=np.int64)
        ),
        "finger_last",
    )
    add(
        "last_reset_not_pending",
        lambda value: value["mutable"]["environment"].__setitem__(
            "_reset_next_step", np.asarray(False, dtype=np.bool_)
        ),
        "finger_last",
    )
    add(
        "integration_dtype",
        lambda value: value["mutable"].__setitem__(
            "integration", value["mutable"]["integration"].astype(np.float32)
        ),
    )
    add(
        "model_array_finger",
        lambda value: value["mutable"]["model_arrays"].pop("site_rgba"),
    )
    add(
        "rng_algorithm",
        lambda value: value["mutable"]["task_rng"].__setitem__("algorithm", "PCG64"),
    )

    def fractional_position(value: dict[str, Any]) -> None:
        value["mutable"]["integration"][0] += 17.0
        value["mutable"]["task_rng"]["position"] = np.asarray(1.5, dtype=np.float64)

    add("rng_fractional_position_with_changed_integration", fractional_position)
    add(
        "rng_flags",
        lambda value: value["mutable"]["task_rng"].__setitem__(
            "has_gauss", np.asarray(True, dtype=np.bool_)
        ),
    )
    add(
        "rng_keys",
        lambda value: value["mutable"]["task_rng"].__setitem__(
            "keys", np.zeros(623, dtype=np.uint32)
        ),
    )
    add(
        "time_step_enum",
        lambda value: value["mutable"]["time_step"].__setitem__(
            "step_type", np.asarray(9, dtype=np.int8)
        ),
    )

    def mutate_observation(value: dict[str, Any]) -> None:
        name = next(iter(value["mutable"]["time_step"]["observation"]))
        value["mutable"]["time_step"]["observation"][name] = np.asarray(
            [0.0], dtype=np.float64
        )

    add("time_step_observation", mutate_observation)
    assert tuple(name for name, _, _ in cases) == CORRUPTION_FAMILIES
    passed: list[str] = []
    candidate_preserved: dict[str, bool] = {}
    for name, base, mutation in cases:
        source, task_id, current, spec, original = contexts[base]
        original_digest = _state_digest(original)
        corrupt = copy.deepcopy(original)
        mutation(corrupt)
        corrupt_digest = _state_digest(corrupt)
        assert corrupt_digest != original_digest, name
        constructed = False

        def forbidden_loader(domain: str, task: str, seed: int) -> Any:
            nonlocal constructed
            constructed = True
            raise AssertionError((domain, task, seed))

        try:
            _restore_state(
                corrupt,
                schema,
                loader=forbidden_loader,
            )
        except AssertionError:
            pass
        else:
            raise AssertionError(f"corruption accepted: {name}")
        assert not constructed, name
        candidate_preserved[name] = _state_digest(corrupt) == corrupt_digest
        assert candidate_preserved[name], name
        assert _state_digest(original) == original_digest, name
        recaptured = _capture_state(source, task_id, current, schema, spec)
        _assert_state_equal(original, recaptured)
        passed.append(name)
    assert tuple(passed) == CORRUPTION_FAMILIES
    assert tuple(candidate_preserved) == CORRUPTION_FAMILIES
    return passed, candidate_preserved


def _write_schema(path: Path) -> dict[str, Any]:
    schema = _build_schema()
    payload = _canonical_bytes(schema)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
        "tasks": len(schema["tasks"]),
    }


def _verify(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    fixture_sha256 = hashlib.sha256(payload).hexdigest()
    schema = json.loads(payload)
    assert _canonical_bytes(schema) == payload
    generated = _canonical_bytes(_build_schema())
    assert generated == payload
    valid_specs, invalid_specs = _verify_dmc_spec_matrix(schema["dmc_spec_schema"])
    roundtrips, trace = _verify_roundtrips(schema)
    corruptions, candidate_preserved = _verify_corruptions(schema)
    return {
        "corruption_candidates_preserved": candidate_preserved,
        "corruption_families": corruptions,
        "dm_control": importlib.metadata.version("dm-control"),
        "dmc_spec_case_counts": {
            "invalid": len(invalid_specs),
            "valid": len(valid_specs),
        },
        "dmc_spec_invalid_cases": invalid_specs,
        "dmc_spec_valid_cases": valid_specs,
        "fixture_sha256": fixture_sha256,
        "fixture_size": len(payload),
        "mujoco": mujoco.__version__,
        "real_state_identity": {
            "camera_override": None,
            "child_index": 0,
            "mode": "proprio",
            "profile": "paper",
            "public_seed": 7,
            "task_count": len(roundtrips),
            "vector_role": "train",
        },
        "restore_order": list(trace),
        "state_record_keys": schema["state_record_schema"]["closed_keys"],
        "tasks": roundtrips,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    schema_parser = subparsers.add_parser("schema")
    schema_parser.add_argument("--output", type=Path, required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--fixture", type=Path, required=True)
    args = parser.parse_args(argv)
    result = (
        _write_schema(args.output)
        if args.command == "schema"
        else _verify(args.fixture)
    )
    print(json.dumps(result, allow_nan=False, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
