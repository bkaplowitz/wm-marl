from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).parents[1]
ARCHITECTURE_PATH = ROOT / "src/world_marl/dreamer_v3_baseline/ARCHITECTURE.md"
PLAN_PATH = ROOT / "src/world_marl/dreamer_v3_baseline/PLAN.md"
FIXTURE_PATH = ROOT / "tests/fixtures/dreamer_v3/dm_control_1_0_17_state_schema.json"
WORKER_PATH = ROOT / "tests/dreamer_v3_dmc_state_worker.py"
TASK_IDS = (
    "acrobot_swingup",
    "ball_in_cup_catch",
    "cartpole_balance",
    "cartpole_balance_sparse",
    "cartpole_swingup",
    "cartpole_swingup_sparse",
    "cheetah_run",
    "finger_spin",
    "finger_turn_easy",
    "finger_turn_hard",
    "hopper_hop",
    "hopper_stand",
    "pendulum_swingup",
    "quadruped_run",
    "quadruped_walk",
    "reacher_easy",
    "reacher_hard",
    "walker_run",
    "walker_stand",
    "walker_walk",
)
RESTORE_ORDER = [
    "validate_closed_candidate",
    "construct_locked_task",
    "copy_complete_model_arrays",
    "mj_setState(INTEGRATION)",
    "mj_step1(legacy_step=True)",
    "restore_task_rng_and_mutable_task_fields",
    "restore_environment_counters_and_adapter_current_time_step",
    "clear_only_enumerated_derived_caches",
]
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

PUBLIC_SEED_MAX = 2**32 - 1 - 10_000
UINT32_MAX = 2**32 - 1
DMC_SPEC_VALID_CASES = (
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
        PUBLIC_SEED_MAX,
        UINT32_MAX,
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
        PUBLIC_SEED_MAX,
        1,
        3,
    ),
)
DMC_SPEC_INVALID_CASES = (
    ("not_mapping", "type"),
    ("closed_key_order", "closed-schema"),
    ("action_repeat_bool", "type"),
    ("action_repeat_range", "range"),
    ("backend_identity", "identity"),
    ("canonical_task_type", "type"),
    ("canonical_task_unknown", "mapping"),
    ("domain_type", "type"),
    ("domain_mapping_mismatch", "mapping"),
    ("suite_task_type", "type"),
    ("suite_task_mapping_mismatch", "mapping"),
    ("profile_type", "type"),
    ("profile_unknown", "enum"),
    ("mode_type", "type"),
    ("mode_unknown", "enum"),
    ("vector_role_type", "type"),
    ("vector_role_unknown", "role"),
    ("public_seed_bool", "type"),
    ("public_seed_negative", "range"),
    ("public_seed_overflow", "range"),
    ("base_seed_bool", "type"),
    ("base_seed_overflow", "range"),
    ("train_base_seed_mismatch", "derived"),
    ("evaluation_offset_mismatch", "derived"),
    ("child_index_bool", "type"),
    ("child_index_negative", "range"),
    ("child_index_overflow", "range"),
    ("child_seed_bool", "type"),
    ("child_seed_mismatch", "derived"),
    ("camera_override_bool", "type"),
    ("camera_override_negative", "range"),
    ("camera_override_out_of_range", "range"),
    ("effective_camera_bool", "type"),
    ("effective_camera_out_of_range", "range"),
    ("effective_camera_mismatch", "derived"),
    ("image_size_container", "type"),
    ("image_size_element_type", "type"),
    ("image_size_value", "identity"),
)


def _run_worker(*arguments: str) -> dict[str, Any]:
    before_environment = dict(os.environ)
    before_modules = {
        name
        for name in sys.modules
        if name.startswith(("dm_control", "mujoco", "glfw"))
    }
    environment = dict(os.environ)
    environment["MUJOCO_GL"] = "off"
    environment.setdefault("UV_CACHE_DIR", "/tmp/wm-marl-uv-cache")
    result = subprocess.run(
        [sys.executable, str(WORKER_PATH), *arguments],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    after_modules = {
        name
        for name in sys.modules
        if name.startswith(("dm_control", "mujoco", "glfw"))
    }
    assert dict(os.environ) == before_environment
    assert after_modules == before_modules
    assert result.stderr == ""
    return json.loads(result.stdout)


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")


def _assert_encoded_schemas(value: Any, path: str = "root") -> None:
    if isinstance(value, dict):
        if set(value) >= {"data_hex", "dtype", "encoding", "shape"}:
            assert value["encoding"] == "c_order_lower_hex", path
            item_size = {
                "|b1": 1,
                "<f4": 4,
                "<f8": 8,
                "<i8": 8,
            }[value["dtype"]]
            elements = 1
            for dimension in value["shape"]:
                elements *= dimension
            assert len(bytes.fromhex(value["data_hex"])) == elements * item_size, path
        if set(value) >= {"serialized", "value"} and isinstance(value["value"], dict):
            encoded = value["value"]
            if set(encoded) >= {"data_hex", "dtype", "shape"}:
                assert encoded["dtype"] == value["serialized"]["dtype"], path
                assert encoded["shape"] == value["serialized"]["shape"], path
        for key, child in value.items():
            _assert_encoded_schemas(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_encoded_schemas(child, f"{path}[{index}]")


def _table(text: str, marker: str) -> list[list[str]]:
    section = text.split(marker, 1)[1]
    lines = section.splitlines()
    start = next(index for index, line in enumerate(lines) if line.startswith("| "))
    table_lines = []
    for line in lines[start:]:
        if not line.startswith("| "):
            break
        table_lines.append(line)
    rows = [
        [cell.strip().strip("`") for cell in line.strip().strip("|").split("|")]
        for line in table_lines
    ]
    assert len(rows) >= 2
    return [rows[0], *rows[2:]]


def _dtype_name(dtype: str) -> str:
    return {"<f4": "float32", "<f8": "float64", "<i8": "int64", "|b1": "bool"}[dtype]


def _shape_text(dtype: str, shape: list[int]) -> str:
    dimensions = ",".join(str(value) for value in shape)
    return f"{_dtype_name(dtype)}[{dimensions}]"


def _static_value(field: dict[str, Any]) -> str:
    encoded = field["value"]
    data = bytes.fromhex(encoded["data_hex"])
    dtype = encoded["dtype"]
    if dtype == "|b1":
        return "true" if bool(data[0]) else "false"
    if dtype == "<i8":
        return str(int.from_bytes(data, byteorder="little", signed=True))
    if dtype == "<f8":
        return repr(struct.unpack("<d", data)[0])
    raise AssertionError(dtype)


def _field_rows(fields: list[dict[str, Any]], role: str) -> str:
    selected = [field for field in fields if field["role"] == role]
    if not selected:
        return "none"
    values = []
    for field in selected:
        if role == "static_compatibility":
            runtime = field["runtime"]["name"]
            values.append(f"{field['name']}: {runtime}={_static_value(field)}")
        else:
            values.append(
                f"{field['name']}: {_shape_text(field['serialized']['dtype'], [])}"
            )
    return "; ".join(values)


def _model_rows(fields: list[dict[str, Any]]) -> str:
    if not fields:
        return "none"
    return "; ".join(
        f"{field['name']}: {_shape_text(field['dtype'], field['shape'])} "
        f"({field['source_mutation']})"
        for field in fields
    )


def _derived_rows(fields: list[dict[str, Any]]) -> str:
    if not fields:
        return "none"
    return "; ".join(f"{field['name']}: {field['restore']}" for field in fields)


def _task_table_rows(schema: dict[str, Any]) -> list[list[str]]:
    rows = []
    for task_id in schema["canonical_task_order"]:
        task = schema["tasks"][task_id]
        sensor = task["sensor_noise"]
        rows.append(
            [
                task_id,
                f"{task['domain']}/{task['suite_task']}",
                task["integration_profile"],
                _field_rows(task["task_fields"], "static_compatibility"),
                _field_rows(task["task_fields"], "mutable_state"),
                _model_rows(task["model_arrays"]),
                _derived_rows(task["derived_caches"]),
                f"{_shape_text(sensor['dtype'], sensor['shape'])}, disabled/zero",
            ]
        )
    return rows


def _integration_table_rows(schema: dict[str, Any]) -> list[list[str]]:
    rows = []
    for name, profile in sorted(
        schema["integration_profiles"].items(), key=lambda item: int(item[0][1:])
    ):
        rows.append(
            [
                name,
                ",".join(str(component["size"]) for component in profile["components"]),
                _shape_text(profile["dtype"], profile["shape"]),
            ]
        )
    return rows


def test_parent_collection_is_renderer_neutral_and_worker_is_isolated(
    tmp_path: Path,
) -> None:
    assert "dm_control" not in globals()
    output = tmp_path / "schema.json"
    result = _run_worker("schema", "--output", str(output))
    assert result["tasks"] == 20
    assert result["size"] == output.stat().st_size
    assert result["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()


def test_child_worker_proves_real_state_and_pure_spec_matrices() -> None:
    result = _run_worker("verify", "--fixture", str(FIXTURE_PATH))
    assert result["dm_control"] == "1.0.17"
    assert result["mujoco"] == "3.1.3"
    assert tuple(result["tasks"]) == TASK_IDS
    assert all(
        value
        == {
            "following_first": True,
            "last_episode": True,
            "mid_episode": True,
        }
        for value in result["tasks"].values()
    )
    assert result["restore_order"] == RESTORE_ORDER
    assert result["state_record_keys"] == [
        "compatibility",
        "dmc_spec",
        "format",
        "format_version",
        "mutable",
    ]
    assert tuple(result["corruption_families"]) == CORRUPTION_FAMILIES
    assert result["corruption_candidates_preserved"] == {
        name: True for name in CORRUPTION_FAMILIES
    }
    assert tuple(map(tuple, result["dmc_spec_valid_cases"])) == DMC_SPEC_VALID_CASES
    assert tuple(map(tuple, result["dmc_spec_invalid_cases"])) == (
        DMC_SPEC_INVALID_CASES
    )
    assert result["dmc_spec_case_counts"] == {
        "invalid": len(DMC_SPEC_INVALID_CASES),
        "valid": len(DMC_SPEC_VALID_CASES),
    }
    assert result["real_state_identity"] == {
        "camera_override": None,
        "child_index": 0,
        "mode": "proprio",
        "profile": "paper",
        "public_seed": 7,
        "task_count": 20,
        "vector_role": "train",
    }
    assert result["fixture_size"] == FIXTURE_PATH.stat().st_size
    assert (
        result["fixture_sha256"]
        == hashlib.sha256(FIXTURE_PATH.read_bytes()).hexdigest()
    )


def test_fixture_regenerates_twice_byte_identically(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first_result = _run_worker("schema", "--output", str(first))
    second_result = _run_worker("schema", "--output", str(second))
    assert first_result == second_result
    assert first.read_bytes() == second.read_bytes() == FIXTURE_PATH.read_bytes()


def test_fixture_and_architecture_tables_are_structurally_identical() -> None:
    payload = FIXTURE_PATH.read_bytes()
    schema = json.loads(payload)
    assert _canonical_bytes(schema) == payload
    assert schema["format"] == "world_marl.dreamer_v3.dmc_state_schema"
    assert schema["format_version"] == 3
    assert schema["state_record_schema"]["closed_keys"] == [
        "compatibility",
        "dmc_spec",
        "format",
        "format_version",
        "mutable",
    ]
    assert b"fixture_sha256" not in payload
    assert schema["backend"]["legacy_step"] is True
    assert tuple(schema["canonical_task_order"]) == TASK_IDS
    assert schema["restore_order"] == RESTORE_ORDER
    assert set(schema["dmc_spec_schema"]["fields"]) == {
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
    }
    _assert_encoded_schemas(schema)
    for task in schema["tasks"].values():
        assert not any(
            field["name"] == "_timeout_progress" for field in task["task_fields"]
        )
    forbidden = ("nullable_float64_scalar", "dict[", "list[", "tuple[int", "...")
    strings = []

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                strings.append(str(key))
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)
        elif isinstance(value, str):
            strings.append(value)

    collect(schema)
    assert not [
        value for value in strings if any(token in value for token in forbidden)
    ]
    architecture = ARCHITECTURE_PATH.read_text()
    integration_table = _table(architecture, "These integration profiles")
    assert integration_table[0] == ["Profile", "Exact component lengths", "Total shape"]
    assert integration_table[1:] == _integration_table_rows(schema)
    task_table = _table(architecture, "The exact per-task state table is:")
    assert task_table[0] == [
        "Canonical task",
        "Load target",
        "Integration",
        "Static task fields",
        "Captured task fields",
        "Complete mutable MjModel arrays",
        "Derived caches",
        "Sensor gate",
    ]
    assert task_table[1:] == _task_table_rows(schema)
    fixture_hash = hashlib.sha256(payload).hexdigest()
    assert fixture_hash in architecture
    assert fixture_hash in PLAN_PATH.read_text()
