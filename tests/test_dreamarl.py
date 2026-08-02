from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import ruamel.yaml as yaml
from world_marl.dreamarl.axes import (
    broadcast_global_batch,
    broadcast_global_sequence,
    fold_agent_batch,
    fold_agent_sequence,
    restore_folded_start_order,
    select_joint_starts,
    unfold_agent_batch,
    unfold_agent_sequence,
)
from world_marl.dreamarl.config import DreaMARLRunSpec
from world_marl.dreamarl.launcher import run_training
from world_marl.dreamarl.parity import (
    ORACLE_COMMIT,
    ORACLE_HASHES,
    verify_m3_reduction_contract,
)
from world_marl.dreamarl.runtime import (
    ALGORITHM_FILES,
    algorithm_entrypoint,
    algorithm_root,
    verify_first_party_source,
)


def _spec(tmp_path: Path, **updates) -> DreaMARLRunSpec:
    values = {
        "experiment_dir": tmp_path / "run",
        "task": "dmc_reacher_easy",
        "seed": 7,
        "train_steps": 250_000,
        "platform": "cpu",
        "python": Path("/usr/bin/python3"),
        "save_every_seconds": 1_800,
        "wandb_project": "world-marl",
        "wandb_entity": "osaze-obahor",
    }
    values.update(updates)
    return DreaMARLRunSpec(**values)


def test_agent_count_changes_geometry_only(tmp_path: Path) -> None:
    one = _spec(tmp_path, num_agents=1)
    many = _spec(tmp_path, num_agents=7)
    one_command = list(one.command)
    many_command = list(many.command)
    index = one_command.index("--agent.num_agents") + 1
    assert one_command[index] == "1"
    assert many_command[index] == "7"
    many_command[index] = "1"
    assert many_command == one_command
    assert one.to_dict()["agent_count_dependent_modules"] == []
    assert many.to_dict()["algorithm_overrides"] == []


def test_single_agent_source_and_regime_match_registered_m3(tmp_path: Path) -> None:
    verification = verify_m3_reduction_contract(_spec(tmp_path))
    assert verification["verified_official_commit"] == ORACLE_COMMIT
    assert verification["verified_algorithm_hashes"] == ORACLE_HASHES
    assert verification["num_agents"] == 1
    assert verification["agent_axis_reduction"] == (
        "identity reshape when num_agents=1"
    )


def test_multi_agent_invocation_keeps_the_m3_regime(tmp_path: Path) -> None:
    verification = verify_m3_reduction_contract(_spec(tmp_path, num_agents=7))
    assert verification["num_agents"] == 7
    assert verification["agent_count_semantics"] == "tensor geometry only"
    assert verification["algorithm_overrides"] == []


def test_meltingpot_invocation_changes_environment_only(tmp_path: Path) -> None:
    spec = _spec(
        tmp_path,
        task="meltingpot_coop_mining",
        num_agents=6,
        train_steps=50_000,
    )
    verification = verify_m3_reduction_contract(spec)
    assert verification["num_agents"] == 6
    assert verification["algorithm_overrides"] == []
    assert spec.to_dict()["configs"] == [
        "meltingpot_vision",
        "jepa_transformer",
    ]


def test_local_memory_is_an_explicit_task_neutral_algorithm_arm(tmp_path: Path) -> None:
    one = _spec(tmp_path, num_agents=1, local_memory=True)
    many = _spec(tmp_path, num_agents=7, local_memory=True)
    assert one.configs[-1] == "local_memory_sidecar"
    assert many.configs == one.configs
    assert one.to_dict()["algorithm_overrides"] == ["local_memory_sidecar"]
    assert many.to_dict()["algorithm_overrides"] == ["local_memory_sidecar"]


def test_unified_memory_is_an_explicit_task_neutral_algorithm_arm(
    tmp_path: Path,
) -> None:
    one = _spec(tmp_path, num_agents=1, unified_memory=True)
    many = _spec(tmp_path, num_agents=7, unified_memory=True)
    assert one.configs[-1] == "local_memory_unified"
    assert many.configs == one.configs
    assert one.to_dict()["algorithm_overrides"] == ["local_memory_unified"]
    assert many.to_dict()["algorithm_overrides"] == ["local_memory_unified"]


def test_memory_architecture_arms_are_mutually_exclusive(tmp_path: Path) -> None:
    with np.testing.assert_raises(ValueError):
        _spec(tmp_path, local_memory=True, unified_memory=True)


def test_unified_memory_override_is_declared_in_base_schema() -> None:
    loader = yaml.YAML(typ="safe")
    configs = loader.load(
        (algorithm_root() / "configs.yaml").read_text(encoding="utf-8")
    )
    base = configs["defaults"]["agent"]["dyn"]["jepa_transformer"]
    assert base["memory_mode"] == "residual"
    assert configs["local_memory_unified"][
        "agent.dyn.jepa_transformer.memory_mode"
    ] == "unified"


def test_agent_count_never_selects_learner_computation() -> None:
    source = (algorithm_root() / "agent.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = []
    learner_methods = {
        "init_policy",
        "init_train",
        "init_report",
        "policy",
        "train",
        "report",
        "_fold_replay",
        "_unfold_replay_updates",
    }
    for function in (
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name in learner_methods
    ):
        for node in ast.walk(function):
            if not isinstance(node, (ast.If, ast.IfExp, ast.Match, ast.While)):
                continue
            test = (
                ast.unparse(node.test)
                if hasattr(node, "test")
                else ast.unparse(node.subject)
            )
            if "num_agents" in test:
                forbidden.append((function.name, test))
    assert forbidden == []


def test_first_party_entrypoint_owns_all_algorithm_files() -> None:
    source = verify_first_party_source()
    assert source["infrastructure_commit"] == ORACLE_COMMIT
    assert algorithm_entrypoint() == algorithm_root() / "main.py"
    assert all((algorithm_root() / name).is_file() for name in ALGORITHM_FILES)


def test_active_learner_does_not_import_the_frozen_oracle() -> None:
    for filename in (
        "agent.py",
        "local_memory.py",
        "rssm.py",
        "transformer_rssm.py",
    ):
        source = (algorithm_root() / filename).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = [
            ast.unparse(node)
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        assert not any("dreamarl.m3" in item or "from .m3" in item for item in imports)


def test_singleton_policy_axis_is_an_identity_reshape() -> None:
    value = np.arange(24, dtype=np.float32).reshape(3, 1, 8)
    folded = fold_agent_batch(value, 1)
    np.testing.assert_array_equal(folded, value[:, 0])
    np.testing.assert_array_equal(unfold_agent_batch(folded, 1), value)


def test_singleton_replay_axis_is_an_identity_reshape() -> None:
    value = np.arange(48, dtype=np.float32).reshape(2, 3, 1, 8)
    folded = fold_agent_sequence(value, 1)
    np.testing.assert_array_equal(folded, value[:, :, 0])
    np.testing.assert_array_equal(unfold_agent_sequence(folded, 1), value)


def test_multi_agent_axis_round_trips_without_value_changes() -> None:
    policy = np.arange(2 * 3 * 5).reshape(2, 3, 5)
    replay = np.arange(2 * 4 * 3 * 5).reshape(2, 4, 3, 5)
    np.testing.assert_array_equal(
        unfold_agent_batch(fold_agent_batch(policy, 3), 3), policy
    )
    np.testing.assert_array_equal(
        unfold_agent_sequence(fold_agent_sequence(replay, 3), 3), replay
    )


def test_joint_start_order_groups_agents_at_each_start() -> None:
    # Value encodes environment, agent, and time.
    grouped = np.zeros((2, 4, 3), np.int32)
    for environment in range(2):
        for time in range(4):
            for agent in range(3):
                grouped[environment, time, agent] = (
                    100 * environment + 10 * time + agent
                )
    folded = fold_agent_sequence(grouped, 3)
    starts = select_joint_starts(folded, 3, 2)
    np.testing.assert_array_equal(
        starts,
        np.array([20, 21, 22, 30, 31, 32, 120, 121, 122, 130, 131, 132]),
    )
    restored = restore_folded_start_order(starts, 3, 2)
    np.testing.assert_array_equal(
        restored,
        np.array([[20, 30], [21, 31], [22, 32], [120, 130], [121, 131], [122, 132]]),
    )


def test_global_fields_are_shared_without_changing_singleton_values() -> None:
    policy = np.arange(4, dtype=np.float32)
    replay = np.arange(12, dtype=np.float32).reshape(4, 3)
    np.testing.assert_array_equal(broadcast_global_batch(policy, 1), policy)
    np.testing.assert_array_equal(broadcast_global_sequence(replay, 1), replay)
    np.testing.assert_array_equal(
        broadcast_global_batch(policy, 3).reshape(4, 3),
        np.repeat(policy[:, None], 3, axis=1),
    )


def test_dry_run_records_first_party_provenance(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    assert run_training(spec, dry_run=True) == 0
    manifest = json.loads(
        (spec.experiment_dir / "launch.json").read_text(encoding="utf-8")
    )
    assert manifest["implementation"] == "first-party DreaMARL"
    assert manifest["command"] == spec.command
    assert manifest["algorithm_overrides"] == []
    assert manifest["agent_axis_native"] is True
    assert manifest["agent_count_dependent_modules"] == []


def test_retired_independent_learner_is_absent() -> None:
    package = Path(__file__).resolve().parents[1] / "src/world_marl/dreamarl"
    retired = {
        "control.py",
        "learner.py",
        "losses.py",
        "replay.py",
        "temporal.py",
        "world_model.py",
    }
    assert retired.isdisjoint(path.name for path in package.glob("*.py"))


def test_report_merges_world_model_metrics() -> None:
    source = (algorithm_root() / "agent.py").read_text(encoding="utf-8")
    assert "metrics.update(mets)" in source
    assert "mets.update(mets)" not in source


def test_locked_baseline_uses_only_uniform_replay() -> None:
    configs = yaml.YAML(typ="safe").load(
        (algorithm_root() / "configs.yaml").read_text(encoding="utf-8")
    )
    assert configs["defaults"]["replay"]["fracs"] == {
        "uniform": 1.0,
        "priority": 0.0,
        "recency": 0.0,
    }
