from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
from world_marl.dreamarl.axes import (
    broadcast_global_batch,
    broadcast_global_sequence,
    fold_agent_batch,
    fold_agent_sequence,
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
    for filename in ("agent.py", "rssm.py", "transformer_rssm.py"):
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
