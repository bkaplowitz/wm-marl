from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import ruamel.yaml as yaml

from world_marl.dreamarl.agent import deterministic
from world_marl.dreamarl.axes import (
    broadcast_global_batch,
    broadcast_global_sequence,
    fold_agent_batch,
    fold_agent_sequence,
    unfold_agent_batch,
    unfold_agent_sequence,
)
from world_marl.dreamarl.config import DreaMARLRunSpec
from world_marl.dreamarl.contracts import verify_run_contract
from world_marl.dreamarl.launcher import run_training
from world_marl.dreamarl.runtime import (
    ALGORITHM_FILES,
    algorithm_entrypoint,
    algorithm_root,
    repository_root,
)


def _spec(tmp_path: Path, **updates) -> DreaMARLRunSpec:
    values = {
        "experiment_dir": tmp_path / "run",
        "task": "meltingpot_externality_mushrooms__dense",
        "seed": 7,
        "train_steps": 50_000,
        "num_agents": 5,
        "platform": "cpu",
        "python": Path("/usr/bin/python3"),
        "save_every_seconds": 1_800,
        "wandb_project": "world-marl",
        "wandb_entity": "osaze-obahor",
    }
    values.update(updates)
    return DreaMARLRunSpec(**values)


def test_run_spec_selects_one_joint_world_algorithm(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    assert spec.configs == ["meltingpot_vision", "joint_world"]
    assert spec.to_dict()["algorithm_overrides"] == []
    assert spec.to_dict()["training_state"].startswith("joint posterior")


def test_single_agent_is_valid_without_being_a_parity_contract(tmp_path: Path) -> None:
    contract = verify_run_contract(_spec(tmp_path, num_agents=1))
    assert contract["single_agent_status"] == (
        "valid reduction, not a numerical parity constraint"
    )
    assert contract["policy_peer_access"] is False


def test_multi_agent_contract_is_joint_and_decentralized(tmp_path: Path) -> None:
    contract = verify_run_contract(_spec(tmp_path, num_agents=7))
    assert contract["world_state_axis"].startswith("one environment state")
    assert contract["world_action_conditioning"] == "synchronous joint action"
    assert contract["critic_information"].startswith("joint latent state")
    assert contract["policy_peer_access"] is False


def test_policy_method_cannot_query_joint_world_state() -> None:
    source = (algorithm_root() / "agent.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    policy = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "policy"
    )
    text = ast.unparse(policy)
    assert "self.world" not in text
    assert "joint_model" not in text
    assert "joint_context" not in text


def test_imagination_samples_all_actions_before_joint_advance() -> None:
    source = (algorithm_root() / "agent.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imagine = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_imagine"
    )
    step = next(
        node
        for node in ast.walk(imagine)
        if isinstance(node, ast.FunctionDef) and node.name == "step"
    )
    statements = [ast.unparse(node) for node in step.body]
    action_index = next(i for i, value in enumerate(statements) if "action = sample" in value)
    advance_index = next(
        i for i, value in enumerate(statements) if "self.world.imagine_step" in value
    )
    assert action_index < advance_index


def test_joint_world_configuration_has_no_optional_adapter_flags() -> None:
    configs = yaml.YAML(typ="safe").load(
        (algorithm_root() / "configs.yaml").read_text(encoding="utf-8")
    )
    assert "joint_world" in configs
    assert configs["joint_world"]["replay.fracs"] == {
        "uniform": 0.5,
        "priority": 0.0,
        "recency": 0.5,
    }
    assert "rec" not in configs["defaults"]["agent"]["loss_scales"]
    assert "dec" not in configs["defaults"]["agent"]
    assert configs["defaults"]["agent"]["local_belief"] == {
        "units": 768,
        "layers": 2,
        "heads": 12,
        "context": 64,
        "ffup": 4,
        "act": "silu",
        "norm": "rms",
        "winit": "trunc_normal_in",
    }
    assert configs["defaults"]["agent"]["joint"]["units"] == 768
    assert configs["defaults"]["agent"]["joint"]["layers"] == 2
    assert configs["defaults"]["agent"]["joint"]["heads"] == 12
    assert configs["defaults"]["agent"]["joint"]["classes"] == 64
    shared_lr = configs["defaults"]["agent"]["opt"]["lr"]
    assert configs["defaults"]["agent"]["belief_lr"] == shared_lr
    assert configs["defaults"]["agent"]["world_lr"] == shared_lr
    assert "structured_local_memory" not in configs
    assert "shared_transition_context" not in configs
    assert "jepa_transformer" not in configs


def test_maintained_agent_is_decoder_free() -> None:
    source = (algorithm_root() / "agent.py").read_text(encoding="utf-8")
    assert "self.dec" not in source
    assert "_decoder_losses" not in source


def test_first_party_runtime_contains_only_maintained_algorithm_files() -> None:
    assert algorithm_entrypoint() == algorithm_root() / "main.py"
    assert repository_root() / "src" / "world_marl" == algorithm_root().parent
    assert all((algorithm_root() / name).is_file() for name in ALGORITHM_FILES)
    assert "joint_model.py" in ALGORITHM_FILES
    assert "local_belief.py" in ALGORITHM_FILES
    assert "perception.py" in ALGORITHM_FILES
    assert "contracts.py" in ALGORITHM_FILES
    assert "joint_context.py" not in ALGORITHM_FILES
    assert "local_memory.py" not in ALGORITHM_FILES
    assert "transformer_rssm.py" not in ALGORITHM_FILES
    assert not (algorithm_root() / "transformer_rssm.py").exists()
    assert not (algorithm_root() / "joint_context.py").exists()
    assert not (algorithm_root() / "local_memory.py").exists()
    assert not (algorithm_root() / "rssm.py").exists()


def test_non_marl_task_is_rejected(tmp_path: Path) -> None:
    spec = _spec(tmp_path, task="dmc_reacher_easy", num_agents=1)
    with np.testing.assert_raises_regex(ValueError, "multi-agent algorithm"):
        _ = spec.configs


def test_agent_axis_round_trips() -> None:
    policy = np.arange(2 * 3 * 5).reshape(2, 3, 5)
    replay = np.arange(2 * 4 * 3 * 5).reshape(2, 4, 3, 5)
    np.testing.assert_array_equal(
        unfold_agent_batch(fold_agent_batch(policy, 3), 3), policy
    )
    np.testing.assert_array_equal(
        unfold_agent_sequence(fold_agent_sequence(replay, 3), 3), replay
    )


def test_global_boundaries_broadcast_without_value_changes() -> None:
    policy = np.arange(4, dtype=np.float32)
    replay = np.arange(12, dtype=np.float32).reshape(4, 3)
    np.testing.assert_array_equal(
        broadcast_global_batch(policy, 3).reshape(4, 3),
        np.repeat(policy[:, None], 3, axis=1),
    )
    np.testing.assert_array_equal(
        unfold_agent_sequence(broadcast_global_sequence(replay, 3), 3),
        np.repeat(replay[:, :, None], 3, axis=2),
    )


def test_dry_run_records_marl_contract(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    assert run_training(spec, dry_run=True) == 0
    manifest = json.loads(
        (spec.experiment_dir / "launch.json").read_text(encoding="utf-8")
    )
    assert manifest["implementation"] == "first-party DreaMARL"
    assert manifest["policy_peer_access"] is False
    assert manifest["world_action_conditioning"] == "synchronous joint action"
    assert manifest["configs"] == ["meltingpot_vision", "joint_world"]


def test_deterministic_policy_uses_distribution_prediction() -> None:
    class Distribution:
        def pred(self):
            return np.asarray([1, 2], np.int32)

    result = deterministic({"action": Distribution()})
    np.testing.assert_array_equal(result["action"], np.asarray([1, 2], np.int32))
