from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dreamarl.config import DreaMARLRunSpec, PUBLIC_ALGORITHMS
from dreamarl.contracts import verify_run_contract
from dreamarl.launcher import run_training
from dreamarl.main import _load_configs, _resolve_config_profiles, _validate_script
from dreamarl.replay import EliteRecentReplay, ExponentialRecency, RecentReplay
from dreamarl.runtime import algorithm_root
from dreamarl.scripts.eval_dreamarl import main as eval_main


def _spec(tmp_path: Path, **updates) -> DreaMARLRunSpec:
    values = {
        "experiment_dir": tmp_path / "local",
        "task": "dmc_reacher_easy",
        "num_agents": 1,
        "algorithm": "local",
        "seed": 7,
        "train_steps": 50_000,
        "platform": "cpu",
        "python": Path("/usr/bin/python3"),
    }
    values.update(updates)
    return DreaMARLRunSpec(**values)


@pytest.mark.parametrize(
    ("algorithm", "profiles", "stage", "rollout_steps", "anchors"),
    (
        ("local", ["smac_vector", "local"], "local", None, None),
        ("ctde-one-step", ["smac_vector", "ctde"], "ctde", 1, 0),
        (
            "ctde-two-step",
            ["smac_vector", "ctde", "ctde_two_step"],
            "ctde",
            2,
            128,
        ),
        (
            "ctde-pcr",
            ["smac_vector", "ctde", "ctde_pcr"],
            "ctde",
            1,
            0,
        ),
    ),
)
def test_public_profiles_resolve_to_complete_configs(
    tmp_path: Path,
    algorithm: str,
    profiles: list[str],
    stage: str,
    rollout_steps: int | None,
    anchors: int | None,
) -> None:
    spec = _spec(
        tmp_path,
        experiment_dir=tmp_path / algorithm,
        task="smac_3m",
        num_agents=3,
        algorithm=algorithm,
    )
    resolved = _resolve_config_profiles(_load_configs(), spec.configs)

    assert spec.configs == profiles
    assert resolved.agent.marl.stage == stage
    assert resolved.agent.loss_scales.posterior_jepa == 2.0
    if rollout_steps is None:
        assert "agent.marl.ctde.rollout_steps" not in resolved.flat
        assert spec.ctde_manifest is None
    else:
        assert resolved.agent.marl.ctde.rollout_steps == rollout_steps
        assert resolved.agent.marl.ctde.multistep.anchors == anchors
        assert resolved.agent.loss_scales.ctde_embedding == 2.0
        assert spec.ctde_manifest["rollout_steps"] == rollout_steps
        assert spec.ctde_manifest["two_step_anchors"] == anchors


def test_default_config_is_clean_single_agent_dmc() -> None:
    assert (algorithm_root() / "configs.yaml").is_file()
    configs = _load_configs()
    defaults = configs["defaults"]

    assert defaults["task"] == "dmc_reacher_easy"
    assert defaults["agent"]["num_agents"] == 1
    assert defaults["agent"]["marl"] == {
        "stage": "local",
        "execution": "strict_decentralized",
    }
    assert "typ" not in defaults["agent"]["dyn"]
    assert "typ" not in defaults["agent"]["enc"]
    for fixed_choice in (
        "objective",
        "embedding_target",
        "embedding_loss",
        "posterior_jepa",
        "dynamics_jepa",
    ):
        assert fixed_choice not in defaults["agent"]
    assert "enabled" not in defaults["agent"]["spatial_jepa"]
    assert "enabled" not in defaults["agent"]["sigreg"]
    assert set(defaults["env"]) == {"dmc", "smac"}
    assert set(defaults["agent"]["spatial_jepa"]) == {
        "mask_ratio",
        "fill_value",
    }
    assert set(configs) == {
        "defaults",
        "local",
        "ctde",
        "ctde_two_step",
        "ctde_pcr",
        "dmc_vision",
        "smac_vector",
        "debug",
    }


@pytest.mark.parametrize("algorithm", PUBLIC_ALGORITHMS)
def test_public_commands_select_profiles_without_internal_stage_overrides(
    tmp_path: Path, algorithm: str
) -> None:
    spec = _spec(
        tmp_path,
        task="smac_3m",
        num_agents=3,
        algorithm=algorithm,
    )
    start = spec.command.index("--configs") + 1
    assert spec.command[start : start + len(spec.configs)] == spec.configs
    assert "--agent.marl.stage" not in spec.command
    assert "--agent.marl.ctde.rollout_steps" not in spec.command


def test_contract_preserves_the_decentralized_execution_boundary(
    tmp_path: Path,
) -> None:
    local = verify_run_contract(_spec(tmp_path))
    ctde = verify_run_contract(
        _spec(
            tmp_path,
            task="smac_3m",
            num_agents=3,
            algorithm="ctde-two-step",
        )
    )

    for contract in (local, ctde):
        assert contract["execution"]["mode"] == "strict_decentralized"
        assert contract["execution"]["policy_peer_access"] is False
        assert contract["execution"]["runtime_communication"] is False
        assert contract["training"]["actor_objective"] == ("score_function_reinforce")
    assert local["ctde"] is None
    assert ctde["ctde"]["training_only"] is True
    assert ctde["ctde"]["rollout_steps"] == 2


def test_environment_and_ctde_boundaries_are_validated(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="singleton visual DMC"):
        _spec(tmp_path, task="dmc_reacher_easy", num_agents=2)
    with pytest.raises(ValueError, match="supports SMAC"):
        _spec(tmp_path, task="unknown_task", num_agents=5)
    with pytest.raises(ValueError, match="at least two agents"):
        _spec(tmp_path, algorithm="ctde-one-step")


def test_recorded_evaluation_protocol_uses_benchmark_defaults(tmp_path: Path) -> None:
    dmc = _spec(tmp_path).to_dict()["evaluation_protocol"]
    smac = _spec(tmp_path, task="smac_3m", num_agents=3).to_dict()[
        "evaluation_protocol"
    ]

    assert dmc == {
        "policy_mode": "deterministic",
        "interval": 0,
        "episodes": 20,
        "envs": 4,
        "seed_offset": 10_000,
    }
    assert smac == {
        "policy_mode": "deterministic",
        "interval": 0,
        "episodes": 32,
        "envs": 1,
        "seed_offset": 50_000,
    }


def test_training_cadence_is_explicit_and_recorded(tmp_path: Path) -> None:
    spec = _spec(tmp_path, train_ratio=1024.0)
    index = spec.command.index("--run.train_ratio")

    assert spec.command[index + 1] == "1024.0"
    assert spec.to_dict()["optimizer_updates_per_environment_step"] == 1.0


def test_recent_replay_keeps_the_exponential_selector_when_empty() -> None:
    replay = RecentReplay(length=4, capacity=32, recency_decay=0.9998, seed=7)
    assert isinstance(replay.sampler, ExponentialRecency)


def test_elite_recent_replay_uses_exact_three_of_sixteen_mixture() -> None:
    class Source:
        def __init__(self, value: int):
            self.value = value

        def __len__(self) -> int:
            return 1

        def sample(self, batch: int, mode: str) -> dict[str, object]:
            assert mode == "train"
            return {"source": np.full((batch, 2), self.value, np.int32)}

    replay = EliteRecentReplay(length=2, capacity=32, elite_capacity=8, seed=7)
    replay.recent = Source(0)
    replay.elite = Source(1)
    batch = replay.sample(16, "train")

    assert batch["source"].shape == (16, 2)
    assert int(batch["source"][:, 0].sum()) == 3


def test_elite_return_threshold_does_not_fall_after_poor_episodes() -> None:
    replay = EliteRecentReplay(
        length=2,
        capacity=32,
        elite_capacity=8,
        elite_min_episodes=2,
        elite_return_window=4,
        seed=7,
    )

    def add_episode(score: float) -> None:
        for index, reward in enumerate((0.0, score)):
            replay.add(
                {
                    "is_first": np.asarray(index == 0),
                    "is_last": np.asarray(index == 1),
                    "reward": np.asarray([reward, reward], np.float32),
                }
            )

    add_episode(10.0)
    add_episode(20.0)
    threshold = replay.threshold
    add_episode(5.0)

    assert replay.completed_episodes == 3
    assert replay.elite_episodes == 2
    assert replay.threshold >= threshold


def test_ctde_v1_2_manifest_records_elite_retention(tmp_path: Path) -> None:
    spec = _spec(
        tmp_path,
        task="smac_3s_vs_4z",
        num_agents=3,
        algorithm="ctde-one-step",
        replay_sampling="elite_recent",
    )

    assert spec.ctde_version == "1.2"
    assert spec.ctde_manifest["stability_replay"] == {
        "recent_sequences_per_batch": 13,
        "elite_sequences_per_batch": 3,
        "elite_fraction": 0.1875,
        "elite_capacity": 12_500,
        "selection": "complete episodes above monotonic rolling p75 team return",
        "return_window_episodes": 256,
        "bootstrap_episodes": 32,
    }
    assert spec.command[spec.command.index("--replay.size") + 1] == "50000"
    assert spec.to_dict()["elite_replay"] == spec.ctde_manifest["stability_replay"]


def test_elite_recent_replay_is_only_available_for_ctde_v1_2(tmp_path: Path) -> None:
    for algorithm in ("local", "ctde-two-step"):
        with pytest.raises(ValueError, match="CTDE v1.2"):
            _spec(
                tmp_path,
                task="smac_3m",
                num_agents=3,
                algorithm=algorithm,
                replay_sampling="elite_recent",
            )


def test_ctde_pcr_records_actor_only_reference_regularization(tmp_path: Path) -> None:
    spec = _spec(
        tmp_path,
        task="smac_3m",
        num_agents=3,
        algorithm="ctde-pcr",
        replay_sampling="recent",
    )
    resolved = _resolve_config_profiles(_load_configs(), spec.configs)
    stability = spec.ctde_manifest["actor_stability"]

    assert spec.ctde_version == "1.2-PCR"
    assert resolved.agent.actor_churn.enabled is True
    assert resolved.agent.actor_churn.beta == 0.02
    assert stability["reference_policy"] == "one_optimizer_update_delayed"
    assert stability["world_model_gradients"] is False
    assert stability["critic_gradients"] is False
    assert spec.to_dict()["policy_churn"] == stability


def test_generic_reporting_modes_are_rejected_for_marl() -> None:
    for script in ("train_eval", "parallel", "parallel_env", "parallel_replay"):
        with pytest.raises(ValueError, match="single-agent reporting"):
            _validate_script(script, 3)
    _validate_script("train", 3)
    _validate_script("parallel", 1)


def test_dry_run_records_one_nested_authoritative_contract(tmp_path: Path) -> None:
    spec = _spec(
        tmp_path,
        task="smac_3m",
        num_agents=3,
        algorithm="ctde-one-step",
    )
    assert run_training(spec, dry_run=True) == 0
    manifest = json.loads(
        (spec.experiment_dir / "launch.json").read_text(encoding="utf-8")
    )

    assert manifest["algorithm"] == "ctde-one-step"
    assert manifest["configs"] == ["smac_vector", "ctde"]
    assert manifest["contract"]["algorithm"] == "ctde-one-step"
    assert manifest["contract"]["ctde"]["training_only"] is True
    assert manifest["actor_objective"] == "score_function_reinforce"
    assert manifest["optimizer_topology"] == "separated"


@pytest.mark.parametrize("algorithm", PUBLIC_ALGORITHMS)
def test_fixed_evaluation_reconstructs_the_recorded_algorithm_and_protocol(
    tmp_path: Path, algorithm: str
) -> None:
    spec = _spec(
        tmp_path,
        experiment_dir=tmp_path / algorithm,
        task="smac_3m",
        num_agents=3,
        algorithm=algorithm,
    )
    assert run_training(spec, dry_run=True) == 0
    checkpoint = spec.logdir / "ckpt" / "checkpoint-123"
    checkpoint.mkdir(parents=True)
    (checkpoint / "done").touch()
    (checkpoint.parent / "latest").write_text(checkpoint.name, encoding="utf-8")

    assert eval_main([str(spec.experiment_dir), "--dry-run"]) == 0
    launch_file = next((spec.experiment_dir / "evaluation").glob("*.launch.json"))
    launch = json.loads(launch_file.read_text(encoding="utf-8"))
    command = launch["command"]

    assert launch["algorithm"] == algorithm
    assert launch["configs"] == spec.configs
    assert launch["episodes"] == 32
    assert launch["envs"] == 1
    assert launch["eval_seed"] == spec.seed + 50_000
    assert command[command.index("--agent.num_agents") + 1] == "3"
    assert command[command.index("--run.eval_eps") + 1] == "32"
    assert command[command.index("--seed") + 1] == str(spec.seed + 50_000)
