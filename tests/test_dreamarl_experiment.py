from __future__ import annotations

import json

from world_marl.dreamarl.config import (
    DreaMARLConfig,
    DynamicsConfig,
    EncoderConfig,
    ImaginationConfig,
    ReplayConfig,
)
from world_marl.dreamarl.experiment import (
    DreaMARLExperimentConfig,
    run_dreamarl_experiment,
)
from world_marl.logging import RunLogger


def test_end_to_end_experiment_uses_latest_policy_and_exact_budget(tmp_path) -> None:
    model = DreaMARLConfig(
        max_agents=2,
        action_dim=5,
        encoder=EncoderConfig(
            embedding_dim=8,
            vector_hidden_dim=16,
            vector_layers=1,
        ),
        dynamics=DynamicsConfig(
            model_dim=8,
            num_layers=1,
            num_heads=2,
            mlp_ratio=2,
            context_length=4,
            stochastic_variables=2,
            stochastic_classes=2,
            cross_agent_layers=1,
            cross_agent_heads=2,
        ),
        imagination=ImaginationConfig(horizon=3),
        replay=ReplayConfig(
            capacity=64,
            sequence_length=4,
            batch_size=2,
        ),
    )
    experiment = DreaMARLExperimentConfig(
        seed=0,
        total_environment_steps=24,
        num_envs=2,
        max_episode_steps=4,
        initial_random_steps=8,
        initial_learner_updates=1,
        collect_steps=2,
        learner_updates_per_collect=1,
        evaluation_interval=8,
        evaluation_episodes=4,
        evaluation_num_envs=2,
        checkpoint_interval=12,
        run_dir=str(tmp_path),
    )
    logger = RunLogger(tmp_path)
    summary = run_dreamarl_experiment(model, experiment, logger)
    logger.close()
    assert summary["status"] == "complete"
    assert summary["environment_steps"] == 24
    assert summary["final_evaluation"]["episodes"] == 4
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["checkpoint_policy"] == "latest"
    assert manifest["checkpoint_search"] is False
    assert manifest["training_budget_excludes_evaluation"] is True
    checkpoint = tmp_path / "checkpoints" / "step_000000024"
    assert (checkpoint / "learner.msgpack").is_file()
    assert (checkpoint / "runtime.npz").is_file()
    assert (checkpoint / "replay.npz").is_file()
