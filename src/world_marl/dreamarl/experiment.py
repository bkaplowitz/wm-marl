"""Resolved, publication-facing DreaMARL experiment protocol."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import subprocess
import time
from typing import Any

import jax
import numpy as np

from world_marl.dreamarl.checkpointing import (
    load_dreamarl_checkpoint,
    save_dreamarl_checkpoint,
)
from world_marl.dreamarl.config import DreaMARLConfig
from world_marl.dreamarl.contracts import (
    sequence_batch_to_jax,
    sequence_batch_to_numpy,
    stack_sequence_batches,
)
from world_marl.dreamarl.environments import CoinGameAdapter
from world_marl.dreamarl.evaluation import evaluate_latest_policy
from world_marl.dreamarl.learner import DreaMARLLearner
from world_marl.dreamarl.replay import JointSequenceReplay
from world_marl.dreamarl.runtime import DreaMARLRuntime
from world_marl.logging import RunLogger, dependency_versions


@dataclass(frozen=True, slots=True)
class DreaMARLExperimentConfig:
    """Environment-step accounting and update schedule, independent of task."""

    seed: int = 0
    total_environment_steps: int = 100_000
    num_envs: int = 64
    max_episode_steps: int = 64
    initial_random_steps: int = 4_096
    initial_learner_updates: int = 64
    collect_steps: int = 16
    learner_updates_per_collect: int = 16
    evaluation_interval: int = 10_000
    evaluation_episodes: int = 128
    evaluation_num_envs: int = 64
    checkpoint_interval: int = 50_000
    run_dir: str = "runs/dreamarl_coin_game_seed0"
    resume_from: str | None = None

    def __post_init__(self) -> None:
        positive = (
            "total_environment_steps",
            "num_envs",
            "max_episode_steps",
            "initial_random_steps",
            "initial_learner_updates",
            "collect_steps",
            "learner_updates_per_collect",
            "evaluation_interval",
            "evaluation_episodes",
            "evaluation_num_envs",
            "checkpoint_interval",
        )
        for name in positive:
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.initial_random_steps > self.total_environment_steps:
            raise ValueError("initial random steps exceed the training budget")
        for name in ("total_environment_steps", "initial_random_steps"):
            if getattr(self, name) % self.num_envs:
                raise ValueError(f"{name} must be divisible by num_envs")


def run_dreamarl_experiment(
    model_config: DreaMARLConfig,
    experiment: DreaMARLExperimentConfig,
    logger: RunLogger,
) -> dict[str, Any]:
    """Train, evaluate, and checkpoint one DreaMARL seed."""

    if model_config.max_agents != 2:
        raise ValueError("CoinGame has exactly two configured agent slots")
    adapter = CoinGameAdapter(
        max_episode_steps=experiment.max_episode_steps
    )
    runtime = DreaMARLRuntime(adapter, model_config)
    learner = DreaMARLLearner(model_config)
    replay = JointSequenceReplay(
        model_config.replay, seed=experiment.seed + 1_000
    )
    run_dir = Path(experiment.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()

    if experiment.resume_from:
        replay.load(experiment.resume_from)
        sample = sequence_batch_to_jax(replay.sample_template())
        learner_template = learner.initialize(
            sample, jax.random.PRNGKey(experiment.seed + 2_000)
        )
        driver_template = runtime.initialize_driver(
            experiment.num_envs,
            jax.random.PRNGKey(experiment.seed + 3_000),
        )
        context_template = runtime.initialize_policy_context(
            driver_template, learner_template.world_model.params
        )
        learner_state, driver, policy_context, restored = (
            load_dreamarl_checkpoint(
                experiment.resume_from,
                learner_template=learner_template,
                driver_template=driver_template,
                policy_context_template=context_template,
                replay=replay,
            )
        )
        environment_steps = int(restored["environment_steps"])
    else:
        driver = runtime.initialize_driver(
            experiment.num_envs,
            jax.random.PRNGKey(experiment.seed + 3_000),
        )
        random_output = runtime.collect_random(
            driver,
            experiment.initial_random_steps // experiment.num_envs,
        )
        driver = random_output.driver
        replay.append(
            sequence_batch_to_numpy(
                random_output.transitions, adapter.spec.agent_ids
            )
        )
        sample = sequence_batch_to_jax(replay.sample_template())
        learner_state = learner.initialize(
            sample, jax.random.PRNGKey(experiment.seed + 2_000)
        )
        initial_batches = _sample_update_stack(
            replay, experiment.initial_learner_updates
        )
        learner_output = learner.train_steps(
            learner_state, initial_batches
        )
        learner_state = learner_output.state
        jax.block_until_ready(learner_state.rng)
        policy_context = runtime.initialize_policy_context(
            driver, learner_state.world_model.params
        )
        environment_steps = experiment.initial_random_steps

    parameter_counts = learner.parameter_counts(learner_state)
    manifest = {
        "algorithm": "DreaMARL",
        "environment": asdict(adapter.spec),
        "model": model_config.to_dict(),
        "experiment": asdict(experiment),
        "parameters": parameter_counts,
        "dependencies": dependency_versions(),
        "git_commit": _git_commit(),
        "checkpoint_policy": "latest",
        "checkpoint_search": False,
        "training_budget_excludes_evaluation": True,
    }
    logger.write_json("manifest.json", manifest)
    logger.update_config(manifest)
    logger.set_train_env_steps(environment_steps)

    cumulative_evaluation_steps = 0
    next_evaluation = _next_boundary(
        environment_steps, experiment.evaluation_interval
    )
    next_checkpoint = _next_boundary(
        environment_steps, experiment.checkpoint_interval
    )
    latest_evaluation: dict[str, Any] | None = None

    while environment_steps < experiment.total_environment_steps:
        remaining_vector_steps = (
            experiment.total_environment_steps - environment_steps
        ) // experiment.num_envs
        collection_steps = min(
            experiment.collect_steps, remaining_vector_steps
        )
        collection_start = time.perf_counter()
        collection = runtime.collect_policy(
            driver,
            policy_context,
            learner_state.world_model.params,
            learner_state.actor.params,
            collection_steps,
            False,
        )
        jax.block_until_ready(collection.driver.key)
        collection_seconds = time.perf_counter() - collection_start
        driver = collection.driver
        policy_context = collection.context
        replay.append(
            sequence_batch_to_numpy(
                collection.transitions, adapter.spec.agent_ids
            )
        )
        added_steps = collection_steps * experiment.num_envs
        environment_steps += added_steps

        updates = max(
            1,
            round(
                experiment.learner_updates_per_collect
                * collection_steps
                / experiment.collect_steps
            ),
        )
        batches = _sample_update_stack(replay, updates)
        learner_start = time.perf_counter()
        learner_output = learner.train_steps(learner_state, batches)
        learner_state = learner_output.state
        jax.block_until_ready(learner_state.rng)
        learner_seconds = time.perf_counter() - learner_start

        completed = np.asarray(collection.metrics.completed)
        completed_returns = np.asarray(
            collection.metrics.episode_returns
        )[completed]
        row = {
            "budget": {
                "train_env_steps": environment_steps,
                "evaluation_env_steps": cumulative_evaluation_steps,
                "replay_size": replay.size,
                "imagined_steps": int(learner_state.actor_updates)
                * model_config.imagination.horizon
                * model_config.replay.batch_size,
            },
            "train": {
                "episode_return_per_agent": (
                    float(np.mean(completed_returns))
                    if completed_returns.size
                    else None
                ),
                "completed_episodes": int(completed_returns.shape[0]),
            },
            "throughput": {
                "collection_env_steps_per_second": added_steps
                / max(collection_seconds, 1e-9),
                "learner_updates_per_second": updates
                / max(learner_seconds, 1e-9),
            },
            **_mean_metrics(learner_output.metrics),
        }
        logger.set_train_env_steps(environment_steps)
        logger.append_metrics(row)

        if environment_steps >= next_evaluation:
            evaluation = evaluate_latest_policy(
                runtime,
                learner_state,
                episodes=experiment.evaluation_episodes,
                num_envs=experiment.evaluation_num_envs,
                seed=experiment.seed + 20_000 + environment_steps,
            )
            latest_evaluation = evaluation.to_dict()
            cumulative_evaluation_steps += evaluation.environment_steps
            logger.append_metrics(
                {
                    "budget": {
                        "train_env_steps": environment_steps,
                        "evaluation_env_steps": cumulative_evaluation_steps,
                    },
                    "evaluation": latest_evaluation,
                }
            )
            next_evaluation += experiment.evaluation_interval

        if environment_steps >= next_checkpoint:
            _save_checkpoint(
                run_dir,
                environment_steps,
                learner_state,
                driver,
                policy_context,
                replay,
                manifest,
            )
            next_checkpoint += experiment.checkpoint_interval

    final_evaluation = evaluate_latest_policy(
        runtime,
        learner_state,
        episodes=experiment.evaluation_episodes,
        num_envs=experiment.evaluation_num_envs,
        seed=experiment.seed + 90_000,
    )
    cumulative_evaluation_steps += final_evaluation.environment_steps
    final_checkpoint = _save_checkpoint(
        run_dir,
        environment_steps,
        learner_state,
        driver,
        policy_context,
        replay,
        manifest,
    )
    summary = {
        "status": "complete",
        "environment_steps": environment_steps,
        "evaluation_environment_steps": cumulative_evaluation_steps,
        "runtime_seconds": time.perf_counter() - start_time,
        "parameters": parameter_counts,
        "final_evaluation": final_evaluation.to_dict(),
        "latest_periodic_evaluation": latest_evaluation,
        "final_checkpoint": str(final_checkpoint),
    }
    logger.write_json("summary.json", summary)
    logger.update_summary(summary)
    return summary


def _sample_update_stack(
    replay: JointSequenceReplay, updates: int
):
    return stack_sequence_batches(
        [
            sequence_batch_to_jax(replay.sample())
            for _ in range(updates)
        ]
    )


def _mean_metrics(metrics: dict[str, jax.Array]) -> dict[str, float]:
    return {
        name: float(np.mean(np.asarray(value)))
        for name, value in metrics.items()
    }


def _next_boundary(current: int, interval: int) -> int:
    return ((current // interval) + 1) * interval


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _save_checkpoint(
    run_dir: Path,
    environment_steps: int,
    learner_state,
    driver,
    policy_context,
    replay,
    manifest,
) -> Path:
    return save_dreamarl_checkpoint(
        run_dir / "checkpoints" / f"step_{environment_steps:09d}",
        learner_state=learner_state,
        driver=driver,
        policy_context=policy_context,
        replay=replay,
        metadata={
            "environment_steps": environment_steps,
            "manifest": manifest,
        },
    )
