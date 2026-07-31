"""Fixed latest-policy evaluation for DreaMARL."""

from __future__ import annotations

from dataclasses import dataclass
import math

import jax
import numpy as np

from world_marl.dreamarl.learner import DreaMARLLearnerState
from world_marl.dreamarl.runtime import DreaMARLRuntime


@dataclass(frozen=True, slots=True)
class DreaMARLEvaluation:
    """Completed deterministic episodes and matched accounting."""

    returns: np.ndarray
    lengths: np.ndarray
    episodes: int
    environment_steps: int

    @property
    def mean_return_per_agent(self) -> float:
        return float(np.mean(self.returns))

    def to_dict(self) -> dict[str, object]:
        per_episode = np.mean(self.returns, axis=-1)
        return {
            "episodes": self.episodes,
            "environment_steps": self.environment_steps,
            "mean_return_per_agent": self.mean_return_per_agent,
            "return_std_per_episode": float(np.std(per_episode)),
            "return_min_per_episode": float(np.min(per_episode)),
            "return_max_per_episode": float(np.max(per_episode)),
            "returns_mean_by_agent": np.mean(self.returns, axis=0).tolist(),
            "returns": self.returns.tolist(),
            "lengths": self.lengths.tolist(),
        }


def evaluate_latest_policy(
    runtime: DreaMARLRuntime,
    learner_state: DreaMARLLearnerState,
    *,
    episodes: int,
    num_envs: int,
    seed: int,
) -> DreaMARLEvaluation:
    """Evaluate the latest parameters without checkpoint search or mutation."""

    if episodes < 1 or num_envs < 1:
        raise ValueError("episodes and num_envs must be positive")
    waves = math.ceil(episodes / num_envs)
    completed_returns: list[np.ndarray] = []
    completed_lengths: list[np.ndarray] = []
    for wave in range(waves):
        driver = runtime.initialize_driver(
            num_envs, jax.random.PRNGKey(seed + wave)
        )
        context = runtime.initialize_policy_context(
            driver, learner_state.world_model.params
        )
        output = runtime.collect_policy(
            driver,
            context,
            learner_state.world_model.params,
            learner_state.actor.params,
            runtime.adapter.spec.max_episode_steps,
            True,
        )
        completed = np.asarray(output.metrics.completed)
        returns = np.asarray(output.metrics.episode_returns)
        lengths = np.asarray(output.metrics.episode_lengths)
        completed_returns.append(returns[completed])
        completed_lengths.append(lengths[completed])
    all_returns = np.concatenate(completed_returns, axis=0)[:episodes]
    all_lengths = np.concatenate(completed_lengths, axis=0)[:episodes]
    if all_returns.shape[0] != episodes:
        raise RuntimeError("evaluation did not produce the requested episodes")
    return DreaMARLEvaluation(
        returns=all_returns.astype(np.float32),
        lengths=all_lengths.astype(np.int32),
        episodes=episodes,
        environment_steps=waves
        * num_envs
        * runtime.adapter.spec.max_episode_steps,
    )
