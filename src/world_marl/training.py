"""Training-loop helpers shared by CLIs and tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState

from world_marl.algs.ippo import RolloutBatch, select_actions
from world_marl.envs.meltingpot_adapter import (
  MeltingPotVectorAdapter,
  flatten_agent_batch,
  unflatten_agent_actions,
)


@dataclass(frozen=True)
class RolloutResult:
  batch: RolloutBatch
  next_observations: np.ndarray
  last_values: jnp.ndarray
  metrics: dict[str, Any]


def collect_rollout(
  adapter: MeltingPotVectorAdapter,
  train_state: TrainState,
  observations: np.ndarray,
  rng: jax.Array,
  *,
  rollout_steps: int,
) -> RolloutResult:
  """Collect a rollout by stepping the Python-side Melting Pot adapter."""
  if rollout_steps < 1:
    raise ValueError("rollout_steps must be >= 1")

  obs_rows = []
  action_rows = []
  log_prob_rows = []
  reward_rows = []
  done_rows = []
  value_rows = []
  completed_returns: list[tuple[float, ...]] = []
  completed_lengths: list[int] = []
  infer_fn = jax.jit(
    lambda state, key, flat_obs: select_actions(
      state,
      key,
      flat_obs,
      deterministic=False,
    )
  )
  value_fn = jax.jit(
    lambda state, flat_obs: state.apply_fn(
      {"params": state.params},
      flat_obs,
    )[1]
  )

  current_observations = observations
  for _ in range(rollout_steps):
    flat_observations = flatten_agent_batch(current_observations)
    rng, action_rng = jax.random.split(rng)
    actions, log_probs, values = infer_fn(
      train_state,
      action_rng,
      jnp.asarray(flat_observations),
    )
    env_actions = unflatten_agent_actions(
      np.asarray(actions),
      num_envs=adapter.num_envs,
      num_agents=adapter.num_agents,
    )
    step = adapter.step(env_actions)

    obs_rows.append(flat_observations)
    action_rows.append(np.asarray(actions, dtype=np.int32))
    log_prob_rows.append(np.asarray(log_probs, dtype=np.float32))
    reward_rows.append(step.rewards.reshape((-1,)))
    done_rows.append(step.dones.reshape((-1,)))
    value_rows.append(np.asarray(values, dtype=np.float32))
    completed_returns.extend(step.completed_returns)
    completed_lengths.extend(step.completed_lengths)
    current_observations = step.observations

  last_flat_observations = flatten_agent_batch(current_observations)
  last_values = value_fn(
    train_state,
    jnp.asarray(last_flat_observations),
  )

  batch = RolloutBatch(
    observations=jnp.asarray(np.stack(obs_rows, axis=0), dtype=jnp.float32),
    actions=jnp.asarray(np.stack(action_rows, axis=0), dtype=jnp.int32),
    log_probs=jnp.asarray(np.stack(log_prob_rows, axis=0), dtype=jnp.float32),
    rewards=jnp.asarray(np.stack(reward_rows, axis=0), dtype=jnp.float32),
    dones=jnp.asarray(np.stack(done_rows, axis=0), dtype=jnp.float32),
    values=jnp.asarray(np.stack(value_rows, axis=0), dtype=jnp.float32),
  )

  completed_array = (
    np.asarray(completed_returns, dtype=np.float32)
    if completed_returns
    else np.asarray([], dtype=np.float32)
  )
  metrics = {
    "rollout_mean_reward": float(batch.rewards.mean()),
    "completed_episodes": len(completed_returns),
    "episode_return_mean": (
      float(completed_array.mean()) if completed_returns else None
    ),
    "episode_length_mean": (
      float(np.mean(completed_lengths)) if completed_lengths else None
    ),
  }
  return RolloutResult(
    batch=batch,
    next_observations=current_observations,
    last_values=last_values,
    metrics=metrics,
  )


def training_window_means(
  rows: list[dict[str, Any]],
  *,
  fraction: float = 1 / 3,
) -> tuple[float, float]:
  """Return early/final means from available training metrics."""
  if not rows:
    return 0.0, 0.0
  values = [
    row["episode_return_mean"]
    if row.get("episode_return_mean") is not None
    else row["rollout_mean_reward"]
    for row in rows
  ]
  window = max(1, int(len(values) * fraction))
  return float(np.mean(values[:window])), float(np.mean(values[-window:]))
