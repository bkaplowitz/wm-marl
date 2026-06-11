"""Verify dependency imports, environment stepping, rollout, and PPO update."""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp

from world_marl.algs.ippo import IPPOConfig, create_train_state, ppo_update
from world_marl.envs.meltingpot_adapter import MeltingPotVectorAdapter
from world_marl.logging import dependency_versions, to_jsonable
from world_marl.training import collect_rollout


def main() -> None:
  adapter = MeltingPotVectorAdapter(
    substrate="coins",
    num_envs=1,
    max_cycles=1000,
    observation_size=22,
  )
  try:
    observations = adapter.reset()
    config = IPPOConfig(
      update_epochs=1,
      num_minibatches=1,
      learning_rate=1e-4,
    )
    key = jax.random.PRNGKey(0)
    key, init_key, rollout_key, update_key = jax.random.split(key, 4)
    train_state = create_train_state(
      init_key,
      adapter.observation_shape,
      adapter.action_dim,
      config,
    )
    rollout = collect_rollout(
      adapter,
      train_state,
      observations,
      rollout_key,
      rollout_steps=4,
    )
    update_fn = jax.jit(lambda state, batch, last_values, rng: ppo_update(
      state,
      batch,
      last_values,
      rng,
      config,
    ))
    train_state, metrics = update_fn(
      train_state,
      rollout.batch,
      rollout.last_values,
      update_key,
    )
    jax.block_until_ready(jnp.asarray(metrics["total_loss"]))
    payload = {
      "status": "ok",
      "versions": dependency_versions(),
      "substrate": adapter.substrate,
      "num_envs": adapter.num_envs,
      "num_agents": adapter.num_agents,
      "observation_shape": adapter.observation_shape,
      "raw_observation_shape": adapter.raw_observation_shape,
      "action_dim": adapter.action_dim,
      "rollout": rollout.metrics,
      "update_metrics": metrics,
    }
    print(json.dumps(to_jsonable(payload), indent=2, sort_keys=True))
  finally:
    adapter.close()


if __name__ == "__main__":
  main()
