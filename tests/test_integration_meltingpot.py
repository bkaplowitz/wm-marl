from __future__ import annotations

import pytest

import jax
import jax.numpy as jnp

from world_marl.algs.ippo import IPPOConfig, create_train_state, select_actions
from world_marl.envs.meltingpot_adapter import (
  MeltingPotVectorAdapter,
  flatten_agent_batch,
)


@pytest.mark.integration
def test_policy_forward_pass_on_real_meltingpot_observations():
  try:
    adapter = MeltingPotVectorAdapter(substrate="coins", num_envs=1, max_cycles=10)
  except Exception as exc:
    pytest.skip(f"Melting Pot runtime unavailable: {exc}")

  try:
    observations = adapter.reset()
    state = create_train_state(
      jax.random.PRNGKey(0),
      adapter.observation_shape,
      adapter.action_dim,
      IPPOConfig(),
    )
    flat = jnp.asarray(flatten_agent_batch(observations))
    actions, log_probs, values = select_actions(state, jax.random.PRNGKey(1), flat)
    assert actions.shape == (adapter.num_agents,)
    assert log_probs.shape == (adapter.num_agents,)
    assert values.shape == (adapter.num_agents,)
  finally:
    adapter.close()
