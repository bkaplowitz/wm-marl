from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import struct

from world_marl.envs.brax_adapter import BraxVectorAdapter, brax_env_name


@struct.dataclass
class _FakeBraxState:
    obs: jax.Array
    reward: jax.Array
    done: jax.Array
    count: jax.Array


class _FakeBraxEnv:
    action_size = 2

    def reset(self, rng):
        seed_value = jax.random.randint(rng, (), minval=0, maxval=1000).astype(
            jnp.float32
        )
        return _FakeBraxState(
            obs=jnp.stack([seed_value, jnp.asarray(0.0), jnp.asarray(0.0)]),
            reward=jnp.asarray(0.0, dtype=jnp.float32),
            done=jnp.asarray(0.0, dtype=jnp.float32),
            count=jnp.asarray(0, dtype=jnp.int32),
        )

    def step(self, state, action):
        count = state.count + 1
        reward = jnp.sum(action)
        done = (count >= 2).astype(jnp.float32)
        return state.replace(
            obs=jnp.stack([state.obs[0], count.astype(jnp.float32), reward]),
            reward=reward,
            done=done,
            count=count,
        )


def test_brax_env_name_parses_name():
    assert brax_env_name("brax:reacher") == "reacher"
    with pytest.raises(ValueError, match="brax:<env_name>"):
        brax_env_name("brax:")


def test_brax_adapter_reset_step_and_completion():
    adapter = BraxVectorAdapter(
        "fake",
        num_envs=2,
        max_cycles=5,
        seed=10,
        env_factory=_FakeBraxEnv,
    )
    try:
        observations = adapter.reset()
        actions = adapter.sample_actions(np.random.default_rng(0))
        first = adapter.step(actions)
        second = adapter.step(np.zeros((2, 1, 2), dtype=np.float32))

        assert adapter.num_agents == 1
        assert adapter.action_dim == 2
        assert adapter.observation_shape == (3,)
        assert observations.shape == (2, 1, 3)
        assert actions.shape == (2, 1, 2)
        assert first.observations.shape == (2, 1, 3)
        assert first.rewards.shape == (2, 1)
        assert second.dones.tolist() == [[1.0], [1.0]]
        assert len(second.completed_returns) == 2
        assert second.completed_lengths == (2, 2)
    finally:
        adapter.close()
