from __future__ import annotations

from flax import struct
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from world_marl.envs.playground_dmc_adapter import (
    PlaygroundDMCAdapter,
    playground_dmc_env_name,
)


@struct.dataclass
class _State:
    obs: jax.Array
    reward: jax.Array
    done: jax.Array


class _FakePlaygroundEnv:
    action_size = 2

    def reset(self, key):
        del key
        return _State(
            obs=jnp.asarray([0.0, 1.0], dtype=jnp.float32),
            reward=jnp.asarray(0.0, dtype=jnp.float32),
            done=jnp.asarray(0.0, dtype=jnp.float32),
        )

    def step(self, state, action):
        next_obs = state.obs + action
        return _State(
            obs=next_obs,
            reward=jnp.sum(action),
            done=jnp.asarray(next_obs[0] >= 2.0, dtype=jnp.float32),
        )


@pytest.mark.parametrize(
    ("env_id", "playground_name"),
    [
        ("point_mass/easy", "PointMass"),
        ("cartpole/swingup", "CartpoleSwingup"),
        ("reacher/easy", "ReacherEasy"),
        ("finger/spin", "FingerSpin"),
        ("walker/walk", "WalkerWalk"),
    ],
)
def test_playground_dmc_env_name_maps_control_suite_tasks(
    env_id: str,
    playground_name: str,
) -> None:
    assert playground_dmc_env_name(env_id) == playground_name


def test_playground_dmc_adapter_exposes_mjx_provenance_and_scanned_collection() -> None:
    adapter = PlaygroundDMCAdapter(
        "point_mass/easy",
        num_envs=2,
        max_cycles=3,
        seed=0,
        env_factory=_FakePlaygroundEnv,
    )

    assert adapter.substrate == "dmc:point_mass/easy"
    assert adapter.observation_shape == (2,)
    assert adapter.action_shape == (2,)
    assert adapter.environment_metadata == {
        "environment_backend": "mujoco_playground",
        "physics_backend": "mjx",
        "suite": "dm_control",
        "playground_environment": "PointMass",
        "observation_mode": "vector",
    }

    observations, actions, rewards, terminals, lasts = adapter.scan_random_sequence(
        4,
        key=jax.random.PRNGKey(1),
        observations=adapter.reset(),
    )
    assert observations.shape == (4, 2, 2)
    assert actions.shape == (4, 2, 2)
    assert rewards.shape == (4, 2)
    assert terminals.shape == (4, 2)
    assert lasts.shape == (4, 2)
    assert np.isfinite(np.asarray(observations)).all()


def test_playground_dmc_rejects_unported_control_suite_task() -> None:
    with pytest.raises(ValueError, match="not available in MuJoCo Playground"):
        playground_dmc_env_name("point_mass/hard")


def test_playground_dmc_cartpole_runs_on_pinned_mjx_stack() -> None:
    pytest.importorskip("mujoco_playground")
    adapter = PlaygroundDMCAdapter(
        "cartpole/swingup",
        num_envs=2,
        max_cycles=3,
        seed=0,
    )

    observations, actions, rewards, terminals, lasts = adapter.scan_random_sequence(
        4,
        key=jax.random.PRNGKey(1),
        observations=adapter.reset(),
    )

    assert observations.shape == (4, 2, 5)
    assert actions.shape == (4, 2, 1)
    assert rewards.shape == (4, 2)
    assert terminals.shape == (4, 2)
    assert lasts.shape == (4, 2)
    assert adapter.environment_metadata["physics_backend"] == "mjx"


def test_playground_dmc_reports_pinned_mjx_incompatibility() -> None:
    pytest.importorskip("mujoco_playground")

    with pytest.raises(RuntimeError, match="pinned MuJoCo/MJX stack"):
        PlaygroundDMCAdapter("point_mass/easy", num_envs=1, max_cycles=3, seed=0)
