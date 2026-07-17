"""JAX-native MuJoCo Playground vector and MJX/Warp vision adapters.

The vision path follows ``google-deepmind/mujoco_playground`` 0.2.0,
``_src/dm_control_suite/cartpole.py``. The local integration converts its
centered grayscale frame stack to ``[0, 1]`` HWC pixels and reuses the
repository's scan-only single-agent adapter contract.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp

from world_marl.envs.brax_adapter import BraxVectorAdapter


_CONTROL_SUITE_TO_PLAYGROUND = {
    "acrobot/swingup": "AcrobotSwingup",
    "acrobot/swingup_sparse": "AcrobotSwingupSparse",
    "ball_in_cup/catch": "BallInCup",
    "cartpole/balance": "CartpoleBalance",
    "cartpole/balance_sparse": "CartpoleBalanceSparse",
    "cartpole/swingup": "CartpoleSwingup",
    "cartpole/swingup_sparse": "CartpoleSwingupSparse",
    "cheetah/run": "CheetahRun",
    "finger/spin": "FingerSpin",
    "finger/turn_easy": "FingerTurnEasy",
    "finger/turn_hard": "FingerTurnHard",
    "fish/swim": "FishSwim",
    "hopper/hop": "HopperHop",
    "hopper/stand": "HopperStand",
    "humanoid/stand": "HumanoidStand",
    "humanoid/walk": "HumanoidWalk",
    "humanoid/run": "HumanoidRun",
    "pendulum/swingup": "PendulumSwingup",
    "point_mass/easy": "PointMass",
    "reacher/easy": "ReacherEasy",
    "reacher/hard": "ReacherHard",
    "swimmer/swimmer6": "SwimmerSwimmer6",
    "walker/run": "WalkerRun",
    "walker/stand": "WalkerStand",
    "walker/walk": "WalkerWalk",
}


def playground_dmc_env_name(env_id: str) -> str:
    if env_id.count("/") != 1:
        raise ValueError("DMC substrates must be formatted as 'dmc:<domain>/<task>'")
    try:
        return _CONTROL_SUITE_TO_PLAYGROUND[env_id]
    except KeyError as exc:
        raise ValueError(
            f"DMC task {env_id!r} is not available in MuJoCo Playground"
        ) from exc


def make_playground_dmc_env(env_id: str) -> Any:
    from mujoco_playground import dm_control_suite

    playground_name = playground_dmc_env_name(env_id)
    try:
        return dm_control_suite.load(playground_name)
    except (AttributeError, NotImplementedError) as exc:
        raise RuntimeError(
            f"DMC task {env_id!r} cannot be loaded by MuJoCo Playground on the "
            f"pinned MuJoCo/MJX stack: {exc}"
        ) from exc


def make_playground_vision_env(
    env_id: str,
    *,
    num_envs: int,
    image_size: int,
    episode_length: int,
) -> Any:
    from mujoco_playground import registry

    config = registry.get_default_config(env_id)
    config.vision = True
    config.impl = "warp"
    config.episode_length = episode_length
    config.vision_config.nworld = num_envs
    config.vision_config.cam_res = (image_size, image_size)
    config.vision_config.render_rgb = True
    return registry.load(env_id, config=config)


def _playground_vision_observation(state: Any) -> jnp.ndarray:
    pixels = jnp.asarray(state.obs["pixels/view_0"], dtype=jnp.float32)
    return jnp.clip(pixels + 0.5, 0.0, 1.0)


class PlaygroundDMCAdapter(BraxVectorAdapter):
    def __init__(
        self,
        env_id: str = "point_mass/easy",
        *,
        num_envs: int = 1,
        max_cycles: int = 1000,
        seed: int = 0,
        env_factory: Callable[[], Any] | None = None,
        auto_reset: bool = True,
    ) -> None:
        playground_name = playground_dmc_env_name(env_id)
        factory = env_factory or (lambda: make_playground_dmc_env(env_id))
        super().__init__(
            playground_name,
            num_envs=num_envs,
            max_cycles=max_cycles,
            seed=seed,
            env_factory=factory,
            auto_reset=auto_reset,
        )
        self.env_id = env_id
        self.substrate = f"dmc:{env_id}"
        self.environment_metadata = {
            "environment_backend": "mujoco_playground",
            "physics_backend": "mjx",
            "suite": "dm_control",
            "playground_environment": playground_name,
            "observation_mode": "vector",
        }


class PlaygroundVisionAdapter(BraxVectorAdapter):
    def __init__(
        self,
        env_id: str = "CartpoleBalance",
        *,
        num_envs: int = 1,
        max_cycles: int = 1000,
        seed: int = 0,
        image_size: int = 64,
        env_factory: Callable[[], Any] | None = None,
        auto_reset: bool = True,
    ) -> None:
        factory = env_factory or (
            lambda: make_playground_vision_env(
                env_id,
                num_envs=num_envs,
                image_size=image_size,
                episode_length=max_cycles,
            )
        )
        super().__init__(
            env_id,
            num_envs=num_envs,
            max_cycles=max_cycles,
            seed=seed,
            env_factory=factory,
            observation_fn=_playground_vision_observation,
            auto_reset=auto_reset,
        )
        if self.observation_shape != (image_size, image_size, 3):
            raise ValueError(
                "MuJoCo Playground vision observations must be HWC frame stacks "
                f"with shape {(image_size, image_size, 3)}, got {self.observation_shape}"
            )
        self.substrate = f"playground-vision:{env_id}"
        self.environment_metadata = {
            "environment_backend": "mujoco_playground",
            "physics_backend": "mjx_warp",
            "renderer_backend": "mjwarp_batch_renderer",
            "playground_environment": env_id,
            "observation_mode": "pixels",
            "pixel_range": [0.0, 1.0],
        }
