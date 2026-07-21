from __future__ import annotations

from collections.abc import Callable
from typing import Any

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
