"""Deterministically seeded DeepMind Control construction."""

from __future__ import annotations


def make_dmc(task: str, *, seed: int, **kwargs):
    """Build the pinned DMC adapter around a reproducibly seeded suite task."""

    from dm_control import suite
    from embodied.envs.dmc import DMC

    domain, task_name = task.split("_", 1)
    suite_domain = "ball_in_cup" if domain == "cup" else domain
    if suite_domain not in suite.TASKS_BY_DOMAIN:
        raise ValueError(
            f"seeded DMC construction only supports dm_control suite tasks, got {task!r}"
        )
    camera = kwargs.pop("camera", -1)
    if camera == -1:
        camera = DMC.DEFAULT_CAMERAS.get(domain, 0)
    environment = suite.load(
        suite_domain,
        task_name,
        task_kwargs={"random": int(seed)},
    )
    return DMC(environment, camera=camera, **kwargs)
