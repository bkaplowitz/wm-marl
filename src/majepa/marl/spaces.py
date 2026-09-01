"""Space transformations at the explicit team-axis boundary."""

from __future__ import annotations

import elements
import numpy as np

from .axes import is_environment_field


def local_observation_spaces(spaces, team_size):
    return {
        key: (
            space
            if is_environment_field(key)
            else remove_agent_axis(key, space, team_size)
        )
        for key, space in spaces.items()
    }


def local_action_spaces(spaces, team_size):
    return {
        key: remove_agent_axis(key, space, team_size) for key, space in spaces.items()
    }


def remove_agent_axis(name, space, team_size):
    if not space.shape or space.shape[0] != team_size:
        raise ValueError(
            f"{name!r} must expose leading agent axis {team_size}, got {space.shape}"
        )
    return elements.Space(
        space.dtype,
        space.shape[1:],
        _remove_bound_axis(name, "low", space.low, team_size),
        _remove_bound_axis(name, "high", space.high, team_size),
    )


def _remove_bound_axis(name, label, bound, team_size):
    if bound is None:
        return None
    values = np.asarray(bound)
    if values.ndim == 0:
        return values
    if values.shape[0] != team_size:
        raise ValueError(
            f"{name!r} {label} bound must expose agent axis {team_size}, "
            f"got {values.shape}"
        )
    if not np.all(values == values[0]):
        raise ValueError(
            f"{name!r} has heterogeneous per-agent {label} bounds; "
            "the shared policy requires homogeneous agents"
        )
    return values[0]


def add_agent_axis(space, team_size):
    low = (
        None
        if space.low is None
        else np.broadcast_to(np.asarray(space.low), (team_size, *space.shape))
    )
    high = (
        None
        if space.high is None
        else np.broadcast_to(np.asarray(space.high), (team_size, *space.shape))
    )
    return elements.Space(space.dtype, (team_size, *space.shape), low, high)


def report_rows(folded_batch, team_size, max_rows=6):
    """Choose a reporting batch without splitting a complete team."""

    if folded_batch % team_size:
        raise ValueError(
            f"folded report batch {folded_batch} is not divisible by {team_size}"
        )
    return min(
        folded_batch,
        max(team_size, (max_rows // team_size) * team_size),
    )
