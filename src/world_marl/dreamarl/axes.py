"""Pure tensor-axis operations used by the DreaMARL agent.

The learner keeps environment and agent axes explicit in replay as
``[batch, time, agent, ...]``. Neural modules share parameters across agents,
so their existing M3 implementation receives ``batch * agent`` trajectories.
These functions only reshape and transpose arrays; they do not change values
or introduce an agent-count-dependent code path.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


GLOBAL_OBSERVATION_KEYS = frozenset({"is_first", "is_last", "is_terminal"})
GLOBAL_REPLAY_KEYS = frozenset({"consec", "stepid"})


def fold_agent_batch(value: Any, num_agents: int) -> Any:
    """Fold ``[batch, agent, ...]`` into ``[batch * agent, ...]``."""

    if value.ndim < 2 or value.shape[1] != num_agents:
        raise ValueError(
            "expected [batch, agent, ...] with agent dimension "
            f"{num_agents}, got {value.shape}"
        )
    return value.reshape((value.shape[0] * num_agents, *value.shape[2:]))


def unfold_agent_batch(value: Any, num_agents: int) -> Any:
    """Restore ``[batch * agent, ...]`` to ``[batch, agent, ...]``."""

    if value.ndim < 1 or value.shape[0] % num_agents:
        raise ValueError(
            f"leading dimension {value.shape[0]} is not divisible by {num_agents}"
        )
    batch = value.shape[0] // num_agents
    return value.reshape((batch, num_agents, *value.shape[1:]))


def fold_agent_sequence(value: Any, num_agents: int) -> Any:
    """Fold ``[batch, time, agent, ...]`` into ``[batch * agent, time, ...]``."""

    if value.ndim < 3 or value.shape[2] != num_agents:
        raise ValueError(
            "expected [batch, time, agent, ...] with agent dimension "
            f"{num_agents}, got {value.shape}"
        )
    axes = (0, 2, 1, *range(3, value.ndim))
    transposed = value.transpose(axes)
    return transposed.reshape(
        (value.shape[0] * num_agents, value.shape[1], *value.shape[3:])
    )


def unfold_agent_sequence(value: Any, num_agents: int) -> Any:
    """Restore ``[batch * agent, time, ...]`` to ``[batch, time, agent, ...]``."""

    if value.ndim < 2 or value.shape[0] % num_agents:
        raise ValueError(
            f"leading dimension {value.shape[0]} is not divisible by {num_agents}"
        )
    batch = value.shape[0] // num_agents
    grouped = value.reshape((batch, num_agents, value.shape[1], *value.shape[2:]))
    axes = (0, 2, 1, *range(3, grouped.ndim))
    return grouped.transpose(axes)


def select_joint_starts(value: Any, num_agents: int, nlast: int) -> Any:
    """Select final starts in ``[environment, start, agent]`` order.

    Local dynamics consume folded trajectories in ``[environment * agent,
    time, ...]`` order. Joint imagination instead requires every agent from a
    particular environment and start time to be adjacent.
    """

    if nlast < 1 or nlast > value.shape[1]:
        raise ValueError((nlast, value.shape))
    grouped = unfold_agent_sequence(value, num_agents)
    selected = grouped[:, -nlast:]
    return selected.reshape(
        (selected.shape[0] * nlast * num_agents, *selected.shape[3:])
    )


def restore_folded_start_order(value: Any, num_agents: int, nlast: int) -> Any:
    """Restore ``[environment * start * agent]`` to folded trajectory order."""

    divisor = num_agents * nlast
    if value.shape[0] % divisor:
        raise ValueError(
            f"leading dimension {value.shape[0]} is not divisible by {divisor}"
        )
    environments = value.shape[0] // divisor
    grouped = value.reshape(
        (environments, nlast, num_agents, *value.shape[1:])
    )
    axes = (0, 2, 1, *range(3, grouped.ndim))
    folded = grouped.transpose(axes)
    return folded.reshape(
        (environments * num_agents, nlast, *value.shape[1:])
    )


def broadcast_global_batch(value: Any, num_agents: int) -> Any:
    """Repeat a joint ``[batch, ...]`` value for each shared-policy agent."""

    expanded = value[:, None]
    shape = (value.shape[0], num_agents, *value.shape[1:])
    return fold_agent_batch(_broadcast_to(expanded, shape), num_agents)


def broadcast_global_sequence(value: Any, num_agents: int) -> Any:
    """Repeat a joint ``[batch, time, ...]`` value for each agent."""

    expanded = value[:, :, None]
    shape = (value.shape[0], value.shape[1], num_agents, *value.shape[2:])
    return fold_agent_sequence(_broadcast_to(expanded, shape), num_agents)


def fold_tree_batch(tree: Any, num_agents: int) -> Any:
    """Fold every array leaf in a policy carry or per-agent output tree."""

    return _map_tree(lambda value: fold_agent_batch(value, num_agents), tree)


def unfold_tree_batch(tree: Any, num_agents: int) -> Any:
    """Restore every array leaf in a policy carry or output tree."""

    return _map_tree(lambda value: unfold_agent_batch(value, num_agents), tree)


def _map_tree(function, tree: Any) -> Any:
    if isinstance(tree, Mapping):
        return type(tree)(
            (key, _map_tree(function, value)) for key, value in tree.items()
        )
    if isinstance(tree, tuple):
        return type(tree)(_map_tree(function, value) for value in tree)
    if isinstance(tree, list):
        return [_map_tree(function, value) for value in tree]
    return function(tree)


def _broadcast_to(value: Any, shape: tuple[int, ...]) -> Any:
    return _xp(value).broadcast_to(value, shape)


def _xp(value: Any):
    module = type(value).__module__.split(".", 1)[0]
    if module in {"jax", "jaxlib"}:
        import jax.numpy as array_module
    else:
        import numpy as array_module
    return array_module
