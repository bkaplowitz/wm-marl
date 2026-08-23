"""Lossless transformations between explicit team and shared-local tensors."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


ENVIRONMENT_FIELDS = frozenset(
    {
        "is_first",
        "is_last",
        "is_terminal",
        "consec",
        "stepid",
        "_environment_step",
    }
)
AGENT_METADATA_FIELDS = frozenset(
    {"agent_present", "agent_alive", "controllable_alive", "action_mask"}
)
MODEL_EXCLUDED_FIELDS = (
    frozenset({"is_first", "is_last", "is_terminal", "reward"}) | AGENT_METADATA_FIELDS
)


@dataclass(frozen=True, slots=True)
class TeamAxis:
    """Describe and transform a fixed homogeneous agent axis.

    Public runtime tensors retain ``[B, A, ...]`` or ``[B, T, A, ...]``.
    Shared local modules consume ``[B * A, ...]`` or ``[B * A, T, ...]``.
    """

    size: int

    def __post_init__(self) -> None:
        if self.size < 1:
            raise ValueError("team size must be positive")

    def fold_batch(self, value: Any) -> Any:
        """Fold ``[B, A, ...]`` into ``[B * A, ...]``."""

        self._expect_axis(value, 1)
        return value.reshape((value.shape[0] * self.size, *value.shape[2:]))

    def unfold_batch(self, value: Any) -> Any:
        """Restore ``[B * A, ...]`` to ``[B, A, ...]``."""

        if not value.shape or value.shape[0] % self.size:
            raise ValueError(
                f"cannot unfold batch shape {value.shape} into teams of {self.size}"
            )
        batch = value.shape[0] // self.size
        return value.reshape((batch, self.size, *value.shape[1:]))

    def fold_sequence(self, value: Any) -> Any:
        """Fold ``[B, T, A, ...]`` into ``[B * A, T, ...]``."""

        self._expect_axis(value, 2)
        axes = (0, 2, 1, *range(3, value.ndim))
        transposed = value.transpose(axes)
        return transposed.reshape(
            (value.shape[0] * self.size, value.shape[1], *value.shape[3:])
        )

    def unfold_sequence(self, value: Any) -> Any:
        """Restore ``[B * A, T, ...]`` to ``[B, T, A, ...]``."""

        if value.ndim < 2 or value.shape[0] % self.size:
            raise ValueError(
                f"cannot unfold sequence shape {value.shape} into teams of {self.size}"
            )
        batch = value.shape[0] // self.size
        grouped = value.reshape((batch, self.size, value.shape[1], *value.shape[2:]))
        axes = (0, 2, 1, *range(3, grouped.ndim))
        return grouped.transpose(axes)

    def group_starts(self, value: Any, starts: int) -> Any:
        """Regroup ``[B * A * S, ...]`` as synchronized ``[B * S, A, ...]``."""

        if starts < 1 or not value.shape or value.shape[0] % (self.size * starts):
            raise ValueError(
                f"cannot group start shape {value.shape} with A={self.size}, S={starts}"
            )
        batch = value.shape[0] // (self.size * starts)
        grouped = value.reshape((batch, self.size, starts, *value.shape[1:]))
        axes = (0, 2, 1, *range(3, grouped.ndim))
        grouped = grouped.transpose(axes)
        return grouped.reshape((batch * starts, self.size, *value.shape[1:]))

    def ungroup_starts(self, value: Any, starts: int) -> Any:
        """Restore synchronized ``[B * S, A, ...]`` to ``[B * A * S, ...]``."""

        self._expect_axis(value, 1)
        if starts < 1 or value.shape[0] % starts:
            raise ValueError(
                f"cannot ungroup start shape {value.shape} with A={self.size}, S={starts}"
            )
        batch = value.shape[0] // starts
        grouped = value.reshape((batch, starts, self.size, *value.shape[2:]))
        axes = (0, 2, 1, *range(3, grouped.ndim))
        grouped = grouped.transpose(axes)
        return grouped.reshape((batch * self.size * starts, *value.shape[2:]))

    def broadcast_batch(self, value: Any) -> Any:
        """Broadcast an environment-level ``[B, ...]`` value across agents."""

        expanded = value[:, None]
        shape = (value.shape[0], self.size, *value.shape[1:])
        return self.fold_batch(_broadcast_to(expanded, shape))

    def broadcast_sequence(self, value: Any) -> Any:
        """Broadcast an environment-level ``[B, T, ...]`` value across agents."""

        expanded = value[:, :, None]
        shape = (value.shape[0], value.shape[1], self.size, *value.shape[2:])
        return self.fold_sequence(_broadcast_to(expanded, shape))

    def fold_tree_batch(self, tree: Any) -> Any:
        return _map_tree(self.fold_batch, tree)

    def unfold_tree_batch(self, tree: Any) -> Any:
        return _map_tree(self.unfold_batch, tree)

    def group_tree_starts(self, tree: Any, starts: int) -> Any:
        return _map_tree(lambda value: self.group_starts(value, starts), tree)

    def ungroup_tree_starts(self, tree: Any, starts: int) -> Any:
        return _map_tree(lambda value: self.ungroup_starts(value, starts), tree)

    def local_policy_data(self, data: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: (
                self.broadcast_batch(value)
                if is_environment_field(key)
                else self.fold_batch(value)
            )
            for key, value in data.items()
        }

    def local_sequence_data(self, data: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: (
                self.broadcast_sequence(value)
                if is_environment_field(key)
                else self.fold_sequence(value)
            )
            for key, value in data.items()
        }

    def unfold_replay_updates(self, updates: Mapping[str, Any]) -> Mapping[str, Any]:
        result = {}
        for key, value in updates.items():
            unfolded = self.unfold_sequence(value)
            result[key] = unfolded[:, :, 0] if key == "stepid" else unfolded
        return type(updates)(result)

    def _expect_axis(self, value: Any, axis: int) -> None:
        if value.ndim <= axis or value.shape[axis] != self.size:
            form = "[B, A, ...]" if axis == 1 else "[B, T, A, ...]"
            raise ValueError(f"expected {form} with A={self.size}, got {value.shape}")


def is_environment_field(name: str) -> bool:
    return name in ENVIRONMENT_FIELDS or name.startswith("log/")


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
    return _array_module(value).broadcast_to(value, shape)


def _array_module(value: Any):
    module = type(value).__module__.split(".", 1)[0]
    if module in {"jax", "jaxlib"}:
        import jax.numpy as array_module
    else:
        import numpy as array_module
    return array_module
