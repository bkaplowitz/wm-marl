"""Lossless categorical replay transport; model states remain dense one-hot."""

import elements
import jax.numpy as jnp
import numpy as np


UNUSED_BEHAVIOR = frozenset({
    "dyn/stoch", "dyn/pair", "dyn/pair_action", "dyn/reset", "dyn/active",
})


def compact_spaces(spaces, stoch, classes):
    if not 1 < classes < 255:
        raise ValueError("Compact replay requires 2–254 categorical classes")
    result = dict(spaces)
    for key in ("dyn/stoch", "dyn/pair"):
        space = result[key]
        leading = space.shape[:-2] if key.endswith("stoch") else space.shape[:-1]
        result[key] = elements.Space(np.uint8, (*leading, stoch))
        if key.endswith("pair"):
            actions = space.shape[-1] - stoch * classes
            result["dyn/pair_action"] = elements.Space(space.dtype, (*leading, actions))
    return result


def pack_entries(data, stoch, classes, xp=jnp):
    """Encode sampled one-hots. Zero is reserved for the all-zero reset state.

    Called inside the policy/learner JIT, before outputs reach CPU replay.
    The two latent fields stay independent because replay writeback can update
    overlapping sampled sequences in different orders.
    """
    result = dict(data)
    for key in ("dyn/stoch", "dyn/pair"):
        if key not in result:
            continue
        value = result[key]
        if key.endswith("pair"):
            result["dyn/pair_action"] = value[..., stoch * classes:]
            value = value[..., :stoch * classes].reshape(*value.shape[:-1], stoch, classes)
        index = xp.argmax(value, axis=-1) + 1
        total = value.sum(axis=-1)
        valid = xp.all((value == 0) | (value == 1), axis=-1) & ((total == 0) | (total == 1))
        index = xp.where(total == 0, 0, index)
        # 255 flags invalid input instead of silently quantizing a soft/NaN state.
        result[key] = xp.where(valid, index, 255).astype(xp.uint8)
    return result


def unpack_entries(data, stoch, classes):
    """Expand after device transfer, before the existing learner operations."""
    result = dict(data)
    categories = jnp.arange(1, classes + 1, dtype=jnp.int32)
    for key in ("dyn/stoch", "dyn/pair"):
        if key not in result:
            continue
        index = result[key]
        value = (index[..., None] == categories).astype(jnp.float32)
        if key.endswith("pair"):
            value = value.reshape(*index.shape[:-1], stoch * classes)
            value = jnp.concatenate([value, result.pop("dyn/pair_action")], axis=-1)
        result[key] = value
    return result


def slim_behavior(data):
    return {key: value for key, value in data.items() if key not in UNUSED_BEHAVIOR}


def filter_transport(agent, data):
    """Reports retain their original sampling draws but omit unused payloads."""
    if not agent.model.config.get("compact_replay", False):
        return data
    for key, value in data.items():
        if key.endswith(("dyn/stoch", "dyn/pair")) and value.dtype == np.uint8:
            if np.any(value == 255):
                raise ValueError(f"Non-categorical state in compact replay field {key}")
    return {key: value for key, value in data.items()
            if key.removeprefix("_behavior_replay/") not in UNUSED_BEHAVIOR
            or not key.startswith("_behavior_replay/")}
