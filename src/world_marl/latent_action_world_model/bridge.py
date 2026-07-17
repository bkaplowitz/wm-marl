"""Empirical latent-code to real-action bridge from expert calibration.

The bridge implements the repository's Genie Appendix E integration rule: a
frozen LAM infers one of six transition codes, every observed expert action is
retained under its inferred code, and rollout actions are sampled uniformly
from that code's recorded list. There is no replay fallback or default action.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True, slots=True)
class ExpertActionBridge:
    actions: jax.Array
    counts: jax.Array
    environment: str
    provenance: Mapping[str, Any]


def _scalar_string(value: np.ndarray, name: str) -> str:
    if value.shape != ():
        raise ValueError(f"calibration {name} must be a scalar string")
    result = str(value.item())
    if not result:
        raise ValueError(f"calibration {name} must be non-empty")
    return result


def load_expert_bridge(
    path: str | Path,
    *,
    infer_codes: Callable[[jax.Array], jax.Array],
) -> ExpertActionBridge:
    required = {"observations", "actions", "is_first", "environment", "provenance"}
    with np.load(path, allow_pickle=False) as calibration:
        missing = required.difference(calibration.files)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"calibration is missing required metadata: {names}")
        observations = np.asarray(calibration["observations"])
        actions = np.asarray(calibration["actions"])
        is_first = np.asarray(calibration["is_first"], dtype=bool)
        environment = _scalar_string(calibration["environment"], "environment")
        provenance_text = _scalar_string(calibration["provenance"], "provenance")

    try:
        provenance = json.loads(provenance_text)
    except json.JSONDecodeError as error:
        raise ValueError("calibration provenance must be valid JSON") from error
    if not isinstance(provenance, dict) or not provenance:
        raise ValueError("calibration provenance must be a non-empty JSON object")
    if observations.ndim != 5 or observations.shape[-1] != 3:
        raise ValueError("calibration observations must be time-major HWC RGB")
    if observations.shape[0] < 2:
        raise ValueError("calibration requires at least two time steps")
    if actions.ndim < 2 or actions.shape[:2] != observations.shape[:2]:
        raise ValueError("calibration actions must share the time/batch prefix")
    if is_first.shape != observations.shape[:2]:
        raise ValueError("calibration is_first must share the time/batch prefix")

    valid = ~is_first[1:]
    current = observations[:-1][valid]
    future = observations[1:][valid]
    transition_actions = actions[:-1][valid]
    videos = jnp.asarray(np.stack([current, future], axis=1))
    codes = np.asarray(infer_codes(videos), dtype=np.int32).reshape(-1)
    if codes.shape != (len(transition_actions),):
        raise ValueError("inferred codes must provide one code per valid transition")
    if np.any((codes < 0) | (codes >= 6)):
        raise ValueError("inferred latent-action codes must be in [0, 6)")

    grouped_actions = [transition_actions[codes == code] for code in range(6)]
    counts = np.asarray([len(group) for group in grouped_actions], dtype=np.int32)
    if np.any(counts == 0):
        raise ValueError("expert calibration must cover all six latent-action codes")
    max_count = int(counts.max())
    padded_actions = np.zeros(
        (6, max_count, *transition_actions.shape[1:]),
        dtype=transition_actions.dtype,
    )
    for code, group in enumerate(grouped_actions):
        padded_actions[code, : len(group)] = group
    return ExpertActionBridge(
        actions=jnp.asarray(padded_actions),
        counts=jnp.asarray(counts),
        environment=environment,
        provenance=provenance,
    )


def sample_real_actions(
    rng: jax.Array,
    actions: jax.Array,
    counts: jax.Array,
    latent_codes: jax.Array,
) -> jax.Array:
    choices = jax.random.uniform(rng, latent_codes.shape)
    indices = jnp.floor(choices * counts[latent_codes]).astype(jnp.int32)
    return actions[latent_codes, indices]
