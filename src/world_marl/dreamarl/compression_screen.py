"""Minimal causal screen for local compression and multi-agent information.

The three variants deliberately share every trainable parameter and differ only
in which frozen input tokens are visible to the predictor:

* ``B``: local belief history and local action.
* ``O_L``: local observation-token history and local action.
* ``O_J``: aligned joint observation-token histories and local action.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .diagnostic_dataset import TrajectorySplit
from .information_ladder import cosine_error, summarize_errors


Array = jax.Array


class ScreenInput(StrEnum):
    BELIEF_LOCAL = "b_local_belief"
    OBSERVATION_LOCAL = "o_l_local_observation"
    OBSERVATION_JOINT = "o_j_joint_observation"


@dataclass(frozen=True, slots=True)
class ScreenConfig:
    feature_width: int = 512
    hidden: int = 256
    heads: int = 4
    temporal_layers: int = 2
    learning_rate: float = 3e-4
    steps: int = 2_000
    batch_trajectories: int = 8
    evaluation_batch_trajectories: int = 8
    grad_clip: float = 10.0
    seed: int = 0
    bootstrap_samples: int = 2_000


@dataclass(frozen=True, slots=True)
class ScreenDataset:
    belief: np.ndarray
    observation: np.ndarray
    action: np.ndarray
    target: np.ndarray
    valid: np.ndarray
    reset: np.ndarray
    reward_event: np.ndarray
    trajectory_id: np.ndarray
    action_dim: int


def _signed_projection(values: np.ndarray, width: int, *, seed: int) -> np.ndarray:
    """Apply a deterministic sparse signed random projection.

    Features are randomly permuted and signed before summation into equally
    sized buckets. Unlike the former compressor, adjacent coordinates are not
    averaged and modality ordering cannot determine the buckets.
    """

    values = np.asarray(values, np.float32)
    flat = values.reshape((*values.shape[:3], -1))
    features = flat.shape[-1]
    generator = np.random.default_rng(seed)
    permutation = generator.permutation(features)
    signs = generator.choice(np.asarray([-1.0, 1.0], np.float32), features)
    projected = flat[..., permutation] * signs
    bucket_size = int(np.ceil(features / width))
    padded_features = width * bucket_size
    if padded_features != features:
        projected = np.pad(
            projected,
            [(0, 0)] * 3 + [(0, padded_features - features)],
        )
    projected = projected.reshape((*flat.shape[:3], width, bucket_size)).sum(-1)
    return (projected / np.sqrt(bucket_size)).astype(np.float32)


def _standardize(
    values: np.ndarray, train_rows: np.ndarray, valid: np.ndarray
) -> np.ndarray:
    selected = values[train_rows][valid[train_rows]]
    mean = selected.mean(0)
    std = selected.std(0)
    return ((values - mean) / np.maximum(std, 1e-4)).astype(np.float32)


def prepare_screen_dataset(
    arrays: Mapping[str, np.ndarray],
    split: TrajectorySplit,
    *,
    feature_width: int,
    action_dim: int,
    projection_seed: int = 0,
) -> ScreenDataset:
    if "source_observation_token" not in arrays:
        raise ValueError(
            "compression screen requires source_observation_token in the dataset"
        )
    valid = np.asarray(arrays["agent_valid"], bool) & np.asarray(
        arrays["action_available"], bool
    )
    belief = _signed_projection(arrays["belief"], feature_width, seed=projection_seed)
    observation = _signed_projection(
        arrays["source_observation_token"],
        feature_width,
        seed=projection_seed + 1,
    )
    belief = _standardize(belief, split.train, valid)
    observation = _standardize(observation, split.train, valid)

    action = np.asarray(arrays["action"], np.int32)
    if action.ndim == 4 and action.shape[-1] == 1:
        action = action[..., 0]
    if action.shape != valid.shape:
        raise ValueError((action.shape, valid.shape))
    if np.any((action[valid] < 0) | (action[valid] >= action_dim)):
        raise ValueError("dataset contains an action outside its declared space")

    if "reset" in arrays:
        reset = np.asarray(arrays["reset"], bool)
        if reset.ndim == 2:
            reset = np.broadcast_to(reset[..., None], valid.shape)
    else:
        reset = np.broadcast_to(
            (np.asarray(arrays["timestep"]) == 0)[..., None], valid.shape
        )
    if reset.shape != valid.shape:
        raise ValueError((reset.shape, valid.shape))

    reward_event = np.asarray(
        arrays.get("reward_event", np.zeros(valid.shape[:2], bool)), bool
    )
    if reward_event.ndim == 2:
        reward_event = np.broadcast_to(reward_event[..., None], valid.shape)
    return ScreenDataset(
        belief=belief,
        observation=observation,
        action=action,
        target=np.asarray(arrays["next_target"], np.float32),
        valid=valid,
        reset=reset,
        reward_event=reward_event,
        trajectory_id=np.asarray(arrays["trajectory_id"]),
        action_dim=action_dim,
    )


def _linear(key: Array, inputs: int, outputs: int) -> Array:
    return jax.random.normal(key, (inputs, outputs)) / jnp.sqrt(inputs)


def init_screen_predictor(
    key: Array,
    config: ScreenConfig,
    *,
    action_dim: int,
    output_dim: int,
) -> dict[str, object]:
    if config.hidden % config.heads:
        raise ValueError("hidden size must be divisible by the number of heads")
    keys = iter(jax.random.split(key, 10 + 6 * config.temporal_layers))
    temporal = []
    for _ in range(config.temporal_layers):
        temporal.append(
            {
                "q": _linear(next(keys), config.hidden, config.hidden),
                "k": _linear(next(keys), config.hidden, config.hidden),
                "v": _linear(next(keys), config.hidden, config.hidden),
                "out": _linear(next(keys), config.hidden, config.hidden),
                "ff1": _linear(next(keys), config.hidden, 4 * config.hidden),
                "ff2": _linear(next(keys), 4 * config.hidden, config.hidden),
            }
        )
    return {
        "input": _linear(next(keys), config.feature_width, config.hidden),
        "input_bias": jnp.zeros((config.hidden,)),
        "action": jax.random.normal(next(keys), (action_dim, config.hidden))
        / jnp.sqrt(config.hidden),
        "set_q": _linear(next(keys), config.hidden, config.hidden),
        "set_k": _linear(next(keys), config.hidden, config.hidden),
        "set_v": _linear(next(keys), config.hidden, config.hidden),
        "set_out": _linear(next(keys), config.hidden, config.hidden),
        "temporal": tuple(temporal),
        "output": _linear(next(keys), config.hidden, output_dim),
        "output_bias": jnp.zeros((output_dim,)),
    }


def _layer_norm(values: Array) -> Array:
    mean = values.mean(-1, keepdims=True)
    variance = jnp.square(values - mean).mean(-1, keepdims=True)
    return (values - mean) * jax.lax.rsqrt(variance + 1e-5)


def _split_heads(values: Array, heads: int) -> Array:
    return values.reshape((*values.shape[:-1], heads, values.shape[-1] // heads))


def _fixed_time_embedding(length: int, hidden: int, dtype: jnp.dtype) -> Array:
    position = jnp.arange(length, dtype=dtype)[:, None]
    frequency = jnp.exp(
        -jnp.log(10_000.0) * jnp.arange(0, hidden, 2, dtype=dtype) / max(hidden, 2)
    )
    embedding = jnp.zeros((length, hidden), dtype=dtype)
    embedding = embedding.at[:, 0::2].set(jnp.sin(position * frequency))
    return embedding.at[:, 1::2].set(jnp.cos(position * frequency[: hidden // 2]))


def screen_predictor(
    params: Mapping[str, object],
    state: Array,
    action: Array,
    valid: Array,
    reset: Array,
    variant: ScreenInput,
    *,
    heads: int,
) -> Array:
    """Predict next beliefs with a shared set-temporal causal Transformer."""

    hidden = jax.nn.silu(
        jnp.einsum("btad,dh->btah", state, params["input"]) + params["input_bias"]
    )
    query = _split_heads(jnp.einsum("btah,hk->btak", hidden, params["set_q"]), heads)
    key = _split_heads(jnp.einsum("btah,hk->btak", hidden, params["set_k"]), heads)
    value = _split_heads(jnp.einsum("btah,hk->btak", hidden, params["set_v"]), heads)
    scores = jnp.einsum("btahd,btshd->btahs", query, key)
    scores /= jnp.sqrt(query.shape[-1])
    agents = state.shape[2]
    visibility = valid[:, :, None, None, :]
    if variant is not ScreenInput.OBSERVATION_JOINT:
        visibility &= jnp.eye(agents, dtype=bool)[None, None, :, None, :]
    scores = jnp.where(visibility, scores, -1e30)
    weights = jax.nn.softmax(scores, -1)
    mixed = jnp.einsum("btahs,btshd->btahd", weights, value)
    mixed = mixed.reshape((*mixed.shape[:3], -1))
    hidden = _layer_norm(hidden + jnp.einsum("btah,hk->btak", mixed, params["set_out"]))
    hidden += params["action"][action]
    hidden += _fixed_time_embedding(hidden.shape[1], hidden.shape[-1], hidden.dtype)[
        None, :, None, :
    ]

    segment = jnp.cumsum(reset.astype(jnp.int32), axis=1).transpose(0, 2, 1)
    time = hidden.shape[1]
    causal = jnp.tril(jnp.ones((time, time), bool))[None, None, None]
    same_segment = segment[:, :, :, None] == segment[:, :, None, :]
    temporal_mask = causal & same_segment[:, :, None]
    temporal_mask &= valid.transpose(0, 2, 1)[:, :, None, None, :]

    for layer in params["temporal"]:
        normalized = _layer_norm(hidden).transpose(0, 2, 1, 3)
        q = _split_heads(jnp.einsum("bath,hk->batk", normalized, layer["q"]), heads)
        k = _split_heads(jnp.einsum("bath,hk->batk", normalized, layer["k"]), heads)
        v = _split_heads(jnp.einsum("bath,hk->batk", normalized, layer["v"]), heads)
        attention = jnp.einsum("baqhd,bakhd->bahqk", q, k)
        attention /= jnp.sqrt(q.shape[-1])
        attention = jnp.where(temporal_mask, attention, -1e30)
        attention = jax.nn.softmax(attention, -1)
        attended = jnp.einsum("bahqk,bakhd->baqhd", attention, v)
        attended = attended.reshape((*attended.shape[:3], -1))
        attended = jnp.einsum("bath,hk->batk", attended, layer["out"])
        hidden = hidden + attended.transpose(0, 2, 1, 3)
        normalized = _layer_norm(hidden)
        feedforward = jax.nn.silu(jnp.einsum("btah,hk->btak", normalized, layer["ff1"]))
        hidden += jnp.einsum("btah,hk->btak", feedforward, layer["ff2"])

    return (
        jnp.einsum("btah,hk->btak", _layer_norm(hidden), params["output"])
        + params["output_bias"]
    )


def _state_for(dataset: ScreenDataset, variant: ScreenInput) -> np.ndarray:
    if variant is ScreenInput.BELIEF_LOCAL:
        return dataset.belief
    return dataset.observation


def train_screen_predictor(
    dataset: ScreenDataset,
    rows: np.ndarray,
    variant: ScreenInput,
    config: ScreenConfig,
) -> tuple[dict[str, object], list[dict[str, float]]]:
    params = init_screen_predictor(
        jax.random.key(config.seed),
        config,
        action_dim=dataset.action_dim,
        output_dim=dataset.target.shape[-1],
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.grad_clip),
        optax.adam(config.learning_rate),
    )
    opt_state = optimizer.init(params)

    @jax.jit
    def update(params, opt_state, state, action, target, valid, reset):
        def objective(current):
            prediction = screen_predictor(
                current,
                state,
                action,
                valid,
                reset,
                variant,
                heads=config.heads,
            )
            error = cosine_error(prediction, target)
            return (error * valid).sum() / jnp.maximum(valid.sum(), 1)

        loss, gradient = jax.value_and_grad(objective)(params)
        updates, next_opt_state = optimizer.update(gradient, opt_state, params)
        return optax.apply_updates(params, updates), next_opt_state, loss

    state = _state_for(dataset, variant)
    generator = np.random.default_rng(config.seed)
    batch_size = min(config.batch_trajectories, rows.size)
    history = []
    for step in range(config.steps):
        selected = generator.choice(rows, batch_size, replace=False)
        params, opt_state, loss = update(
            params,
            opt_state,
            jnp.asarray(state[selected]),
            jnp.asarray(dataset.action[selected]),
            jnp.asarray(dataset.target[selected]),
            jnp.asarray(dataset.valid[selected]),
            jnp.asarray(dataset.reset[selected]),
        )
        if step == 0 or step == config.steps - 1 or (step + 1) % 100 == 0:
            history.append({"step": step + 1, "loss": float(loss)})
    return params, history


def evaluate_screen_predictor(
    params: Mapping[str, object],
    dataset: ScreenDataset,
    rows: np.ndarray,
    variant: ScreenInput,
    config: ScreenConfig,
) -> dict[str, np.ndarray]:
    pieces = []
    batch_size = max(1, config.evaluation_batch_trajectories)
    state = _state_for(dataset, variant)
    for start in range(0, rows.size, batch_size):
        selected = rows[start : start + batch_size]
        prediction = screen_predictor(
            params,
            jnp.asarray(state[selected]),
            jnp.asarray(dataset.action[selected]),
            jnp.asarray(dataset.valid[selected]),
            jnp.asarray(dataset.reset[selected]),
            variant,
            heads=config.heads,
        )
        pieces.append(np.asarray(cosine_error(prediction, dataset.target[selected])))
    error = np.concatenate(pieces, 0)
    valid = dataset.valid[rows]
    return {
        "error": error,
        "valid": valid,
        "reward_event": dataset.reward_event[rows],
        "trajectory_id": np.broadcast_to(
            dataset.trajectory_id[rows, None, None], valid.shape
        ),
        "agent": np.broadcast_to(np.arange(valid.shape[-1])[None, None], valid.shape),
    }


def parameter_count(params: Mapping[str, object]) -> int:
    return sum(int(value.size) for value in jax.tree.leaves(params))


def summarize_screen_errors(
    evaluation: Mapping[str, np.ndarray], config: ScreenConfig
) -> dict[str, object]:
    return summarize_errors(
        evaluation,
        seed=config.seed,
        bootstrap_samples=config.bootstrap_samples,
    )
