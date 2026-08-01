"""Equal-capacity one-step information ladder for frozen DreaMARL replay."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .diagnostic_dataset import TrajectorySplit, trajectory_bootstrap_interval


Array = jax.Array


class Rung(StrEnum):
    X0_LOCAL = "x0_local_belief_local_action"
    X1_JOINT_ACTION = "x1_local_belief_joint_action"
    X2_JOINT_BELIEF = "x2_joint_belief_joint_action"
    X3_JOINT_OBSERVATION = "x3_joint_observation_joint_action"
    X4_ORACLE_STATE = "x4_oracle_state_joint_action"


@dataclass(frozen=True, slots=True)
class LadderConfig:
    feature_width: int = 512
    hidden: int = 256
    learning_rate: float = 3e-4
    steps: int = 2_000
    batch_trajectories: int = 8
    grad_clip: float = 10.0
    seed: int = 0
    bootstrap_samples: int = 2_000


@dataclass(frozen=True, slots=True)
class PreparedDataset:
    belief: np.ndarray
    observation: np.ndarray | None
    oracle: np.ndarray | None
    action: np.ndarray
    target: np.ndarray
    valid: np.ndarray
    reward_event: np.ndarray
    trajectory_id: np.ndarray
    action_dim: int


def available_rungs(arrays: Mapping[str, np.ndarray]) -> tuple[Rung, ...]:
    result = [Rung.X0_LOCAL, Rung.X1_JOINT_ACTION, Rung.X2_JOINT_BELIEF]
    if "observation" in arrays:
        result.append(Rung.X3_JOINT_OBSERVATION)
    if "oracle_state" in arrays:
        result.append(Rung.X4_ORACLE_STATE)
    return tuple(result)


def _compress(values: np.ndarray, width: int) -> np.ndarray:
    """Deterministically compress a modality without trainable capacity."""

    values = np.asarray(values, np.float32)
    flat = values.reshape((*values.shape[:3], -1))
    features = flat.shape[-1]
    if features == width:
        return flat
    if features < width:
        return np.pad(flat, [(0, 0)] * 3 + [(0, width - features)])
    block = int(np.ceil(features / width))
    padded = np.pad(flat, [(0, 0)] * 3 + [(0, width * block - features)])
    mask = np.pad(
        np.ones((features,), np.float32), (0, width * block - features)
    ).reshape(width, block)
    denominator = np.maximum(mask.sum(-1), 1.0)
    return padded.reshape((*flat.shape[:3], width, block)).sum(-1) / denominator


def _observation_features(observation: np.ndarray, width: int) -> np.ndarray:
    observation = np.asarray(observation)
    if observation.ndim < 6:
        return _compress(observation, width)
    height, image_width = observation.shape[-3:-1]
    rows = np.linspace(0, height - 1, min(16, height)).astype(np.int32)
    columns = np.linspace(0, image_width - 1, min(16, image_width)).astype(
        np.int32
    )
    sampled = observation[..., rows, :, :][..., columns, :]
    return _compress(sampled.astype(np.float32) / 255.0, width)


def _standardize(
    values: np.ndarray, train_rows: np.ndarray, valid: np.ndarray
) -> np.ndarray:
    selected = values[train_rows]
    selected_valid = valid[train_rows]
    flat = selected[selected_valid]
    mean = flat.mean(0)
    std = flat.std(0)
    return ((values - mean) / np.maximum(std, 1e-4)).astype(np.float32)


def prepare_dataset(
    arrays: Mapping[str, np.ndarray],
    split: TrajectorySplit,
    *,
    feature_width: int,
    action_dim: int,
) -> PreparedDataset:
    valid = np.asarray(arrays["agent_valid"], bool) & np.asarray(
        arrays["action_available"], bool
    )
    belief = _compress(np.asarray(arrays["belief"]), feature_width)
    belief = _standardize(belief, split.train, valid)
    observation = None
    if "observation" in arrays:
        observation = _observation_features(arrays["observation"], feature_width)
        observation = _standardize(observation, split.train, valid)
    oracle = None
    if "oracle_state" in arrays:
        source = np.asarray(arrays["oracle_state"])
        source = np.broadcast_to(source[:, :, None], (*valid.shape, *source.shape[2:]))
        oracle = _compress(source, feature_width)
        oracle = _standardize(oracle, split.train, valid)
    action = np.asarray(arrays["action"], np.int32)
    if action.ndim == 4 and action.shape[-1] == 1:
        action = action[..., 0]
    if action.shape != valid.shape:
        raise ValueError((action.shape, valid.shape))
    if np.any((action[valid] < 0) | (action[valid] >= action_dim)):
        raise ValueError("dataset contains an action outside the declared action space")
    return PreparedDataset(
        belief=belief,
        observation=observation,
        oracle=oracle,
        action=action,
        target=np.asarray(arrays["next_target"], np.float32),
        valid=valid,
        reward_event=np.broadcast_to(
            np.asarray(arrays.get("reward_event", np.zeros(valid.shape[:2], bool)))[
                ..., None
            ],
            valid.shape,
        ),
        trajectory_id=np.asarray(arrays["trajectory_id"]),
        action_dim=action_dim,
    )


def _init_linear(key: Array, input_dim: int, output_dim: int) -> Array:
    return jax.random.normal(key, (input_dim, output_dim)) / jnp.sqrt(input_dim)


def init_predictor(
    key: Array, feature_width: int, action_dim: int, hidden: int, output_dim: int
) -> dict[str, Array]:
    keys = jax.random.split(key, 4)
    return {
        "state_kernel": _init_linear(keys[0], feature_width, hidden),
        "state_bias": jnp.zeros((hidden,)),
        "action_embedding": jax.random.normal(keys[1], (action_dim, hidden))
        / jnp.sqrt(hidden),
        "mix_kernel": _init_linear(keys[2], 2 * hidden, hidden),
        "mix_bias": jnp.zeros((hidden,)),
        "output_kernel": _init_linear(keys[3], hidden, output_dim),
        "output_bias": jnp.zeros((output_dim,)),
    }


def init_residual(
    key: Array, hidden: int, output_dim: int
) -> dict[str, Array]:
    return {
        "mix_kernel": _init_linear(key, 2 * hidden, hidden),
        "mix_bias": jnp.zeros((hidden,)),
        "output_kernel": jnp.zeros((hidden, output_dim)),
        "output_bias": jnp.zeros((output_dim,)),
    }


def _leave_one_out(values: Array, valid: Array) -> Array:
    weights = valid.astype(values.dtype)[..., None]
    total = (values * weights).sum(-2, keepdims=True)
    count = weights.sum(-2, keepdims=True)
    return (total - values * weights) / jnp.maximum(count - weights, 1.0)


def predictor(
    params: Mapping[str, Array],
    state: Array,
    action: Array,
    valid: Array,
    rung: Rung,
) -> Array:
    state_hidden = jax.nn.silu(
        jnp.einsum("...ad,dh->...ah", state, params["state_kernel"])
        + params["state_bias"]
    )
    action_hidden = params["action_embedding"][action]
    local = state_hidden + action_hidden
    if rung is Rung.X0_LOCAL:
        context = jnp.zeros_like(local)
    elif rung is Rung.X1_JOINT_ACTION:
        context = _leave_one_out(action_hidden, valid)
    else:
        context = _leave_one_out(state_hidden + action_hidden, valid)
    fused = jax.nn.silu(
        jnp.einsum(
            "...ad,dh->...ah",
            jnp.concatenate([local, context], -1),
            params["mix_kernel"],
        )
        + params["mix_bias"]
    )
    return (
        jnp.einsum("...ah,hd->...ad", fused, params["output_kernel"])
        + params["output_bias"]
    )


def residual_predictor(
    base_params: Mapping[str, Array],
    residual_params: Mapping[str, Array],
    belief: Array,
    expanded_state: Array,
    action: Array,
    valid: Array,
    rung: Rung,
) -> Array:
    if rung is Rung.X0_LOCAL:
        raise ValueError("X0 is the frozen base, not an expanded residual rung")
    base = jax.lax.stop_gradient(
        predictor(base_params, belief, action, valid, Rung.X0_LOCAL)
    )
    belief_hidden = jax.nn.silu(
        jnp.einsum(
            "...ad,dh->...ah", belief, base_params["state_kernel"]
        )
        + base_params["state_bias"]
    )
    state_hidden = jax.nn.silu(
        jnp.einsum(
            "...ad,dh->...ah", expanded_state, base_params["state_kernel"]
        )
        + base_params["state_bias"]
    )
    action_hidden = base_params["action_embedding"][action]
    local = belief_hidden + action_hidden
    if rung is Rung.X1_JOINT_ACTION:
        context = _leave_one_out(action_hidden, valid)
    else:
        context = _leave_one_out(state_hidden + action_hidden, valid)
    hidden = jax.nn.silu(
        jnp.einsum(
            "...ad,dh->...ah",
            jnp.concatenate([local, context], -1),
            residual_params["mix_kernel"],
        )
        + residual_params["mix_bias"]
    )
    residual = (
        jnp.einsum(
            "...ah,hd->...ad", hidden, residual_params["output_kernel"]
        )
        + residual_params["output_bias"]
    )
    return base + residual


def cosine_error(prediction: Array, target: Array) -> Array:
    prediction /= jnp.sqrt(jnp.square(prediction).sum(-1, keepdims=True) + 1e-8)
    target /= jnp.sqrt(jnp.square(target).sum(-1, keepdims=True) + 1e-8)
    return 1.0 - (prediction * target).sum(-1)


def _state_for(dataset: PreparedDataset, rung: Rung) -> np.ndarray:
    if rung in {Rung.X0_LOCAL, Rung.X1_JOINT_ACTION, Rung.X2_JOINT_BELIEF}:
        return dataset.belief
    if rung is Rung.X3_JOINT_OBSERVATION and dataset.observation is not None:
        return dataset.observation
    if rung is Rung.X4_ORACLE_STATE and dataset.oracle is not None:
        return dataset.oracle
    raise ValueError(f"dataset cannot provide inputs for {rung.value}")


def train_predictor(
    dataset: PreparedDataset,
    rows: np.ndarray,
    rung: Rung,
    config: LadderConfig,
) -> tuple[dict[str, Array], list[dict[str, float]]]:
    params = init_predictor(
        jax.random.key(config.seed),
        config.feature_width,
        dataset.action_dim,
        config.hidden,
        dataset.target.shape[-1],
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.grad_clip),
        optax.adam(config.learning_rate),
    )
    opt_state = optimizer.init(params)

    @jax.jit
    def update(params, opt_state, state, action, target, valid):
        def objective(current):
            prediction = predictor(current, state, action, valid, rung)
            errors = cosine_error(prediction, target)
            denominator = jnp.maximum(valid.sum(), 1)
            return (errors * valid).sum() / denominator

        loss, gradients = jax.value_and_grad(objective)(params)
        updates, opt_state = optimizer.update(gradients, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    state = _state_for(dataset, rung)
    generator = np.random.default_rng(config.seed)
    batch = min(config.batch_trajectories, rows.size)
    history = []
    for step in range(config.steps):
        selected = generator.choice(rows, batch, replace=False)
        params, opt_state, loss = update(
            params,
            opt_state,
            jnp.asarray(state[selected]),
            jnp.asarray(dataset.action[selected]),
            jnp.asarray(dataset.target[selected]),
            jnp.asarray(dataset.valid[selected]),
        )
        if step == 0 or step == config.steps - 1 or (step + 1) % 100 == 0:
            history.append({"step": step + 1, "loss": float(loss)})
    return params, history


def train_residual_predictor(
    base_params: Mapping[str, Array],
    dataset: PreparedDataset,
    rows: np.ndarray,
    rung: Rung,
    config: LadderConfig,
) -> tuple[dict[str, Array], list[dict[str, float]]]:
    if rung is Rung.X0_LOCAL:
        raise ValueError("cannot train an X0 residual")
    params = init_residual(
        jax.random.key(config.seed + 10_000),
        config.hidden,
        dataset.target.shape[-1],
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.grad_clip),
        optax.adam(config.learning_rate),
    )
    opt_state = optimizer.init(params)

    @jax.jit
    def update(params, opt_state, belief, state, action, target, valid):
        def objective(current):
            prediction = residual_predictor(
                base_params, current, belief, state, action, valid, rung
            )
            errors = cosine_error(prediction, target)
            return (errors * valid).sum() / jnp.maximum(valid.sum(), 1)

        loss, gradients = jax.value_and_grad(objective)(params)
        updates, opt_state = optimizer.update(gradients, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    expanded_state = _state_for(dataset, rung)
    generator = np.random.default_rng(config.seed + 10_000)
    batch = min(config.batch_trajectories, rows.size)
    history = []
    for step in range(config.steps):
        selected = generator.choice(rows, batch, replace=False)
        params, opt_state, loss = update(
            params,
            opt_state,
            jnp.asarray(dataset.belief[selected]),
            jnp.asarray(expanded_state[selected]),
            jnp.asarray(dataset.action[selected]),
            jnp.asarray(dataset.target[selected]),
            jnp.asarray(dataset.valid[selected]),
        )
        if step == 0 or step == config.steps - 1 or (step + 1) % 100 == 0:
            history.append({"step": step + 1, "loss": float(loss)})
    return params, history


def evaluate_predictor(
    params: Mapping[str, Array],
    dataset: PreparedDataset,
    rows: np.ndarray,
    rung: Rung,
) -> dict[str, np.ndarray]:
    state = _state_for(dataset, rung)[rows]
    prediction = predictor(
        params,
        jnp.asarray(state),
        jnp.asarray(dataset.action[rows]),
        jnp.asarray(dataset.valid[rows]),
        rung,
    )
    return {
        "error": np.asarray(cosine_error(prediction, jnp.asarray(dataset.target[rows]))),
        "valid": dataset.valid[rows],
        "reward_event": dataset.reward_event[rows],
        "trajectory_id": np.broadcast_to(
            dataset.trajectory_id[rows, None, None], dataset.valid[rows].shape
        ),
        "agent": np.broadcast_to(
            np.arange(dataset.valid.shape[-1])[None, None], dataset.valid[rows].shape
        ),
    }


def evaluate_residual_predictor(
    base_params: Mapping[str, Array],
    residual_params: Mapping[str, Array],
    dataset: PreparedDataset,
    rows: np.ndarray,
    rung: Rung,
) -> dict[str, np.ndarray]:
    prediction = residual_predictor(
        base_params,
        residual_params,
        jnp.asarray(dataset.belief[rows]),
        jnp.asarray(_state_for(dataset, rung)[rows]),
        jnp.asarray(dataset.action[rows]),
        jnp.asarray(dataset.valid[rows]),
        rung,
    )
    return {
        "error": np.asarray(
            cosine_error(prediction, jnp.asarray(dataset.target[rows]))
        ),
        "valid": dataset.valid[rows],
        "reward_event": dataset.reward_event[rows],
        "trajectory_id": np.broadcast_to(
            dataset.trajectory_id[rows, None, None], dataset.valid[rows].shape
        ),
        "agent": np.broadcast_to(
            np.arange(dataset.valid.shape[-1])[None, None],
            dataset.valid[rows].shape,
        ),
    }


def summarize_errors(
    evaluation: Mapping[str, np.ndarray],
    *,
    seed: int,
    bootstrap_samples: int,
) -> dict[str, object]:
    valid = np.asarray(evaluation["valid"], bool)
    errors = np.asarray(evaluation["error"])[valid]
    trajectories = np.asarray(evaluation["trajectory_id"])[valid]
    reward_event = np.asarray(evaluation["reward_event"])[valid]
    agents = np.asarray(evaluation["agent"])[valid]
    result: dict[str, object] = {
        "overall": trajectory_bootstrap_interval(
            errors,
            trajectories,
            seed=seed,
            samples=bootstrap_samples,
        ),
        "by_agent": {
            str(agent): float(errors[agents == agent].mean())
            for agent in np.unique(agents)
        },
    }
    if reward_event.any() and (~reward_event).any():
        result["reward_event"] = trajectory_bootstrap_interval(
            errors[reward_event],
            trajectories[reward_event],
            seed=seed + 1,
            samples=bootstrap_samples,
        )
        result["non_reward"] = trajectory_bootstrap_interval(
            errors[~reward_event],
            trajectories[~reward_event],
            seed=seed + 2,
            samples=bootstrap_samples,
        )
    return result


def summarize_delta(
    previous: Mapping[str, np.ndarray],
    current: Mapping[str, np.ndarray],
    *,
    seed: int,
    bootstrap_samples: int,
) -> dict[str, float]:
    if not np.array_equal(previous["valid"], current["valid"]):
        raise ValueError("rung evaluations have different validity masks")
    valid = np.asarray(current["valid"], bool)
    improvement = (
        np.asarray(previous["error"])[valid] - np.asarray(current["error"])[valid]
    )
    trajectory = np.asarray(current["trajectory_id"])[valid]
    return trajectory_bootstrap_interval(
        improvement,
        trajectory,
        seed=seed,
        samples=bootstrap_samples,
    )
