"""Small categorical next-state baseline for JaxMARL CoinGame diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.training.train_state import TrainState
from jaxmarl.environments.coin_game.coin_game import MOVES

from world_marl.envs.jaxmarl_coin_adapter import coin_game_reward_done
from world_marl.world_model import VectorTransitionBatch

GRID_SIZE = 3
NUM_CELLS = GRID_SIZE * GRID_SIZE
NUM_ENTITIES = 4
DEFAULT_ACTION_DIM = 5


@dataclass(frozen=True)
class SoftmaxBaselineData:
    """CoinGame transitions decoded into categorical entity-cell targets."""

    states: np.ndarray
    positions: np.ndarray
    actions: np.ndarray
    next_states: np.ndarray
    next_positions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    action_dim: int = DEFAULT_ACTION_DIM
    num_agents: int = 2

    @property
    def num_transitions(self) -> int:
        return int(self.actions.shape[0])


@dataclass(frozen=True)
class SoftmaxBaselineConfig:
    """Training config for the categorical next-state baseline."""

    hidden_dims: tuple[int, ...] = (256, 256)
    learning_rate: float = 1e-3
    batch_size: int = 256
    train_steps: int = 1000
    max_grad_norm: float = 1.0
    stochastic_target_weight: float = 32.0


@dataclass(frozen=True)
class SoftmaxBaselinePredictions:
    """Model logits for next entity positions."""

    next_position_logits: np.ndarray

    @property
    def next_positions(self) -> np.ndarray:
        return np.argmax(self.next_position_logits, axis=-1).astype(np.int32)


@dataclass(frozen=True)
class SoftmaxBaselineTargets:
    """Categorical supervision for next entity positions."""

    distributions: np.ndarray
    weights: np.ndarray

    @property
    def entropy(self) -> float:
        clipped = np.clip(self.distributions, 1e-12, 1.0)
        return float(-np.sum(self.distributions * np.log(clipped), axis=-1).mean())


class DiscreteCoinSoftmaxModel(nn.Module):
    """MLP predicting categorical next-cell logits for each entity."""

    num_agents: int
    action_dim: int
    hidden_dims: tuple[int, ...]

    @nn.compact
    def __call__(self, positions: jax.Array, actions: jax.Array) -> jax.Array:
        batch_size = positions.shape[0]
        position_features = jax.nn.one_hot(positions, NUM_CELLS).reshape(
            (batch_size, -1)
        )
        action_features = jax.nn.one_hot(actions, self.action_dim).reshape(
            (batch_size, -1)
        )
        x = jnp.concatenate([position_features, action_features], axis=-1)
        for width in self.hidden_dims:
            x = nn.relu(nn.Dense(width)(x))
        logits = nn.Dense(self.num_agents * NUM_ENTITIES * NUM_CELLS)(x)
        return logits.reshape((batch_size, self.num_agents, NUM_ENTITIES, NUM_CELLS))


def prepare_softmax_data(
    batch: VectorTransitionBatch,
    *,
    action_dim: int = DEFAULT_ACTION_DIM,
) -> SoftmaxBaselineData:
    """Decode a shared vector transition batch for the categorical baseline."""
    states = np.asarray(batch.states, dtype=np.float32)
    actions = np.asarray(batch.actions, dtype=np.int32)
    next_states = np.asarray(batch.next_states, dtype=np.float32)
    rewards = np.asarray(batch.rewards, dtype=np.float32)
    dones = np.asarray(batch.dones, dtype=np.float32)

    if states.ndim != 3 or tuple(states.shape[1:]) != (2, 36):
        raise ValueError(
            "CoinGame softmax baseline expects states shaped [N, 2, 36], "
            f"got {states.shape}"
        )
    if next_states.shape != states.shape:
        raise ValueError(
            f"next state shape mismatch: {next_states.shape} vs {states.shape}"
        )
    if actions.shape != states.shape[:2]:
        raise ValueError(
            f"action shape mismatch: {actions.shape} vs {states.shape[:2]}"
        )
    if rewards.shape != states.shape[:2] or dones.shape != states.shape[:2]:
        raise ValueError("rewards and dones must be shaped [N, 2]")
    if action_dim != DEFAULT_ACTION_DIM:
        raise ValueError("CoinGame softmax baseline currently expects five actions")

    return SoftmaxBaselineData(
        states=states,
        positions=decode_coin_positions(states),
        actions=actions,
        next_states=next_states,
        next_positions=decode_coin_positions(next_states),
        rewards=rewards,
        dones=dones,
        action_dim=action_dim,
        num_agents=states.shape[1],
    )


def split_softmax_data(
    data: SoftmaxBaselineData,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[SoftmaxBaselineData, SoftmaxBaselineData]:
    """Shuffle transitions into train and validation splits."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if data.num_transitions < 2:
        raise ValueError("at least two transitions are required")

    validation_size = int(round(data.num_transitions * validation_fraction))
    validation_size = min(max(1, validation_size), data.num_transitions - 1)
    indices = np.random.default_rng(seed).permutation(data.num_transitions)
    validation_indices = indices[:validation_size]
    train_indices = indices[validation_size:]
    return _take_softmax_data(data, train_indices), _take_softmax_data(
        data,
        validation_indices,
    )


def decode_coin_positions(observations: np.ndarray) -> np.ndarray:
    """Decode flat CoinGame vectors into entity cell ids.

    The return shape is ``[transition, agent, entity]``. Entity order is
    red player, blue player, red coin, blue coin in each agent's observation
    frame.
    """
    observations = np.asarray(observations, dtype=np.float32)
    if observations.ndim != 3 or observations.shape[1:] != (2, 36):
        raise ValueError(
            f"CoinGame observations must be shaped [N, 2, 36], got {observations.shape}"
        )
    grids = observations.reshape((observations.shape[0], 2, GRID_SIZE, GRID_SIZE, 4))
    flat = grids.reshape((observations.shape[0], 2, NUM_CELLS, 4))
    return np.argmax(flat, axis=2).astype(np.int32)


def encode_coin_positions(positions: np.ndarray) -> np.ndarray:
    """Encode entity cell ids into flat CoinGame vector observations."""
    positions = np.asarray(positions, dtype=np.int32)
    if positions.ndim != 3 or positions.shape[1:] != (2, NUM_ENTITIES):
        raise ValueError(f"positions must be shaped [N, 2, 4], got {positions.shape}")
    if np.any((positions < 0) | (positions >= NUM_CELLS)):
        raise ValueError("positions must be valid grid cell ids")

    observations = np.zeros(
        (positions.shape[0], 2, NUM_CELLS, NUM_ENTITIES),
        dtype=np.float32,
    )
    transition_index = np.arange(positions.shape[0])[:, None, None]
    agent_index = np.arange(2)[None, :, None]
    entity_index = np.arange(NUM_ENTITIES)[None, None, :]
    observations[transition_index, agent_index, positions, entity_index] = 1.0
    return observations.reshape((positions.shape[0], 2, 36))


def softmax_target_distributions(
    data: SoftmaxBaselineData,
    *,
    stochastic_target_weight: float = 1.0,
) -> SoftmaxBaselineTargets:
    """Build categorical targets for deterministic and stochastic transitions."""
    if stochastic_target_weight <= 0.0:
        raise ValueError("stochastic_target_weight must be positive")
    distributions = np.eye(NUM_CELLS, dtype=np.float32)[data.next_positions]
    weights = np.ones(data.next_positions.shape, dtype=np.float32)
    uniform = np.full((NUM_CELLS,), 1.0 / NUM_CELLS, dtype=np.float32)
    red_collected, blue_collected = collected_coin_masks(data.positions, data.actions)
    terminal = np.any(data.dones > 0.0, axis=1)
    nonterminal = ~terminal

    red_respawn = red_collected & nonterminal
    blue_respawn = blue_collected & nonterminal
    distributions[red_respawn, 0, 2, :] = uniform
    distributions[red_respawn, 1, 3, :] = uniform
    weights[red_respawn, 0, 2] = stochastic_target_weight
    weights[red_respawn, 1, 3] = stochastic_target_weight
    distributions[blue_respawn, 0, 3, :] = uniform
    distributions[blue_respawn, 1, 2, :] = uniform
    weights[blue_respawn, 0, 3] = stochastic_target_weight
    weights[blue_respawn, 1, 2] = stochastic_target_weight
    distributions[terminal, :, :, :] = uniform
    weights[terminal, :, :] = stochastic_target_weight
    return SoftmaxBaselineTargets(distributions=distributions, weights=weights)


def create_softmax_train_state(
    rng: jax.Array,
    *,
    config: SoftmaxBaselineConfig,
    num_agents: int = 2,
    action_dim: int = DEFAULT_ACTION_DIM,
) -> TrainState:
    """Initialize the categorical baseline train state."""
    model = DiscreteCoinSoftmaxModel(
        num_agents=num_agents,
        action_dim=action_dim,
        hidden_dims=config.hidden_dims,
    )
    params = model.init(
        rng,
        jnp.zeros((1, num_agents, NUM_ENTITIES), dtype=jnp.int32),
        jnp.zeros((1, num_agents), dtype=jnp.int32),
    )["params"]
    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(config.learning_rate),
        ),
    )


def softmax_baseline_loss(
    params: Any,
    apply_fn: Any,
    positions: jax.Array,
    actions: jax.Array,
    next_positions: jax.Array,
    target_distributions: jax.Array,
    target_weights: jax.Array,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Distributional next-position loss and sampled-label accuracy metrics."""
    logits = apply_fn({"params": params}, positions, actions)
    per_entity_ce = optax.softmax_cross_entropy(logits, target_distributions)
    weighted_ce = per_entity_ce * target_weights
    predicted = jnp.argmax(logits, axis=-1)
    entity_accuracy = jnp.mean((predicted == next_positions).astype(jnp.float32))
    full_exact = jnp.mean(
        jnp.all(predicted == next_positions, axis=(1, 2)).astype(jnp.float32)
    )
    weight_sum = jnp.maximum(jnp.sum(target_weights), 1.0)
    loss = jnp.sum(weighted_ce) / weight_sum
    per_entity_entropy = -jnp.sum(
        target_distributions * jnp.log(jnp.clip(target_distributions, 1e-12, 1.0)),
        axis=-1,
    )
    target_entropy = jnp.sum(per_entity_entropy * target_weights) / weight_sum
    return loss, {
        "loss": loss,
        "distributional_position_cross_entropy": loss,
        "target_distribution_entropy": target_entropy,
        "target_distribution_kl": loss - target_entropy,
        "entity_accuracy": entity_accuracy,
        "full_state_exact_accuracy": full_exact,
    }


@jax.jit
def softmax_baseline_train_step(
    state: TrainState,
    positions: jax.Array,
    actions: jax.Array,
    next_positions: jax.Array,
    target_distributions: jax.Array,
    target_weights: jax.Array,
) -> tuple[TrainState, dict[str, jax.Array]]:
    """Run one optimizer update on an already sampled minibatch."""

    def loss_fn(params):
        return softmax_baseline_loss(
            params,
            state.apply_fn,
            positions,
            actions,
            next_positions,
            target_distributions,
            target_weights,
        )

    (_loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    return state.apply_gradients(grads=grads), metrics


def train_softmax_baseline(
    rng: jax.Array,
    train_data: SoftmaxBaselineData,
    *,
    config: SoftmaxBaselineConfig,
    progress_callback: Callable[[int, dict[str, float]], None] | None = None,
) -> tuple[TrainState, list[dict[str, float]]]:
    """Train the categorical next-state baseline."""
    if config.train_steps < 1:
        raise ValueError("train_steps must be >= 1")
    if config.batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    rng, init_key = jax.random.split(rng)
    state = create_softmax_train_state(
        init_key,
        config=config,
        num_agents=train_data.num_agents,
        action_dim=train_data.action_dim,
    )
    positions = jnp.asarray(train_data.positions, dtype=jnp.int32)
    actions = jnp.asarray(train_data.actions, dtype=jnp.int32)
    next_positions = jnp.asarray(train_data.next_positions, dtype=jnp.int32)
    targets = softmax_target_distributions(
        train_data,
        stochastic_target_weight=config.stochastic_target_weight,
    )
    target_distributions = jnp.asarray(targets.distributions, dtype=jnp.float32)
    target_weights = jnp.asarray(targets.weights, dtype=jnp.float32)
    train_size = int(train_data.num_transitions)
    rows: list[dict[str, float]] = []

    for step_index in range(config.train_steps):
        rng, step_key = jax.random.split(rng)
        indices = jax.random.randint(
            step_key,
            shape=(config.batch_size,),
            minval=0,
            maxval=train_size,
        )
        state, metrics = softmax_baseline_train_step(
            state,
            positions[indices],
            actions[indices],
            next_positions[indices],
            target_distributions[indices],
            target_weights[indices],
        )
        row = {key: float(value) for key, value in metrics.items()}
        rows.append(row)
        if progress_callback is not None:
            progress_callback(step_index + 1, row)

    jax.block_until_ready(state.params)
    return state, rows


def predict_softmax_baseline(
    train_state: TrainState,
    data: SoftmaxBaselineData,
) -> SoftmaxBaselinePredictions:
    """Predict next-position logits for a dataset."""
    logits = train_state.apply_fn(
        {"params": train_state.params},
        jnp.asarray(data.positions, dtype=jnp.int32),
        jnp.asarray(data.actions, dtype=jnp.int32),
    )
    return SoftmaxBaselinePredictions(
        next_position_logits=np.asarray(logits, dtype=np.float32)
    )


def evaluate_softmax_baseline(
    train_data: SoftmaxBaselineData,
    validation_data: SoftmaxBaselineData,
    predictions: SoftmaxBaselinePredictions,
) -> dict[str, Any]:
    """Evaluate next-state prediction against simple categorical baselines."""
    predicted = predictions.next_positions
    target = validation_data.next_positions
    entity_matches = predicted == target
    full_matches = np.all(entity_matches, axis=(1, 2))
    deterministic_mask = deterministic_transition_mask(validation_data)
    stochastic_mask = ~deterministic_mask
    marginal_modes = position_marginal_modes(train_data.next_positions)
    marginal = np.repeat(
        marginal_modes.reshape(1, *marginal_modes.shape),
        validation_data.num_transitions,
        axis=0,
    )
    persistence = validation_data.positions
    expected_deterministic = expected_deterministic_next_positions(
        validation_data.positions,
        validation_data.actions,
    )
    analytic_matches = np.all(expected_deterministic == target, axis=(1, 2))
    marginal_exact = np.all(marginal == target, axis=(1, 2))
    persistence_exact = np.all(persistence == target, axis=(1, 2))
    respawn_metrics = stochastic_respawn_metrics(validation_data, predictions)
    target_distributions = softmax_target_distributions(validation_data)
    distributional_ce = distributional_cross_entropy_positions(
        predictions.next_position_logits,
        target_distributions.distributions,
    )
    return {
        "position_cross_entropy": categorical_cross_entropy_positions(
            predictions.next_position_logits,
            target,
        ),
        "distributional_position_cross_entropy": distributional_ce,
        "target_distribution_entropy": target_distributions.entropy,
        "target_distribution_kl": distributional_ce - target_distributions.entropy,
        "entity_accuracy": float(entity_matches.mean()),
        "agent_exact_accuracy": np.all(entity_matches, axis=2)
        .mean(axis=0)
        .astype(float)
        .tolist(),
        "full_state_exact_accuracy": float(full_matches.mean()),
        "deterministic_transition_fraction": float(deterministic_mask.mean()),
        "deterministic_entity_accuracy": masked_mean(
            entity_matches,
            deterministic_mask,
        ),
        "deterministic_full_state_exact_accuracy": masked_mean(
            full_matches,
            deterministic_mask,
        ),
        "stochastic_transition_fraction": float(stochastic_mask.mean()),
        "stochastic_full_state_exact_accuracy": masked_mean(
            full_matches,
            stochastic_mask,
        ),
        "marginal_full_state_exact_accuracy": float(marginal_exact.mean()),
        "persistence_full_state_exact_accuracy": float(persistence_exact.mean()),
        "analytic_deterministic_full_state_exact_accuracy": masked_mean(
            analytic_matches,
            deterministic_mask,
        ),
        "model_beats_marginal": bool(full_matches.mean() > marginal_exact.mean()),
        "model_beats_persistence": bool(full_matches.mean() > persistence_exact.mean()),
        "valid_prediction_fraction": valid_position_fraction(predicted),
        "reward": reward_prediction_metrics(validation_data),
        "respawn": respawn_metrics,
        "reward_event_fraction": float(
            np.any(np.abs(validation_data.rewards) > 1e-6, axis=1).mean()
        ),
        "done_fraction": float(np.any(validation_data.dones > 0.0, axis=1).mean()),
    }


def summarize_softmax_outcome(
    metrics: dict[str, Any],
    *,
    finite_losses: bool,
    reload_passed: bool,
    min_deterministic_exact: float,
    max_respawn_uniform_kl: float = 0.25,
) -> tuple[bool, dict[str, bool]]:
    """Build pass/fail criteria for the diagnostic baseline test."""
    deterministic_exact = metrics["deterministic_full_state_exact_accuracy"]
    respawn_kl = metrics["respawn"]["uniform_target_kl"]
    criteria = {
        "finite_losses": bool(finite_losses),
        "reload_passed": bool(reload_passed),
        "valid_predictions": bool(metrics["valid_prediction_fraction"] == 1.0),
        "beats_marginal": bool(metrics["model_beats_marginal"]),
        "beats_persistence": bool(metrics["model_beats_persistence"]),
        "deterministic_exact_high": bool(
            deterministic_exact is not None
            and deterministic_exact >= min_deterministic_exact
        ),
        "respawn_distribution_calibrated": bool(
            respawn_kl is None or respawn_kl <= max_respawn_uniform_kl
        ),
    }
    return all(criteria.values()), criteria


def categorical_cross_entropy_positions(
    logits: np.ndarray,
    targets: np.ndarray,
) -> float:
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int32)
    probs = softmax_np(logits, axis=-1)
    flat_targets = targets.reshape(-1)
    flat_probs = probs.reshape((-1, NUM_CELLS))
    selected = flat_probs[np.arange(flat_targets.shape[0]), flat_targets]
    return float(-np.log(np.clip(selected, 1e-12, 1.0)).mean())


def distributional_cross_entropy_positions(
    logits: np.ndarray,
    targets: np.ndarray,
) -> float:
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    probs = softmax_np(logits, axis=-1)
    return float(-(targets * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=-1).mean())


def position_marginal_modes(next_positions: np.ndarray) -> np.ndarray:
    modes = np.zeros(next_positions.shape[1:], dtype=np.int32)
    for index in np.ndindex(next_positions.shape[1:]):
        values = next_positions[(slice(None), *index)]
        counts = np.bincount(values, minlength=NUM_CELLS)
        modes[index] = int(np.argmax(counts))
    return modes


def deterministic_transition_mask(data: SoftmaxBaselineData) -> np.ndarray:
    """Transitions whose next state is deterministic from state/action alone."""
    collected = coin_collected(data.positions, data.actions)
    done = np.any(data.dones > 0.0, axis=1)
    return np.logical_not(np.logical_or(collected, done))


def reward_prediction_metrics(
    data: SoftmaxBaselineData,
    *,
    atol: float = 1e-6,
) -> dict[str, Any]:
    """Check environment reward helper consistency on the same transitions."""
    predicted_rewards, _ = coin_game_reward_done(
        jnp.asarray(data.states, dtype=jnp.float32),
        jnp.asarray(data.actions, dtype=jnp.int32),
        jnp.asarray(data.next_states, dtype=jnp.float32),
    )
    predicted = np.asarray(predicted_rewards, dtype=np.float32)
    target = np.asarray(data.rewards, dtype=np.float32)
    if predicted.shape != target.shape:
        raise ValueError(f"reward shape mismatch: {predicted.shape} vs {target.shape}")
    per_agent_exact = np.isclose(predicted, target, atol=atol)
    transition_exact = np.all(per_agent_exact, axis=1)
    nonterminal = ~np.any(data.dones > 0.0, axis=1)
    predicted_event = np.any(np.abs(predicted) > atol, axis=1)
    target_event = np.any(np.abs(target) > atol, axis=1)
    return {
        "mse": mse(predicted, target),
        "per_agent_exact_accuracy": float(per_agent_exact.mean()),
        "transition_exact_accuracy": float(transition_exact.mean()),
        "nonterminal_transition_exact_accuracy": masked_mean(
            transition_exact,
            nonterminal,
        ),
        "nonterminal_mse": masked_mse(predicted, target, nonterminal),
        "event_accuracy": float((predicted_event == target_event).mean()),
        "nonterminal_event_accuracy": masked_mean(
            predicted_event == target_event,
            nonterminal,
        ),
        "nonterminal_fraction": float(nonterminal.mean()),
    }


def stochastic_respawn_metrics(
    data: SoftmaxBaselineData,
    predictions: SoftmaxBaselinePredictions,
) -> dict[str, Any]:
    """Evaluate predicted distributions for random coin respawn targets."""
    logits = np.asarray(predictions.next_position_logits, dtype=np.float64)
    targets = np.asarray(data.next_positions, dtype=np.int32)
    nonterminal = ~np.any(data.dones > 0.0, axis=1)
    red_collected, blue_collected = collected_coin_masks(data.positions, data.actions)
    masks_and_entities = (
        (red_collected & nonterminal, 2, "red_coin"),
        (blue_collected & nonterminal, 3, "blue_coin"),
    )

    selected_logits = []
    selected_targets = []
    counts: dict[str, int] = {}
    for mask, entity_index, name in masks_and_entities:
        counts[f"{name}_respawn_count"] = int(mask.sum())
        if bool(mask.any()):
            selected_logits.append(logits[mask, 0, entity_index, :])
            selected_targets.append(targets[mask, 0, entity_index])

    if not selected_logits:
        return {
            "num_respawn_targets": 0,
            **counts,
            "cross_entropy": None,
            "uniform_cross_entropy": float(np.log(NUM_CELLS)),
            "top1_accuracy": None,
            "top3_accuracy": None,
            "mean_target_probability": None,
            "uniform_target_probability": 1.0 / NUM_CELLS,
            "mean_entropy": None,
            "uniform_entropy": float(np.log(NUM_CELLS)),
            "uniform_target_cross_entropy": None,
            "uniform_target_kl": None,
            "mean_distribution_tv_to_uniform": None,
            "aggregate_distribution_tv_to_uniform": None,
            "empirical_target_tv_to_uniform": None,
        }

    respawn_logits = np.concatenate(selected_logits, axis=0)
    respawn_targets = np.concatenate(selected_targets, axis=0)
    probs = softmax_np(respawn_logits, axis=-1)
    selected = probs[np.arange(respawn_targets.shape[0]), respawn_targets]
    sorted_indices = np.argsort(respawn_logits, axis=-1)
    top1 = sorted_indices[:, -1]
    top3 = sorted_indices[:, -3:]
    uniform = np.full((NUM_CELLS,), 1.0 / NUM_CELLS, dtype=np.float64)
    aggregate = probs.mean(axis=0)
    target_counts = np.bincount(respawn_targets, minlength=NUM_CELLS).astype(np.float64)
    empirical = target_counts / max(float(target_counts.sum()), 1.0)
    entropy = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=-1)
    uniform_target_ce = -np.sum(
        uniform.reshape(1, -1) * np.log(np.clip(probs, 1e-12, 1.0)),
        axis=-1,
    )
    return {
        "num_respawn_targets": int(respawn_targets.shape[0]),
        **counts,
        "cross_entropy": float(-np.log(np.clip(selected, 1e-12, 1.0)).mean()),
        "uniform_cross_entropy": float(np.log(NUM_CELLS)),
        "top1_accuracy": float((top1 == respawn_targets).mean()),
        "top3_accuracy": float((top3 == respawn_targets[:, None]).any(axis=1).mean()),
        "mean_target_probability": float(selected.mean()),
        "uniform_target_probability": 1.0 / NUM_CELLS,
        "mean_entropy": float(entropy.mean()),
        "uniform_entropy": float(np.log(NUM_CELLS)),
        "uniform_target_cross_entropy": float(uniform_target_ce.mean()),
        "uniform_target_kl": float(uniform_target_ce.mean() - np.log(NUM_CELLS)),
        "mean_distribution_tv_to_uniform": float(
            0.5 * np.abs(probs - uniform.reshape(1, -1)).sum(axis=1).mean()
        ),
        "aggregate_distribution_tv_to_uniform": float(
            0.5 * np.abs(aggregate - uniform).sum()
        ),
        "empirical_target_tv_to_uniform": float(
            0.5 * np.abs(empirical - uniform).sum()
        ),
    }


def coin_collected(positions: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Return whether either player collects either coin after applying actions."""
    red_collected, blue_collected = collected_coin_masks(positions, actions)
    return red_collected | blue_collected


def collected_coin_masks(
    positions: np.ndarray,
    actions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return masks for red-coin and blue-coin collection events."""
    moved = _moved_player_and_coin_positions(positions, actions)
    red_red = np.all(moved.red_player == moved.red_coin, axis=-1)
    red_blue = np.all(moved.red_player == moved.blue_coin, axis=-1)
    blue_red = np.all(moved.blue_player == moved.red_coin, axis=-1)
    blue_blue = np.all(moved.blue_player == moved.blue_coin, axis=-1)
    return red_red | blue_red, red_blue | blue_blue


def expected_deterministic_next_positions(
    positions: np.ndarray,
    actions: np.ndarray,
) -> np.ndarray:
    """Next positions when no coin respawn/reset randomness is involved."""
    agent0 = np.asarray(positions, dtype=np.int32)[:, 0]
    moved = _moved_player_and_coin_positions(positions, actions)
    new_red = row_col_to_cell(moved.red_player)
    new_blue = row_col_to_cell(moved.blue_player)
    red_coin = agent0[:, 2]
    blue_coin = agent0[:, 3]
    expected_agent0 = np.stack([new_red, new_blue, red_coin, blue_coin], axis=1)
    expected_agent1 = np.stack([new_blue, new_red, blue_coin, red_coin], axis=1)
    return np.stack([expected_agent0, expected_agent1], axis=1).astype(np.int32)


def sample_predictions(
    data: SoftmaxBaselineData,
    predictions: SoftmaxBaselinePredictions,
    *,
    count: int,
) -> list[dict[str, Any]]:
    """Small JSON-friendly prediction table."""
    count = min(max(0, count), data.num_transitions)
    predicted = predictions.next_positions
    reward_metrics = reward_prediction_metrics(data)
    rows = []
    deterministic = deterministic_transition_mask(data)
    for index in range(count):
        rows.append(
            {
                "index": index,
                "state_positions": data.positions[index].astype(int).tolist(),
                "joint_action": data.actions[index].astype(int).tolist(),
                "target_next_positions": data.next_positions[index]
                .astype(int)
                .tolist(),
                "predicted_next_positions": predicted[index].astype(int).tolist(),
                "exact": bool(
                    np.array_equal(predicted[index], data.next_positions[index])
                ),
                "deterministic_transition": bool(deterministic[index]),
                "rewards": data.rewards[index].astype(float).tolist(),
                "dones": data.dones[index].astype(float).tolist(),
            }
        )
    if rows:
        rows[0]["reward_helper_nonterminal_exact_accuracy"] = reward_metrics[
            "nonterminal_transition_exact_accuracy"
        ]
    return rows


def valid_position_fraction(positions: np.ndarray) -> float:
    positions = np.asarray(positions)
    valid = (positions >= 0) & (positions < NUM_CELLS)
    return float(valid.mean())


def masked_mean(values: np.ndarray, mask: np.ndarray) -> float | None:
    values = np.asarray(values)
    mask = np.asarray(mask, dtype=bool)
    if not bool(mask.any()):
        return None
    return float(values[mask].mean())


def mse(predicted: np.ndarray, target: np.ndarray) -> float:
    diff = np.asarray(predicted, dtype=np.float32) - np.asarray(
        target, dtype=np.float32
    )
    return float(np.square(diff).mean())


def masked_mse(
    predicted: np.ndarray, target: np.ndarray, mask: np.ndarray
) -> float | None:
    mask = np.asarray(mask, dtype=bool)
    if not bool(mask.any()):
        return None
    diff = np.asarray(predicted, dtype=np.float32) - np.asarray(
        target, dtype=np.float32
    )
    return float(np.square(diff[mask]).mean())


def softmax_np(logits: np.ndarray, *, axis: int = -1) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def cell_to_row_col(cells: np.ndarray) -> np.ndarray:
    cells = np.asarray(cells, dtype=np.int32)
    return np.stack([cells // GRID_SIZE, cells % GRID_SIZE], axis=-1)


def row_col_to_cell(row_col: np.ndarray) -> np.ndarray:
    row_col = np.asarray(row_col, dtype=np.int32)
    return (row_col[..., 0] % GRID_SIZE) * GRID_SIZE + (row_col[..., 1] % GRID_SIZE)


@dataclass(frozen=True)
class _MovedCoinPositions:
    red_player: np.ndarray
    blue_player: np.ndarray
    red_coin: np.ndarray
    blue_coin: np.ndarray


def _moved_player_and_coin_positions(
    positions: np.ndarray,
    actions: np.ndarray,
) -> _MovedCoinPositions:
    positions = np.asarray(positions, dtype=np.int32)
    actions = np.asarray(actions, dtype=np.int32)
    if positions.ndim != 3 or positions.shape[1:] != (2, NUM_ENTITIES):
        raise ValueError(f"positions must be shaped [N, 2, 4], got {positions.shape}")
    if actions.shape != positions.shape[:2]:
        raise ValueError(f"actions must be shaped [N, 2], got {actions.shape}")

    agent0 = positions[:, 0]
    red_player = cell_to_row_col(agent0[:, 0])
    blue_player = cell_to_row_col(agent0[:, 1])
    red_coin = cell_to_row_col(agent0[:, 2])
    blue_coin = cell_to_row_col(agent0[:, 3])
    moves = np.asarray(MOVES, dtype=np.int32)
    moved_red = (red_player + moves[actions[:, 0]]) % GRID_SIZE
    moved_blue = (blue_player + moves[actions[:, 1]]) % GRID_SIZE
    return _MovedCoinPositions(
        red_player=moved_red,
        blue_player=moved_blue,
        red_coin=red_coin,
        blue_coin=blue_coin,
    )


def _take_softmax_data(
    data: SoftmaxBaselineData,
    indices: np.ndarray,
) -> SoftmaxBaselineData:
    return SoftmaxBaselineData(
        states=data.states[indices],
        positions=data.positions[indices],
        actions=data.actions[indices],
        next_states=data.next_states[indices],
        next_positions=data.next_positions[indices],
        rewards=data.rewards[indices],
        dones=data.dones[indices],
        action_dim=data.action_dim,
        num_agents=data.num_agents,
    )
