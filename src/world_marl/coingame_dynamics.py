"""Discrete CoinGame dynamics modeling.

This module is the clean first world-model milestone:

    p(next_joint_state | state, joint_action)

The state is the native JaxMARL CoinGame vector observation. Each agent observes
a flattened ``3 x 3 x 4`` one-hot grid, so the model predicts each next entity
cell with a categorical head instead of treating grid positions as continuous
numbers.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.training.train_state import TrainState

from world_marl.evaluation import PolicyFn

GRID_SIZE = 3
NUM_CELLS = GRID_SIZE * GRID_SIZE
NUM_ENTITIES = 4
DEFAULT_ACTION_DIM = 5

# JaxMARL CoinGame actions: right, left, up, down, stay.
MOVES = np.asarray(
  [
    [0, 1],
    [0, -1],
    [1, 0],
    [-1, 0],
    [0, 0],
  ],
  dtype=np.int32,
)


@dataclass(frozen=True)
class CoinDynamicsDataset:
  """Collected CoinGame transition rows."""

  observations: np.ndarray
  actions: np.ndarray
  rewards: np.ndarray
  dones: np.ndarray
  next_observations: np.ndarray
  action_dim: int
  num_agents: int
  num_envs: int
  rollout_steps: int

  @property
  def num_transitions(self) -> int:
    return int(self.actions.shape[0])


@dataclass(frozen=True)
class CoinDynamicsData:
  """CoinGame transitions decoded into entity-cell targets."""

  positions: np.ndarray
  actions: np.ndarray
  next_positions: np.ndarray
  rewards: np.ndarray
  dones: np.ndarray
  action_dim: int = DEFAULT_ACTION_DIM
  num_agents: int = 2

  @property
  def num_transitions(self) -> int:
    return int(self.actions.shape[0])


@dataclass(frozen=True)
class CoinDynamicsConfig:
  """Training config for the categorical CoinGame dynamics model."""

  hidden_dims: tuple[int, ...] = (256, 256)
  learning_rate: float = 1e-3
  batch_size: int = 256
  train_steps: int = 1000
  max_grad_norm: float = 1.0


@dataclass(frozen=True)
class CoinDynamicsPredictions:
  """Model logits for next entity positions."""

  next_position_logits: np.ndarray

  @property
  def next_positions(self) -> np.ndarray:
    return np.argmax(self.next_position_logits, axis=-1).astype(np.int32)


class DiscreteCoinDynamicsModel(nn.Module):
  """Predict next CoinGame entity cells from state and joint action."""

  num_agents: int = 2
  action_dim: int = DEFAULT_ACTION_DIM
  hidden_dims: tuple[int, ...] = (256, 256)

  @nn.compact
  def __call__(self, positions: jax.Array, actions: jax.Array) -> jax.Array:
    position_features = jax.nn.one_hot(
      positions.astype(jnp.int32),
      NUM_CELLS,
    ).reshape((positions.shape[0], -1))
    action_features = jax.nn.one_hot(
      actions.astype(jnp.int32),
      self.action_dim,
    ).reshape((actions.shape[0], -1))
    x = jnp.concatenate([position_features, action_features], axis=-1)
    for hidden_dim in self.hidden_dims:
      x = nn.Dense(hidden_dim)(x)
      x = nn.silu(x)
    logits = nn.Dense(self.num_agents * NUM_ENTITIES * NUM_CELLS)(x)
    return logits.reshape(
      (positions.shape[0], self.num_agents, NUM_ENTITIES, NUM_CELLS)
    )


def collect_coin_dynamics_dataset(
  adapter,
  rng: np.random.Generator,
  *,
  rollout_steps: int,
  policy_fn: PolicyFn | None = None,
  progress_callback: Callable[[int], None] | None = None,
) -> CoinDynamicsDataset:
  """Collect ``(state, joint_action, reward, done, next_state)`` transitions."""
  if rollout_steps < 1:
    raise ValueError("rollout_steps must be >= 1")

  observations = adapter.reset()
  obs_rows: list[np.ndarray] = []
  action_rows: list[np.ndarray] = []
  reward_rows: list[np.ndarray] = []
  done_rows: list[np.ndarray] = []
  next_obs_rows: list[np.ndarray] = []
  expected_shape = (adapter.num_envs, adapter.num_agents)

  for step_index in range(rollout_steps):
    if policy_fn is None:
      actions = adapter.sample_actions(rng)
    else:
      actions = np.asarray(policy_fn(observations), dtype=np.int32)
    if actions.shape != expected_shape:
      raise ValueError(f"actions must have shape {expected_shape}, got {actions.shape}")

    step = adapter.step(actions)
    obs_rows.append(np.asarray(observations, dtype=np.float32).copy())
    action_rows.append(actions.copy())
    reward_rows.append(step.rewards.copy())
    done_rows.append(step.dones.copy())
    next_obs_rows.append(np.asarray(step.observations, dtype=np.float32).copy())
    observations = step.observations
    if progress_callback is not None:
      progress_callback(step_index + 1)

  return CoinDynamicsDataset(
    observations=np.concatenate(obs_rows, axis=0).astype(np.float32),
    actions=np.concatenate(action_rows, axis=0).astype(np.int32),
    rewards=np.concatenate(reward_rows, axis=0).astype(np.float32),
    dones=np.concatenate(done_rows, axis=0).astype(np.float32),
    next_observations=np.concatenate(next_obs_rows, axis=0).astype(np.float32),
    action_dim=int(adapter.action_dim),
    num_agents=int(adapter.num_agents),
    num_envs=int(adapter.num_envs),
    rollout_steps=int(rollout_steps),
  )


def prepare_coin_dynamics_data(dataset: CoinDynamicsDataset) -> CoinDynamicsData:
  """Decode native CoinGame vector observations into categorical positions."""
  if dataset.num_agents != 2 or dataset.action_dim != DEFAULT_ACTION_DIM:
    raise ValueError("CoinGame dynamics currently expects two agents and five actions")
  if tuple(dataset.observations.shape[1:]) != (2, 36):
    raise ValueError(
      "CoinGame dynamics expects observations shaped [N, 2, 36], "
      f"got {dataset.observations.shape}"
    )
  return CoinDynamicsData(
    positions=decode_coin_positions(dataset.observations),
    actions=dataset.actions,
    next_positions=decode_coin_positions(dataset.next_observations),
    rewards=dataset.rewards,
    dones=dataset.dones,
    action_dim=dataset.action_dim,
    num_agents=dataset.num_agents,
  )


def split_coin_dynamics_data(
  data: CoinDynamicsData,
  *,
  validation_fraction: float,
  seed: int,
) -> tuple[CoinDynamicsData, CoinDynamicsData]:
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
  return _take_coin_dynamics(data, train_indices), _take_coin_dynamics(
    data, validation_indices
  )


def decode_coin_positions(observations: np.ndarray) -> np.ndarray:
  """Decode ``[N, 2, 36]`` or ``[N, 2, 3, 3, 4]`` observations to cell ids."""
  observations = np.asarray(observations, dtype=np.float32)
  if observations.ndim == 3 and observations.shape[-1] == 36:
    grids = observations.reshape(
      (observations.shape[0], observations.shape[1], GRID_SIZE, GRID_SIZE, NUM_ENTITIES)
    )
  elif observations.ndim == 5 and observations.shape[2:] == (
    GRID_SIZE,
    GRID_SIZE,
    NUM_ENTITIES,
  ):
    grids = observations
  else:
    raise ValueError(
      "expected CoinGame observations shaped [N, 2, 36] or [N, 2, 3, 3, 4], "
      f"got {observations.shape}"
    )
  flat = grids.reshape((grids.shape[0], grids.shape[1], NUM_CELLS, NUM_ENTITIES))
  return np.argmax(flat, axis=2).astype(np.int32)


def encode_coin_positions(positions: np.ndarray) -> np.ndarray:
  """Encode entity-cell ids back into flattened CoinGame observations."""
  positions = np.asarray(positions, dtype=np.int32)
  if positions.ndim != 3 or positions.shape[1:] != (2, NUM_ENTITIES):
    raise ValueError(f"expected positions shaped [N, 2, 4], got {positions.shape}")
  if np.any(positions < 0) or np.any(positions >= NUM_CELLS):
    raise ValueError("CoinGame positions must be in [0, 8]")
  observations = np.zeros((positions.shape[0], 2, NUM_CELLS, NUM_ENTITIES), dtype=np.float32)
  rows = np.arange(positions.shape[0])[:, None, None]
  agents = np.arange(2)[None, :, None]
  entities = np.arange(NUM_ENTITIES)[None, None, :]
  observations[rows, agents, positions, entities] = 1.0
  return observations.reshape((positions.shape[0], 2, 36))


def create_coin_dynamics_train_state(
  rng: jax.Array,
  *,
  config: CoinDynamicsConfig,
  num_agents: int = 2,
  action_dim: int = DEFAULT_ACTION_DIM,
) -> TrainState:
  """Initialize a discrete dynamics train state."""
  model = DiscreteCoinDynamicsModel(
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


def coin_dynamics_loss(
  params: Any,
  apply_fn: Any,
  positions: jax.Array,
  actions: jax.Array,
  next_positions: jax.Array,
) -> tuple[jax.Array, dict[str, jax.Array]]:
  """Categorical next-position loss and accuracy metrics."""
  logits = apply_fn({"params": params}, positions, actions)
  per_entity_ce = optax.softmax_cross_entropy_with_integer_labels(
    logits,
    next_positions,
  )
  predicted = jnp.argmax(logits, axis=-1)
  entity_accuracy = jnp.mean((predicted == next_positions).astype(jnp.float32))
  full_exact = jnp.mean(
    jnp.all(predicted == next_positions, axis=(1, 2)).astype(jnp.float32)
  )
  loss = jnp.mean(per_entity_ce)
  return loss, {
    "loss": loss,
    "position_cross_entropy": loss,
    "entity_accuracy": entity_accuracy,
    "full_state_exact_accuracy": full_exact,
  }


@jax.jit
def coin_dynamics_train_step(
  state: TrainState,
  positions: jax.Array,
  actions: jax.Array,
  next_positions: jax.Array,
) -> tuple[TrainState, dict[str, jax.Array]]:
  """Run one optimizer update on an already sampled minibatch."""

  def loss_fn(params):
    return coin_dynamics_loss(params, state.apply_fn, positions, actions, next_positions)

  (_loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
  return state.apply_gradients(grads=grads), metrics


def train_coin_dynamics_model(
  rng: jax.Array,
  train_data: CoinDynamicsData,
  *,
  config: CoinDynamicsConfig,
  progress_callback: Callable[[int, dict[str, float]], None] | None = None,
) -> tuple[TrainState, list[dict[str, float]]]:
  """Train the categorical dynamics model."""
  if config.train_steps < 1:
    raise ValueError("train_steps must be >= 1")
  if config.batch_size < 1:
    raise ValueError("batch_size must be >= 1")

  rng, init_key = jax.random.split(rng)
  state = create_coin_dynamics_train_state(
    init_key,
    config=config,
    num_agents=train_data.num_agents,
    action_dim=train_data.action_dim,
  )
  positions = jnp.asarray(train_data.positions, dtype=jnp.int32)
  actions = jnp.asarray(train_data.actions, dtype=jnp.int32)
  next_positions = jnp.asarray(train_data.next_positions, dtype=jnp.int32)
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
    state, metrics = coin_dynamics_train_step(
      state,
      positions[indices],
      actions[indices],
      next_positions[indices],
    )
    row = {key: float(value) for key, value in metrics.items()}
    rows.append(row)
    if progress_callback is not None:
      progress_callback(step_index + 1, row)

  jax.block_until_ready(state.params)
  return state, rows


def predict_coin_dynamics(
  train_state: TrainState,
  data: CoinDynamicsData,
) -> CoinDynamicsPredictions:
  """Predict next-position logits for a dataset."""
  logits = train_state.apply_fn(
    {"params": train_state.params},
    jnp.asarray(data.positions, dtype=jnp.int32),
    jnp.asarray(data.actions, dtype=jnp.int32),
  )
  return CoinDynamicsPredictions(next_position_logits=np.asarray(logits, dtype=np.float32))


def evaluate_coin_dynamics(
  train_data: CoinDynamicsData,
  validation_data: CoinDynamicsData,
  predictions: CoinDynamicsPredictions,
) -> dict[str, Any]:
  """Evaluate next-joint-state prediction against simple baselines."""
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
  return {
    "position_cross_entropy": categorical_cross_entropy_positions(
      predictions.next_position_logits,
      target,
    ),
    "entity_accuracy": float(entity_matches.mean()),
    "agent_exact_accuracy": np.all(entity_matches, axis=2).mean(axis=0).astype(float).tolist(),
    "full_state_exact_accuracy": float(full_matches.mean()),
    "deterministic_transition_fraction": float(deterministic_mask.mean()),
    "deterministic_entity_accuracy": masked_mean(entity_matches, deterministic_mask),
    "deterministic_full_state_exact_accuracy": masked_mean(full_matches, deterministic_mask),
    "stochastic_transition_fraction": float(stochastic_mask.mean()),
    "stochastic_full_state_exact_accuracy": masked_mean(full_matches, stochastic_mask),
    "marginal_full_state_exact_accuracy": float(marginal_exact.mean()),
    "persistence_full_state_exact_accuracy": float(persistence_exact.mean()),
    "analytic_deterministic_full_state_exact_accuracy": masked_mean(
      analytic_matches,
      deterministic_mask,
    ),
    "model_beats_marginal": bool(full_matches.mean() > marginal_exact.mean()),
    "model_beats_persistence": bool(full_matches.mean() > persistence_exact.mean()),
    "valid_prediction_fraction": valid_position_fraction(predicted),
    "reward_event_fraction": float(np.any(np.abs(validation_data.rewards) > 1e-6, axis=1).mean()),
    "done_fraction": float(np.any(validation_data.dones > 0.0, axis=1).mean()),
  }


def summarize_coin_dynamics_outcome(
  metrics: dict[str, Any],
  *,
  finite_losses: bool,
  reload_passed: bool,
  min_deterministic_exact: float,
) -> tuple[bool, dict[str, bool]]:
  """Build pass/fail criteria for the first dynamics milestone."""
  deterministic_exact = metrics["deterministic_full_state_exact_accuracy"]
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
  }
  return all(criteria.values()), criteria


def categorical_cross_entropy_positions(logits: np.ndarray, targets: np.ndarray) -> float:
  """Mean CE for ``[..., class]`` logits and integer position targets."""
  logits = np.asarray(logits, dtype=np.float64)
  targets = np.asarray(targets, dtype=np.int32)
  shifted = logits - logits.max(axis=-1, keepdims=True)
  log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
  rows = np.arange(targets.shape[0])[:, None, None]
  agents = np.arange(targets.shape[1])[None, :, None]
  entities = np.arange(targets.shape[2])[None, None, :]
  selected = log_probs[rows, agents, entities, targets]
  return float(-selected.mean())


def position_marginal_modes(next_positions: np.ndarray) -> np.ndarray:
  """Most frequent next cell for each agent/entity in the training data."""
  next_positions = np.asarray(next_positions, dtype=np.int32)
  modes = np.zeros(next_positions.shape[1:], dtype=np.int32)
  for agent in range(next_positions.shape[1]):
    for entity in range(next_positions.shape[2]):
      counts = np.bincount(next_positions[:, agent, entity], minlength=NUM_CELLS)
      modes[agent, entity] = int(np.argmax(counts))
  return modes


def deterministic_transition_mask(data: CoinDynamicsData) -> np.ndarray:
  """Transitions whose next state is deterministic from state/action alone."""
  collected = coin_collected(data.positions, data.actions)
  done = np.any(data.dones > 0.0, axis=1)
  return np.logical_not(np.logical_or(collected, done))


def coin_collected(positions: np.ndarray, actions: np.ndarray) -> np.ndarray:
  """Return whether either player collects either coin after applying actions."""
  agent0 = np.asarray(positions, dtype=np.int32)[:, 0]
  actions = np.asarray(actions, dtype=np.int32)
  red_pos = cell_to_row_col(agent0[:, 0])
  blue_pos = cell_to_row_col(agent0[:, 1])
  red_coin = cell_to_row_col(agent0[:, 2])
  blue_coin = cell_to_row_col(agent0[:, 3])
  new_red = (red_pos + MOVES[actions[:, 0]]) % GRID_SIZE
  new_blue = (blue_pos + MOVES[actions[:, 1]]) % GRID_SIZE
  return (
    np.all(new_red == red_coin, axis=-1)
    | np.all(new_red == blue_coin, axis=-1)
    | np.all(new_blue == red_coin, axis=-1)
    | np.all(new_blue == blue_coin, axis=-1)
  )


def expected_deterministic_next_positions(
  positions: np.ndarray,
  actions: np.ndarray,
) -> np.ndarray:
  """Next positions when no coin respawn/reset randomness is involved."""
  agent0 = np.asarray(positions, dtype=np.int32)[:, 0]
  actions = np.asarray(actions, dtype=np.int32)
  red_pos = cell_to_row_col(agent0[:, 0])
  blue_pos = cell_to_row_col(agent0[:, 1])
  new_red = row_col_to_cell((red_pos + MOVES[actions[:, 0]]) % GRID_SIZE)
  new_blue = row_col_to_cell((blue_pos + MOVES[actions[:, 1]]) % GRID_SIZE)
  red_coin = agent0[:, 2]
  blue_coin = agent0[:, 3]
  expected_agent0 = np.stack([new_red, new_blue, red_coin, blue_coin], axis=1)
  expected_agent1 = np.stack([new_blue, new_red, blue_coin, red_coin], axis=1)
  return np.stack([expected_agent0, expected_agent1], axis=1).astype(np.int32)


def sample_predictions(
  data: CoinDynamicsData,
  predictions: CoinDynamicsPredictions,
  *,
  count: int,
) -> list[dict[str, Any]]:
  """Small JSON-friendly prediction table."""
  count = min(max(0, count), data.num_transitions)
  predicted = predictions.next_positions
  rows = []
  deterministic = deterministic_transition_mask(data)
  for index in range(count):
    rows.append(
      {
        "index": index,
        "state_positions": data.positions[index].astype(int).tolist(),
        "joint_action": data.actions[index].astype(int).tolist(),
        "target_next_positions": data.next_positions[index].astype(int).tolist(),
        "predicted_next_positions": predicted[index].astype(int).tolist(),
        "exact": bool(np.array_equal(predicted[index], data.next_positions[index])),
        "deterministic_transition": bool(deterministic[index]),
        "rewards": data.rewards[index].astype(float).tolist(),
        "dones": data.dones[index].astype(float).tolist(),
      }
    )
  return rows


def valid_position_fraction(positions: np.ndarray) -> float:
  positions = np.asarray(positions, dtype=np.int32)
  return float(((positions >= 0) & (positions < NUM_CELLS)).mean())


def masked_mean(values: np.ndarray, mask: np.ndarray) -> float | None:
  values = np.asarray(values)
  mask = np.asarray(mask, dtype=bool)
  if not bool(mask.any()):
    return None
  return float(values[mask].mean())


def cell_to_row_col(cells: np.ndarray) -> np.ndarray:
  cells = np.asarray(cells, dtype=np.int32)
  return np.stack([cells // GRID_SIZE, cells % GRID_SIZE], axis=-1)


def row_col_to_cell(row_col: np.ndarray) -> np.ndarray:
  row_col = np.asarray(row_col, dtype=np.int32)
  return row_col[..., 0] * GRID_SIZE + row_col[..., 1]


def _take_coin_dynamics(data: CoinDynamicsData, indices: np.ndarray) -> CoinDynamicsData:
  return CoinDynamicsData(
    positions=data.positions[indices],
    actions=data.actions[indices],
    next_positions=data.next_positions[indices],
    rewards=data.rewards[indices],
    dones=data.dones[indices],
    action_dim=data.action_dim,
    num_agents=data.num_agents,
  )
