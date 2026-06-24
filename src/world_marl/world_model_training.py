"""Helpers for collecting vector-state batches for prefit world models."""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState

from flow_matching.train import topk_checkpoint_merge, wsd_block_size_schedule
from world_marl.envs.meltingpot_adapter import MeltingPotVectorAdapter
from world_marl.training import build_central_observations
from world_marl.world_model import (
    VectorTransitionBatch,
    _apply_vector_policy,
    _num_factors,
    train_world_model_step,
)


def flatten_state_observations(observations: np.ndarray) -> np.ndarray:
    """Flatten local observations while preserving env and agent axes."""
    observations = np.asarray(observations, dtype=np.float32)
    if observations.ndim < 3:
        raise ValueError("expected observations shaped [env, agent, ...]")
    return observations.reshape((observations.shape[0], observations.shape[1], -1))


def collect_random_transition_batch(
    adapter: MeltingPotVectorAdapter,
    observations: np.ndarray,
    rng: np.random.Generator,
    *,
    rollout_steps: int,
) -> tuple[VectorTransitionBatch, np.ndarray, jnp.ndarray]:
    """Collect vector-state transitions using adapter-sampled random actions."""
    if rollout_steps < 1:
        raise ValueError("rollout_steps must be >= 1")

    current_observations = observations
    rows = _TransitionRows()
    for _ in range(rollout_steps):
        states = flatten_state_observations(current_observations)
        actions = adapter.sample_actions(rng)
        step = adapter.step(actions)
        rows.append(
            states=states,
            actions=actions,
            next_states=flatten_state_observations(step.observations),
            rewards=step.rewards,
            dones=step.dones,
        )
        current_observations = step.observations

    batch = rows.to_batch()
    return batch, current_observations, batch.states


def collect_policy_transition_batch(
    adapter: MeltingPotVectorAdapter,
    train_state: TrainState,
    observations: np.ndarray,
    rng: jax.Array,
    *,
    rollout_steps: int,
    algorithm: str,
) -> tuple[VectorTransitionBatch, np.ndarray, jax.Array, jnp.ndarray]:
    """Collect vector-state transitions using the current IPPO/MAPPO policy."""
    if rollout_steps < 1:
        raise ValueError("rollout_steps must be >= 1")
    if algorithm not in {"ippo", "mappo"}:
        raise ValueError(f"unsupported algorithm {algorithm!r}")

    current_observations = observations
    rows = _TransitionRows()
    for _ in range(rollout_steps):
        states = flatten_state_observations(current_observations)
        flat_states = states.reshape((adapter.num_envs * adapter.num_agents, -1))
        rng, action_key = jax.random.split(rng)
        central_states = (
            jnp.asarray(
                build_central_observations(
                    states,
                    observation_mode="vector",
                ).reshape((adapter.num_envs * adapter.num_agents, -1))
            )
            if algorithm == "mappo"
            else None
        )
        policy, _ = _apply_vector_policy(
            train_state, jnp.asarray(flat_states), central_states
        )
        actions = np.asarray(policy.sample(seed=action_key), dtype=np.int32).reshape(
            (adapter.num_envs, adapter.num_agents)
        )
        step = adapter.step(actions)
        rows.append(
            states=states,
            actions=actions,
            next_states=flatten_state_observations(step.observations),
            rewards=step.rewards,
            dones=step.dones,
        )
        current_observations = step.observations

    batch = rows.to_batch()
    return batch, current_observations, rng, batch.states


def concatenate_transition_batches(
    batches: Sequence[VectorTransitionBatch],
) -> VectorTransitionBatch:
    """Concatenate non-empty transition batches along the batch dimension."""
    if not batches:
        raise ValueError("expected at least one transition batch")
    return VectorTransitionBatch(
        states=jnp.concatenate([batch.states for batch in batches], axis=0),
        actions=jnp.concatenate([batch.actions for batch in batches], axis=0),
        next_states=jnp.concatenate([batch.next_states for batch in batches], axis=0),
        rewards=jnp.concatenate([batch.rewards for batch in batches], axis=0),
        dones=jnp.concatenate([batch.dones for batch in batches], axis=0),
    )


def fit_world_model_steps(
    model_state: TrainState,
    rng: jax.Array,
    batch: VectorTransitionBatch,
    config,
    *,
    steps: int,
) -> tuple[TrainState, jax.Array, jnp.ndarray, jnp.ndarray]:
    """Run full-batch world-model fitting steps.

    Returns the updated state, the advanced rng, the final step loss, and the
    per-step loss history (length ``steps``) for plotting fit convergence.

    For ``flow_type == "llada2"`` with ``config.wsd_merge_k > 1`` the run is split
    into ``wsd_merge_k`` contiguous segments sharing one global WSD curriculum; the
    params after each segment are kept and the lowest-loss half are averaged via
    Weight-Space Merge (§4.3, ``topk_checkpoint_merge``) into the returned state.
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")

    merge_k = config.wsd_merge_k if config.flow_type == "llada2" else 1
    if merge_k <= 1:
        model_state, rng, loss_history = _fit_world_model_updates(
            model_state, rng, batch, config, steps=steps
        )
        return model_state, rng, loss_history[-1], loss_history

    seg = max(1, steps // merge_k)
    snapshots: list[tuple[float, object]] = []
    histories: list[jnp.ndarray] = []
    offset = 0
    for c in range(merge_k):
        n = steps - offset if c == merge_k - 1 else seg
        model_state, rng, hist = _fit_world_model_updates(
            model_state,
            rng,
            batch,
            config,
            steps=n,
            step_offset=offset,
            total_steps=steps,
        )
        snapshots.append((float(hist[-1]), model_state.params))
        histories.append(hist)
        offset += n

    snapshots.sort(key=lambda item: item[0])
    top = max(1, merge_k // 2)
    merged = topk_checkpoint_merge([params for _, params in snapshots[:top]])
    model_state = model_state.replace(params=merged)
    loss_history = jnp.concatenate(histories)
    return model_state, rng, loss_history[-1], loss_history


@partial(
    jax.jit, static_argnames=("config", "steps", "step_offset", "total_steps")
)
def _fit_world_model_updates(
    model_state: TrainState,
    rng: jax.Array,
    batch: VectorTransitionBatch,
    config,
    *,
    steps: int,
    step_offset: int = 0,
    total_steps: int | None = None,
) -> tuple[TrainState, jax.Array, jnp.ndarray]:
    """Fused full-batch fitting: one ``lax.scan`` step per gradient update.

    The carry is ``(model_state, rng)`` and ``scan`` stacks each step's loss into
    the returned history. For ``flow_type == "llada2"`` each step is also handed
    its WSD block size (§4.1) and an annealed §7.1 masked-embedding noise std,
    indexed by a *global* step (``step_offset`` + local) so the warmup/stable/decay
    curriculum stays continuous across a segmented (checkpoint-merged) run.
    """
    if config.flow_type == "llada2":
        total = steps if total_steps is None else total_steps
        d = _num_factors(config)
        divisors = tuple(s for s in range(1, d + 1) if d % s == 0)
        block_sizes = jnp.asarray(
            [
                wsd_block_size_schedule(
                    step_offset + s,
                    total,
                    divisors=divisors,
                    warmup_frac=config.wsd_warmup_frac,
                    stable_frac=config.wsd_stable_frac,
                )
                for s in range(steps)
            ],
            dtype=jnp.int32,
        )
        global_steps = jnp.arange(
            step_offset, step_offset + steps, dtype=jnp.float32
        )
        noise_iters = config.masked_embed_noise_iters

        def llada2_update(carry, xs):
            state, rng = carry
            block_size, gstep = xs
            rng, fit_key, noise_key = jax.random.split(rng, 3)
            if noise_iters > 0:
                std = config.mask_noise_std * jnp.maximum(
                    0.0, 1.0 - gstep / noise_iters
                )
            else:
                std = 0.0
            state, loss = train_world_model_step(
                state,
                fit_key,
                batch,
                config,
                block_size=block_size,
                mask_noise_std=std,
                noise_rng=noise_key,
            )
            return (state, rng), loss

        (model_state, rng), loss_history = jax.lax.scan(
            llada2_update, (model_state, rng), (block_sizes, global_steps)
        )
        return model_state, rng, loss_history

    def update(carry, _):
        state, rng = carry
        rng, fit_key = jax.random.split(rng)
        state, loss = train_world_model_step(state, fit_key, batch, config)
        return (state, rng), loss

    (model_state, rng), loss_history = jax.lax.scan(
        update, (model_state, rng), xs=None, length=steps
    )
    return model_state, rng, loss_history


def sample_initial_states(
    states: jnp.ndarray,
    rng: jax.Array,
    *,
    num_envs: int,
) -> jnp.ndarray:
    """Sample model-rollout initial states from a collected state pool."""
    if states.shape[0] < 1:
        raise ValueError("expected at least one collected state")
    indices = jax.random.randint(rng, (num_envs,), minval=0, maxval=states.shape[0])
    return states[indices]


class _TransitionRows:
    def __init__(self) -> None:
        self.states: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.next_states: list[np.ndarray] = []
        self.rewards: list[np.ndarray] = []
        self.dones: list[np.ndarray] = []

    def append(
        self,
        *,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        self.states.append(states)
        self.actions.append(np.asarray(actions, dtype=np.int32))
        self.next_states.append(next_states)
        self.rewards.append(np.asarray(rewards, dtype=np.float32))
        self.dones.append(np.asarray(dones, dtype=np.float32))

    def to_batch(self) -> VectorTransitionBatch:
        states = np.concatenate(self.states, axis=0)
        return VectorTransitionBatch(
            states=jnp.asarray(states, dtype=jnp.float32),
            actions=jnp.asarray(np.concatenate(self.actions, axis=0), dtype=jnp.int32),
            next_states=jnp.asarray(
                np.concatenate(self.next_states, axis=0),
                dtype=jnp.float32,
            ),
            rewards=jnp.asarray(
                np.concatenate(self.rewards, axis=0),
                dtype=jnp.float32,
            ),
            dones=jnp.asarray(np.concatenate(self.dones, axis=0), dtype=jnp.float32),
        )
