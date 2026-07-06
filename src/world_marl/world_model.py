"""Vector-state world-model glue for model-based PPO rollouts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from flow_matching.llada2 import (
    BlockDiffusionTransformer,
    create_llada2_train_state,
    llada2_bdlm_loss,
    llada2_train_step,
    sample_llada2_block_diffusion,
)
from flow_matching.models import (
    MLPVectorField,
    TokenizedDiscreteDenoiser,
    TokenizedDiscreteTransformer,
)
from flow_matching.simulate import (
    sample_marginal_discrete_flow_model,
    sample_marginal_flow_model,
)
from flow_matching.train import (
    conditioned_discrete_flow_matching_loss,
    conditioned_discrete_train_step,
    conditioned_flow_matching_loss,
    conditioned_train_step,
    create_conditioned_train_state,
    create_discrete_conditioned_train_state,
)
from world_marl.algs.ippo import RolloutBatch
from world_marl.algs.mappo import MAPPORolloutBatch
from world_marl.training import RolloutResult, build_vector_central

# (states, env_actions, next_states) -> (rewards, dones), each [env, agent].
RewardDoneFn = Callable[
    [jnp.ndarray, jnp.ndarray, jnp.ndarray],
    tuple[jnp.ndarray, jnp.ndarray],
]


class VectorTransitionBatch(NamedTuple):
    states: jnp.ndarray
    actions: jnp.ndarray
    next_states: jnp.ndarray
    rewards: jnp.ndarray
    dones: jnp.ndarray


@dataclass(frozen=True)
class VectorWorldModelConfig:
    state_dim: int
    num_agents: int
    action_dim: int
    hidden_dims: tuple[int, ...] = (128, 128)
    learning_rate: float = 1e-3
    integration_steps: int = 8
    flow_type: str = "gaussian"
    # Categorical cardinality V for discrete flow matching (0 = continuous). For
    # CoinGame each (3, 3, 4) grid is 4 channels one-hot over 9 cells, so V = 9.
    num_categories: int = 0
    # Discrete denoiser architecture (only consulted when flow_type == "discrete").
    discrete_arch: str = "mlp"
    # Transformer-arm capacity, shared by the discrete-transformer and llada2 arms
    # (frozen dataclass -> all fields are hashable, safe as static jit args).
    model_dim: int = 64
    num_heads: int = 4
    ffn_hidden_dims: tuple[int, ...] = (256, 256)


@dataclass(frozen=True)
class LLaDA2WorldModelConfig(VectorWorldModelConfig):
    """LLaDA2.0 block-diffusion arm knobs; required when ``flow_type == "llada2"``.

    num_layers and the per-layer MoE expert width derive from the inherited
    ffn_hidden_dims, so capacity has a single source of truth across the
    transformer arms. Stays frozen/hashable like the base (static jit arg).
    """

    flow_type: str = "llada2"
    block_size: int = 4
    num_experts: int = 1  # 1 -> dense SwiGLU (no router); >1 -> MoE top-k
    expert_top_k: int = 1
    mask_schedule: str = "linear"
    alpha_min: float = 0.15  # mask-rate band lower bound (§5.1)
    alpha_max: float = 0.95  # mask-rate band upper bound (§5.1)
    cap_lambda: float = 0.1  # CAP confidence-loss weight (eq 6)
    moe_aux_coeff: float = 0.01  # load-balancing aux weight
    complementary_masking: bool = True  # §5.1 complementary pair
    confidence_threshold: float = 0.9  # §5.4 sampler commit threshold
    steps_per_block: int = 4  # §5.4 refinement passes per block
    rope_base: float = 10000.0
    rope_scaling: float = 1.0  # YaRN inference scaling (1.0 = off)
    rope_original_max_position: int = 32
    masked_embed_noise_iters: int = 0  # §7.1 stabilizer length (0 = from-scratch)
    mask_noise_std: float = 0.1  # §7.1 noise std (annealed to 0 over the iters)
    wsd_enabled: bool = True  # §4.1 curriculum on; False -> constant block_size
    wsd_warmup_frac: float = 0.3  # §4.1 block-size curriculum phase fractions
    wsd_stable_frac: float = 0.4
    wsd_merge_k: int = 1  # §4.3 top-k checkpoint merge (1 = single, no merge)


def create_world_model_state(
    key: jax.Array,
    config: VectorWorldModelConfig,
) -> TrainState:
    if config.flow_type == "llada2":
        if not isinstance(config, LLaDA2WorldModelConfig):
            raise TypeError("flow_type 'llada2' requires an LLaDA2WorldModelConfig")
        model = BlockDiffusionTransformer(
            num_categories=config.num_categories,
            num_actions=config.action_dim,
            model_dim=config.model_dim,
            num_heads=config.num_heads,
            num_layers=len(config.ffn_hidden_dims),
            ffn_hidden_dim=config.ffn_hidden_dims[0],
            num_experts=config.num_experts,
            expert_top_k=config.expert_top_k,
            rope_base=config.rope_base,
            rope_scaling=config.rope_scaling,
            rope_original_max_position=config.rope_original_max_position,
        )
        return create_llada2_train_state(
            key,
            model,
            config.learning_rate,
            num_factors=_num_factors(config),
            num_action_tokens=config.num_agents,
        )
    if config.flow_type == "discrete":
        if config.discrete_arch == "transformer":
            model = TokenizedDiscreteTransformer(
                num_categories=config.num_categories,
                model_dim=config.model_dim,
                num_heads=config.num_heads,
                ffn_hidden_dims=config.ffn_hidden_dims,
            )
        else:
            model = TokenizedDiscreteDenoiser(
                num_categories=config.num_categories,
                hidden_dims=config.hidden_dims,
            )
        return create_discrete_conditioned_train_state(
            key,
            model,
            config.learning_rate,
            num_factors=_num_factors(config),
            cond_dim=_cond_dim(config),
        )
    model = MLPVectorField(hidden_dims=config.hidden_dims)
    return create_conditioned_train_state(
        key,
        model,
        config.learning_rate,
        dim=_transition_dim(config),
        cond_dim=_cond_dim(config),
    )


def world_model_loss(
    params: Any,
    apply_fn: Any,
    key: jax.Array,
    batch: VectorTransitionBatch,
    config: VectorWorldModelConfig,
) -> jnp.ndarray:
    if config.flow_type == "llada2":
        if not isinstance(config, LLaDA2WorldModelConfig):
            raise TypeError("flow_type 'llada2' requires an LLaDA2WorldModelConfig")
        prev_tokens = _pack_discrete_tokens(batch.states, config)
        x0 = _pack_discrete_tokens(batch.next_states, config)
        return llada2_bdlm_loss(
            params,
            apply_fn,
            key,
            x0,
            prev_tokens,
            batch.actions,
            config.num_categories,
            block_size=config.block_size,
            mask_schedule_name=config.mask_schedule,
            alpha_min=config.alpha_min,
            alpha_max=config.alpha_max,
            complementary=config.complementary_masking,
            cap_lambda=config.cap_lambda,
            moe_aux_coeff=config.moe_aux_coeff,
        )
    cond_vars = _pack_cond_vars(batch.states, batch.actions, config)
    if config.flow_type == "discrete":
        z = _pack_discrete_tokens(batch.next_states, config)
        return conditioned_discrete_flow_matching_loss(
            params, apply_fn, key, z, cond_vars, config.num_categories
        )
    x1 = _pack_transition(batch.next_states, config)
    return conditioned_flow_matching_loss(
        params, apply_fn, key, x1, cond_vars, config.flow_type
    )


@partial(jax.jit, static_argnames="config")
def train_world_model_step(
    state: TrainState,
    key: jax.Array,
    batch: VectorTransitionBatch,
    config: VectorWorldModelConfig,
    *,
    block_size: jax.Array | None = None,
    mask_noise_std: jax.Array | float = 0.0,
    noise_rng: jax.Array | None = None,
) -> tuple[TrainState, jnp.ndarray]:
    """One world-model gradient step.

    ``block_size``/``mask_noise_std``/``noise_rng`` are only consulted for the
    ``llada2`` flow; they stay *traced* (not static) so the WSD curriculum can thread
    a per-step block size and an annealed §7.1 noise std through one fused ``scan``
    with no recompiles. ``block_size is None`` (the default for the other flows) is a
    structural check on absence, never a value comparison.
    """
    if config.flow_type == "llada2":
        if not isinstance(config, LLaDA2WorldModelConfig):
            raise TypeError("flow_type 'llada2' requires an LLaDA2WorldModelConfig")
        prev_tokens = _pack_discrete_tokens(batch.states, config)
        x0 = _pack_discrete_tokens(batch.next_states, config)
        bs = config.block_size if block_size is None else block_size
        return llada2_train_step(
            state,
            key,
            x0,
            prev_tokens,
            batch.actions,
            config.num_categories,
            block_size=bs,
            mask_schedule_name=config.mask_schedule,
            alpha_min=config.alpha_min,
            alpha_max=config.alpha_max,
            complementary=config.complementary_masking,
            cap_lambda=config.cap_lambda,
            moe_aux_coeff=config.moe_aux_coeff,
            mask_noise_std=mask_noise_std,
            noise_rng=noise_rng,
        )
    cond_vars = _pack_cond_vars(batch.states, batch.actions, config)
    if config.flow_type == "discrete":
        z = _pack_discrete_tokens(batch.next_states, config)
        return conditioned_discrete_train_step(
            state, key, z, cond_vars, config.num_categories
        )
    x1 = _pack_transition(batch.next_states, config)
    return conditioned_train_step(state, key, x1, cond_vars, config.flow_type)


def predict_next(
    state: TrainState,
    key: jax.Array,
    states: jnp.ndarray,
    actions: jnp.ndarray,
    config: VectorWorldModelConfig,
) -> jnp.ndarray:
    """Sample next-states from the conditioned flow (next-state only)."""
    if config.flow_type == "llada2":
        if not isinstance(config, LLaDA2WorldModelConfig):
            raise TypeError("flow_type 'llada2' requires an LLaDA2WorldModelConfig")
        prev_tokens = _pack_discrete_tokens(states, config)
        tokens = sample_llada2_block_diffusion(
            state.apply_fn,
            state.params,
            key,
            prev_tokens,
            actions,
            num_factors=_num_factors(config),
            num_categories=config.num_categories,
            block_size=config.block_size,
            steps_per_block=config.steps_per_block,
            confidence_threshold=config.confidence_threshold,
        )
        return _unpack_discrete_onehot(tokens, config)
    cond_vars = _pack_cond_vars(states, actions, config)
    if config.flow_type == "discrete":
        tokens = sample_marginal_discrete_flow_model(
            state.apply_fn,
            state.params,
            key,
            cond_vars,
            num_factors=_num_factors(config),
            num_categories=config.num_categories,
            steps=config.integration_steps,
        )
        return _unpack_discrete_onehot(tokens, config)
    transition = sample_marginal_flow_model(
        state.apply_fn,
        state.params,
        key,
        cond_vars,
        dim=_transition_dim(config),
        steps=config.integration_steps,
    )
    return _unpack_transition(transition, config)


def simulate_ippo_model_rollout(
    model_state: TrainState,
    policy_state: TrainState,
    initial_states: jnp.ndarray,
    rng: jax.Array,
    *,
    rollout_steps: int,
    config: VectorWorldModelConfig,
    reward_done_fn: RewardDoneFn,
) -> RolloutResult:
    return _simulate_model_rollout(
        model_state,
        policy_state,
        initial_states,
        rng,
        rollout_steps=rollout_steps,
        config=config,
        algorithm="ippo",
        reward_done_fn=reward_done_fn,
    )


def simulate_mappo_model_rollout(
    model_state: TrainState,
    policy_state: TrainState,
    initial_states: jnp.ndarray,
    rng: jax.Array,
    *,
    rollout_steps: int,
    config: VectorWorldModelConfig,
    reward_done_fn: RewardDoneFn,
) -> RolloutResult:
    return _simulate_model_rollout(
        model_state,
        policy_state,
        initial_states,
        rng,
        rollout_steps=rollout_steps,
        config=config,
        algorithm="mappo",
        reward_done_fn=reward_done_fn,
    )


def _simulate_model_rollout(
    model_state: TrainState,
    policy_state: TrainState,
    initial_states: jnp.ndarray,
    rng: jax.Array,
    *,
    rollout_steps: int,
    config: VectorWorldModelConfig,
    algorithm: str,
    reward_done_fn: RewardDoneFn,
) -> RolloutResult:
    if rollout_steps < 1:
        raise ValueError("rollout_steps must be >= 1")
    if algorithm not in {"ippo", "mappo"}:
        raise ValueError(f"unsupported algorithm {algorithm!r}")
    is_mappo = algorithm == "mappo"

    stacked, final_states, last_values = _imagined_rollout(
        model_state,
        policy_state,
        initial_states,
        rng,
        rollout_steps=rollout_steps,
        config=config,
        is_mappo=is_mappo,
        reward_done_fn=reward_done_fn,
    )

    common = {
        "observations": stacked["observations"],
        "actions": stacked["actions"],
        "log_probs": stacked["log_probs"],
        "rewards": stacked["rewards"],
        "dones": stacked["dones"],
        "values": stacked["values"],
    }
    if is_mappo:
        batch = MAPPORolloutBatch(
            central_observations=stacked["central_observations"],
            **common,
        )
    else:
        batch = RolloutBatch(**common)
    # float() metrics stay on the host, outside the jitted scan above.
    mean_reward = float(jnp.mean(batch.rewards))
    return RolloutResult(
        batch=batch,
        next_observations=final_states,
        last_values=last_values,
        metrics={
            "rollout_mean_reward": mean_reward,
            "model_rollout_mean_reward": mean_reward,
        },
    )


@partial(
    jax.jit,
    static_argnames=("rollout_steps", "config", "is_mappo", "reward_done_fn"),
)
def _imagined_rollout(
    model_state: TrainState,
    policy_state: TrainState,
    initial_states: jnp.ndarray,
    rng: jax.Array,
    *,
    rollout_steps: int,
    config: VectorWorldModelConfig,
    is_mappo: bool,
    reward_done_fn: RewardDoneFn,
) -> tuple[dict[str, jnp.ndarray], jnp.ndarray, jnp.ndarray]:
    """Fused imagined rollout: one ``lax.scan`` step per imagined timestep. This prevents interdevice unloading and loading that would otherwise slow down the runtime substantially and keeps all training on the GPU.

    The carry is ``(rng, current_states)`` and ``scan`` stacks every per-step
    transition along axis 0.``.
    This is safe:
    - ``reward_done_fn`` is a static argument because it is a plain callable, not a
    pytree
    - a module-level provider such as ``coin_game_reward_done`` keeps a
    stable identity, so this compiles once and reuses the cache across every PPO
    update
    - The inner Euler integrator in ``predict_next`` is itself a
    ``lax.scan``, so the two nest.

    """
    num_envs = initial_states.shape[0]
    num_actors = num_envs * config.num_agents

    def step(carry, _):
        rng, current_states = carry
        flat_states = current_states.reshape((num_actors, config.state_dim))
        central_states = (
            build_vector_central(current_states, jnp).reshape((num_actors, -1))
            if is_mappo
            else None
        )
        rng, action_key, model_key = jax.random.split(rng, 3)
        # Distribution over actions and value estimates from the current policy.
        policy, values = _apply_vector_policy(policy_state, flat_states, central_states)
        actions = policy.sample(seed=action_key).astype(jnp.int32)
        log_probs = policy.log_prob(actions)
        env_actions = actions.reshape((num_envs, config.num_agents))
        # World model supplies next-states; rewards/dones come from the callback.
        next_states = predict_next(
            model_state, model_key, current_states, env_actions, config
        )
        rewards, dones = _reward_done(
            reward_done_fn, current_states, env_actions, next_states
        )
        outputs = {
            "observations": flat_states,
            "actions": actions,
            "log_probs": log_probs,
            "rewards": rewards.reshape((num_actors,)),
            "dones": dones.reshape((num_actors,)),
            "values": values,
        }
        if is_mappo:
            outputs["central_observations"] = central_states
        return (rng, next_states), outputs

    (rng, final_states), stacked = jax.lax.scan(
        step, (rng, initial_states), xs=None, length=rollout_steps
    )

    last_flat = final_states.reshape((num_actors, config.state_dim))
    last_central = (
        build_vector_central(final_states, jnp).reshape((num_actors, -1))
        if is_mappo
        else None
    )
    last_values = _apply_vector_policy(policy_state, last_flat, last_central)[1]
    return stacked, final_states, last_values


def _apply_vector_policy(
    policy_state: TrainState,
    flat_states: jnp.ndarray,
    central_states: jnp.ndarray | None,
) -> tuple[Any, jnp.ndarray]:
    """Apply an MLP policy, passing central observations only for MAPPO."""
    if central_states is None:
        return policy_state.apply_fn({"params": policy_state.params}, flat_states)
    return policy_state.apply_fn(
        {"params": policy_state.params}, flat_states, central_states
    )


def _reward_done(
    reward_done_fn: RewardDoneFn,
    states: jnp.ndarray,
    env_actions: jnp.ndarray,
    next_states: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Rewards/dones from the callback. None is not acceptable"""
    rewards, dones = reward_done_fn(states, env_actions, next_states)
    return (
        jnp.asarray(rewards, dtype=jnp.float32),
        jnp.asarray(dones, dtype=jnp.float32),
    )


def _pack_cond_vars(
    states: jnp.ndarray,
    actions: jnp.ndarray,
    config: VectorWorldModelConfig,
) -> jnp.ndarray:
    flat_actions_dim = config.num_agents * config.action_dim
    flat_states = states.reshape((states.shape[0], _flat_state_dim(config)))
    action_features = jax.nn.one_hot(actions, config.action_dim).reshape(
        (actions.shape[0], flat_actions_dim)
    )
    return jnp.concatenate([flat_states, action_features], axis=-1)


def _pack_transition(
    next_states: jnp.ndarray,
    config: VectorWorldModelConfig,
) -> jnp.ndarray:
    return next_states.reshape((next_states.shape[0], _flat_state_dim(config)))


def _unpack_transition(
    transition: jnp.ndarray,
    config: VectorWorldModelConfig,
) -> jnp.ndarray:
    return transition.reshape(
        (transition.shape[0], config.num_agents, config.state_dim)
    )


def _channels_per_agent(config: VectorWorldModelConfig) -> int:
    """Categorical groups inside one agent's grid (CoinGame: 4 channels)."""
    return config.state_dim // config.num_categories


def _num_factors(config: VectorWorldModelConfig) -> int:
    """Total categorical factors d packed across agents (CoinGame: 2*4 = 8)."""
    return config.num_agents * _channels_per_agent(config)


def _pack_discrete_tokens(
    next_states: jnp.ndarray,
    config: VectorWorldModelConfig,
) -> jnp.ndarray:
    """Decode one-hot grids ``(B, A, state_dim)`` to integer tokens ``(B, d)``.

    CoinGame flattens each ``(3, 3, 4)`` grid in C-order, so the layout is
    position-major / channel-minor: index ``= pos*C + ch``. Reshaping to
    ``(B, A, V, C)`` therefore puts positions on axis -2 and channels on axis -1,
    and ``argmax`` over the V positions recovers each channel's occupied cell --
    the same decode ``coin_game_reward_done`` trusts. A plain contiguous
    ``reshape(B, d, V)`` would scramble channels into positions.
    """
    num_categories = config.num_categories
    channels = _channels_per_agent(config)
    grid = next_states.reshape(
        (next_states.shape[0], config.num_agents, num_categories, channels)
    )
    tokens = jnp.argmax(grid, axis=2)  # (B, A, C): occupied position per channel
    return tokens.reshape((next_states.shape[0], _num_factors(config)))


def _unpack_discrete_onehot(
    tokens: jnp.ndarray,
    config: VectorWorldModelConfig,
) -> jnp.ndarray:
    """Inverse of :func:`_pack_discrete_tokens`: tokens ``(B, d)`` -> one-hot grids.

    Rebuilds the strided position-major / channel-minor layout so the returned
    ``(B, A, state_dim)`` one-hot floats are byte-compatible with env states.
    """
    num_categories = config.num_categories
    channels = _channels_per_agent(config)
    per_agent = tokens.reshape((tokens.shape[0], config.num_agents, channels))
    onehot = jax.nn.one_hot(per_agent, num_categories)  # (B, A, C, V)
    grid = jnp.swapaxes(onehot, -1, -2)  # (B, A, V, C): position-major
    return grid.reshape((tokens.shape[0], config.num_agents, config.state_dim))


def _flat_state_dim(config: VectorWorldModelConfig) -> int:
    return config.num_agents * config.state_dim


def _transition_dim(config: VectorWorldModelConfig) -> int:
    return _flat_state_dim(config)


def _cond_dim(config: VectorWorldModelConfig) -> int:
    return _flat_state_dim(config) + config.num_agents * config.action_dim
