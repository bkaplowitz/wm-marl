"""Config + create/train/sample dispatch for the generative world-model arms.

Three arms share one interface, all conditioned on (previous observation,
action) and predicting the next observation:

- ``discrete-transformer``: CTMC discrete flow matching over quantile-binned
  observation tokens (``TokenizedDiscreteTransformer`` + tau-leaping sampler).
- ``continuous-transformer``: continuous (linear-interpolant) flow matching
  directly on observation floats (``ContinuousTokenTransformer`` + Euler ODE).
- ``llada2``: LLaDA2.0 block diffusion over the same observation tokens, with
  actions supplied as prefix tokens rather than a conditioning vector.

``predict_next`` is float-in/float-out for every arm: token arms encode/decode
internally, so the imagination loop never handles tokens. All coupling to the
quantile binning lives here and in :mod:`world_marl.genwm.tokenizer`; the
underlying ``flow_matching`` train/sample functions are reused unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from flow_matching.llada2 import (
    BlockDiffusionTransformer,
    create_llada2_train_state,
    llada2_train_step,
    sample_llada2_block_diffusion,
)
from flow_matching.models import TokenizedDiscreteTransformer
from flow_matching.simulate import (
    sample_marginal_discrete_flow_model,
    sample_marginal_flow_model,
)
from flow_matching.train import (
    conditioned_discrete_train_step,
    conditioned_train_step,
    create_conditioned_train_state,
    create_discrete_conditioned_train_state,
)
from world_marl.genwm.models import ContinuousTokenTransformer
from world_marl.genwm.tokenizer import (
    QuantileTokenizer,
    decode_tokens,
    encode_tokens,
)

GENWM_ARMS = ("discrete-transformer", "continuous-transformer", "llada2")


@dataclass(frozen=True)
class GenWMConfig:
    """Frozen (hashable) generative world-model config, safe as a static jit arg.

    ``action_dim`` is the number of discrete actions or the continuous action
    vector dimension. Transformer capacity has one source of truth
    (``model_dim`` x ``num_layers`` with ``mlp_ratio`` FFN width) across arms;
    LLaDA2's scalar ``ffn_hidden_dim`` and the parent's per-layer tuple both
    derive from it, so the arms stay parameter-matched.
    """

    arm: str
    obs_dim: int
    action_dim: int
    action_mode: str  # "discrete" | "continuous"
    obs_bins: int = 32  # observation-token vocabulary for the token arms
    action_bins: int = 8  # llada2 action-token vocabulary (continuous actions)
    model_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    mlp_ratio: int = 4
    learning_rate: float = 1e-3
    integration_steps: int = 8  # Euler / CTMC steps for the flow arms
    flow_type: str = "linear"  # continuous-transformer interpolant
    # LLaDA2 knobs; defaults mirror LLaDA2WorldModelConfig in world_model.py.
    block_size: int = 4
    steps_per_block: int = 4
    confidence_threshold: float = 0.9
    mask_schedule: str = "linear"
    alpha_min: float = 0.15
    alpha_max: float = 0.95
    complementary_masking: bool = True
    cap_lambda: float = 0.1
    moe_aux_coeff: float = 0.01

    def __post_init__(self) -> None:
        if self.arm not in GENWM_ARMS:
            raise ValueError(f"arm must be one of {GENWM_ARMS}, got {self.arm!r}")
        if self.action_mode not in ("discrete", "continuous"):
            raise ValueError(f"unsupported action_mode {self.action_mode!r}")

    @property
    def cond_dim(self) -> int:
        # One-hot discrete actions and raw continuous action vectors are both
        # action_dim wide.
        return self.obs_dim + self.action_dim

    @property
    def num_action_tokens(self) -> int:
        return 1 if self.action_mode == "discrete" else self.action_dim

    @property
    def ffn_hidden_dims(self) -> tuple[int, ...]:
        return (self.mlp_ratio * self.model_dim,) * self.num_layers


def action_features(actions: jax.Array, config: GenWMConfig) -> jax.Array:
    """Actions -> the float feature block shared by cond_vars and the head."""
    if config.action_mode == "discrete":
        return jax.nn.one_hot(actions, config.action_dim, dtype=jnp.float32)
    return actions.astype(jnp.float32)


def action_token_ids(
    actions: jax.Array,
    action_tokenizer: QuantileTokenizer | None,
    config: GenWMConfig,
) -> jax.Array:
    """Actions -> llada2 prefix token ids ``(B, num_action_tokens)``."""
    if config.action_mode == "discrete":
        return actions.astype(jnp.int32)[:, None]
    if action_tokenizer is None:
        raise ValueError("continuous llada2 arm requires an action tokenizer")
    return encode_tokens(action_tokenizer, actions)


def create_genwm_state(key: jax.Array, config: GenWMConfig) -> TrainState:
    if config.arm == "discrete-transformer":
        model = TokenizedDiscreteTransformer(
            num_categories=config.obs_bins,
            model_dim=config.model_dim,
            num_heads=config.num_heads,
            ffn_hidden_dims=config.ffn_hidden_dims,
        )
        return create_discrete_conditioned_train_state(
            key,
            model,
            config.learning_rate,
            num_factors=config.obs_dim,
            cond_dim=config.cond_dim,
        )
    if config.arm == "continuous-transformer":
        model = ContinuousTokenTransformer(
            model_dim=config.model_dim,
            num_heads=config.num_heads,
            ffn_hidden_dims=config.ffn_hidden_dims,
        )
        return create_conditioned_train_state(
            key,
            model,
            config.learning_rate,
            dim=config.obs_dim,
            cond_dim=config.cond_dim,
        )
    num_actions = (
        config.action_dim if config.action_mode == "discrete" else config.action_bins
    )
    model = BlockDiffusionTransformer(
        num_categories=config.obs_bins,
        num_actions=num_actions,
        model_dim=config.model_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        ffn_hidden_dim=config.mlp_ratio * config.model_dim,
    )
    return create_llada2_train_state(
        key,
        model,
        config.learning_rate,
        num_factors=config.obs_dim,
        num_action_tokens=config.num_action_tokens,
    )


@partial(jax.jit, static_argnames=("config",))
def genwm_train_step(
    state: TrainState,
    key: jax.Array,
    observations: jax.Array,
    actions: jax.Array,
    next_observations: jax.Array,
    obs_tokenizer: QuantileTokenizer,
    action_tokenizer: QuantileTokenizer | None,
    config: GenWMConfig,
) -> tuple[TrainState, jax.Array]:
    """One optimizer update on a batch of real (s, a, s') transitions.

    Callers must pre-filter terminal transitions: with auto-resetting envs the
    stored s' at a terminal step is the post-reset observation.
    """
    if config.arm == "llada2":
        return llada2_train_step(
            state,
            key,
            encode_tokens(obs_tokenizer, next_observations),
            encode_tokens(obs_tokenizer, observations),
            action_token_ids(actions, action_tokenizer, config),
            config.obs_bins,
            block_size=config.block_size,
            mask_schedule_name=config.mask_schedule,
            alpha_min=config.alpha_min,
            alpha_max=config.alpha_max,
            complementary=config.complementary_masking,
            cap_lambda=config.cap_lambda,
            moe_aux_coeff=config.moe_aux_coeff,
        )
    cond_vars = jnp.concatenate(
        [observations, action_features(actions, config)], axis=-1
    )
    if config.arm == "discrete-transformer":
        return conditioned_discrete_train_step(
            state,
            key,
            encode_tokens(obs_tokenizer, next_observations),
            cond_vars,
            config.obs_bins,
        )
    return conditioned_train_step(
        state, key, next_observations, cond_vars, config.flow_type
    )


@partial(jax.jit, static_argnames=("config",))
def genwm_predict_next(
    state: TrainState,
    key: jax.Array,
    observations: jax.Array,
    actions: jax.Array,
    obs_tokenizer: QuantileTokenizer,
    action_tokenizer: QuantileTokenizer | None,
    config: GenWMConfig,
) -> jax.Array:
    """Sample next observations ``(B, obs_dim)`` floats from the fitted model."""
    if config.arm == "llada2":
        tokens = sample_llada2_block_diffusion(
            state.apply_fn,
            state.params,
            key,
            encode_tokens(obs_tokenizer, observations),
            action_token_ids(actions, action_tokenizer, config),
            num_factors=config.obs_dim,
            num_categories=config.obs_bins,
            block_size=config.block_size,
            steps_per_block=config.steps_per_block,
            confidence_threshold=config.confidence_threshold,
        )
        return decode_tokens(obs_tokenizer, tokens)
    cond_vars = jnp.concatenate(
        [observations, action_features(actions, config)], axis=-1
    )
    if config.arm == "discrete-transformer":
        tokens = sample_marginal_discrete_flow_model(
            state.apply_fn,
            state.params,
            key,
            cond_vars,
            num_factors=config.obs_dim,
            num_categories=config.obs_bins,
            steps=config.integration_steps,
        )
        return decode_tokens(obs_tokenizer, tokens)
    return sample_marginal_flow_model(
        state.apply_fn,
        state.params,
        key,
        cond_vars,
        dim=config.obs_dim,
        steps=config.integration_steps,
    )
