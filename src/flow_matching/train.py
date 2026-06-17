"""Training helpers for conditional flow matching."""

from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from flow_matching.distributions import GaussianMixture2D, sample_gmm
from flow_matching.paths import (
    conditional_vector_field,
    flow_schedule,
    sample_conditional_path,
)


def create_train_state(
    key: jax.Array,
    model: flax.linen.Module,
    learning_rate: float,
    dim: int = 2,
) -> TrainState:
    """Initialize model parameters and an Adam optimizer."""
    init_x = jnp.zeros((1, dim))
    init_t = jnp.zeros((1, 1))
    key, train_state_key = jax.random.split(key)
    params = model.init(train_state_key, init_x, init_t)["params"]
    tx = optax.adam(learning_rate)
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def flow_matching_loss(
    params: Any,
    apply_fn: Any,  # model.apply
    key: jax.Array,
    gmm: GaussianMixture2D,
    batch_size: int,
) -> jax.Array:
    """Compute the conditional flow-matching MSE loss."""
    key, key_x1, key_t, key_xt = jax.random.split(key, 4)
    # sample x1, t uniformly, xt from x1, t
    x1 = sample_gmm(key_x1, gmm, batch_size)
    t = jax.random.uniform(key_t, shape=(batch_size, 1))
    xt = sample_conditional_path(key_xt, x1, t)
    target_flow = conditional_vector_field(xt, x1, t)  # flow from model
    model_flow = apply_fn({"params": params}, xt, t)  # model flow estimated
    return jnp.mean(
        (target_flow - model_flow) ** 2
    )  # mse loss on flow matching objective


@partial(jax.jit, static_argnames="batch_size")
def train_step(
    state: TrainState,
    key: jax.Array,
    gmm: GaussianMixture2D,
    batch_size: int,
) -> tuple[TrainState, jax.Array]:
    """Run one optimizer update."""
    # Get loss and semigradient of loss
    grad_fn = jax.value_and_grad(flow_matching_loss)
    # Compute grad
    loss, grads = grad_fn(state.params, state.apply_fn, key, gmm, batch_size)
    # update trainstate
    state = state.apply_gradients(grads=grads)
    return state, loss


def create_conditioned_train_state(
    key: jax.Array,
    model: flax.linen.Module,
    learning_rate: float,
    *,
    dim: int,
    cond_dim: int,
) -> TrainState:
    """Initialize a conditioned vector-field model and its Adam optimizer.

    ``dim`` is the size of the target ``x``; ``cond_dim`` is the size of the
    conditioning variables threaded into the model alongside ``(x, t)``.
    """
    key, init_key = jax.random.split(key)
    params = model.init(
        init_key,
        jnp.zeros((1, dim)),
        jnp.zeros((1, 1)),
        jnp.zeros((1, cond_dim)),
    )["params"]
    tx = optax.adam(learning_rate)
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def conditioned_flow_matching_loss(
    params: Any,
    apply_fn: Any,  # model.apply
    key: jax.Array,
    x1: jax.Array,
    cond_vars: jax.Array,
    flow_type: str = "gaussian",
) -> jax.Array:
    """Conditional FM MSE loss with conditioning passed straight to the model."""
    alpha, alpha_dt, beta, beta_dt = flow_schedule(flow_type)
    key_t, key_xt = jax.random.split(key)
    t = jax.random.uniform(key_t, shape=(x1.shape[0], 1))
    xt = sample_conditional_path(key_xt, x1, t, alpha, beta)
    target_flow = conditional_vector_field(xt, x1, t, alpha, alpha_dt, beta, beta_dt)
    model_flow = apply_fn({"params": params}, xt, t, cond_vars)  # already x1-sized
    return jnp.mean((target_flow - model_flow) ** 2)


@partial(jax.jit, static_argnames="flow_type")
def conditioned_train_step(
    state: TrainState,
    key: jax.Array,
    x1: jax.Array,
    cond_vars: jax.Array,
    flow_type: str = "gaussian",
) -> tuple[TrainState, jax.Array]:
    """Run one optimizer update for the conditioned vector field."""
    loss, grads = jax.value_and_grad(conditioned_flow_matching_loss)(
        state.params, state.apply_fn, key, x1, cond_vars, flow_type
    )
    return state.apply_gradients(grads=grads), loss
