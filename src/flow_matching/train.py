"""Training helpers for conditional flow matching."""

from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from flow_matching.distributions import GaussianMixture2D, sample_gmm
from flow_matching.paths import conditional_vector_field, sample_conditional_path


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
