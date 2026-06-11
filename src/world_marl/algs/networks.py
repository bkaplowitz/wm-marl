"""Neural network modules for PPO-style MARL algorithms."""

from __future__ import annotations

import distrax
import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal


class CNNActorCritic(nn.Module):
  """Shared CNN actor-critic for RGB Melting Pot observations."""

  action_dim: int
  activation: str = "relu"

  @nn.compact
  def __call__(self, observations: jnp.ndarray) -> tuple[distrax.Categorical, jnp.ndarray]:
    if self.activation == "tanh":
      activation = nn.tanh
    elif self.activation == "relu":
      activation = nn.relu
    else:
      raise ValueError(f"unsupported activation {self.activation!r}")

    x = observations.astype(jnp.float32)
    x = nn.Conv(
      features=32,
      kernel_size=(5, 5),
      padding="SAME",
      kernel_init=orthogonal(np.sqrt(2.0)),
      bias_init=constant(0.0),
    )(x)
    x = activation(x)
    x = nn.Conv(
      features=32,
      kernel_size=(3, 3),
      padding="SAME",
      kernel_init=orthogonal(np.sqrt(2.0)),
      bias_init=constant(0.0),
    )(x)
    x = activation(x)
    x = nn.Conv(
      features=32,
      kernel_size=(3, 3),
      padding="SAME",
      kernel_init=orthogonal(np.sqrt(2.0)),
      bias_init=constant(0.0),
    )(x)
    x = activation(x)
    x = x.reshape((x.shape[0], -1))
    embedding = nn.Dense(
      features=128,
      kernel_init=orthogonal(np.sqrt(2.0)),
      bias_init=constant(0.0),
    )(x)
    embedding = activation(embedding)

    actor_hidden = nn.Dense(
      features=64,
      kernel_init=orthogonal(np.sqrt(2.0)),
      bias_init=constant(0.0),
    )(embedding)
    actor_hidden = activation(actor_hidden)
    logits = nn.Dense(
      features=self.action_dim,
      kernel_init=orthogonal(0.01),
      bias_init=constant(0.0),
    )(actor_hidden)

    critic_hidden = nn.Dense(
      features=64,
      kernel_init=orthogonal(np.sqrt(2.0)),
      bias_init=constant(0.0),
    )(embedding)
    critic_hidden = activation(critic_hidden)
    value = nn.Dense(
      features=1,
      kernel_init=orthogonal(1.0),
      bias_init=constant(0.0),
    )(critic_hidden)

    return distrax.Categorical(logits=logits), jnp.squeeze(value, axis=-1)


class CNNMAPPOActorCritic(nn.Module):
  """CNN actor with a centralized CNN critic for MAPPO."""

  action_dim: int
  activation: str = "relu"

  @nn.compact
  def __call__(
    self,
    observations: jnp.ndarray,
    central_observations: jnp.ndarray,
  ) -> tuple[distrax.Categorical, jnp.ndarray]:
    if self.activation == "tanh":
      activation = nn.tanh
    elif self.activation == "relu":
      activation = nn.relu
    else:
      raise ValueError(f"unsupported activation {self.activation!r}")

    actor_embedding = _cnn_embedding(
      observations,
      activation=activation,
      name="actor",
    )
    actor_hidden = nn.Dense(
      features=64,
      kernel_init=orthogonal(np.sqrt(2.0)),
      bias_init=constant(0.0),
      name="actor_hidden",
    )(actor_embedding)
    actor_hidden = activation(actor_hidden)
    logits = nn.Dense(
      features=self.action_dim,
      kernel_init=orthogonal(0.01),
      bias_init=constant(0.0),
      name="actor_logits",
    )(actor_hidden)

    critic_embedding = _cnn_embedding(
      central_observations,
      activation=activation,
      name="critic",
    )
    critic_hidden = nn.Dense(
      features=64,
      kernel_init=orthogonal(np.sqrt(2.0)),
      bias_init=constant(0.0),
      name="critic_hidden",
    )(critic_embedding)
    critic_hidden = activation(critic_hidden)
    value = nn.Dense(
      features=1,
      kernel_init=orthogonal(1.0),
      bias_init=constant(0.0),
      name="critic_value",
    )(critic_hidden)

    return distrax.Categorical(logits=logits), jnp.squeeze(value, axis=-1)


def _cnn_embedding(
  observations: jnp.ndarray,
  *,
  activation,
  name: str,
) -> jnp.ndarray:
  x = observations.astype(jnp.float32)
  x = nn.Conv(
    features=32,
    kernel_size=(5, 5),
    padding="SAME",
    kernel_init=orthogonal(np.sqrt(2.0)),
    bias_init=constant(0.0),
    name=f"{name}_conv_0",
  )(x)
  x = activation(x)
  x = nn.Conv(
    features=32,
    kernel_size=(3, 3),
    padding="SAME",
    kernel_init=orthogonal(np.sqrt(2.0)),
    bias_init=constant(0.0),
    name=f"{name}_conv_1",
  )(x)
  x = activation(x)
  x = nn.Conv(
    features=32,
    kernel_size=(3, 3),
    padding="SAME",
    kernel_init=orthogonal(np.sqrt(2.0)),
    bias_init=constant(0.0),
    name=f"{name}_conv_2",
  )(x)
  x = activation(x)
  x = x.reshape((x.shape[0], -1))
  x = nn.Dense(
    features=128,
    kernel_init=orthogonal(np.sqrt(2.0)),
    bias_init=constant(0.0),
    name=f"{name}_dense",
  )(x)
  return activation(x)
