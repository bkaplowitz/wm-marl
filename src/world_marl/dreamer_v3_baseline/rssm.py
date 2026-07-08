from __future__ import annotations

from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp

from world_marl.dreamer_v3_baseline.config import RSSMConfig


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class RSSMState:
    deterministic: jax.Array
    stochastic: jax.Array
    logits: jax.Array

    def tree_flatten(self) -> tuple[tuple[jax.Array, jax.Array, jax.Array], None]:
        return (self.deterministic, self.stochastic, self.logits), None

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: None,
        children: tuple[jax.Array, jax.Array, jax.Array],
    ) -> RSSMState:
        del aux_data
        deterministic, stochastic, logits = children
        return cls(deterministic=deterministic, stochastic=stochastic, logits=logits)


def categorical_straight_through(logits: jax.Array) -> tuple[jax.Array, jax.Array]:
    probs = jax.nn.softmax(logits, axis=-1)
    hard = jax.nn.one_hot(jnp.argmax(probs, axis=-1), logits.shape[-1])
    straight_through = hard - jax.lax.stop_gradient(probs) + probs
    return straight_through.astype(jnp.float32), probs


def initial_rssm_state(*, batch_size: int, config: RSSMConfig) -> RSSMState:
    deterministic = jnp.zeros(
        (batch_size, config.deterministic_size), dtype=jnp.float32
    )
    logits = jnp.zeros(
        (batch_size, config.stochastic_size, config.discrete_classes),
        dtype=jnp.float32,
    )
    stochastic, _ = categorical_straight_through(logits)
    return RSSMState(
        deterministic=deterministic,
        stochastic=stochastic,
        logits=logits,
    )


def flatten_rssm_state(state: RSSMState) -> jax.Array:
    return jnp.concatenate(
        [
            state.deterministic,
            state.stochastic.reshape((state.stochastic.shape[0], -1)),
        ],
        axis=-1,
    )


class DreamerRSSM(nn.Module):
    config: RSSMConfig
    action_dim: int

    @nn.compact
    def __call__(
        self,
        prev_state: RSSMState,
        actions: jax.Array,
        embed: jax.Array,
    ) -> tuple[RSSMState, RSSMState]:
        prev_features = jnp.concatenate(
            [flatten_rssm_state(prev_state), actions], axis=-1
        )
        prior_hidden = nn.silu(
            nn.Dense(self.config.hidden_size, name="prior_hidden")(prev_features)
        )
        prior_deterministic = nn.tanh(
            nn.Dense(self.config.deterministic_size, name="prior_deterministic")(
                prior_hidden
            )
        )
        prior_logits = nn.Dense(
            self.config.stochastic_size * self.config.discrete_classes,
            name="prior_logits",
        )(prior_deterministic)
        prior_logits = prior_logits.reshape(
            (-1, self.config.stochastic_size, self.config.discrete_classes)
        )
        prior_stochastic, _ = categorical_straight_through(prior_logits)
        prior = RSSMState(
            deterministic=prior_deterministic,
            stochastic=prior_stochastic,
            logits=prior_logits,
        )

        posterior_input = jnp.concatenate([prior.deterministic, embed], axis=-1)
        posterior_hidden = nn.silu(
            nn.Dense(self.config.hidden_size, name="posterior_hidden")(posterior_input)
        )
        posterior_logits = nn.Dense(
            self.config.stochastic_size * self.config.discrete_classes,
            name="posterior_logits",
        )(posterior_hidden)
        posterior_logits = posterior_logits.reshape(
            (-1, self.config.stochastic_size, self.config.discrete_classes)
        )
        posterior_stochastic, _ = categorical_straight_through(posterior_logits)
        posterior = RSSMState(
            deterministic=prior.deterministic,
            stochastic=posterior_stochastic,
            logits=posterior_logits,
        )
        return prior, posterior
