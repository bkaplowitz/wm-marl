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


def unimix_logits(logits: jax.Array, unimix: float) -> tuple[jax.Array, jax.Array]:
    probs = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
    if unimix:
        uniform = jnp.full_like(probs, 1.0 / probs.shape[-1])
        probs = (1.0 - unimix) * probs + unimix * uniform
    return jnp.log(probs), probs


def categorical_straight_through(
    logits: jax.Array,
    key: jax.Array,
    *,
    unimix: float,
) -> tuple[jax.Array, jax.Array]:
    mixed_logits, probs = unimix_logits(logits, unimix)
    indices = jax.random.categorical(key, mixed_logits, axis=-1)
    hard = jax.nn.one_hot(indices, logits.shape[-1], dtype=jnp.float32)
    straight_through = jax.lax.stop_gradient(hard) + (
        probs - jax.lax.stop_gradient(probs)
    )
    return straight_through, probs


def initial_rssm_state(*, batch_size: int, config: RSSMConfig) -> RSSMState:
    deterministic = jnp.zeros(
        (batch_size, config.deterministic_size), dtype=jnp.float32
    )
    stochastic = jnp.zeros(
        (batch_size, config.stochastic_size, config.discrete_classes),
        dtype=jnp.float32,
    )
    logits = jnp.full_like(stochastic, -jnp.log(float(config.discrete_classes)))
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


def reset_rssm_state(
    state: RSSMState,
    is_first: jax.Array,
    *,
    config: RSSMConfig,
) -> RSSMState:
    initial = initial_rssm_state(batch_size=state.deterministic.shape[0], config=config)
    deterministic_mask = is_first.astype(bool).reshape((-1, 1))
    stochastic_mask = is_first.astype(bool).reshape((-1, 1, 1))
    return RSSMState(
        deterministic=jnp.where(
            deterministic_mask, initial.deterministic, state.deterministic
        ),
        stochastic=jnp.where(stochastic_mask, initial.stochastic, state.stochastic),
        logits=jnp.where(stochastic_mask, initial.logits, state.logits),
    )


class NormedLinear(nn.Module):
    features: int

    @nn.compact
    def __call__(self, inputs: jax.Array) -> jax.Array:
        x = nn.Dense(self.features)(inputs)
        x = nn.RMSNorm(epsilon=1e-4)(x)
        return nn.silu(x)


class BlockLinear(nn.Module):
    features: int
    blocks: int
    use_bias: bool = True

    @nn.compact
    def __call__(self, inputs: jax.Array) -> jax.Array:
        input_size = inputs.shape[-1]
        if input_size % self.blocks or self.features % self.blocks:
            raise ValueError("block-linear dimensions must be divisible by blocks")
        input_per_block = input_size // self.blocks
        output_per_block = self.features // self.blocks
        kernel = self.param(
            "kernel",
            nn.initializers.lecun_normal(),
            (self.blocks, input_per_block, output_per_block),
        )
        grouped = inputs.reshape((*inputs.shape[:-1], self.blocks, input_per_block))
        output = jnp.einsum("...bi,bio->...bo", grouped, kernel)
        output = output.reshape((*inputs.shape[:-1], self.features))
        if self.use_bias:
            bias = self.param("bias", nn.initializers.zeros_init(), (self.features,))
            output = output + bias
        return output


class DreamerRSSM(nn.Module):
    config: RSSMConfig
    action_dim: int

    def setup(self) -> None:
        self.deterministic_embed = NormedLinear(
            self.config.hidden_size,
            name="deterministic_embed",
        )
        self.stochastic_embed = NormedLinear(
            self.config.hidden_size,
            name="stochastic_embed",
        )
        self.action_embed = NormedLinear(
            self.config.hidden_size,
            name="action_embed",
        )
        self.block_gru_hidden = BlockLinear(
            self.config.deterministic_size,
            self.config.blocks,
            name="block_gru_hidden",
        )
        self.block_gru_hidden_norm = nn.RMSNorm(
            epsilon=1e-4,
            name="block_gru_hidden_norm",
        )
        self.block_gru_gates = BlockLinear(
            3 * self.config.deterministic_size,
            self.config.blocks,
            name="block_gru_gates",
        )
        self.prior_hidden = tuple(
            NormedLinear(self.config.hidden_size, name=f"prior_hidden_{index}")
            for index in range(self.config.prior_layers)
        )
        self.prior_logits = nn.Dense(
            self.config.stochastic_size * self.config.discrete_classes,
            use_bias=False,
            name="prior_logits",
        )
        self.posterior_hidden = tuple(
            NormedLinear(self.config.hidden_size, name=f"posterior_hidden_{index}")
            for index in range(self.config.posterior_layers)
        )
        self.posterior_logits = nn.Dense(
            self.config.stochastic_size * self.config.discrete_classes,
            use_bias=False,
            name="posterior_logits",
        )

    def _block_gru(
        self,
        deterministic: jax.Array,
        stochastic: jax.Array,
        actions: jax.Array,
    ) -> jax.Array:
        stochastic = stochastic.reshape((stochastic.shape[0], -1))
        actions = actions / jax.lax.stop_gradient(jnp.maximum(1.0, jnp.abs(actions)))
        mixed = jnp.concatenate(
            [
                self.deterministic_embed(deterministic),
                self.stochastic_embed(stochastic),
                self.action_embed(actions),
            ],
            axis=-1,
        )
        mixed = jnp.repeat(mixed[:, None, :], self.config.blocks, axis=1)
        grouped_deterministic = deterministic.reshape(
            deterministic.shape[0],
            self.config.blocks,
            self.config.deterministic_size // self.config.blocks,
        )
        hidden_input = jnp.concatenate([grouped_deterministic, mixed], axis=-1)
        hidden_input = hidden_input.reshape((deterministic.shape[0], -1))
        hidden = self.block_gru_hidden(hidden_input)
        hidden = nn.silu(self.block_gru_hidden_norm(hidden))
        gates = self.block_gru_gates(hidden)
        gates = gates.reshape(
            deterministic.shape[0],
            self.config.blocks,
            3,
            self.config.deterministic_size // self.config.blocks,
        )
        reset, candidate, update = jnp.moveaxis(gates, 2, 0)
        reset = reset.reshape(deterministic.shape)
        candidate = candidate.reshape(deterministic.shape)
        update = update.reshape(deterministic.shape)
        reset = jax.nn.sigmoid(reset)
        candidate = jnp.tanh(reset * candidate)
        update = jax.nn.sigmoid(update - 1.0)
        return update * candidate + (1.0 - update) * deterministic

    def _state_from_logits(
        self,
        deterministic: jax.Array,
        raw_logits: jax.Array,
        key: jax.Array,
    ) -> RSSMState:
        raw_logits = raw_logits.reshape(
            (-1, self.config.stochastic_size, self.config.discrete_classes)
        )
        stochastic, probs = categorical_straight_through(
            raw_logits,
            key,
            unimix=self.config.unimix,
        )
        return RSSMState(
            deterministic=deterministic,
            stochastic=stochastic,
            logits=jnp.log(probs),
        )

    def prior(
        self,
        prev_state: RSSMState,
        actions: jax.Array,
        key: jax.Array,
    ) -> RSSMState:
        if actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"expected action feature width {self.action_dim}, got "
                f"{actions.shape[-1]}"
            )
        deterministic = self._block_gru(
            prev_state.deterministic,
            prev_state.stochastic,
            actions,
        )
        hidden = deterministic
        for layer in self.prior_hidden:
            hidden = layer(hidden)
        return self._state_from_logits(
            deterministic,
            self.prior_logits(hidden),
            key,
        )

    def __call__(
        self,
        prev_state: RSSMState,
        actions: jax.Array,
        embed: jax.Array,
        key: jax.Array,
    ) -> tuple[RSSMState, RSSMState]:
        prior_key, posterior_key = jax.random.split(key)
        prior = self.prior(prev_state, actions, prior_key)
        hidden = jnp.concatenate([prior.deterministic, embed], axis=-1)
        for layer in self.posterior_hidden:
            hidden = layer(hidden)
        posterior = self._state_from_logits(
            prior.deterministic,
            self.posterior_logits(hidden),
            posterior_key,
        )
        return prior, posterior
