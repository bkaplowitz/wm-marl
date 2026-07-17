"""Frozen-world-model reward and continuation heads.

The reward distribution reuses this repository's DreamerV3 255-bin symlog
two-hot contract. The continuation distribution is Bernoulli with BCE logits.
Inputs are always stop-gradient features so fitting these heads cannot update a
tokenizer, LAM, or dynamics model.
"""

from collections.abc import Sequence
from typing import NamedTuple

from flax import linen as nn
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import optax

from world_marl.dreamer_v3_baseline.imagination import decode_two_hot_logits
from world_marl.dreamer_v3_baseline.losses import symexp_two_hot


class RewardContinueOutputs(NamedTuple):
    reward_logits: jax.Array
    continue_logits: jax.Array

    @property
    def continue_probability(self) -> jax.Array:
        return jax.nn.sigmoid(self.continue_logits)


class RewardContinueHeads(nn.Module):
    hidden_dims: Sequence[int] = (256,)
    reward_bins: int = 255

    @nn.compact
    def __call__(self, features: jax.Array) -> RewardContinueOutputs:
        values = jax.lax.stop_gradient(features).astype(jnp.float32)
        for index, dim in enumerate(self.hidden_dims):
            values = nn.silu(nn.Dense(dim, name=f"hidden_{index}")(values))
        reward_logits = nn.Dense(
            self.reward_bins,
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.zeros_init(),
            name="reward_logits",
        )(values)
        continue_logits = nn.Dense(1, name="continue_logits")(values)[..., 0]
        return RewardContinueOutputs(reward_logits, continue_logits)


def reward_continue_loss(
    outputs: RewardContinueOutputs,
    rewards: jax.Array,
    continues: jax.Array,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    reward_targets = symexp_two_hot(
        rewards,
        num_bins=outputs.reward_logits.shape[-1],
    )
    reward_loss = optax.softmax_cross_entropy(
        outputs.reward_logits,
        reward_targets,
    ).mean()
    continue_loss = optax.sigmoid_binary_cross_entropy(
        outputs.continue_logits,
        continues.astype(jnp.float32),
    ).mean()
    loss = reward_loss + continue_loss
    return loss, {
        "loss": loss,
        "reward_loss": reward_loss,
        "continue_loss": continue_loss,
    }


def decode_reward(reward_logits: jax.Array) -> jax.Array:
    return decode_two_hot_logits(reward_logits)


def create_head_train_state(
    rng: jax.Array,
    module: RewardContinueHeads,
    example_features: jax.Array,
    learning_rate: optax.ScalarOrSchedule,
) -> TrainState:
    variables = module.init(rng, example_features)
    return TrainState.create(
        apply_fn=module.apply,
        params=variables["params"],
        tx=optax.adam(learning_rate),
    )


def head_train_step(
    state: TrainState,
    features: jax.Array,
    rewards: jax.Array,
    continues: jax.Array,
) -> tuple[TrainState, dict[str, jax.Array]]:
    def loss_fn(params):
        outputs = state.apply_fn({"params": params}, features)
        return reward_continue_loss(outputs, rewards, continues)

    (_, metrics), gradients = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    return state.apply_gradients(grads=gradients), metrics


def scan_head_updates(
    state: TrainState,
    features: jax.Array,
    rewards: jax.Array,
    continues: jax.Array,
) -> tuple[TrainState, dict[str, jax.Array]]:
    def update(current_state, inputs):
        batch_features, batch_rewards, batch_continues = inputs
        return head_train_step(
            current_state,
            batch_features,
            batch_rewards,
            batch_continues,
        )

    return jax.lax.scan(update, state, (features, rewards, continues))
