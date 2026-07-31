"""Compiled optimizer state and update transactions for DreaMARL."""

from __future__ import annotations

from functools import partial
from typing import Any, NamedTuple

from flax.training import train_state
import jax
import jax.numpy as jnp
import optax

from world_marl.dreamarl.config import DreaMARLConfig
from world_marl.dreamarl.contracts import JaxMultiAgentSequenceBatch
from world_marl.dreamarl.control import (
    CentralizedCritic,
    SharedActor,
    actor_loss,
    critic_loss,
)
from world_marl.dreamarl.losses import (
    encoder_params,
    update_ema,
    world_model_loss,
)
from world_marl.dreamarl.world_model import DreaMARLWorldModel


class DreaMARLLearnerState(NamedTuple):
    """All trainable and target state needed for exact continuation."""

    world_model: train_state.TrainState
    actor: train_state.TrainState
    critic: train_state.TrainState
    target_encoder: Any
    target_critic: Any
    return_low: jax.Array
    return_high: jax.Array
    return_normalization_initialized: jax.Array
    rng: jax.Array
    world_updates: jax.Array
    actor_updates: jax.Array
    critic_updates: jax.Array


class LearnerStepOutput(NamedTuple):
    state: DreaMARLLearnerState
    metrics: dict[str, jax.Array]


class DreaMARLLearner:
    """Own final-grade modules and expose atomic compiled update methods."""

    def __init__(self, config: DreaMARLConfig) -> None:
        self.config = config
        self.world_model = DreaMARLWorldModel(config)
        self.actor = SharedActor(config)
        self.critic = CentralizedCritic(config)

    def initialize(
        self,
        sample: JaxMultiAgentSequenceBatch,
        key: jax.Array,
    ) -> DreaMARLLearnerState:
        """Initialize every parameter tree from a representative replay batch."""

        model_key, infer_key, actor_key, critic_key, state_key = jax.random.split(
            key, 5
        )
        model_params = self.world_model.init(
            model_key, sample, infer_key
        )["params"]
        inference = self.world_model.apply(
            {"params": model_params}, sample, infer_key
        )
        belief = self.world_model.apply(
            {"params": model_params},
            inference.final_state,
            method=self.world_model.belief,
        )
        actor_params = self.actor.init(
            actor_key,
            belief,
            inference.final_state.agent_alive,
            inference.final_state.action_mask,
        )["params"]
        critic_params = self.critic.init(
            critic_key,
            belief,
            inference.final_state.agent_alive,
        )["params"]
        optimizer = self.config.optimizer
        world_state = train_state.TrainState.create(
            apply_fn=self.world_model.apply,
            params=model_params,
            tx=_optimizer(
                optimizer.world_model_learning_rate,
                optimizer.world_model_grad_clip,
            ),
        )
        actor_state = train_state.TrainState.create(
            apply_fn=self.actor.apply,
            params=actor_params,
            tx=_optimizer(
                optimizer.actor_learning_rate,
                optimizer.actor_critic_grad_clip,
            ),
        )
        critic_state = train_state.TrainState.create(
            apply_fn=self.critic.apply,
            params=critic_params,
            tx=_optimizer(
                optimizer.critic_learning_rate,
                optimizer.actor_critic_grad_clip,
            ),
        )
        return DreaMARLLearnerState(
            world_model=world_state,
            actor=actor_state,
            critic=critic_state,
            target_encoder=encoder_params(model_params),
            target_critic=critic_params,
            return_low=jnp.asarray(0.0, jnp.float32),
            return_high=jnp.asarray(1.0, jnp.float32),
            return_normalization_initialized=jnp.asarray(False),
            rng=state_key,
            world_updates=jnp.asarray(0, jnp.int32),
            actor_updates=jnp.asarray(0, jnp.int32),
            critic_updates=jnp.asarray(0, jnp.int32),
        )

    @partial(jax.jit, static_argnums=0)
    def world_model_step(
        self,
        state: DreaMARLLearnerState,
        batch: JaxMultiAgentSequenceBatch,
    ) -> LearnerStepOutput:
        """Apply one world-model update and its encoder EMA atomically."""

        next_key, loss_key = jax.random.split(state.rng)

        def objective(params):
            output = world_model_loss(
                self.world_model,
                params,
                state.target_encoder,
                batch,
                loss_key,
                self.config,
            )
            return output.loss, output.metrics

        (loss, metrics), gradients = jax.value_and_grad(
            objective, has_aux=True
        )(state.world_model.params)
        world_state = state.world_model.apply_gradients(grads=gradients)
        target_encoder = update_ema(
            state.target_encoder,
            encoder_params(world_state.params),
            self.config.optimizer.target_encoder_decay,
        )
        next_state = state._replace(
            world_model=world_state,
            target_encoder=target_encoder,
            rng=next_key,
            world_updates=state.world_updates + 1,
        )
        metrics = {
            **{f"world_model/{name}": value for name, value in metrics.items()},
            "world_model/gradient_norm": optax.global_norm(gradients),
            "world_model/update": state.world_updates + 1,
            "world_model/loss": loss,
        }
        return LearnerStepOutput(next_state, metrics)

    @partial(jax.jit, static_argnums=0)
    def actor_critic_step(
        self,
        state: DreaMARLLearnerState,
        batch: JaxMultiAgentSequenceBatch,
    ) -> LearnerStepOutput:
        """Apply one coherent-imagination actor and critic update."""

        next_key, infer_key, imagination_key = jax.random.split(state.rng, 3)
        inference = self.world_model.apply(
            {"params": state.world_model.params}, batch, infer_key
        )
        start_state = jax.tree.map(
            jax.lax.stop_gradient, inference.final_state
        )

        def actor_objective(params):
            output = actor_loss(
                self.actor,
                params,
                self.critic,
                state.target_critic,
                self.world_model,
                state.world_model.params,
                start_state,
                imagination_key,
                self.config,
                state.return_low,
                state.return_high,
                state.return_normalization_initialized,
            )
            return output.loss, (output.metrics, output.imagination)

        (actor_value, (actor_metrics, imagination)), actor_gradients = (
            jax.value_and_grad(actor_objective, has_aux=True)(
                state.actor.params
            )
        )
        actor_state = state.actor.apply_gradients(grads=actor_gradients)

        def critic_objective(params):
            return critic_loss(self.critic, params, imagination)

        (critic_value, critic_metrics), critic_gradients = jax.value_and_grad(
            critic_objective, has_aux=True
        )(state.critic.params)
        critic_state = state.critic.apply_gradients(grads=critic_gradients)
        target_critic = update_ema(
            state.target_critic,
            critic_state.params,
            self.config.imagination.target_critic_decay,
        )
        next_state = state._replace(
            actor=actor_state,
            critic=critic_state,
            target_critic=target_critic,
            return_low=actor_metrics["return_normalization_low"],
            return_high=actor_metrics["return_normalization_high"],
            return_normalization_initialized=jnp.asarray(True),
            rng=next_key,
            actor_updates=state.actor_updates + 1,
            critic_updates=state.critic_updates + 1,
        )
        metrics = {
            **{
                f"actor/{name}": value
                for name, value in actor_metrics.items()
            },
            **{
                f"critic/{name}": value
                for name, value in critic_metrics.items()
            },
            "actor/loss": actor_value,
            "actor/gradient_norm": optax.global_norm(actor_gradients),
            "actor/update": state.actor_updates + 1,
            "critic/loss": critic_value,
            "critic/gradient_norm": optax.global_norm(critic_gradients),
            "critic/update": state.critic_updates + 1,
        }
        return LearnerStepOutput(next_state, metrics)

    def parameter_counts(self, state: DreaMARLLearnerState) -> dict[str, int]:
        """Return component and total trainable parameter counts."""

        counts = {
            "world_model": _parameter_count(state.world_model.params),
            "actor": _parameter_count(state.actor.params),
            "critic": _parameter_count(state.critic.params),
        }
        return {**counts, "total": sum(counts.values())}

    @partial(jax.jit, static_argnums=0)
    def train_steps(
        self,
        state: DreaMARLLearnerState,
        batches: JaxMultiAgentSequenceBatch,
    ) -> LearnerStepOutput:
        """Run alternating model/control updates over a prefetched batch stack."""

        def step(current, batch):
            world_output = self.world_model_step(current, batch)
            control_output = self.actor_critic_step(world_output.state, batch)
            metrics = {**world_output.metrics, **control_output.metrics}
            return control_output.state, metrics

        next_state, metrics = jax.lax.scan(step, state, batches)
        return LearnerStepOutput(next_state, metrics)


def _optimizer(learning_rate: float, gradient_clip: float) -> optax.GradientTransformation:
    return optax.chain(
        optax.clip_by_global_norm(gradient_clip),
        optax.adam(learning_rate, eps=1e-5),
    )


def _parameter_count(params: Any) -> int:
    return int(sum(value.size for value in jax.tree.leaves(params)))
