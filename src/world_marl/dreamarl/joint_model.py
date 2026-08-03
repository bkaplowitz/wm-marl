"""Authoritative joint latent dynamics for DreaMARL training and imagination."""

import math

import embodied
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import optax


f32 = jnp.float32
sg = jax.lax.stop_gradient


class JointWorldModel(nj.Module):
    """Permutation-equivariant posterior and prior over a team state.

    The carry has exactly one global token per environment plus one deterministic
    and one categorical stochastic state per agent.  Every predicted local
    outcome is derived from this same state sample.
    """

    units: int = 512
    layers: int = 2
    heads: int = 8
    ffup: int = 4
    stoch: int = 32
    classes: int = 32
    hidden: int = 512
    act: str = "silu"
    norm: str = "rms"
    unimix: float = 0.01
    free_nats: float = 1.0
    winit: str = "trunc_normal_in"

    def __init__(
        self,
        act_space,
        embedding_dim: int,
        belief_dim: int,
        num_agents: int,
        **kw,
    ):
        if num_agents < 1:
            raise ValueError("num_agents must be positive")
        if self.units % self.heads:
            raise ValueError("joint width must be divisible by attention heads")
        self.act_space = act_space
        self.embedding_dim = int(embedding_dim)
        self.belief_dim = int(belief_dim)
        self.num_agents = int(num_agents)
        self.action_dim = _encoded_action_dim(act_space)
        self.kw = kw

    @property
    def entry_space(self):
        return {}

    def initial(self, batch_size: int):
        dtype = nn.COMPUTE_DTYPE
        return {
            "global": jnp.zeros((batch_size, self.units), dtype),
            "deter": jnp.zeros((batch_size, self.num_agents, self.units), dtype),
            "stoch": jnp.zeros(
                (batch_size, self.num_agents, self.stoch, self.classes), dtype
            ),
            "logit": jnp.zeros(
                (batch_size, self.num_agents, self.stoch, self.classes), dtype
            ),
        }

    def observe(
        self,
        carry,
        embeddings,
        beliefs,
        previous_actions,
        reset,
        training,
        single=False,
    ):
        carry, embeddings, beliefs, previous_actions = nn.cast(
            (carry, embeddings, beliefs, previous_actions)
        )
        if single:
            return self._observe_step(
                carry, embeddings, beliefs, previous_actions, reset, training
            )
        carry, outputs = nj.scan(
            lambda state, inputs: self._observe_step(
                state, *inputs, training=training
            ),
            carry,
            (embeddings, beliefs, previous_actions, reset),
            axis=1,
        )
        return carry, outputs

    def _observe_step(
        self, carry, embeddings, beliefs, previous_actions, reset, training
    ):
        _assert_agent_tensor(embeddings, self.num_agents, self.embedding_dim)
        _assert_agent_tensor(beliefs, self.num_agents, self.belief_dim)
        prior = self.imagine_step(carry, previous_actions, reset, training)

        observation = jnp.concatenate([embeddings, beliefs], -1)
        observation = self.sub(
            "observation_projection", nn.Linear, self.units, winit=self.winit
        )(observation)
        global_input = prior["global"] + observation.mean(1)
        agent_input = prior["deter"] + observation
        global_state, agent_state = self._set_transform(
            "posterior", global_input, agent_input, training
        )
        logits = self._logits("posterior_logits", agent_state)
        stochastic = nn.cast(self._dist(logits).sample(seed=nj.seed()))
        state = nn.cast(
            {
                "global": global_state,
                "deter": agent_state,
                "stoch": stochastic,
                "logit": logits,
            }
        )
        prediction = self.predict_embedding(prior)
        outputs = {
            **state,
            "logit": logits,
            "prior_global": prior["global"],
            "prior_deter": prior["deter"],
            "prior_stoch": prior["stoch"],
            "prior_logit": prior["logit"],
            "pred_embedding": prediction,
        }
        return state, outputs

    def imagine_step(self, carry, action, reset, training):
        carry = self._reset(carry, reset)
        encoded_action = self.encode_action(action)
        encoded_action = nn.mask(encoded_action, ~reset)
        stochastic = carry["stoch"].reshape(
            (*carry["stoch"].shape[:-2], self.stoch * self.classes)
        )
        agent_input = jnp.concatenate(
            [carry["deter"], stochastic, encoded_action], -1
        )
        agent_input = self.sub(
            "transition_agent_input", nn.Linear, self.units, winit=self.winit
        )(agent_input)
        pooled = agent_input.mean(1)
        global_input = self.sub(
            "transition_global_input", nn.Linear, self.units, winit=self.winit
        )(jnp.concatenate([carry["global"], pooled], -1))
        global_state, agent_state = self._set_transform(
            "transition", global_input, agent_input, training
        )
        logits = self._logits("prior_logits", agent_state)
        stochastic = nn.cast(self._dist(logits).sample(seed=nj.seed()))
        return nn.cast(
            {
                "global": global_state,
                "deter": agent_state,
                "stoch": stochastic,
                "logit": logits,
            }
        )

    def loss(
        self,
        carry,
        embeddings,
        beliefs,
        previous_actions,
        reset,
        training,
        target_embeddings=None,
    ):
        target_embeddings = embeddings if target_embeddings is None else target_embeddings
        carry, features = self.observe(
            carry,
            embeddings,
            beliefs,
            previous_actions,
            reset,
            training,
        )
        prior = self._dist(features["prior_logit"])
        posterior = self._dist(features["logit"])
        dyn = self._dist(sg(features["logit"])).kl(prior)
        rep = posterior.kl(self._dist(sg(features["prior_logit"])))
        if self.free_nats:
            dyn = jnp.maximum(dyn, self.free_nats)
            rep = jnp.maximum(rep, self.free_nats)
        latent = optax.losses.cosine_distance(
            sg(target_embeddings),
            features["pred_embedding"],
            axis=-1,
            epsilon=1e-8,
        )
        latent = nn.mask(latent, ~reset[..., None])
        losses = {"dyn": dyn, "rep": rep, "latent_1": latent}
        metrics = {
            "dyn_ent": prior.entropy().mean(),
            "rep_ent": posterior.entropy().mean(),
            "world_model/h1_latent_error": latent.mean(),
        }
        return carry, features, losses, metrics

    def overshoot_loss(
        self,
        posterior_features,
        actions,
        target_embeddings,
        resets,
        horizons=(2, 4, 8),
        training=True,
    ):
        """Open-loop JEPA loss from posterior starts without crossing resets."""

        total = jnp.zeros((), f32)
        metrics = {}
        weights = {2: 0.5, 4: 0.25, 8: 0.125}
        time = resets.shape[1]
        for horizon in horizons:
            if horizon >= time:
                continue
            state = {
                key: posterior_features[key][:, :-horizon]
                for key in ("global", "deter", "stoch", "logit")
            }
            batch, starts = state["global"].shape[:2]
            state = jax.tree.map(
                lambda value: value.reshape((batch * starts, *value.shape[2:])),
                state,
            )
            valid = jnp.ones((batch, starts), bool)
            for offset in range(horizon):
                step_actions = {
                    key: value[:, offset : offset + starts].reshape(
                        (batch * starts, *value.shape[2:])
                    )
                    for key, value in actions.items()
                }
                next_reset = resets[:, offset + 1 : offset + starts + 1]
                valid &= ~next_reset
                state = self.imagine_step(
                    state,
                    step_actions,
                    next_reset.reshape((batch * starts,)),
                    training,
                )
                state = {
                    key: state[key] for key in ("global", "deter", "stoch", "logit")
                }
            prediction = self.predict_embedding(state).reshape(
                (batch, starts, self.num_agents, self.embedding_dim)
            )
            target = target_embeddings[:, horizon:]
            error = optax.losses.cosine_distance(
                sg(target), prediction, axis=-1, epsilon=1e-8
            )
            masked = jnp.where(valid[..., None], error, 0.0)
            denominator = jnp.maximum(1, valid.sum() * self.num_agents)
            mean_error = masked.sum() / denominator
            total += weights.get(horizon, 1.0) * mean_error
            metrics[f"world_model/h{horizon}_latent_error"] = mean_error
        return total, metrics

    def predict_embedding(self, state):
        local = self.agent_feature(state)
        x = self.sub("predictor_hidden", nn.Linear, self.hidden, winit=self.winit)(local)
        x = nn.act(self.act)(self.sub("predictor_norm", nn.Norm, self.norm)(x))
        return self.sub(
            "predictor_output", nn.Linear, self.embedding_dim, winit=self.winit
        )(x)

    def agent_feature(self, state):
        stochastic = state["stoch"].reshape(
            (*state["stoch"].shape[:-2], self.stoch * self.classes)
        )
        global_token = jnp.broadcast_to(
            state["global"][..., None, :], state["deter"].shape
        )
        return jnp.concatenate([global_token, state["deter"], stochastic], -1)

    def team_feature(self, state):
        stochastic = state["stoch"].reshape(
            (*state["stoch"].shape[:-2], self.stoch * self.classes)
        )
        return jnp.concatenate(
            [state["global"], state["deter"].mean(-2), stochastic.mean(-2)], -1
        )

    def encode_action(self, action):
        encoded = nn.DictConcat(self.act_space, 1)(action)
        return encoded / sg(jnp.maximum(1, jnp.abs(encoded)))

    def _set_transform(self, name, global_input, agent_input, training):
        global_type = self.value(
            "global_type", nn.init("trunc_normal"), (self.units,), f32
        )
        agent_type = self.value(
            "agent_type", nn.init("trunc_normal"), (self.units,), f32
        )
        tokens = jnp.concatenate(
            [
                global_input[:, None] + nn.cast(global_type),
                agent_input + nn.cast(agent_type),
            ],
            1,
        )
        tokens = self.sub(
            name,
            nn.Transformer,
            units=self.units,
            layers=self.layers,
            heads=self.heads,
            ffup=self.ffup,
            act=self.act,
            norm=self.norm,
            rope=False,
            winit=self.winit,
        )(tokens, training=training)
        return tokens[:, 0], tokens[:, 1:]

    def _logits(self, name, inputs):
        logits = self.sub(
            name,
            nn.Linear,
            self.stoch * self.classes,
            winit=self.winit,
            outscale=1.0,
        )(inputs)
        return logits.reshape(
            (*logits.shape[:-1], self.stoch, self.classes)
        )

    def _dist(self, logits):
        output = embodied.jax.outs.OneHot(logits, self.unimix)
        return embodied.jax.outs.Agg(output, 1, jnp.sum)

    def _reset(self, carry, reset):
        initial = self.initial(reset.shape[0])
        return jax.tree.map(lambda old, new: nn.where(reset, new, old), carry, initial)


def _assert_agent_tensor(value, num_agents: int, width: int) -> None:
    if value.ndim != 3 or value.shape[1:] != (num_agents, width):
        raise ValueError(
            f"expected [batch, agent={num_agents}, width={width}], got {value.shape}"
        )


def _encoded_action_dim(act_space) -> int:
    total = 0
    for space in act_space.values():
        width = math.prod(space.shape) if space.shape else 1
        if space.discrete:
            classes = np.asarray(space.classes).reshape(-1)
            if not (classes == classes[0]).all():
                raise ValueError("discrete action elements must share cardinality")
            width *= int(classes[0])
        total += width
    return int(total)
