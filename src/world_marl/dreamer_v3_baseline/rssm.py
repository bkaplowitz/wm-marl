from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax import struct

from world_marl.dreamer_v3_baseline.config import RSSMConfig
from world_marl.dreamer_v3_baseline.distributions import (
    AggregateOutput,
    OneHotOutput,
)
from world_marl.dreamer_v3_baseline.networks import (
    BlockGRU,
    Linear,
    RMSNorm,
    TensorSpace,
    _activation,
)
from world_marl.dreamer_v3_baseline.oracle import (
    PAPER_REVISION,
    UPSTREAM_CURRENT_REVISION,
    OracleSourceSpec,
    register_oracle_source_spec,
)


Array = jax.Array
_f32 = jnp.float32
_DEFAULT_COMPUTE_DTYPE = jnp.bfloat16

_RSSM_SOURCE_HASHES = {
    "dreamerv3/agent.py": (
        "adce8e4274bc098c218bf9a20fd3327545f0ad7d850b5fe328597382e91b5269"
    ),
    "dreamerv3/configs.yaml": (
        "9dff9c7062e3e33951cb54c6dd4b598aaf7e56e18e2cff39c812eaa797bcfcfc"
    ),
    "dreamerv3/rssm.py": (
        "d6d50166914e94fb8bd17a5d5dbda9d42cdd37b85819bb1e9fff3a64d4ad2eb6"
    ),
    "embodied/jax/heads.py": (
        "437641cde21e7f9e3f69b88ad8f6b7e7c22e54eec8c5b19eef6127afde1a9b3f"
    ),
    "embodied/jax/nets.py": (
        "9a1c0c71ad7d3596572a44416e78434f777d8f4dbcbe8ca0dd6b86bb8246392c"
    ),
    "embodied/jax/outs.py": (
        "7e80691f175c71be614f089023cce3a809e0d026c6d5ce89bf566d5f11eb3ed0"
    ),
}

RSSM_SOURCE_SPEC = OracleSourceSpec(
    name="rssm",
    revision_hashes={
        PAPER_REVISION: _RSSM_SOURCE_HASHES,
        UPSTREAM_CURRENT_REVISION: _RSSM_SOURCE_HASHES,
    },
    execution_dtypes=("bfloat16", "float32"),
)
register_oracle_source_spec(RSSM_SOURCE_SPEC)


@struct.dataclass
class RSSMState:
    deter: Array
    stoch: Array

    @property
    def deterministic(self) -> Array:
        return self.deter

    @property
    def stochastic(self) -> Array:
        return self.stoch


@struct.dataclass
class RSSMTrajectory:
    posterior: RSSMState | None
    prior: RSSMState | None
    posterior_logits: Array | None
    prior_logits: Array
    features: Array
    final_state: RSSMState
    mode: Literal["observe", "imagine"] = struct.field(pytree_node=False)

    def __post_init__(self) -> None:
        if self.mode == "observe":
            if self.posterior is None or self.posterior_logits is None:
                raise ValueError(
                    "observe trajectory requires posterior state and logits"
                )
            if self.prior is not None:
                raise ValueError(
                    "observe trajectory must not contain sampled prior state"
                )
            state = self.posterior
            logits = self.posterior_logits
        elif self.mode == "imagine":
            if self.prior is None:
                raise ValueError("imagine trajectory requires sampled prior state")
            if self.posterior is not None or self.posterior_logits is not None:
                raise ValueError("imagine trajectory must not contain posterior values")
            state = self.prior
            logits = self.prior_logits
        else:
            raise ValueError(f"unknown RSSM trajectory mode: {self.mode}")
        if state.deter.ndim < 3 or state.stoch.ndim < 4:
            raise ValueError("RSSM trajectory states must have batch and time axes")
        leading = state.deter.shape[:-1]
        if len(leading) != 2 or state.stoch.shape[:-2] != leading:
            raise ValueError("RSSM trajectory state leading shapes do not match")
        if logits.shape[:-2] != leading or self.prior_logits.shape[:-2] != leading:
            raise ValueError("RSSM trajectory logits leading shapes do not match")
        if logits.shape[-2:] != state.stoch.shape[-2:]:
            raise ValueError(
                "RSSM trajectory logits and stochastic shapes do not match"
            )
        if self.prior_logits.shape[-2:] != state.stoch.shape[-2:]:
            raise ValueError("RSSM prior logits and stochastic shapes do not match")
        expected_width = state.deter.shape[-1] + math.prod(state.stoch.shape[-2:])
        if self.features.shape != (*leading, expected_width):
            raise ValueError("RSSM trajectory feature shape does not match states")
        batch = leading[0]
        if self.final_state.deter.shape != (batch, state.deter.shape[-1]):
            raise ValueError("RSSM final deterministic state shape does not match")
        if self.final_state.stoch.shape != (batch, *state.stoch.shape[-2:]):
            raise ValueError("RSSM final stochastic state shape does not match")


def ninjax_scan_sample_keys(root_seed: Array, length: int) -> Array:
    if length <= 0:
        raise ValueError("scan length must be positive")
    contexts = jax.random.split(root_seed, length + 1)[1:]
    return jax.vmap(lambda key: jax.random.split(key, 16)[1])(contexts)


class RSSM(nn.Module):
    config: RSSMConfig
    action_spaces: Mapping[str, TensorSpace]
    param_dtype: Any = _f32
    compute_dtype: Any = _DEFAULT_COMPUTE_DTYPE

    def setup(self) -> None:
        if not self.action_spaces:
            raise ValueError("RSSM requires at least one action space")
        action_dim = 0
        for space in self.action_spaces.values():
            size = math.prod(space.shape or (1,))
            if space.discrete:
                classes = space.class_values.reshape(-1)
                if not len(classes) or not np.all(classes == classes[0]):
                    raise ValueError("RSSM actions require uniform discrete classes")
                size *= int(classes[0])
            action_dim += size
        self.core = BlockGRU(
            self.config,
            action_dim,
            param_dtype=self.param_dtype,
            compute_dtype=self.compute_dtype,
            name="core",
        )
        self.prior_layers = tuple(
            Linear(
                self.config.hidden,
                initializer=self.config.initializer,
                param_dtype=self.param_dtype,
                compute_dtype=self.compute_dtype,
                name=f"prior{index}",
            )
            for index in range(self.config.image_layers)
        )
        self.prior_norms = tuple(
            RMSNorm(
                param_dtype=self.param_dtype,
                compute_dtype=self.compute_dtype,
                name=f"prior{index}norm",
            )
            for index in range(self.config.image_layers)
        )
        self.prior_logit = Linear(
            self.config.stoch * self.config.classes,
            initializer=self.config.initializer,
            output_scale=self.config.output_scale,
            param_dtype=self.param_dtype,
            compute_dtype=self.compute_dtype,
            name="priorlogit",
        )
        self.obs_layers = tuple(
            Linear(
                self.config.hidden,
                initializer=self.config.initializer,
                param_dtype=self.param_dtype,
                compute_dtype=self.compute_dtype,
                name=f"obs{index}",
            )
            for index in range(self.config.observation_layers)
        )
        self.obs_norms = tuple(
            RMSNorm(
                param_dtype=self.param_dtype,
                compute_dtype=self.compute_dtype,
                name=f"obs{index}norm",
            )
            for index in range(self.config.observation_layers)
        )
        self.obs_logit = Linear(
            self.config.stoch * self.config.classes,
            initializer=self.config.initializer,
            output_scale=self.config.output_scale,
            param_dtype=self.param_dtype,
            compute_dtype=self.compute_dtype,
            name="obslogit",
        )

    @property
    def entry_space(self) -> dict[str, TensorSpace]:
        return {
            "deter": TensorSpace((self.config.deter,), "float32"),
            "stoch": TensorSpace((self.config.stoch, self.config.classes), "float32"),
        }

    def _validate_state(self, state: RSSMState) -> None:
        if state.deter.shape[:-1] != state.stoch.shape[:-2]:
            raise ValueError("RSSM state leading shapes do not match")
        if state.deter.shape[-1:] != (self.config.deter,):
            raise ValueError("RSSM deterministic state shape does not match config")
        if state.stoch.shape[-2:] != (self.config.stoch, self.config.classes):
            raise ValueError("RSSM stochastic state shape does not match config")
        if not jnp.issubdtype(state.deter.dtype, jnp.floating):
            raise TypeError("RSSM deterministic state must be floating")
        if not jnp.issubdtype(state.stoch.dtype, jnp.floating):
            raise TypeError("RSSM stochastic state must be floating")

    def _cast_state(self, state: RSSMState) -> RSSMState:
        self._validate_state(state)
        return RSSMState(
            state.deter.astype(self.compute_dtype),
            state.stoch.astype(self.compute_dtype),
        )

    def initial(self, batch_size: int) -> RSSMState:
        if batch_size <= 0:
            raise ValueError("RSSM batch size must be positive")
        return RSSMState(
            jnp.zeros((batch_size, self.config.deter), self.compute_dtype),
            jnp.zeros(
                (batch_size, self.config.stoch, self.config.classes),
                self.compute_dtype,
            ),
        )

    def reset(self, state: RSSMState, is_first: Array) -> RSSMState:
        state = self._cast_state(state)
        reset = jnp.asarray(is_first)
        if reset.dtype != jnp.bool_:
            raise TypeError("RSSM reset flag must be boolean")
        if reset.shape != state.deter.shape[:-1]:
            raise ValueError("RSSM reset shape must match state leading shape")
        deter_mask = reset[(...,) + (None,)]
        stoch_mask = reset[(...,) + (None, None)]
        return RSSMState(
            jnp.where(deter_mask, jnp.zeros_like(state.deter), state.deter),
            jnp.where(stoch_mask, jnp.zeros_like(state.stoch), state.stoch),
        )

    def getfeat(self, state: RSSMState) -> Array:
        state = self._cast_state(state)
        flat = state.stoch.reshape((*state.stoch.shape[:-2], -1))
        return jnp.concatenate([state.deter, flat], -1)

    @staticmethod
    def _mask(value: Array, available: Array) -> Array:
        expanded = available[(...,) + (None,) * (value.ndim - available.ndim)]
        return jnp.where(expanded, value, jnp.zeros_like(value))

    def _flatten_action(self, actions: Mapping[str, Array]) -> Array:
        if set(actions) != set(self.action_spaces):
            raise ValueError("RSSM action keys do not match declared spaces")
        result = []
        leading = None
        for key in sorted(self.action_spaces):
            space = self.action_spaces[key]
            value = jnp.asarray(actions[key])
            if value.ndim < len(space.shape):
                raise ValueError(f"RSSM action rank mismatch: {key}")
            bdims = value.ndim - len(space.shape)
            current_leading = value.shape[:bdims]
            if value.shape[bdims:] != space.shape:
                raise ValueError(f"RSSM action shape mismatch: {key}")
            if leading is None:
                leading = current_leading
            elif leading != current_leading:
                raise ValueError("RSSM action leading shapes do not match")
            event_axes = tuple(range(bdims, value.ndim))
            if jnp.issubdtype(value.dtype, jnp.floating):
                available = value != -jnp.inf
            elif jnp.issubdtype(value.dtype, jnp.signedinteger):
                available = value != -1
            elif (
                jnp.issubdtype(value.dtype, jnp.unsignedinteger) or value.dtype == bool
            ):
                available = jnp.ones(value.shape, bool)
            else:
                raise TypeError(f"unsupported RSSM action dtype: {value.dtype}")
            if event_axes:
                available = available.all(event_axes)
            value = self._mask(value, available)
            if space.discrete:
                classes = space.class_values.reshape(-1)
                if not np.all(classes == classes[0]):
                    raise ValueError("RSSM actions require uniform discrete classes")
                value = jax.nn.one_hot(
                    value.astype(jnp.int32),
                    int(classes[0]),
                    dtype=self.compute_dtype,
                )
            else:
                value = value.astype(self.compute_dtype)
            value = self._mask(value, available)
            result.append(value.reshape((*current_leading, -1)))
        return jnp.concatenate(result, -1)

    def _prior_logits(self, deter: Array) -> Array:
        value = jnp.asarray(deter).astype(self.compute_dtype)
        if value.shape[-1] != self.config.deter:
            raise ValueError("RSSM prior deter width does not match config")
        for layer, norm in zip(self.prior_layers, self.prior_norms, strict=True):
            value = _activation(self.config.activation, norm(layer(value)))
        value = self.prior_logit(value)
        return value.reshape(
            (*value.shape[:-1], self.config.stoch, self.config.classes)
        )

    def _posterior_logits(self, deter: Array, tokens: Array) -> Array:
        deter = jnp.asarray(deter).astype(self.compute_dtype)
        tokens = jnp.asarray(tokens).astype(self.compute_dtype)
        if tokens.shape[: deter.ndim - 1] != deter.shape[:-1]:
            raise ValueError("RSSM token leading shape does not match state")
        tokens = tokens.reshape((*deter.shape[:-1], -1))
        value = tokens if self.config.absolute else jnp.concatenate([deter, tokens], -1)
        for layer, norm in zip(self.obs_layers, self.obs_norms, strict=True):
            value = _activation(self.config.activation, norm(layer(value)))
        value = self.obs_logit(value)
        return value.reshape(
            (*value.shape[:-1], self.config.stoch, self.config.classes)
        )

    def _dist(self, logits: Array) -> AggregateOutput:
        return AggregateOutput(OneHotOutput(logits, self.config.unimix), 1, jnp.sum)

    def _sample(self, logits: Array, sample_key: Array) -> Array:
        return self._dist(logits).sample(sample_key).astype(self.compute_dtype)

    def img_step(
        self,
        state: RSSMState,
        action: Mapping[str, Array],
        sample_key: Array,
    ) -> tuple[RSSMState, Array]:
        state = self._cast_state(state)
        action_value = self._flatten_action(action)
        if action_value.shape[:-1] != state.deter.shape[:-1]:
            raise ValueError("RSSM action leading shape does not match state")
        deter = self.core(state.deter, state.stoch, action_value)
        logits = self._prior_logits(deter)
        return RSSMState(deter, self._sample(logits, sample_key)), logits

    def obs_step(
        self,
        state: RSSMState,
        action: Mapping[str, Array],
        tokens: Array,
        is_first: Array,
        sample_key: Array,
    ) -> tuple[RSSMState, Array]:
        state = self.reset(state, is_first)
        action_value = self._flatten_action(action)
        reset = jnp.asarray(is_first)
        action_value = self._mask(action_value, ~reset)
        deter = self.core(state.deter, state.stoch, action_value)
        logits = self._posterior_logits(deter, tokens)
        return RSSMState(deter, self._sample(logits, sample_key)), logits

    def observe(
        self,
        initial: RSSMState,
        tokens: Array,
        actions: Mapping[str, Array],
        is_first: Array,
        sample_keys: Array,
    ) -> RSSMTrajectory:
        initial = self._cast_state(initial)
        if tokens.ndim < 3 or tokens.shape[:2] != is_first.shape:
            raise ValueError("RSSM observe tokens/reset require [B,T] leading axes")
        batch, time = is_first.shape
        if time <= 0:
            raise ValueError("RSSM observe sequence length must be positive")
        if initial.deter.shape[:-1] != (batch,):
            raise ValueError("RSSM observe initial batch does not match inputs")
        if sample_keys.shape[0] != time:
            raise ValueError("RSSM observe requires one sample key per time step")
        time_actions = jax.tree.map(lambda value: jnp.swapaxes(value, 0, 1), actions)
        inputs = (
            jnp.swapaxes(tokens, 0, 1),
            time_actions,
            jnp.swapaxes(is_first, 0, 1),
            sample_keys,
        )

        if self.is_initializing():
            first_reset = is_first[:, 0]
            first_state = self.reset(initial, first_reset)
            first_action = jax.tree.map(lambda value: value[:, 0], actions)
            action_value = self._flatten_action(first_action)
            action_value = self._mask(action_value, ~first_reset)
            deter = self.core(first_state.deter, first_state.stoch, action_value)
            self._posterior_logits(deter, tokens[:, 0])
            self._prior_logits(deter)

        def step(carry, values):
            step_tokens, step_actions, reset, key = values
            state, logits = self.obs_step(carry, step_actions, step_tokens, reset, key)
            return state, (state, logits)

        final, (states, posterior_logits) = jax.lax.scan(step, initial, inputs)
        posterior = jax.tree.map(lambda value: jnp.swapaxes(value, 0, 1), states)
        posterior_logits = jnp.swapaxes(posterior_logits, 0, 1)
        prior_logits = self._prior_logits(posterior.deter)
        return RSSMTrajectory(
            posterior,
            None,
            posterior_logits,
            prior_logits,
            self.getfeat(posterior),
            final,
            "observe",
        )

    def imagine(
        self,
        initial: RSSMState,
        actions: Mapping[str, Array],
        sample_keys: Array,
    ) -> RSSMTrajectory:
        initial = self._cast_state(initial)
        leaves = jax.tree.leaves(actions)
        if not leaves or leaves[0].ndim < 2:
            raise ValueError("RSSM imagine actions require [B,T] leading axes")
        batch, time = leaves[0].shape[:2]
        if time <= 0:
            raise ValueError("RSSM imagine sequence length must be positive")
        if initial.deter.shape[:-1] != (batch,) or sample_keys.shape[0] != time:
            raise ValueError("RSSM imagine initial/key shapes do not match actions")
        time_actions = jax.tree.map(lambda value: jnp.swapaxes(value, 0, 1), actions)

        if self.is_initializing():
            first_action = jax.tree.map(lambda value: value[:, 0], actions)
            action_value = self._flatten_action(first_action)
            deter = self.core(initial.deter, initial.stoch, action_value)
            self._prior_logits(deter)

        def step(carry, values):
            action, key = values
            state, logits = self.img_step(carry, action, key)
            return state, (state, logits)

        final, (states, prior_logits) = jax.lax.scan(
            step, initial, (time_actions, sample_keys)
        )
        prior = jax.tree.map(lambda value: jnp.swapaxes(value, 0, 1), states)
        prior_logits = jnp.swapaxes(prior_logits, 0, 1)
        return RSSMTrajectory(
            None,
            prior,
            None,
            prior_logits,
            self.getfeat(prior),
            final,
            "imagine",
        )

    def dyn_loss(self, posterior_logits: Array, prior_logits: Array) -> Array:
        posterior = self._dist(jax.lax.stop_gradient(posterior_logits))
        prior = self._dist(prior_logits)
        value = posterior.kl(prior)
        if self.config.free_nats:
            value = jnp.maximum(value, self.config.free_nats)
        return value

    def rep_loss(self, posterior_logits: Array, prior_logits: Array) -> Array:
        posterior = self._dist(posterior_logits)
        prior = self._dist(jax.lax.stop_gradient(prior_logits))
        value = posterior.kl(prior)
        if self.config.free_nats:
            value = jnp.maximum(value, self.config.free_nats)
        return value

    def truncate(self, entries: RSSMState, carry: RSSMState | None = None) -> RSSMState:
        del carry
        if entries.deter.ndim != 3 or entries.stoch.ndim != 4:
            raise ValueError("RSSM truncate requires [B,T] entries")
        return jax.tree.map(lambda value: value[:, -1], entries)

    def starts(self, entries: RSSMState, carry: RSSMState, nlast: int) -> RSSMState:
        if nlast <= 0 or nlast > entries.deter.shape[1]:
            raise ValueError("RSSM starts nlast is outside sequence length")
        self._validate_state(carry)
        batch = carry.deter.shape[0]
        if entries.deter.shape[0] != batch:
            raise ValueError("RSSM starts carry batch does not match entries")
        return jax.tree.map(
            lambda value: value[:, -nlast:].reshape((batch * nlast, *value.shape[2:])),
            entries,
        )


DreamerRSSM = RSSM


def flatten_rssm_state(state: RSSMState) -> Array:
    flat = state.stoch.reshape((*state.stoch.shape[:-2], -1))
    return jnp.concatenate([state.deter, flat], -1)


def initial_rssm_state(*, batch_size: int, config: RSSMConfig) -> RSSMState:
    return RSSMState(
        jnp.zeros((batch_size, config.deter), _DEFAULT_COMPUTE_DTYPE),
        jnp.zeros((batch_size, config.stoch, config.classes), _DEFAULT_COMPUTE_DTYPE),
    )


def reset_rssm_state(
    state: RSSMState,
    is_first: Array,
    *,
    config: RSSMConfig,
) -> RSSMState:
    if state.deter.shape[-1] != config.deter:
        raise ValueError("RSSM state/config mismatch")
    reset = jnp.asarray(is_first, bool)
    if reset.shape != state.deter.shape[:-1]:
        raise ValueError("RSSM reset shape must match state")
    return RSSMState(
        jnp.where(reset[..., None], jnp.zeros_like(state.deter), state.deter),
        jnp.where(reset[..., None, None], jnp.zeros_like(state.stoch), state.stoch),
    )


__all__ = [
    "DreamerRSSM",
    "RSSM",
    "RSSMState",
    "RSSMTrajectory",
    "RSSM_SOURCE_SPEC",
    "ninjax_scan_sample_keys",
]
