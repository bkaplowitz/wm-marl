from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from flax import struct
from jax import Array as JaxArray

from .networks import TensorSpace
from .rssm import RSSMState


Array = npt.NDArray[np.generic]


def _plain(value: object, label: str, seen: set[int] | None = None) -> None:
    if seen is None:
        seen = set()
    if type(value) is dict:
        identity = id(value)
        if identity in seen:
            raise ValueError(f"{label} contains a mutable alias")
        seen.add(identity)
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"{label} keys must be exact strings")
            _plain(item, label, seen)
        return
    if type(value) is np.ndarray:
        identity = id(value)
        if identity in seen:
            raise ValueError(f"{label} contains a mutable alias")
        seen.add(identity)
        if value.dtype.hasobject:
            raise TypeError(f"{label} cannot contain object arrays")
        return
    if value is None or type(value) in (str, bytes, bool, int, float):
        return
    raise TypeError(f"{label} contains a nonprimitive leaf")


def _exact_record(value: object, keys: set[str], label: str) -> Mapping[str, Any]:
    if type(value) is not dict:
        raise TypeError(f"{label} must be a plain dictionary")
    if set(value) != keys:
        raise ValueError(f"{label} keys differ")
    return value


def _validated_leaf(
    value: object,
    space: TensorSpace,
    leading_shape: tuple[int, ...],
    label: str,
) -> Array:
    array = np.asarray(value)
    expected = (*leading_shape, *space.shape)
    if array.dtype != np.dtype(space.dtype) or array.shape != expected:
        raise ValueError(f"{label} has wrong dtype or shape")
    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
        raise ValueError(f"{label} must be finite")
    return array.copy()


def validate_action_tree(
    action: Mapping[str, object],
    action_spaces: Mapping[str, TensorSpace],
    leading_shape: tuple[int, ...],
) -> dict[str, Array]:
    if not isinstance(action, Mapping) or set(action) != set(action_spaces):
        raise ValueError("action tree schema differs")
    if type(leading_shape) is not tuple or any(
        type(size) is not int or size < 0 for size in leading_shape
    ):
        raise TypeError("action leading shape must be a tuple of nonnegative integers")
    return {
        name: _validated_leaf(action[name], space, leading_shape, f"action {name}")
        for name, space in sorted(action_spaces.items())
    }


def validate_replay_row(
    row: Mapping[str, object],
    action_spaces: Mapping[str, TensorSpace],
    leading_shape: tuple[int, ...],
) -> None:
    action = validate_action_tree(
        {name: row[name] for name in action_spaces}, action_spaces, leading_shape
    )
    is_last = np.asarray(row["is_last"])
    is_terminal = np.asarray(row["is_terminal"])
    if is_last.dtype != np.dtype(bool) or is_terminal.dtype != np.dtype(bool):
        raise ValueError("replay boundary flags must have bool dtype")
    if is_last.shape != leading_shape or is_terminal.shape != leading_shape:
        raise ValueError("replay boundary flags have wrong shape")
    if np.any(is_terminal & ~is_last):
        raise ValueError("is_terminal requires is_last")
    for value in action.values():
        mask = is_last.reshape(
            (*leading_shape, *((1,) * (value.ndim - len(leading_shape))))
        )
        if np.any(np.where(mask, value, 0) != 0):
            raise ValueError("final replay row action must be zero")


@struct.dataclass
class AgentCarry:
    encoder: Mapping[str, Any]
    rssm: RSSMState
    decoder: Mapping[str, Any]
    prev_action: Mapping[str, Any]

    def state_dict(self) -> dict[str, object]:
        return {
            "decoder": {
                name: np.asarray(value).copy() for name, value in self.decoder.items()
            },
            "encoder": {
                name: np.asarray(value).copy() for name, value in self.encoder.items()
            },
            "prev_action": {
                name: np.asarray(value).copy()
                for name, value in self.prev_action.items()
            },
            "rssm": {
                "deter": np.asarray(self.rssm.deter).copy(),
                "stoch": np.asarray(self.rssm.stoch).copy(),
            },
        }

    @classmethod
    def from_state(
        cls,
        state: object,
        agent: DreamerAgent,
        expected_leading_shape: tuple[int, ...],
    ) -> AgentCarry:
        _plain(state, "AgentCarry state")
        record = _exact_record(
            state, {"decoder", "encoder", "prev_action", "rssm"}, "AgentCarry state"
        )
        encoder = agent._validate_aux_carry(
            record["encoder"],
            agent.encoder_carry_spaces,
            expected_leading_shape,
            "encoder",
        )
        decoder = agent._validate_aux_carry(
            record["decoder"],
            agent.decoder_carry_spaces,
            expected_leading_shape,
            "decoder",
        )
        previous = validate_action_tree(
            record["prev_action"], agent.action_spaces, expected_leading_shape
        )
        rssm = _exact_record(record["rssm"], {"deter", "stoch"}, "AgentCarry RSSM")
        config = agent.rssm_config
        deter = np.asarray(rssm["deter"])
        stoch = np.asarray(rssm["stoch"])
        dtype = np.dtype(agent.compute_dtype)
        if deter.dtype != dtype or deter.shape != (
            *expected_leading_shape,
            config.deter,
        ):
            raise ValueError("AgentCarry deter has wrong dtype or shape")
        if stoch.dtype != dtype or stoch.shape != (
            *expected_leading_shape,
            config.stoch,
            config.classes,
        ):
            raise ValueError("AgentCarry stoch has wrong dtype or shape")
        return cls(
            encoder,
            RSSMState(jnp.asarray(deter.copy()), jnp.asarray(stoch.copy())),
            decoder,
            previous,
        )


@dataclass(frozen=True)
class DreamerAgent:
    obs_space: Mapping[str, TensorSpace]
    act_space: Mapping[str, TensorSpace]
    config: Any

    def __post_init__(self) -> None:
        object.__setattr__(self, "obs_space", dict(sorted(self.obs_space.items())))
        object.__setattr__(self, "act_space", dict(sorted(self.act_space.items())))
        if not self.act_space:
            raise ValueError("DreamerAgent action spaces cannot be empty")

    @property
    def observation_spaces(self) -> Mapping[str, TensorSpace]:
        return self.obs_space

    @property
    def action_spaces(self) -> Mapping[str, TensorSpace]:
        return self.act_space

    @property
    def rssm_config(self):
        return self.config.rssm

    @property
    def compute_dtype(self) -> str:
        return np.dtype(self.config.compute_dtype).name

    @property
    def encoder_carry_spaces(self) -> Mapping[str, TensorSpace]:
        return {}

    @property
    def decoder_carry_spaces(self) -> Mapping[str, TensorSpace]:
        return {}

    def _validate_aux_carry(
        self,
        value: object,
        spaces: Mapping[str, TensorSpace],
        leading_shape: tuple[int, ...],
        label: str,
    ) -> dict[str, Array]:
        record = _exact_record(value, set(spaces), f"AgentCarry {label}")
        return {
            name: _validated_leaf(record[name], space, leading_shape, f"{label} {name}")
            for name, space in spaces.items()
        }

    def initial(self, batch_size: int) -> AgentCarry:
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("agent batch size must be positive")
        dtype = jnp.dtype(self.compute_dtype)
        rssm = RSSMState(
            jnp.zeros((batch_size, self.rssm_config.deter), dtype),
            jnp.zeros(
                (batch_size, self.rssm_config.stoch, self.rssm_config.classes), dtype
            ),
        )
        previous = {
            name: jnp.zeros((batch_size, *space.shape), jnp.dtype(space.dtype))
            for name, space in self.action_spaces.items()
        }
        return AgentCarry({}, rssm, {}, previous)

    def apply_replay_context(
        self,
        carry: AgentCarry,
        data: Mapping[str, object],
    ) -> tuple[
        AgentCarry,
        dict[str, JaxArray],
        dict[str, JaxArray],
        JaxArray,
    ]:
        actions = {name: jnp.asarray(data[name]) for name in self.action_spaces}
        if not actions:
            raise ValueError("replay data has no action tree")
        batch, raw_time = next(iter(actions.values())).shape[:2]
        for name, space in self.action_spaces.items():
            if actions[name].dtype != np.dtype(space.dtype) or actions[name].shape != (
                batch,
                raw_time,
                *space.shape,
            ):
                raise ValueError(f"replay action {name} has wrong dtype or shape")
        step_ids = jnp.asarray(data["stepid"])
        if step_ids.dtype != np.uint8 or step_ids.shape != (batch, raw_time, 20):
            raise ValueError("replay step ids have wrong dtype or shape")
        previous = {
            name: jnp.concatenate(
                [jnp.asarray(carry.prev_action[name])[:, None], value[:, :-1]], axis=1
            )
            for name, value in actions.items()
        }
        context = self.config.sequence.context
        observations = {
            name: jnp.asarray(data[name]) for name in self.observation_spaces
        }
        rssm = carry.rssm
        if context > 0:
            if raw_time <= context:
                raise ValueError("replay context consumes the whole sequence")
            consec = jnp.asarray(data["consec"])
            if consec.shape != (batch, raw_time):
                raise ValueError("replay consec has wrong shape")
            first = consec[:, 0] == 0
            deter_entries = jnp.asarray(data["dyn/deter"])
            stoch_entries = jnp.asarray(data["dyn/stoch"])
            reconstructed_deter = deter_entries[:, context - 1]
            reconstructed_stoch = stoch_entries[:, context - 1]
            rssm = RSSMState(
                jnp.where(
                    first[:, None],
                    reconstructed_deter,
                    jnp.asarray(carry.rssm.deter),
                ),
                jnp.where(
                    first[:, None, None],
                    reconstructed_stoch,
                    jnp.asarray(carry.rssm.stoch),
                ),
            )
            observations = {
                name: value[:, context:] for name, value in observations.items()
            }
            normal_previous = {
                name: value[:, context:] for name, value in previous.items()
            }
            replay_previous = {
                name: value[:, context - 1 : -1] for name, value in actions.items()
            }
            previous = {
                name: jnp.where(
                    first.reshape((batch, *((1,) * (value.ndim - 1)))),
                    replay_previous[name],
                    value,
                )
                for name, value in normal_previous.items()
            }
            step_ids = step_ids[:, context:]
        outgoing = AgentCarry(
            carry.encoder,
            rssm,
            carry.decoder,
            {name: value[:, -1].copy() for name, value in actions.items()},
        )
        return outgoing, observations, previous, step_ids


__all__ = [
    "AgentCarry",
    "DreamerAgent",
    "validate_action_tree",
    "validate_replay_row",
]
