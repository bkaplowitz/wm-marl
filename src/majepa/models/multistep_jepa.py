"""Training-only direct multi-step JEPA predictor for CTDE replay."""

from collections.abc import Sequence

import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj

f32 = jnp.float32
sg = jax.lax.stop_gradient


def isolated_creation_call(function, salt, *args, **kwargs):
    """Initialize auxiliary modules on an independent reproducible RNG stream."""

    if not nj.creating():
        return function(*args, **kwargs)
    context = nj.context()
    outer_seed = context.seed
    outer_reserve = context.reserve
    if outer_seed is None:
        return function(*args, **kwargs)
    context.seed = jax.random.fold_in(outer_seed, int(salt))
    context.reserve = []
    try:
        return function(*args, **kwargs)
    finally:
        context.seed = outer_seed
        context.reserve = outer_reserve


class ActionConditionedMultiStepJEPA(nj.Module):
    """Predict EMA futures from joint roots and focal action tails.

    The joint hidden already includes synchronized action a_t. Each horizon
    head additionally consumes the focal agent's recorded actions a_{t+1:t+h-1}.
    This auxiliary predictor is trained on replay and is not the simulator used
    to produce PPO trajectories.
    """

    width: int = 256
    layers: int = 2
    units: int = 512
    act: str = "silu"
    norm: str = "rms"
    winit: str = "trunc_normal_in"

    def __init__(
        self,
        action_count: int,
        action_low: int,
        target_dim: int,
        horizons: Sequence[int],
        max_horizon: int,
        **kwargs,
    ):
        self.action_count = int(action_count)
        self.action_low = int(action_low)
        self.target_dim = int(target_dim)
        self.horizons = tuple(int(horizon) for horizon in horizons)
        self.max_horizon = int(max_horizon)
        if self.action_count < 2 or self.target_dim < 1:
            raise ValueError("multi-step JEPA needs categorical actions and a target")
        if (
            not self.horizons
            or tuple(sorted(set(self.horizons))) != self.horizons
            or min(self.horizons) < 1
            or max(self.horizons) > self.max_horizon
        ):
            raise ValueError(
                "multi-step JEPA horizons must be sorted unique positives within K"
            )
        if self.max_horizon != max(self.horizons):
            raise ValueError("multi-step JEPA K must equal the largest horizon")
        if self.width < 1 or self.layers < 1 or self.units < 1:
            raise ValueError("multi-step JEPA widths and layer count must be positive")
        del kwargs

    def __call__(
        self,
        joint_hidden,
        action_windows,
        *,
        selected_horizon=None,
    ):
        """Apply direct heads to joint roots and focal replay-action tails."""

        if joint_hidden.ndim != 4:
            raise ValueError(
                f"multi-step shared hidden must be [B,R,A,D], got {joint_hidden.shape}"
            )
        if action_windows.shape != (*joint_hidden.shape[:-1], self.max_horizon):
            raise ValueError(
                "multi-step action windows must be [B,R,A,K], got "
                f"{action_windows.shape} for roots {joint_hidden.shape} and "
                f"K={self.max_horizon}"
            )
        if selected_horizon is not None:
            selected_horizon = int(selected_horizon)
            if selected_horizon not in self.horizons:
                raise ValueError(
                    f"selected horizon {selected_horizon} not in {self.horizons}"
                )
        root = self.sub("joint_projection", nn.Linear, self.width, winit=self.winit)(
            nn.cast(joint_hidden)
        )
        root = nn.act(self.act)(self.sub("joint_norm", nn.Norm, self.norm)(root))
        action_index = action_windows.astype(jnp.int32) - self.action_low
        in_range = (action_index >= 0) & (action_index < self.action_count)
        onehot = jax.nn.one_hot(
            jnp.clip(action_index, 0, self.action_count - 1),
            self.action_count,
            dtype=f32,
        )
        onehot *= in_range[..., None].astype(f32)
        predictions = {}
        for horizon in self.horizons:
            if selected_horizon is not None and horizon != selected_horizon:
                continue
            # ``joint_hidden_t`` already consumed a_t.  Supplying only positions
            # 1..h-1 avoids duplicating that action while preserving the complete
            # a_t..a_{t+h-1} conditioning across the two inputs.
            tail_positions = (jnp.arange(self.max_horizon) >= 1) & (
                jnp.arange(self.max_horizon) < horizon
            )
            prefix = onehot * tail_positions.astype(f32)[None, None, None, :, None]
            prefix = prefix.reshape((*prefix.shape[:-2], -1))
            action = self.sub(
                f"h{horizon}_action", nn.Linear, self.width, winit=self.winit
            )(nn.cast(prefix))
            action = nn.act(self.act)(
                self.sub(f"h{horizon}_action_norm", nn.Norm, self.norm)(action)
            )
            fused = jnp.concatenate([root, action], axis=-1)
            hidden = self.sub(
                f"h{horizon}_trunk",
                nn.MLP,
                self.layers,
                self.units,
                act=self.act,
                norm=self.norm,
                winit=self.winit,
            )(fused)
            prediction = self.sub(
                f"h{horizon}_prediction",
                nn.Linear,
                self.target_dim,
                winit=self.winit,
            )(hidden)
            predictions[horizon] = prediction
        return predictions


__all__ = [
    "ActionConditionedMultiStepJEPA",
    "isolated_creation_call",
]
