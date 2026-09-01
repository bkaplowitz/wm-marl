"""Training-only direct multi-step JEPA predictor for CTDE replay."""

from collections.abc import Sequence

import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj

f32 = jnp.float32
sg = jax.lax.stop_gradient


def isolated_creation_call(function, salt, *args, **kwargs):
    """Create treatment parameters without advancing the base RNG stream."""

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


class TeammateActionPlanGRU(nj.Module):
    """Shared causal future-peer planner anchored to stopped TBv2 beliefs.

    One decoder is reused for every peer and every future step. Its state is
    initialized from the stopped focal root and centered current-action belief.
    To emit ``q^k`` it consumes the last causal focal-prefix action
    ``a_{t+k-1}`` and the previous predicted peer-action expectation. Delta logits are zero
    initialized over the stopped TBv2 ``q0`` baseline.
    """

    units: int = 256
    act: str = "silu"
    norm: str = "rms"
    winit: str = "trunc_normal_in"

    def __init__(
        self,
        action_count: int,
        action_low: int,
        peers: int,
        max_horizon: int,
        **kwargs,
    ):
        self.action_count = int(action_count)
        self.action_low = int(action_low)
        self.peers = int(peers)
        self.max_horizon = int(max_horizon)
        if self.action_count < 2 or self.peers < 1 or self.max_horizon < 2:
            raise ValueError(
                "teammate action plan needs peers, categorical actions, and K >= 2"
            )
        if self.units < 1:
            raise ValueError("teammate action plan units must be positive")
        del kwargs

    def __call__(self, local_root, action_windows, q0_logits, q0_context):
        if local_root.ndim != 4:
            raise ValueError(
                f"teammate plan roots must be [B,R,A,F], got {local_root.shape}"
            )
        batch_shape = local_root.shape[:-1]
        if action_windows.shape != (*batch_shape, self.max_horizon):
            raise ValueError("teammate plan actions do not align with local roots")
        belief_shape = (*batch_shape, self.peers, self.action_count)
        if q0_logits.shape != belief_shape or q0_context.shape != belief_shape:
            raise ValueError(
                "teammate q0 logits/context do not align with plan roots: "
                f"{q0_logits.shape}, {q0_context.shape}, expected {belief_shape}"
            )

        local_root = nn.cast(sg(local_root))
        q0_logits = sg(q0_logits.astype(f32))
        q0_context = nn.cast(sg(q0_context))
        repeated_root = jnp.broadcast_to(
            local_root[..., None, :], (*batch_shape, self.peers, local_root.shape[-1])
        )
        initial = jnp.concatenate([repeated_root, q0_context], axis=-1)
        carry = self.sub("initial_projection", nn.Linear, self.units, winit=self.winit)(
            initial
        )
        carry = nn.act(self.act)(self.sub("initial_norm", nn.Norm, self.norm)(carry))

        action_index = action_windows.astype(jnp.int32) - self.action_low
        in_range = (action_index >= 0) & (action_index < self.action_count)
        action_onehot = jax.nn.one_hot(
            jnp.clip(action_index, 0, self.action_count - 1),
            self.action_count,
            dtype=f32,
        )
        action_onehot *= in_range[..., None].astype(f32)
        previous_expectation = jax.nn.softmax(q0_logits, axis=-1)
        outputs = []
        decoder = self.sub(
            "decoder", nn.GRU, units=self.units, norm=self.norm, winit=self.winit
        )
        delta_head = self.sub(
            "delta_logits",
            nn.Linear,
            self.action_count,
            winit=self.winit,
            outscale=0.0,
        )
        for step in range(1, self.max_horizon):
            own_action = jnp.broadcast_to(
                action_onehot[..., step - 1, None, :],
                (*batch_shape, self.peers, self.action_count),
            )
            decoder_input = nn.cast(
                jnp.concatenate([own_action, previous_expectation], axis=-1)
            )
            carry, output = decoder.step(
                carry,
                decoder_input,
                jnp.zeros(batch_shape + (self.peers,), bool),
            )
            logits = q0_logits + delta_head(output).astype(f32)
            outputs.append(logits)
            previous_expectation = jax.nn.softmax(logits, axis=-1)
        return jnp.stack(outputs, axis=-3)


class ActionConditionedMultiStepJEPA(nj.Module):
    """Predict EMA futures from joint roots, own actions, and teammate belief.

    The shared hidden already encodes the stopped factual local state, joint
    root context, and synchronized action ``a_t``. Each head additionally sees
    only its focal agent's future replay-action tail ``a_{t+1:t+h-1}`` and a
    detached, own-observation-conditioned TBv2 context from the same factual
    root. The belief path is an additive zero-initialized residual, so adding it
    cannot perturb the standalone predictor at initialization and a uniform
    teammate belief remains exactly inert after training.
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

    def _belief_residual(self, root, action, belief_context, horizon):
        """Return a zero-initialized per-head belief-conditioned correction."""

        belief = self.sub(
            f"h{horizon}_belief_projection",
            nn.Linear,
            self.width,
            bias=False,
            winit=self.winit,
        )(nn.cast(sg(belief_context)))
        belief = nn.act(self.act)(
            self.sub(f"h{horizon}_belief_norm", nn.Norm, self.norm)(belief)
        )
        value = jnp.concatenate([belief, root * belief, action * belief], axis=-1)
        for index in range(self.layers):
            value = self.sub(
                f"h{horizon}_belief_fusion{index}",
                nn.Linear,
                self.units,
                bias=False,
                winit=self.winit,
            )(value)
            value = nn.act(self.act)(
                self.sub(f"h{horizon}_belief_fusion_norm{index}", nn.Norm, self.norm)(
                    value
                )
            )
        return self.sub(
            f"h{horizon}_belief_prediction",
            nn.Linear,
            self.target_dim,
            bias=False,
            winit=self.winit,
            outscale=0.0,
        )(value)

    def _aggregate_belief_plan(self, belief_plan):
        """Mean-pool the stopped per-peer action plan."""

        peer_plan = self.sub(
            "belief_peer_projection",
            nn.Linear,
            self.width,
            bias=False,
            winit=self.winit,
        )(nn.cast(sg(belief_plan)))
        peer_plan = nn.act(self.act)(
            self.sub("belief_peer_norm", nn.Norm, self.norm)(peer_plan)
        )
        return peer_plan.mean(axis=-2)

    def __call__(
        self,
        joint_hidden,
        action_windows,
        belief_plan=None,
        *,
        selected_horizon=None,
    ):
        """Apply direct heads to live roots, own tails, and stopped TBv2 context."""

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
        if belief_plan is not None:
            if (
                belief_plan.ndim != 6
                or belief_plan.shape[:3] != joint_hidden.shape[:3]
                or belief_plan.shape[3] != self.max_horizon - 1
                or belief_plan.shape[4] < 1
                or belief_plan.shape[5] != self.action_count
            ):
                raise ValueError(
                    "multi-step belief plan must be [B,R,A,K-1,P,C], got "
                    f"{belief_plan.shape} for roots {joint_hidden.shape}"
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
        pooled_plan = (
            isolated_creation_call(
                self._aggregate_belief_plan,
                0x4D534250,
                belief_plan,
            )
            if belief_plan is not None
            else None
        )
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
            if pooled_plan is None:
                predictions[horizon] = prediction
            else:
                plan_positions = jnp.arange(self.max_horizon - 1) < horizon - 1
                belief_context = (
                    pooled_plan * plan_positions.astype(f32)[None, None, None, :, None]
                )
                belief_context = belief_context.reshape((*belief_context.shape[:3], -1))
                belief_residual = isolated_creation_call(
                    self._belief_residual,
                    0x4D534250 + horizon,
                    root,
                    action,
                    belief_context,
                    horizon,
                )
                predictions[horizon] = prediction + belief_residual
        return predictions


__all__ = [
    "ActionConditionedMultiStepJEPA",
    "TeammateActionPlanGRU",
    "isolated_creation_call",
]
