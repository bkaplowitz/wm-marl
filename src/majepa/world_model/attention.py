"""Causal temporal attention and its incremental key/value cache."""

import math

import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj

f32 = jnp.float32


def _reset_array(value, reset):
    shape = (reset.shape[0],) + (1,) * (value.ndim - 1)
    return jnp.where(reset.reshape(shape), jnp.zeros_like(value), value)


class CausalTransformer(nj.Module):
    """Causal sequence model with equivalent parallel and recurrent paths."""

    units: int = 512
    output: int = 2048
    layers: int = 2
    heads: int = 8
    context: int = 64
    ffup: int = 4
    act: str = "silu"
    norm: str = "rms"
    winit: str = "trunc_normal_in"
    condition_mode: str = "add"

    def __init__(self, pair_dim, **kw):
        self.pair_dim = pair_dim
        self.kw = kw
        if self.condition_mode not in {"add", "adaln"}:
            raise ValueError(
                "causal Transformer condition_mode must be 'add' or 'adaln'"
            )

    def initial(self, batch_size):
        head_dim = self.units // self.heads
        dtype = nn.COMPUTE_DTYPE
        return {
            "keys": jnp.zeros(
                (batch_size, self.layers, self.context, self.heads, head_dim),
                dtype,
            ),
            "values": jnp.zeros(
                (batch_size, self.layers, self.context, self.heads, head_dim),
                dtype,
            ),
            "valid": jnp.zeros((batch_size, self.context), bool),
            "position": -jnp.ones((batch_size,), jnp.int32),
        }

    def sequence(self, cache, previous_pairs, resets, condition=None):
        """Process replay in parallel and retain every imagination-start cache."""

        batch, length = previous_pairs.shape[:2]
        projected = self.sub(
            "pair_projection", nn.Linear, self.units, winit=self.winit
        )(nn.cast(previous_pairs))
        if condition is not None:
            condition = nn.cast(condition)
            if condition.shape[:2] != projected.shape[:2]:
                raise ValueError(
                    "causal Transformer condition must share batch/time axes: "
                    f"{condition.shape} versus {projected.shape}"
                )
        start = self.value("start_token", nn.init("trunc_normal"), (self.units,), f32)
        start = nn.cast(jnp.broadcast_to(start, projected.shape))
        x = jnp.where(resets[..., None], start, projected)
        positions = _episode_positions(resets, cache["position"])
        segments = jnp.cumsum(resets.astype(jnp.int32), axis=1)

        old_segments = jnp.zeros((batch, self.context), jnp.int32)
        key_segments = jnp.concatenate([old_segments, segments], axis=1)
        key_valid = jnp.concatenate(
            [cache["valid"], jnp.ones((batch, length), bool)], axis=1
        )
        old_mask = cache["valid"][:, None, :] & (segments[:, :, None] == 0)
        causal = jnp.arange(length)[None, None, :] <= jnp.arange(length)[None, :, None]
        new_mask = causal & (segments[:, :, None] == segments[:, None, :])
        attention_mask = jnp.concatenate([old_mask, new_mask], axis=-1)
        old_positions = (
            cache["position"][:, None]
            - jnp.arange(self.context - 1, -1, -1, dtype=jnp.int32)[None]
        )
        key_positions = jnp.concatenate([old_positions, positions], axis=1)
        within_window = (key_positions[:, None] <= positions[:, :, None]) & (
            key_positions[:, None] > positions[:, :, None] - self.context
        )
        attention_mask &= within_window

        snapshot_indices = (
            jnp.arange(length)[:, None] + jnp.arange(self.context)[None, :] + 1
        )
        gathered_valid = key_valid[:, snapshot_indices]
        gathered_segments = key_segments[:, snapshot_indices]
        snapshot_valid = gathered_valid & (gathered_segments == segments[:, :, None])

        key_snapshots = []
        value_snapshots = []
        for index in range(self.layers):
            with nj.scope(f"layer{index}"):
                modulation = None
                if condition is not None and self.condition_mode == "add":
                    x = x + self.sub(
                        "condition", nn.Linear, self.units, winit=self.winit
                    )(condition)
                elif condition is not None:
                    modulation = self.sub(
                        "condition_adaln",
                        nn.Linear,
                        6 * self.units,
                        winit=self.winit,
                        outscale=0.0,
                    )(nn.act(self.act)(condition))
                    modulation = jnp.split(modulation, 6, axis=-1)
                residual = x
                normed = self.sub("attention_norm", nn.Norm, self.norm)(x)
                if modulation is not None:
                    shift, scale, attention_gate = modulation[:3]
                    normed = normed * (1 + scale) + shift
                qkv = self.sub("qkv", nn.Linear, 3 * self.units, winit=self.winit)(
                    normed
                )
                query, key, value = jnp.split(qkv, 3, axis=-1)
                shape = (batch, length, self.heads, self.units // self.heads)
                query, key, value = [
                    item.reshape(shape) for item in (query, key, value)
                ]
                query = _rope_f32(query, positions)
                key = _rope_f32(key, positions)
                key_bank = jnp.concatenate([cache["keys"][:, index], key], axis=1)
                value_bank = jnp.concatenate([cache["values"][:, index], value], axis=1)
                logits = jnp.einsum("bthd,bshd->bhts", query, key_bank)
                logits = f32(logits) / math.sqrt(key.shape[-1])
                logits = jnp.where(attention_mask[:, None], logits, -1e30)
                weights = jax.nn.softmax(logits, axis=-1).astype(x.dtype)
                attended = jnp.einsum("bhts,bshd->bthd", weights, value_bank)
                attended = attended.reshape((batch, length, self.units))
                attended = self.sub(
                    "attention_out", nn.Linear, self.units, winit=self.winit
                )(attended)
                if modulation is not None:
                    attended = attention_gate * attended
                x = residual + attended

                residual = x
                x = self.sub("ffn_norm", nn.Norm, self.norm)(x)
                if modulation is not None:
                    shift, scale, ffn_gate = modulation[3:]
                    x = x * (1 + scale) + shift
                x = self.sub(
                    "ffn_in", nn.Linear, self.units * self.ffup, winit=self.winit
                )(x)
                x = nn.act(self.act)(x)
                x = self.sub("ffn_out", nn.Linear, self.units, winit=self.winit)(x)
                if modulation is not None:
                    x = ffn_gate * x
                x = residual + x

                key_window = key_bank[:, snapshot_indices]
                value_window = value_bank[:, snapshot_indices]
                mask = snapshot_valid[..., None, None]
                key_snapshots.append(jnp.where(mask, key_window, 0))
                value_snapshots.append(jnp.where(mask, value_window, 0))

        x = self.sub("output_norm", nn.Norm, self.norm)(x)
        states = self.sub("state_projection", nn.Linear, self.output, winit=self.winit)(
            x
        )
        key_snapshots = jnp.stack(key_snapshots, axis=2)
        value_snapshots = jnp.stack(value_snapshots, axis=2)
        snapshots = {
            "keys": key_snapshots,
            "values": value_snapshots,
            "valid": snapshot_valid,
            "position": positions,
        }
        final = {
            "keys": key_snapshots[:, -1],
            "values": value_snapshots[:, -1],
            "valid": snapshot_valid[:, -1],
            "position": positions[:, -1],
        }
        return nn.cast(final), nn.cast(states), nn.cast(snapshots)

    def step(self, cache, previous_pair, reset, condition=None):
        """Advance one timestep using the same weights as ``sequence``."""

        batch = previous_pair.shape[0]
        cache = {
            key: _reset_array(value, reset)
            for key, value in cache.items()
            if key != "position"
        } | {"position": jnp.where(reset, -1, cache["position"])}
        position = cache["position"] + 1
        projected = self.sub(
            "pair_projection", nn.Linear, self.units, winit=self.winit
        )(nn.cast(previous_pair))
        if condition is not None:
            condition = nn.cast(condition)
            if condition.shape[0] != projected.shape[0]:
                raise ValueError(
                    "causal Transformer condition must share the batch axis: "
                    f"{condition.shape} versus {projected.shape}"
                )
        start = self.value("start_token", nn.init("trunc_normal"), (self.units,), f32)
        x = jnp.where(reset[:, None], nn.cast(start), projected)
        valid = jnp.concatenate(
            [cache["valid"][:, 1:], jnp.ones((batch, 1), bool)], axis=1
        )
        next_keys = []
        next_values = []
        for index in range(self.layers):
            with nj.scope(f"layer{index}"):
                modulation = None
                if condition is not None and self.condition_mode == "add":
                    x = x + self.sub(
                        "condition", nn.Linear, self.units, winit=self.winit
                    )(condition)
                elif condition is not None:
                    modulation = self.sub(
                        "condition_adaln",
                        nn.Linear,
                        6 * self.units,
                        winit=self.winit,
                        outscale=0.0,
                    )(nn.act(self.act)(condition))
                    modulation = jnp.split(modulation, 6, axis=-1)
                residual = x
                normed = self.sub("attention_norm", nn.Norm, self.norm)(x)
                if modulation is not None:
                    shift, scale, attention_gate = modulation[:3]
                    normed = normed * (1 + scale) + shift
                qkv = self.sub("qkv", nn.Linear, 3 * self.units, winit=self.winit)(
                    normed
                )
                query, key, value = jnp.split(qkv, 3, axis=-1)
                shape = (batch, self.heads, self.units // self.heads)
                query, key, value = [
                    item.reshape(shape) for item in (query, key, value)
                ]
                timestamp = position[:, None]
                query = _rope_f32(query[:, None], timestamp)[:, 0]
                key = _rope_f32(key[:, None], timestamp)[:, 0]
                keys = jnp.concatenate(
                    [cache["keys"][:, index, 1:], key[:, None]], axis=1
                )
                values = jnp.concatenate(
                    [cache["values"][:, index, 1:], value[:, None]], axis=1
                )
                logits = jnp.einsum("bhd,bthd->bht", query, keys)
                logits = f32(logits) / math.sqrt(key.shape[-1])
                logits = jnp.where(valid[:, None], logits, -1e30)
                weights = jax.nn.softmax(logits, axis=-1).astype(x.dtype)
                attended = jnp.einsum("bht,bthd->bhd", weights, values)
                attended = attended.reshape((batch, self.units))
                attended = self.sub(
                    "attention_out", nn.Linear, self.units, winit=self.winit
                )(attended)
                if modulation is not None:
                    attended = attention_gate * attended
                x = residual + attended

                residual = x
                x = self.sub("ffn_norm", nn.Norm, self.norm)(x)
                if modulation is not None:
                    shift, scale, ffn_gate = modulation[3:]
                    x = x * (1 + scale) + shift
                x = self.sub(
                    "ffn_in", nn.Linear, self.units * self.ffup, winit=self.winit
                )(x)
                x = nn.act(self.act)(x)
                x = self.sub("ffn_out", nn.Linear, self.units, winit=self.winit)(x)
                if modulation is not None:
                    x = ffn_gate * x
                x = residual + x
                next_keys.append(keys)
                next_values.append(values)

        x = self.sub("output_norm", nn.Norm, self.norm)(x)
        state = self.sub("state_projection", nn.Linear, self.output, winit=self.winit)(
            x
        )
        next_cache = {
            "keys": jnp.stack(next_keys, axis=1),
            "values": jnp.stack(next_values, axis=1),
            "valid": valid,
            "position": position,
        }
        return nn.cast(next_cache), nn.cast(state)


def _episode_positions(resets, initial_position):
    def advance(position, reset):
        position = jnp.where(reset, 0, position + 1)
        return position, position

    _, positions = jax.lax.scan(advance, initial_position, resets.T)
    return positions.T


def _rope_f32(x, timestamps, maxlen=4096):
    """Apply RoPE with FP32 angles and trigonometry, then restore input dtype."""

    if x.shape[-1] % 2:
        raise ValueError(f"RoPE feature width must be even, got {x.shape[-1]}")
    timestamps = jnp.asarray(timestamps, f32)
    frequencies = (2.0 / x.shape[-1]) * jnp.arange(x.shape[-1] // 2, dtype=f32)
    timescales = jnp.asarray(maxlen, f32) ** frequencies
    radians = timestamps[..., None] / timescales
    sine = jnp.sin(radians)[..., None, :]
    cosine = jnp.cos(radians)[..., None, :]
    left, right = jnp.split(f32(x), 2, axis=-1)
    rotated = jnp.concatenate(
        [left * cosine - right * sine, right * cosine + left * sine],
        axis=-1,
    )
    return rotated.astype(x.dtype)
