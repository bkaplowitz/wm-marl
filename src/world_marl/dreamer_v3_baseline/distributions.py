from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp


Array = jax.Array
Aggregate = Callable[[Array, Sequence[int]], Array]
_f32 = jnp.float32
_i32 = jnp.int32
_stop_gradient = jax.lax.stop_gradient


def symlog(value: Array) -> Array:
    value = jnp.asarray(value)
    return jnp.sign(value) * jnp.log1p(jnp.abs(value))


def symexp(value: Array) -> Array:
    value = jnp.asarray(value)
    return jnp.sign(value) * jnp.expm1(jnp.abs(value))


class _Output:
    def __repr__(self) -> str:
        prediction = self.pred()
        return f"{type(self).__name__}({prediction.dtype}, shape={prediction.shape})"

    def pred(self) -> Array:
        raise NotImplementedError

    def loss(self, target: Array) -> Array:
        return -self.logp(_stop_gradient(target))

    def sample(self, seed: Array, shape: tuple[int, ...] = ()) -> Array:
        raise NotImplementedError

    def logp(self, event: Array) -> Array:
        raise NotImplementedError

    def prob(self, event: Array) -> Array:
        return jnp.exp(self.logp(event))

    def entropy(self) -> Array:
        raise NotImplementedError

    def kl(self, other: Any) -> Array:
        raise NotImplementedError


class AggregateOutput(_Output):
    def __init__(
        self,
        output: _Output,
        dims: int,
        aggregate: Aggregate = jnp.sum,
    ) -> None:
        self.output = output
        self.axes = [-index for index in range(1, dims + 1)]
        self.aggregate = aggregate

    def __repr__(self) -> str:
        prediction = self.pred()
        name = type(self.output).__name__
        return f"{name}({prediction.dtype}, shape={prediction.shape}, agg={len(self.axes)})"

    def pred(self) -> Array:
        return self.output.pred()

    def loss(self, target: Array) -> Array:
        return self.aggregate(self.output.loss(target), self.axes)

    def sample(self, seed: Array, shape: tuple[int, ...] = ()) -> Array:
        return self.output.sample(seed, shape)

    def logp(self, event: Array) -> Array:
        return self.output.logp(event).sum(self.axes)

    def prob(self, event: Array) -> Array:
        return self.output.prob(event).sum(self.axes)

    def entropy(self) -> Array:
        return self.aggregate(self.output.entropy(), self.axes)

    def kl(self, other: Any) -> Array:
        assert isinstance(other, AggregateOutput), other
        return self.aggregate(self.output.kl(other.output), self.axes)


class MSEOutput(_Output):
    def __init__(
        self,
        mean: Array,
        squash: Callable[[Array], Array] | None = None,
    ) -> None:
        self.mean = _f32(mean)
        self.squash = squash or (lambda value: value)

    def pred(self) -> Array:
        return self.mean

    def loss(self, target: Array) -> Array:
        assert jnp.issubdtype(target.dtype, jnp.floating), target.dtype
        assert self.mean.shape == target.shape, (self.mean.shape, target.shape)
        target = _stop_gradient(self.squash(_f32(target)))
        return jnp.square(self.mean - target)


class NormalOutput(_Output):
    def __init__(self, mean: Array, stddev: Array | float = 1.0) -> None:
        self.mean = _f32(mean)
        self.stddev = jnp.broadcast_to(_f32(stddev), self.mean.shape)

    @classmethod
    def bounded(
        cls,
        raw_mean: Array,
        raw_stddev: Array,
        min_stddev: float,
        max_stddev: float,
    ) -> NormalOutput:
        stddev = (max_stddev - min_stddev) * jax.nn.sigmoid(
            raw_stddev + 2.0
        ) + min_stddev
        return cls(jnp.tanh(raw_mean), stddev)

    def pred(self) -> Array:
        return self.mean

    def sample(self, seed: Array, shape: tuple[int, ...] = ()) -> Array:
        sample = jax.random.normal(seed, shape + self.mean.shape, _f32)
        return sample * self.stddev + self.mean

    def logp(self, event: Array) -> Array:
        assert jnp.issubdtype(event.dtype, jnp.floating), event.dtype
        return jax.scipy.stats.norm.logpdf(_f32(event), self.mean, self.stddev)

    def entropy(self) -> Array:
        return 0.5 * jnp.log(2 * jnp.pi * jnp.square(self.stddev)) + 0.5

    def kl(self, other: Any) -> Array:
        assert isinstance(other, type(self)), (self, other)
        return 0.5 * (
            jnp.square(self.stddev / other.stddev)
            + jnp.square(other.mean - self.mean) / jnp.square(other.stddev)
            + 2 * jnp.log(other.stddev)
            - 2 * jnp.log(self.stddev)
            - 1
        )


class BinaryOutput(_Output):
    def __init__(self, logit: Array) -> None:
        self.logit = _f32(logit)

    def pred(self) -> Array:
        return self.logit > 0

    def logp(self, event: Array) -> Array:
        event = _f32(event)
        logp = jax.nn.log_sigmoid(self.logit)
        lognotp = jax.nn.log_sigmoid(-self.logit)
        return event * logp + (1 - event) * lognotp

    def sample(self, seed: Array, shape: tuple[int, ...] = ()) -> Array:
        probability = jax.nn.sigmoid(self.logit)
        return jax.random.bernoulli(
            seed,
            probability,
            shape + self.logit.shape,
        )


class CategoricalOutput(_Output):
    def __init__(self, logits: Array, unimix: float = 0.0) -> None:
        logits = _f32(logits)
        if unimix:
            probabilities = jax.nn.softmax(logits, -1)
            uniform = jnp.ones_like(probabilities) / probabilities.shape[-1]
            probabilities = (1 - unimix) * probabilities + unimix * uniform
            logits = jnp.log(probabilities)
        self.logits = logits

    def pred(self) -> Array:
        return jnp.argmax(self.logits, -1)

    def sample(self, seed: Array, shape: tuple[int, ...] = ()) -> Array:
        return jax.random.categorical(
            seed,
            self.logits,
            -1,
            shape + self.logits.shape[:-1],
        )

    def logp(self, event: Array) -> Array:
        onehot = jax.nn.one_hot(event, self.logits.shape[-1])
        return (jax.nn.log_softmax(self.logits, -1) * onehot).sum(-1)

    def entropy(self) -> Array:
        log_probability = jax.nn.log_softmax(self.logits, -1)
        probability = jax.nn.softmax(self.logits, -1)
        return -(probability * log_probability).sum(-1)

    def kl(self, other: Any) -> Array:
        assert isinstance(other, CategoricalOutput), other
        log_probability = jax.nn.log_softmax(self.logits, -1)
        log_other = jax.nn.log_softmax(other.logits, -1)
        probability = jax.nn.softmax(self.logits, -1)
        return (probability * (log_probability - log_other)).sum(-1)


class OneHotOutput(_Output):
    def __init__(self, logits: Array, unimix: float = 0.0) -> None:
        self.dist = CategoricalOutput(logits, unimix)

    def pred(self) -> Array:
        return self._onehot_with_grad(self.dist.pred())

    def sample(self, seed: Array, shape: tuple[int, ...] = ()) -> Array:
        return self._onehot_with_grad(self.dist.sample(seed, shape))

    def logp(self, event: Array) -> Array:
        return (jax.nn.log_softmax(self.dist.logits, -1) * event).sum(-1)

    def entropy(self) -> Array:
        return self.dist.entropy()

    def kl(self, other: Any) -> Array:
        assert isinstance(other, OneHotOutput), other
        return self.dist.kl(other.dist)

    def _onehot_with_grad(self, index: Array) -> Array:
        value = jax.nn.one_hot(index, self.dist.logits.shape[-1], dtype=_f32)
        probabilities = jax.nn.softmax(self.dist.logits, -1)
        return _stop_gradient(value) + (probabilities - _stop_gradient(probabilities))


class TwoHotOutput(_Output):
    def __init__(
        self,
        logits: Array,
        bins: int | Array = 255,
        squash: Callable[[Array], Array] | None = None,
        unsquash: Callable[[Array], Array] | None = None,
    ) -> None:
        logits = _f32(logits)
        resolved_bins = _symexp_bins(bins) if isinstance(bins, int) else bins
        assert logits.shape[-1] == len(resolved_bins), (
            logits.shape,
            len(resolved_bins),
        )
        assert resolved_bins.dtype == _f32, resolved_bins.dtype
        self.logits = logits
        self.probs = jax.nn.softmax(logits)
        self.bins = jnp.array(resolved_bins)
        self.squash = squash or (lambda value: value)
        self.unsquash = unsquash or (lambda value: value)

    def pred(self) -> Array:
        count = self.logits.shape[-1]
        if count % 2 == 1:
            midpoint = (count - 1) // 2
            prob_negative = self.probs[..., :midpoint]
            prob_zero = self.probs[..., midpoint : midpoint + 1]
            prob_positive = self.probs[..., midpoint + 1 :]
            bin_negative = self.bins[..., :midpoint]
            bin_zero = self.bins[..., midpoint : midpoint + 1]
            bin_positive = self.bins[..., midpoint + 1 :]
            weighted_average = (prob_zero * bin_zero).sum(-1) + (
                (prob_negative * bin_negative)[..., ::-1] + prob_positive * bin_positive
            ).sum(-1)
        else:
            midpoint = count // 2
            prob_negative = self.probs[..., :midpoint]
            prob_positive = self.probs[..., midpoint:]
            bin_negative = self.bins[..., :midpoint]
            bin_positive = self.bins[..., midpoint:]
            weighted_average = (
                (prob_negative * bin_negative)[..., ::-1] + prob_positive * bin_positive
            ).sum(-1)
        return self.unsquash(weighted_average)

    def loss(self, target: Array) -> Array:
        assert target.dtype == _f32, target.dtype
        target = _stop_gradient(self.squash(target))
        below = (self.bins <= target[..., None]).astype(_i32).sum(-1) - 1
        above = len(self.bins) - (self.bins > target[..., None]).astype(_i32).sum(-1)
        below = jnp.clip(below, 0, len(self.bins) - 1)
        above = jnp.clip(above, 0, len(self.bins) - 1)
        equal = below == above
        distance_below = jnp.where(
            equal,
            1,
            jnp.abs(self.bins[below] - target),
        )
        distance_above = jnp.where(
            equal,
            1,
            jnp.abs(self.bins[above] - target),
        )
        total = distance_below + distance_above
        weight_below = distance_above / total
        weight_above = distance_below / total
        target_distribution = (
            jax.nn.one_hot(below, len(self.bins)) * weight_below[..., None]
            + jax.nn.one_hot(above, len(self.bins)) * weight_above[..., None]
        )
        log_prediction = self.logits - jax.scipy.special.logsumexp(
            self.logits,
            -1,
            keepdims=True,
        )
        return -(target_distribution * log_prediction).sum(-1)


def _symexp_bins(count: int) -> Array:
    if count <= 1:
        raise ValueError("two-hot outputs require at least two bins")
    if count % 2 == 1:
        half = jnp.linspace(-20, 0, (count - 1) // 2 + 1, dtype=_f32)
        half = symexp(half)
        return jnp.concatenate([half, -half[:-1][::-1]], 0)
    half = jnp.linspace(-20, 0, count // 2, dtype=_f32)
    half = symexp(half)
    return jnp.concatenate([half, -half[::-1]], 0)


__all__ = [
    "AggregateOutput",
    "BinaryOutput",
    "CategoricalOutput",
    "MSEOutput",
    "NormalOutput",
    "OneHotOutput",
    "TwoHotOutput",
    "symexp",
    "symlog",
]
