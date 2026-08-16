"""Optimizer construction for the DreaMARL learner."""

import re

import embodied.jax
import optax


class OptimizationMixin:
    def _build_optimizer(self, config):
        return self._make_opt(**config.opt)

    def _make_opt(
        self,
        lr: float = 4e-5,
        agc: float = 0.3,
        eps: float = 1e-20,
        beta1: float = 0.9,
        beta2: float = 0.999,
        momentum: bool = True,
        nesterov: bool = False,
        wd: float = 0.0,
        wdregex: str = r"/kernel$",
        schedule: str = "const",
        warmup: int = 1000,
        anneal: int = 0,
    ):
        chain = [
            embodied.jax.opt.clip_by_agc(agc),
            embodied.jax.opt.scale_by_rms(beta2, eps),
        ]
        if momentum:
            chain.append(embodied.jax.opt.scale_by_momentum(beta1, nesterov))
        if wd:
            assert not wdregex[0].isnumeric(), wdregex
            pattern = re.compile(wdregex)

            def wdmask(params):
                return {key: bool(pattern.search(key)) for key in params}

            chain.append(optax.add_decayed_weights(wd, wdmask))
        assert anneal > 0 or schedule == "const"
        if schedule == "const":
            sched = optax.constant_schedule(lr)
        elif schedule == "linear":
            sched = optax.linear_schedule(lr, 0.1 * lr, anneal - warmup)
        elif schedule == "cosine":
            sched = optax.cosine_decay_schedule(lr, anneal - warmup, 0.1 * lr)
        else:
            raise NotImplementedError(schedule)
        if warmup:
            ramp = optax.linear_schedule(0.0, lr, warmup)
            sched = optax.join_schedules([ramp, sched], [warmup])
        chain.append(optax.scale_by_learning_rate(sched))
        return optax.chain(*chain)
