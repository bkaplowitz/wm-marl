"""Optimizer construction for the DreaMARL learner."""

import math
import re

import embodied.jax
from embodied.jax import internal, nets
import jax
import jax.numpy as jnp
import ninjax as nj
import optax


f32 = jnp.float32
i32 = jnp.int32
sg = jax.lax.stop_gradient


class GroupedOptimizer(nj.Module):
    """Apply independent optimizer transforms to disjoint module groups.

    The loss is evaluated once and differentiated jointly. Parameter ownership
    then routes world-model, actor, and critic gradients through independent
    optimizer state and learning-rate schedules without changing the forward
    pass or the sampled imagination.
    """

    summary_depth: int = 2

    def __init__(self, groups, modules=None):
        self.groups = {
            key: (tuple(modules), optimizer)
            for key, (modules, optimizer) in groups.items()
        }
        grouped_modules = [
            module for group, _ in self.groups.values() for module in group
        ]
        if len({id(module) for module in grouped_modules}) != len(grouped_modules):
            raise ValueError("optimizer module groups must be disjoint")
        self.modules = tuple(modules or grouped_modules)
        if {id(module) for module in self.modules} != {
            id(module) for module in grouped_modules
        }:
            raise ValueError("optimizer module order must contain every group module")
        self.step = {
            key: nj.Variable(jnp.array, 0, i32, name=f"{key}_step")
            for key in self.groups
        }
        self.scaling = nets.COMPUTE_DTYPE == jnp.float16
        if self.scaling:
            self.grad_scale = nj.Variable(jnp.array, 1e4, f32, name="grad_scale")
            self.good_steps = nj.Variable(jnp.array, 0, i32, name="good_steps")

    def __call__(self, lossfn, *args, has_aux=False, **kwargs):
        def lossfn2(*inner_args, **inner_kwargs):
            outputs = lossfn(*inner_args, **inner_kwargs)
            loss, aux = outputs if has_aux else (outputs, None)
            assert loss.dtype == f32, (self.name, loss.dtype)
            assert loss.shape == (), (self.name, loss.shape)
            if self.scaling:
                loss *= sg(self.grad_scale.read())
            return loss, aux

        loss, params, grads, aux = nj.grad(lossfn2, self.modules, has_aux=True)(
            *args, **kwargs
        )
        if self.scaling:
            loss *= 1 / self.grad_scale.read()

        axes = internal.get_data_axes()
        if axes:
            grads = jax.tree.map(lambda value: jax.lax.pmean(value, axes), grads)
        if self.scaling:
            invscale = 1 / self.grad_scale.read()
            grads = jax.tree.map(lambda value: value * invscale, grads)

        all_updates = {}
        metrics = {f"{self.name}/loss": loss.mean()}
        assigned = set()
        finite = True
        for key, (modules, optimizer) in self.groups.items():
            prefixes = tuple(f"{module.path}/" for module in modules)
            group_params = {
                name: value
                for name, value in params.items()
                if name.startswith(prefixes)
            }
            group_grads = {name: grads[name] for name in group_params}
            if not group_params:
                raise ValueError(f"optimizer group {key!r} has no parameters")
            overlap = assigned.intersection(group_params)
            if overlap:
                raise ValueError(f"optimizer groups overlap at {sorted(overlap)}")
            assigned.update(group_params)

            state = self.sub(f"{key}_state", nj.Tree, optimizer.init, group_params)
            updates, new_state = optimizer.update(
                group_grads, state.read(), group_params
            )
            group_finite = jnp.isfinite(optax.global_norm(group_grads))
            finite = finite & group_finite
            state.write(new_state)
            all_updates.update(updates)
            self.step[key].write(self.step[key].read() + i32(group_finite))

            counts = {
                name: math.prod(value.shape) for name, value in group_params.items()
            }
            prefix = f"{self.name}/{key}"
            metrics.update(
                {
                    f"{prefix}/updates": self.step[key].read(),
                    f"{prefix}/grad_norm": optax.global_norm(group_grads),
                    f"{prefix}/grad_rms": nets.rms(group_grads),
                    f"{prefix}/update_rms": nets.rms(updates),
                    f"{prefix}/param_rms": nets.rms(group_params),
                    f"{prefix}/param_count": jnp.asarray(sum(counts.values()), f32),
                }
            )
            if nj.creating():
                print(self._summarize_group(key, counts))

        missing = set(params).difference(assigned)
        if missing:
            raise ValueError(f"optimizer parameters have no owner: {sorted(missing)}")
        nj.context().update(optax.apply_updates(params, all_updates))

        if self.scaling:
            self._update_scale(finite)
            metrics[f"{self.name}/grad_scale"] = self.grad_scale.read()
            metrics[f"{self.name}/grad_overflow"] = f32(~finite)
        return (metrics, aux) if has_aux else metrics

    def _update_scale(self, finite):
        keep = finite & (self.good_steps.read() < 1000)
        incr = finite & (self.good_steps.read() >= 1000)
        decr = ~finite
        self.good_steps.write(i32(keep) * (self.good_steps.read() + 1))
        self.grad_scale.write(
            jnp.clip(
                f32(keep) * self.grad_scale.read()
                + f32(incr) * self.grad_scale.read() * 2
                + f32(decr) * self.grad_scale.read() / 2,
                1e-4,
                1e5,
            )
        )

    def _summarize_group(self, key, counts):
        total = sum(counts.values())
        return f"Optimizer {self.name}/{key} has {total:,} parameters"


class OptimizationMixin:
    def _build_optimizer(self, config):
        return self._make_opt(**config.opt)

    def _build_grouped_optimizer(
        self,
        modules,
        world_modules,
        actor_modules,
        critic_modules,
        *,
        matched=False,
    ):
        actor_config = self.config.opt if matched else self.config.actor_opt
        critic_config = self.config.opt if matched else self.config.critic_opt
        return GroupedOptimizer(
            {
                "world": (world_modules, self._make_opt(**self.config.opt)),
                "actor": (actor_modules, self._make_opt(**actor_config)),
                "critic": (critic_modules, self._make_opt(**critic_config)),
            },
            modules=modules,
            summary_depth=1,
            name="opt",
        )

    def _build_ppo_optimizers(self, world_modules, actor_modules, critic_modules):
        model_opt = GroupedOptimizer(
            {
                "world": (world_modules, self._make_opt(**self.config.opt)),
                "critic": (critic_modules, self._make_opt(**self.config.opt)),
            },
            modules=[*world_modules, *critic_modules],
            summary_depth=1,
            name="opt",
        )
        actor_opt = embodied.jax.Optimizer(
            actor_modules,
            self._make_opt(**self.config.opt),
            summary_depth=1,
            name="actor_opt",
        )
        return model_opt, actor_opt

    def _build_jecc_optimizer(
        self,
        modules,
        world_modules,
        actor_modules,
        critic_modules,
        jecc_modules,
    ):
        """Build the four disjoint optimizer groups used by JECC."""

        return GroupedOptimizer(
            {
                "world": (world_modules, self._make_opt(**self.config.opt)),
                "actor": (actor_modules, self._make_opt(**self.config.opt)),
                "critic": (critic_modules, self._make_opt(**self.config.opt)),
                "jecc": (
                    jecc_modules,
                    self._make_opt(**self.config.marl.jecc.opt),
                ),
            },
            modules=modules,
            summary_depth=1,
            name="opt",
        )

    def _build_ctde_optimizer(
        self,
        modules,
        local_world_modules,
        joint_world_modules,
        actor_modules,
        critic_modules,
    ):
        """Build the four disjoint optimizer groups used by CTDE."""

        matched = self.config.opt
        joint = self.config.marl.ctde.opt
        return GroupedOptimizer(
            {
                "local_world": (
                    local_world_modules,
                    self._make_opt(**matched),
                ),
                "joint_world": (
                    joint_world_modules,
                    self._make_opt(**joint),
                ),
                "actor": (actor_modules, self._make_opt(**matched)),
                "critic": (critic_modules, self._make_opt(**matched)),
            },
            modules=modules,
            summary_depth=1,
            name="opt",
        )

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
