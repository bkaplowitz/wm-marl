"""Optimizer construction for the MA-JEPA learner."""

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

    def __call__(
        self,
        lossfn,
        *args,
        has_aux=False,
        active_groups=None,
        skip_groups=(),
        **kwargs,
    ):
        active_groups = dict(active_groups or {})
        unknown = set(active_groups).difference(self.groups)
        if unknown:
            raise ValueError(f"unknown optimizer activity groups: {sorted(unknown)}")
        skip_groups = frozenset(skip_groups)
        unknown = skip_groups.difference(self.groups)
        if unknown:
            raise ValueError(f"unknown optimizer groups to skip: {sorted(unknown)}")
        overlap = skip_groups.intersection(active_groups)
        if overlap:
            raise ValueError(
                "optimizer groups cannot be both skipped and dynamically gated: "
                f"{sorted(overlap)}"
            )

        def lossfn2(*inner_args, **inner_kwargs):
            outputs = lossfn(*inner_args, **inner_kwargs)
            loss, aux = outputs if has_aux else (outputs, None)
            assert loss.dtype == f32, (self.name, loss.dtype)
            assert loss.shape == (), (self.name, loss.shape)
            if self.scaling:
                loss *= sg(self.grad_scale.read())
            return loss, aux

        grad_modules = tuple(
            module
            for key, (modules, _) in self.groups.items()
            if key not in skip_groups
            for module in modules
        )
        if not grad_modules:
            raise ValueError("optimizer cannot skip every parameter group")
        loss, params, grads, aux = nj.grad(lossfn2, grad_modules, has_aux=True)(
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
            gated = key in active_groups
            active = jnp.asarray(active_groups.get(key, True), bool)
            if active.shape:
                raise ValueError(
                    f"optimizer group activity must be scalar, got {active.shape}"
                )
            prefixes = tuple(f"{module.path}/" for module in modules)
            group_params = {
                name: value
                for name, value in params.items()
                if name.startswith(prefixes)
            }
            group_grads = {name: grads[name] for name in group_params}
            if not group_params:
                if key not in skip_groups:
                    raise ValueError(f"optimizer group {key!r} has no parameters")
                metrics.update(
                    {
                        f"{self.name}/{key}/updates": self.step[key].read(),
                        f"{self.name}/{key}/active": jnp.asarray(0.0, f32),
                        f"{self.name}/{key}/skipped": jnp.asarray(1.0, f32),
                        f"{self.name}/{key}/param_count": jnp.asarray(0.0, f32),
                    }
                )
                continue
            overlap = assigned.intersection(group_params)
            if overlap:
                raise ValueError(f"optimizer groups overlap at {sorted(overlap)}")
            assigned.update(group_params)

            group_finite = jnp.isfinite(optax.global_norm(group_grads))
            if key in skip_groups:
                # PPO owns these groups in separate proximal epochs. Skipping
                # here must leave parameters, moments, and schedule counters
                # completely untouched.
                updates = jax.tree.map(jnp.zeros_like, group_params)
            else:
                state = self.sub(f"{key}_state", nj.Tree, optimizer.init, group_params)
                old_state = state.read()
                updates, new_state = optimizer.update(
                    group_grads, old_state, group_params
                )
                finite = finite & group_finite
                # A disabled group is a literal optimizer freeze: parameters,
                # moments, schedule counters, and update counters all stay fixed.
                if gated:
                    state.write(
                        jax.tree.map(
                            lambda new, old: jnp.where(active, new, old),
                            new_state,
                            old_state,
                        )
                    )
                    updates = jax.tree.map(
                        lambda value: jnp.where(active, value, jnp.zeros_like(value)),
                        updates,
                    )
                else:
                    state.write(new_state)
            all_updates.update(updates)
            if key not in skip_groups:
                self.step[key].write(self.step[key].read() + i32(active & group_finite))

            counts = {
                name: math.prod(value.shape) for name, value in group_params.items()
            }
            prefix = f"{self.name}/{key}"
            metrics.update(
                {
                    f"{prefix}/updates": self.step[key].read(),
                    f"{prefix}/active": f32(active & (key not in skip_groups)),
                    f"{prefix}/skipped": jnp.asarray(key in skip_groups, f32),
                    f"{prefix}/grad_norm": optax.global_norm(group_grads),
                    f"{prefix}/grad_rms": nets.rms(group_grads),
                    f"{prefix}/update_rms": nets.rms(updates),
                    f"{prefix}/param_rms": nets.rms(group_params),
                    f"{prefix}/param_count": jnp.asarray(sum(counts.values()), f32),
                }
            )
            if len(modules) > 1:
                for module in (
                    module
                    for module in modules
                    if module.name.startswith("ctde_teammate_")
                ):
                    module_prefix = f"{module.path}/"
                    module_grads = {
                        name: value
                        for name, value in group_grads.items()
                        if name.startswith(module_prefix)
                    }
                    module_updates = {
                        name: value
                        for name, value in updates.items()
                        if name.startswith(module_prefix)
                    }
                    if module_grads:
                        metrics[f"{prefix}/{module.name}_grad_norm"] = (
                            optax.global_norm(module_grads)
                        )
                        metrics[f"{prefix}/{module.name}_update_norm"] = (
                            optax.global_norm(module_updates)
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

    def step_group(
        self,
        key,
        lossfn,
        *args,
        has_aux=False,
        active=True,
        **kwargs,
    ):
        """Optimize exactly one parameter group for one PPO epoch."""

        if key not in self.groups:
            raise ValueError(f"unknown optimizer group: {key!r}")
        modules, optimizer = self.groups[key]
        active = jnp.asarray(active, bool)
        if active.shape:
            raise ValueError(
                f"optimizer group activity must be scalar, got {active.shape}"
            )

        def lossfn2(*inner_args, **inner_kwargs):
            outputs = lossfn(*inner_args, **inner_kwargs)
            loss, aux = outputs if has_aux else (outputs, None)
            assert loss.dtype == f32, (self.name, key, loss.dtype)
            assert loss.shape == (), (self.name, key, loss.shape)
            if self.scaling:
                loss *= sg(self.grad_scale.read())
            return loss, aux

        loss, params, grads, aux = nj.grad(lossfn2, modules, has_aux=True)(
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

        prefixes = tuple(f"{module.path}/" for module in modules)
        unexpected = [name for name in params if not name.startswith(prefixes)]
        if unexpected:
            raise ValueError(
                f"optimizer group {key!r} captured foreign parameters: {unexpected}"
            )
        if not params:
            raise ValueError(f"optimizer group {key!r} has no parameters")

        state = self.sub(f"{key}_state", nj.Tree, optimizer.init, params)
        old_state = state.read()
        updates, new_state = optimizer.update(grads, old_state, params)
        grad_norm = optax.global_norm(grads)
        finite = jnp.isfinite(grad_norm)
        apply = active & finite
        state.write(
            jax.tree.map(
                lambda new, old: jnp.where(apply, new, old),
                new_state,
                old_state,
            )
        )
        updates = jax.tree.map(
            lambda value: jnp.where(apply, value, jnp.zeros_like(value)),
            updates,
        )
        nj.context().update(optax.apply_updates(params, updates))
        self.step[key].write(self.step[key].read() + i32(apply))

        counts = {name: math.prod(value.shape) for name, value in params.items()}
        if nj.creating():
            print(self._summarize_group(key, counts))
        prefix = f"{self.name}/{key}"
        metrics = {
            f"{prefix}/loss": loss.mean(),
            f"{prefix}/updates": self.step[key].read(),
            f"{prefix}/active": f32(apply),
            f"{prefix}/skipped": jnp.asarray(0.0, f32),
            f"{prefix}/grad_norm": grad_norm,
            f"{prefix}/grad_rms": nets.rms(grads),
            f"{prefix}/update_rms": nets.rms(updates),
            f"{prefix}/param_rms": nets.rms(params),
            f"{prefix}/param_count": jnp.asarray(sum(counts.values()), f32),
        }
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
        actor = dict(matched)
        actor["lr"] = float(self.config.ppo.actor_lr)
        actor["warmup"] = int(self.config.ppo.optimizer_warmup)
        actor["update_every"] = 1
        critic = dict(matched)
        critic["lr"] = float(self.config.ppo.critic_lr)
        critic["warmup"] = int(self.config.ppo.optimizer_warmup)
        critic["update_every"] = 1
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
                "actor": (actor_modules, self._make_opt(**actor)),
                "critic": (critic_modules, self._make_opt(**critic)),
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
        update_every: int = 1,
    ):
        if update_every < 1:
            raise ValueError("optimizer update interval must be positive")
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
        optimizer = optax.chain(*chain)
        if update_every > 1:
            optimizer = optax.MultiSteps(
                optimizer,
                every_k_schedule=update_every,
                use_grad_mean=True,
            )
        return optimizer
