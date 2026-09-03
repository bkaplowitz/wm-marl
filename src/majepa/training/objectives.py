"""Pure actor and critic objectives for latent imagination and replay."""

import chex
import jax.numpy as jnp

from .common import f32, sg


def normalized_policy_entropy(distribution, entropy):
    """Normalize entropy to [0, 1] using the distribution's live support."""

    if hasattr(distribution, "logits"):
        support = (distribution.logits > -1e20).sum(axis=-1)
        maximum = jnp.log(jnp.maximum(support, 1)).astype(jnp.float32)
        return entropy / jnp.maximum(maximum, 1e-8)
    if hasattr(distribution, "minent") and hasattr(distribution, "maxent"):
        minimum = jnp.asarray(distribution.minent)
        maximum = jnp.asarray(distribution.maxent)
        return (entropy - minimum) / jnp.maximum(maximum - minimum, 1e-8)
    raise TypeError("normalized entropy requires logits or entropy bounds")


def imag_loss(
    act,
    rew,
    con,
    policy,
    value,
    slowvalue,
    retnorm,
    valnorm,
    advnorm,
    update,
    valid=None,
    contdisc=True,
    slowtar=True,
    horizon=333,
    lam=0.95,
    actent=3e-4,
    normalize_entropy=False,
    slowreg=1.0,
):
    losses = {}
    metrics = {}

    voffset, vscale = valnorm.stats()
    val = value.pred() * vscale + voffset
    slowval = slowvalue.pred() * vscale + voffset
    tarval = slowval if slowtar else val
    disc = 1 if contdisc else 1 - 1 / horizon
    weight = jnp.cumprod(disc * con, 1) / disc
    last = jnp.zeros_like(con)
    term = 1 - con
    ret = lambda_return(last, term, rew, tarval, tarval, disc, lam)

    norm_valid = None if valid is None else valid[:, : ret.shape[1]]
    loss_valid = (
        jnp.ones_like(ret)
        if norm_valid is None
        else norm_valid.astype(jnp.float32)
        / jnp.maximum(norm_valid.astype(jnp.float32).mean(), 1e-8)
    )
    roffset, rscale = retnorm(ret, update, norm_valid)
    adv = (ret - tarval[:, :-1]) / rscale
    aoffset, ascale = advnorm(adv, update, norm_valid)
    adv_normed = (adv - aoffset) / ascale
    logpi = sum([dist.logp(sg(act[key]))[:, :-1] for key, dist in policy.items()])
    full_ents = {key: dist.entropy() for key, dist in policy.items()}
    ents = {key: entropy[:, :-1] for key, entropy in full_ents.items()}
    bonus_ents = ents
    if normalize_entropy:
        bonus_ents = {
            key: normalized_policy_entropy(policy[key], full_ents[key])[:, :-1]
            for key in policy
        }
    policy_loss = (
        loss_valid
        * sg(weight[:, :-1])
        * -(logpi * sg(adv_normed) + actent * sum(bonus_ents.values()))
    )
    losses["policy"] = policy_loss

    voffset, vscale = valnorm(ret, update, norm_valid)
    tar_normed = (ret - voffset) / vscale
    tar_padded = jnp.concatenate([tar_normed, 0 * tar_normed[:, -1:]], 1)
    losses["value"] = (
        loss_valid
        * sg(weight[:, :-1])
        * (value.loss(sg(tar_padded)) + slowreg * value.loss(sg(slowvalue.pred())))[
            :, :-1
        ]
    )

    ret_normed = (ret - roffset) / rscale
    value_error = val[:, :-1] - ret
    metric_weight = (
        jnp.ones_like(ret) if norm_valid is None else norm_valid.astype(jnp.float32)
    )
    metric_count = jnp.maximum(metric_weight.sum(), 1.0)
    metric_selected = metric_weight.astype(bool)
    value_error_mean = jnp.where(metric_selected, value_error, 0.0).sum() / metric_count
    value_target_mean = jnp.where(metric_selected, ret, 0.0).sum() / metric_count
    value_error_variance = (
        jnp.where(
            metric_selected, jnp.square(value_error - value_error_mean), 0.0
        ).sum()
        / metric_count
    )
    value_target_variance = (
        jnp.where(metric_selected, jnp.square(ret - value_target_mean), 0.0).sum()
        / metric_count
    )
    metrics["adv"] = adv.mean()
    metrics["adv_std"] = adv.std()
    metrics["adv_mag"] = jnp.abs(adv).mean()
    metrics["rew"] = rew.mean()
    metrics["con"] = con.mean()
    metrics["ret"] = ret_normed.mean()
    metrics["ret_raw"] = jnp.where(metric_selected, ret, 0.0).sum() / metric_count
    root_valid = (
        jnp.ones_like(ret[:, 0])
        if norm_valid is None
        else norm_valid[:, 0].astype(jnp.float32)
    )
    metrics["ret_root_raw"] = (ret[:, 0] * root_valid).sum() / jnp.maximum(
        root_valid.sum(), 1.0
    )
    reward_return = (weight * rew).sum(axis=1)
    metrics["reward_return_raw"] = (reward_return * root_valid).sum() / jnp.maximum(
        root_valid.sum(), 1.0
    )
    metrics["val"] = val.mean()
    metrics["val_raw"] = (
        jnp.where(metric_selected, val[:, :-1], 0.0).sum() / metric_count
    )
    metrics["critic_root_raw"] = (val[:, 0] * root_valid).sum() / jnp.maximum(
        root_valid.sum(), 1.0
    )
    metrics["critic_rollout_raw"] = (
        jnp.where(metric_selected, val[:, 1:], 0.0).sum() / metric_count
    )
    metrics["tar"] = tar_normed.mean()
    metrics["weight"] = weight.mean()
    metrics["slowval"] = slowval.mean()
    metrics["critic/value_rmse"] = jnp.sqrt(
        jnp.where(metric_selected, jnp.square(value_error), 0.0).sum() / metric_count
    )
    metrics["critic/value_bias"] = value_error_mean
    metrics["critic/value_explained_variance"] = 1.0 - (
        value_error_variance / jnp.maximum(value_target_variance, 1e-8)
    )
    metrics["ret_min"] = ret_normed.min()
    metrics["ret_max"] = ret_normed.max()
    metrics["ret_rate"] = (jnp.abs(ret_normed) >= 1.0).mean()
    metrics["actor_entropy/coefficient"] = jnp.asarray(actent, jnp.float32)
    for key in act:
        entropy = ents[key].mean()
        metrics[f"ent/{key}"] = entropy
        metrics[f"actor_entropy/bonus_{key}"] = bonus_ents[key].mean()
        metrics[f"act_abs/{key}"] = jnp.abs(act[key][:, :-1]).mean()
        if jnp.issubdtype(act[key].dtype, jnp.floating):
            metrics[f"act_saturation/{key}"] = (
                jnp.abs(act[key][:, :-1]) >= 0.95
            ).mean()
        if hasattr(policy[key], "minent"):
            lo = jnp.asarray(policy[key].minent).mean()
            hi = jnp.asarray(policy[key].maxent).mean()
            metrics[f"rand/{key}"] = (entropy - lo) / jnp.maximum(hi - lo, 1e-8)

    return (
        losses,
        {
            "ret": ret,
            "adv": adv,
            "adv_normed": adv_normed,
            "logpi": logpi,
            "weight": weight[:, :-1],
        },
        metrics,
    )


def repl_loss(
    last,
    term,
    rew,
    boot,
    value,
    slowvalue,
    valnorm,
    update=True,
    valid=None,
    slowreg=1.0,
    slowtar=True,
    horizon=333,
    lam=0.95,
):
    losses = {}

    voffset, vscale = valnorm.stats()
    val = value.pred() * vscale + voffset
    slowval = slowvalue.pred() * vscale + voffset
    tarval = slowval if slowtar else val
    disc = 1 - 1 / horizon
    weight = f32(~last)
    ret = lambda_return(last, term, rew, tarval, boot, disc, lam)

    norm_valid = None if valid is None else valid[:, : ret.shape[1]]
    loss_valid = (
        jnp.ones_like(ret)
        if norm_valid is None
        else norm_valid.astype(jnp.float32)
        / jnp.maximum(norm_valid.astype(jnp.float32).mean(), 1e-8)
    )
    voffset, vscale = valnorm(ret, update, norm_valid)
    ret_normed = (ret - voffset) / vscale
    ret_padded = jnp.concatenate([ret_normed, 0 * ret_normed[:, -1:]], 1)
    losses["repval"] = (
        loss_valid
        * weight[:, :-1]
        * (value.loss(sg(ret_padded)) + slowreg * value.loss(sg(slowvalue.pred())))[
            :, :-1
        ]
    )
    value_error = val[:, :-1] - ret
    metric_weight = (
        jnp.ones_like(ret) if norm_valid is None else norm_valid.astype(jnp.float32)
    )
    metric_count = jnp.maximum(metric_weight.sum(), 1.0)
    metric_selected = metric_weight.astype(bool)
    value_error_mean = jnp.where(metric_selected, value_error, 0.0).sum() / metric_count
    value_target_mean = jnp.where(metric_selected, ret, 0.0).sum() / metric_count
    value_error_variance = (
        jnp.where(
            metric_selected, jnp.square(value_error - value_error_mean), 0.0
        ).sum()
        / metric_count
    )
    value_target_variance = (
        jnp.where(metric_selected, jnp.square(ret - value_target_mean), 0.0).sum()
        / metric_count
    )
    metrics = {
        "critic/value_rmse": jnp.sqrt(
            jnp.where(metric_selected, jnp.square(value_error), 0.0).sum()
            / metric_count
        ),
        "critic/value_bias": value_error_mean,
        "critic/value_explained_variance": 1.0
        - value_error_variance / jnp.maximum(value_target_variance, 1e-8),
    }
    return losses, {"ret": ret, "slowval": slowval}, metrics


def lambda_return(last, term, rew, val, boot, disc, lam):
    chex.assert_equal_shape((last, term, rew, val, boot))
    rets = [boot[:, -1]]
    live = (1 - f32(term))[:, 1:] * disc
    cont = (1 - f32(last))[:, 1:] * lam
    interm = rew[:, 1:] + (1 - cont) * live * boot[:, 1:]
    for index in reversed(range(live.shape[1])):
        rets.append(interm[:, index] + live[:, index] * cont[:, index] * rets[-1])
    return jnp.stack(list(reversed(rets))[:-1], 1)
