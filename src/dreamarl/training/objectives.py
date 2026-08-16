"""Pure actor and critic objectives for latent imagination and replay."""

import chex
import jax.numpy as jnp

from .common import f32, sg


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
    contdisc=True,
    slowtar=True,
    horizon=333,
    lam=0.95,
    actent=3e-4,
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

    roffset, rscale = retnorm(ret, update)
    adv = (ret - tarval[:, :-1]) / rscale
    aoffset, ascale = advnorm(adv, update)
    adv_normed = (adv - aoffset) / ascale
    logpi = sum([dist.logp(sg(act[key]))[:, :-1] for key, dist in policy.items()])
    ents = {key: dist.entropy()[:, :-1] for key, dist in policy.items()}
    policy_loss = sg(weight[:, :-1]) * -(
        logpi * sg(adv_normed) + actent * sum(ents.values())
    )
    losses["policy"] = policy_loss

    voffset, vscale = valnorm(ret, update)
    tar_normed = (ret - voffset) / vscale
    tar_padded = jnp.concatenate([tar_normed, 0 * tar_normed[:, -1:]], 1)
    losses["value"] = (
        sg(weight[:, :-1])
        * (value.loss(sg(tar_padded)) + slowreg * value.loss(sg(slowvalue.pred())))[
            :, :-1
        ]
    )

    ret_normed = (ret - roffset) / rscale
    metrics["adv"] = adv.mean()
    metrics["adv_std"] = adv.std()
    metrics["adv_mag"] = jnp.abs(adv).mean()
    metrics["rew"] = rew.mean()
    metrics["con"] = con.mean()
    metrics["ret"] = ret_normed.mean()
    metrics["val"] = val.mean()
    metrics["tar"] = tar_normed.mean()
    metrics["weight"] = weight.mean()
    metrics["slowval"] = slowval.mean()
    metrics["ret_min"] = ret_normed.min()
    metrics["ret_max"] = ret_normed.max()
    metrics["ret_rate"] = (jnp.abs(ret_normed) >= 1.0).mean()
    for key in act:
        entropy = ents[key].mean()
        metrics[f"ent/{key}"] = entropy
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

    voffset, vscale = valnorm(ret, update)
    ret_normed = (ret - voffset) / vscale
    ret_padded = jnp.concatenate([ret_normed, 0 * ret_normed[:, -1:]], 1)
    losses["repval"] = (
        weight[:, :-1]
        * (value.loss(sg(ret_padded)) + slowreg * value.loss(sg(slowvalue.pred())))[
            :, :-1
        ]
    )
    return losses, {"ret": ret, "slowval": slowval}, {}


def lambda_return(last, term, rew, val, boot, disc, lam):
    chex.assert_equal_shape((last, term, rew, val, boot))
    rets = [boot[:, -1]]
    live = (1 - f32(term))[:, 1:] * disc
    cont = (1 - f32(last))[:, 1:] * lam
    interm = rew[:, 1:] + (1 - cont) * live * boot[:, 1:]
    for index in reversed(range(live.shape[1])):
        rets.append(interm[:, index] + live[:, index] * cont[:, index] * rets[-1])
    return jnp.stack(list(reversed(rets))[:-1], 1)
