"""Categorical latent building blocks shared by DreaMARL dynamics."""

import embodied.jax
import embodied.jax.nets as nn
import jax.numpy as jnp
import ninjax as nj


class CategoricalLatent(nj.Module):
    """Categorical latent prior, posterior, and representation predictor."""

    deter: int = 8192
    hidden: int = 1024
    stoch: int = 32
    classes: int = 64
    norm: str = "rms"
    act: str = "silu"
    unimix: float = 0.01
    outscale: float = 1.0
    imglayers: int = 2

    def __init__(self, act_space, enc_output, **kwargs):
        self.act_space = act_space
        self.enc_output = enc_output
        self.kw = kwargs

    def predictor(self, value, name="pred"):
        hidden_name = "pred0" if name == "pred" else f"{name}0"
        norm_name = "pred0norm" if name == "pred" else f"{name}0norm"
        output_name = "pred_out" if name == "pred" else f"{name}_out"
        value = self.sub(hidden_name, nn.Linear, self.hidden, **self.kw)(value)
        value = nn.act(self.act)(self.sub(norm_name, nn.Norm, self.norm)(value))
        return self.sub(output_name, nn.Linear, self.enc_output, **self.kw)(value)

    def _prior(self, feature):
        value = feature
        for index in range(self.imglayers):
            value = self.sub(f"prior{index}", nn.Linear, self.hidden, **self.kw)(value)
            value = nn.act(self.act)(
                self.sub(f"prior{index}norm", nn.Norm, self.norm)(value)
            )
        return self._logit("priorlogit", value)

    def _logit(self, name, value):
        kwargs = dict(**self.kw, outscale=self.outscale)
        value = self.sub(name, nn.Linear, self.stoch * self.classes, **kwargs)(value)
        return value.reshape(value.shape[:-1] + (self.stoch, self.classes))

    def _dist(self, logits):
        distribution = embodied.jax.outs.OneHot(logits, self.unimix)
        return embodied.jax.outs.Agg(distribution, 1, jnp.sum)

    def distribution(self, logits):
        """Return the maintained categorical latent distribution."""

        return self._dist(logits)
