"""Online policy execution for the DreaMARL agent."""

import elements
import jax
import jax.numpy as jnp
from ..models.heads import apply_action_mask, apply_predicted_action_mask
from .common import predict, sample


class PolicyMixin:
    def policy(self, carry, obs, mode="train"):
        enc_carry, dyn_carry, dec_carry, prevact = carry
        kwargs = dict(training=False, single=True)
        reset = obs["is_first"]
        enc_carry, enc_entry, tokens = self.enc(enc_carry, obs, reset, **kwargs)
        dyn_carry, dyn_entry, feat, _ = self.observe_dynamics(
            dyn_carry, tokens, prevact, reset, obs, **kwargs
        )
        if self.dec is not None:
            dec_carry, dec_entry, _ = self.dec(dec_carry, feat, reset, **kwargs)
        else:
            dec_entry = {}
        tensor = self.feat2tensor(feat)
        policy = self.policy_distribution(
            tensor,
            bdims=1,
            action_mask=obs.get("action_mask"),
        )
        if mode in {"train", "eval_sample"}:
            act = sample(policy)
        elif mode == "eval":
            act = predict(policy)
        else:
            raise ValueError(f"unknown policy mode: {mode!r}")
        out = {
            "finite": elements.tree.flatdict(
                jax.tree.map(
                    lambda value: jnp.isfinite(value).all(range(1, value.ndim)),
                    dict(obs=obs, carry=carry, tokens=tokens, feat=feat, act=act),
                )
            )
        }
        carry = (enc_carry, dyn_carry, dec_carry, act)
        if self.config.replay_context:
            entries = dict(
                enc=enc_entry,
                dyn=self.policy_dynamics_replay_entries(dyn_entry),
            )
            if self.dec is not None:
                entries["dec"] = dec_entry
            out.update(elements.tree.flatdict(entries))
        return carry, act, out

    def policy_distribution(self, tensor, bdims, action_mask=None):
        policy = self.pol(tensor, bdims=bdims)
        if getattr(self, "action_mask_key", None) is None:
            return policy
        if action_mask is None:
            output = self.actmask(tensor, bdims=bdims)
            binary = output.output if hasattr(output, "output") else output
            return apply_predicted_action_mask(
                policy,
                jax.lax.stop_gradient(binary.logit),
                self.action_mask_key,
            )
        return apply_action_mask(policy, action_mask, self.action_mask_key)

    def observe_dynamics(self, carry, tokens, action, reset, obs, training, single):
        del obs
        return self.dyn.observe(carry, tokens, action, reset, training, single=single)
