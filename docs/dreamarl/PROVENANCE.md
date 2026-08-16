# DreaMARL Provenance

DreaMARL is implemented as first-party source under
`src/dreamarl`. Its maintained runtime does not import model code
from Dreamer-CDP or DreamerV3.

The algorithmic reference is the official `danijar/dreamerv3` revision:

```text
e3f02248693a79dc8b0ebd62c93683888ddaccfe
```

That checkout is pinned under `external/dreamerv3` and supplies the Embodied
runtime plus an immutable numerical oracle. Tests compare the first-party
encoder, categorical prior, actor-critic, and configuration against this
revision. Decoder, RSSM, ViT, and masking controls are isolated under
`src/dreamarl/ablations`.

The maintained algorithm keeps DreamerV3's stochastic latent, reward and
continuation heads, actor-critic objectives, replay semantics, and optimizer.
It replaces the RSSM temporal core with a first-party causal Transformer and
replaces reconstruction with posterior, action-conditioned dynamics, and
masked-spatial EMA-target prediction, regularized by SIGReg.

The production launcher has no model-family switches. Historical controls use
the separate ablation launcher and manifest, so their options cannot alter the
canonical configuration.

The multi-agent implementation adds the first-party reversible agent-axis
bridge in `src/dreamarl/marl/core.py`, synchronous team imagination, and one
shared peer-conditioned transition in `src/dreamarl/world_model/transformer.py`.
For each focal agent, stopped-gradient peer latent-action tokens are projected,
masked, averaged, and injected through a zero-initialized bounded residual
gate. Actors and critics remain local. With `A=1`, the peer set is empty and the
complete training update is exactly the locked single-agent learner.

The pinned Dreamer-CDP checkout remains an isolated historical baseline. It is
not imported by DreaMARL and none of its split learning rates, large cosine-loss
coefficient, or modified actor/checkpoint behavior are part of the maintained
algorithm.

The official MARIE source is pinned under `external/marie` at revision:

```text
5dc114f78e9f35389b843e05f01c455988451d0e
```

It is an immutable multi-agent benchmark and mechanism reference. The launcher
rejects tracked source modifications. DreaMARL does not import MARIE model code.
No MARIE mechanism is part of the maintained DreaMARL algorithm.

Every launch records the official DreamerV3 revision, resolved architecture,
and exact command. Publication artifacts must additionally name the repository
commit that produced them; the launcher does not walk and hash source files to
maintain a redundant second revision scheme.

Dreamer-derived source remains subject to the license in
`src/dreamarl/LICENSE`.
