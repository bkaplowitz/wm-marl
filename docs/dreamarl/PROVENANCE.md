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

The V-JEPA 2.1 ablation is derived from Meta's official
`facebookresearch/vjepa2` `train_2_1/vitb16/pretrain-256px-16f.yaml` recipe.
The LeWorldModel ablation is derived from the official `lucas-maes/le-wm`
training and model configurations. Their manifests identify the causal
integration boundary: visual and representation settings are transferred,
while DreaMARL's causal temporal model, behavior learner, replay protocol, and
environments remain fixed.

The maintained algorithm keeps DreamerV3's stochastic latent, reward and
continuation heads, actor-critic objectives, replay semantics, and optimizer.
It replaces the RSSM temporal core with a first-party causal Transformer and
replaces reconstruction with posterior, action-conditioned dynamics, and
masked-spatial EMA-target prediction, regularized by SIGReg.

The production launcher has no model-family switches. Historical controls use
the separate ablation launcher and manifest, so their options cannot alter the
canonical configuration.

The multi-agent implementation adds the first-party reversible agent-axis
bridge in `src/dreamarl/marl/core.py` and synchronized team imagination. B0
applies the shared local learner independently to every agent: actors, critics,
and world-model transitions receive no peer tensor or runtime communication.
With `A=1`, the complete training update remains exactly the locked
single-agent learner.

B1 adds a first-party, training-only EMA team-slot teacher and whole-agent
masking objective at the explicit team-axis boundary. The online branch must
predict the complete current team slots from visible local embeddings and
visible local histories. A second predictor preserves each local-state/action
pair before set pooling and predicts the next complete EMA team state from the
joint replay action. B1 inputs are stop-gradient, so the maintained local
single-agent learner is not reshaped by either team loss. Balanced permutation-invariant matching anchors every
source and predicted slot to mean-centered, agent-relative active local EMA
embeddings, with an
additional coverage loss for completely hidden agents and explicit slot
anti-collapse penalties. These modules are absent for `A=1`. In B1 they are
excluded from policy synchronization, online collection, and imagination.

B2 is the first controlled CTDE stage. It reuses the posterior JEPA predictor
to reconstruct active local EMA-content predictions from executable local world
states, combines them with the grouped causal local histories into eight team
slots, and supplies the stopped-gradient flattened belief to both centralized
value heads. Replay and recursive imagination call the same belief constructor.
The B1 action branch consumes these causal predicted embeddings in B2. No target
encoder value, team slot, or peer tensor enters the actor or runtime policy.

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

Melting Pot uses the pinned `dm-meltingpot==2.4.0`, `dmlab2d==1.0.0`, and
`shimmy==2.0.1` environment stack. DreaMARL supplies each seed while constructing
the Lab2D substrate; Shimmy documents its `reset(seed)` argument as ignored.
This controls the backend seed stream but does not make trajectories bitwise
deterministic. Identically seeded Lab2D instances may diverge under identical
actions, and manifests record that limitation explicitly.

Dreamer-derived source remains subject to the license in
`src/dreamarl/LICENSE`.
