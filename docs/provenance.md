# DreaMARL Provenance

## First-party implementation

The maintained DreaMARL algorithm is implemented under `src/dreamarl`. This
includes the local encoder, categorical latent state, causal Transformer,
joint-embedding objectives, actor and critic integration, explicit agent-axis
runtime, DreaMARL-CTDE joint simulator, centralized attention critic, SMAC
adapter, replay objectives, and evaluation diagnostics.

DreaMARL does not import the learned model implementations in the pinned
comparison repositories. The DreamerV3 checkout supplies the Embodied runtime
and reference modules used for numerical parity tests. Comparison launchers run
the other upstream programs as isolated processes.

## DreamerV3 foundation

The algorithmic and runtime reference is the official
[`danijar/dreamerv3`](https://github.com/danijar/dreamerv3) revision:

```text
e3f02248693a79dc8b0ebd62c93683888ddaccfe
```

The checkout is pinned at `external/dreamerv3`. DreaMARL retains DreamerV3's
categorical stochastic state, reward and continuation distributions,
Dreamer-style latent imagination, score-function actor, lambda-return value
learning, slow value target, return-range normalization, and replay/runtime
infrastructure.

DreaMARL changes the learned world representation in two central ways:

- a causal Transformer replaces the RSSM recurrent transition;
- EMA-target posterior, action-conditioned, and masked-spatial prediction
  replace pixel reconstruction in the maintained model.

The first-party convolutional encoder, categorical prior, behavior objectives,
and configuration are checked against the pinned source where equivalence is
claimed. DreamerV3 reconstruction and RSSM variants remain isolated scientific
controls under `src/dreamarl/ablations`, alongside compact visual, masking,
target, and SIGReg controls used by the paper.

## Multi-agent implementation

The reversible `[B,T,A,...] <-> [B*A,T,...]` agent-axis bridge and synchronized
team bookkeeping are first-party DreaMARL code. Independent DreaMARL shares the
local parameter tree while keeping world state, value, and actor information
observation-local. At one agent, the wrapper reduces to the single-agent path.

DreaMARL-CTDE is also first-party code. It adds:

- a permutation-equivariant agent interaction model without agent IDs;
- a causal, joint-action-conditioned JEPA simulator;
- next-local-embedding, reward, continuation, availability, and liveness
  prediction;
- centralized imagined transitions completed through the existing local
  posterior interface;
- a centralized attention critic over stopped synchronized local states;
- an optional bounded two-step self-fed objective with a last-step-only gradient
  boundary.

The online actor remains the local DreaMARL actor. Joint modules and the
centralized critic are training-only and are absent from policy synchronization,
collection, and evaluation.

## External comparison sources

The repository pins four upstream implementations as Git submodules:

| Comparison | Repository | Pinned revision | Role |
| --- | --- | --- | --- |
| DreamerV3 | [`danijar/dreamerv3`](https://github.com/danijar/dreamerv3) | `e3f02248693a79dc8b0ebd62c93683888ddaccfe` | Runtime, parity oracle, and DMC score archive |
| MARIE | [`breez3young/MARIE`](https://github.com/breez3young/MARIE) | `5dc114f78e9f35389b843e05f01c455988451d0e` | Peer-reviewed multi-agent world-model reference |
| Dreamer-CDP | [`fmi-basel/Dreamer-CDP`](https://github.com/fmi-basel/Dreamer-CDP) | `a851fa3e3d70b624b094ee1810ad4bb602346092` | Isolated single-agent comparison tooling |
| NE-Dreamer | [`corl-team/nedreamer`](https://github.com/corl-team/nedreamer) | `11cd3a978b83743f795cbfa81c2e095344912c17` | Isolated single-agent comparison tooling |

The launch adapters verify the expected checkout and record upstream commands,
seeds, budgets, and normalized artifact locations. No hyperparameter or
mechanism from an external implementation becomes part of DreaMARL merely by
being available through these launchers.

MARIE is a mechanism and benchmark reference, not an imported component.
DreaMARL-CTDE follows the same high-level centralized-training/decentralized-
execution requirement but uses predictive local embeddings instead of MARIE's
reconstructed VQ observations. Details of the attempted upstream reproduction
are recorded in [the MARIE reproducibility note](reproducibility/marie.md).

## Scientific controls

The public ablation package is deliberately narrower than the development
history. It retains the controls required to identify the maintained model:
RSSM versus causal Transformer dynamics, reconstruction versus joint-embedding
prediction, compact visual backbones, spatial mask topology, online versus EMA
targets, and SIGReg. Impractical large visual recipes and failed multi-agent
mechanisms are not shipped as supported model families.

## Environment provenance

### DMC

Visual single-agent results use DeepMind Control Suite tasks through the
DreamerV3/Embodied environment interface. Launch manifests record the task,
observation mode, action repeat, image resolution, seed, and environment-step
budget.

### SMAC

DreaMARL-CTDE uses SMAC v1 with StarCraft II 4.10, difficulty 7, continuing
episodes, native shared team reward, fixed roster slots, and legal-action
masks. The adapter leaves the benchmark reward used for learning unchanged and
logs corrected damage, shield, death, survival, timeout, and action diagnostics
alongside it. Corrected diagnostics are evaluation evidence, not reward shaping.

The isolated MARIE reproduction pins the official SMAC source at:

```text
d6aab33f76abc3849c50463a8592a84f59a5ef84
```

## Experiment identity

Every launch records the resolved model configuration, environment protocol,
seed, budget, external revisions, and exact command. Publication artifacts must
also record the DreaMARL Git commit that produced them. The repository commit is
the source identity; the launcher intentionally does not maintain a second,
redundant source-tree hashing scheme.

Historical experiment-stage identifiers can occur in old checkpoints and
manifests. Publication text and new result tables use the semantic names
DreaMARL, Independent DreaMARL, DreaMARL-CTDE one-step, and DreaMARL-CTDE
two-step.

## License boundary

First-party and Dreamer-derived DreaMARL source is released under the root MIT
[LICENSE](../LICENSE). Each Git submodule retains its own license and
copyright notices. [NOTICE.md](../NOTICE.md) records the pinned components
and their role; it does not replace any upstream license.
