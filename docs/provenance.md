# DreaMARL provenance

## First-party implementation

The maintained algorithm is implemented under `src/dreamarl`. It includes the
local categorical latent world model, causal Transformer, joint-embedding
objectives, explicit agent-axis runtime, centralized joint simulator and
critic, TBv2 teammate belief, identity-aware future peer-plan attention,
shared actor, replay, SMAC adapter, and held-out evaluation path.

## Runtime foundation

DreaMARL uses the official
[`danijar/dreamerv3`](https://github.com/danijar/dreamerv3) runtime pinned at:

```text
e3f02248693a79dc8b0ebd62c93683888ddaccfe
```

The checkout supplies Embodied/NinjaX runtime facilities. The learned DreaMARL
model is first-party code. It retains categorical stochastic states, latent
imagination, score-function actor learning, lambda-return value learning, a
slow value target, return normalization, and replay/runtime infrastructure.
It replaces reconstruction with stopped-EMA predictive embedding objectives
and adds the multi-agent architecture documented in
[`final_dreamarl.md`](final_dreamarl.md).

## Multi-agent contract

Training uses synchronized `[batch, time, agent, ...]` data. The centralized
joint simulator and attention critic may use team information during training.
The executed policy uses only each focal agent's local history, its locally
predicted TBv2 teammate belief, and the shared actor parameters. Centralized
states, peer observations, future labels, the peer-plan decoder, and the
multi-step JEPA predictor are not synchronized to execution workers.

## Environment

The supported benchmark is SMAC v1 with StarCraft II 4.10, difficulty 7,
continuing episodes, native shared reward, fixed roster slots, and native legal
action masks. Diagnostics do not alter the benchmark reward.

## Experiment identity

Every launch records the resolved configuration, task, seed, budget, command,
and source revision. This branch accepts only manifests whose algorithm is
`final-dreamarl`. The repository commit is the source identity.

## License boundary

First-party and Dreamer-derived source is released under the root MIT
[`LICENSE`](../LICENSE). The pinned runtime retains its upstream license and
copyright notices.
