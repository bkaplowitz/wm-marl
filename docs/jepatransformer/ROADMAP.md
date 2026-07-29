# JEPA Transformer Research Programme

## Goal

Develop a reconstruction-free stochastic Transformer world model for visual
multi-agent reinforcement learning. The model predicts future per-agent
embeddings, advances all agents in one coherent joint imagined world, and
trains decentralized policies with centralized value learning.

The programme promotes a method only when it preserves control performance and
demonstrates a measurable benefit in world-model cost, robustness, or
adaptation to changing teammate policies.

## Fixed Experimental Rules

- Official baselines run from immutable upstream checkouts.
- Training curves use real environment transitions on the x-axis.
- Evaluation uses deterministic actions from the latest policy.
- Evaluation transitions are reported separately from training transitions.
- No real-environment checkpoint search is permitted.
- Every promotion comparison holds all components fixed except the component
  named by that milestone.
- Rejected alternatives remain in their experiment commits and do not become
  dormant switches in the maintained algorithm.

The machine-readable protocol is
`configs/jepatransformer/visual_dmc_protocol.toml`.

## Temporal Contract

For a posterior rollout, the temporal state at step `t` contains only tokens
strictly before observation `o_t`:

```text
h_t = Transformer((z_0, a_0), ..., (z_{t-1}, a_{t-1}))
z_t ~ q(z_t | h_t, E(o_t))
a_t ~ pi(a_t | h_t, z_t)
h_{t+1} = Transformer.append(z_t, a_t)
```

During imagination, `z_t` is sampled from `p(z_t | h_t)` instead. The model
predicts the next embedding, reward, and continuation from the resulting next
state. Tests must establish causal masking, cache equivalence, sequence-boundary
reset, and exact target alignment before control experiments begin.

## Milestones

### M0: Reproducible foundation

Archive the vector prototype, pin all reference implementations and papers,
record licenses and dependency environments, and freeze the visual benchmark
protocol. Produce a novelty matrix and a source-verification command.

### M1: Official visual baselines

Reproduce unmodified DreamerV3 and NE-Dreamer on visual Walker Walk and Cheetah
Run with seeds 0 and 1. Normalize both into the same transition accounting and
latest-policy evaluation artifacts without modifying upstream source.

### M2: Exact JEPA-RSSM

Port only the official Dreamer-CDP delta into the pinned Dreamer backbone:
decoder removal, continuous deterministic predictor, target definition,
negative-cosine objective, gradient routing, and delta-specific optimization.
Use exact NE-Dreamer as the single pre-registered fallback. Apply the 250K
four-task gate in the protocol manifest.

### M3: Stochastic JEPA Transformer

Replace only the RSSM recurrent transition with a causal Transformer that is
the actual temporal state used by the prior, prediction heads, actor, critic,
and imagination. Promote it only if it retains at least 95% of JEPA-RSSM
aggregate AUC, improves a memory-sensitive task, and has acceptable recurrent
rollout cost.

### M4: Encoder selection

At fixed temporal dynamics, compare the Dreamer CNN with a small
parameter-matched ViT on four visual tasks and one distraction condition.
Promote the ViT only for a material control or robustness gain within the
declared compute allowance.

### M5: Single-agent lock

Evaluate one configuration on eight diverse visual tasks, three seeds, and
500K transitions. If the gate passes, run the final 20-task, five-seed,
one-million-transition study.

### M6: Visual multi-agent baseline

Reproduce MATWM on vector coordination, PettingZoo visual control, and one
MeltingPot task. Validate explicit agent-axis replay, lifecycle boundaries,
decentralized action selection, and imagined policy updates. If its linked code
remains unavailable, clearly label and validate a paper-faithful
reimplementation rather than calling it official.

### M7: Matched MARL JEPA substitution

Keep the MATWM-style learner and multi-agent structure fixed while replacing
visual reconstruction with per-agent next-embedding prediction. Require
matched control and a measurable world-model efficiency benefit.

### M8: Coherent joint imagination

Implement the distinct algorithm: shared local temporal dynamics, one
joint-action-conditioned cross-agent block, per-agent stochastic beliefs,
decentralized shared actors, and a centralized critic. Every imagined step must
advance every agent from the same joint state and joint action.

### M9: Policy-shift study

Replace one teammate at a fixed training transition and measure post-shift
return AUC, model-error recovery, transitions and wall-clock time to fixed and
relative recovery targets, final performance, and error by replay age.

## Decision Rule

Proceed to publication-scale evaluation only if the evidence establishes all
three claims:

1. Reconstruction-free stochastic visual world models support policy learning.
2. The method provides a measured efficiency, robustness, or adaptation gain.
3. Coherent joint imagination improves multi-agent control rather than only
   prediction metrics.
