# DreaMARL Frozen Representation Diagnostics

## Question

Does correctly aligned information from other agents improve the frozen
DreaMARL world model, and at which representation interface?

## Contract

- No environment interaction and no online learner updates.
- Frozen checkpoint and replay for each environment.
- 32 non-overlapping replay windows sampled from distinct replay chunks.
- 64 context steps reconstruct the causal Transformer cache.
- 32 logged-action open-loop prediction steps are evaluated.
- Training and validation are split by replay window.
- Only a zero-initialized, permutation-equivariant residual adapter is trained.
- Predictor probes are parameter matched at 397,888 parameters.
- Correct, agent-shuffled, environment-shuffled, and null controls are reported.
- Raw categorical KL includes the model's 1% uniform mixture and is measured
  before free-nats clipping.

The checkpoints are the latest complete periodic checkpoints: step 38,880 for
Pure Coordination and step 46,330 for Externality. The upstream loop did not
write a final checkpoint at normal exit; future runs must add a final save.

## Baseline World Model

| Environment | Teacher-forced cosine | Teacher-forced raw KL | Open-loop cosine mean | h1 | h8 | h16 | h32 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pure Coordination | 0.01646 | 1.2133 | 0.03641 | 0.02776 | 0.02214 | 0.02066 | 0.08850 |
| Externality Dense | 0.06326 | 2.9922 | 0.12728 | 0.08374 | 0.11850 | 0.12662 | 0.15512 |

Externality has a clear compounding-error problem. Its cosine error nearly
doubles from h1 to h32. Pure Coordination is substantially easier for the
frozen world model.

## Controlled Interventions

Externality validation after 100 adapter updates:

| Intervention | Metric | Baseline | Correct context | Shuffled context | Interpretation |
|---|---|---:|---:|---:|---|
| Other actions -> predictor | cosine | 0.12728 | 0.12636 | 0.12636 | Small capacity gain; no alignment use |
| Other latents -> predictor | cosine | 0.12728 | 0.12631 | 0.12630 | Small capacity gain; no alignment use |
| Paired latent/action -> predictor | cosine | 0.12728 | 0.12632 | 0.12632 | Pairing gives no held-out benefit |
| Paired latent/action -> prior | raw KL | 2.99217 | 2.89073 | 2.88934 | Lower KL, but shuffled is equally good |
| Recurrent paired context -> predictor | cosine | 0.12728 | 0.12631 | 0.12631 | Persistence does not reveal aligned signal |

At 1,000 updates, aligned latent and paired predictor probes overfit and become
worse than baseline: 0.12950 and 0.12974 respectively. The prior also worsens
to 3.11093. Null and shuffled controls remain as good as or better than correct
context. Pure Coordination shows the same absence of a correct-context gap.

## Decision

No tested joint-conditioning interface should be promoted into the learner.
The probes can fit replay, but their held-out gain does not depend on correctly
aligned cross-agent information. This rejects the simple hypothesis that the
current representation is adequate and only lacks a generic agent mixer.

The evidence instead supports the plan's Outcome E: the present local JEPA
target is not organized so a small shared interaction module can exploit the
relevant multi-agent variables. A generic joint-action or joint-latent mixer
would add complexity without an identified mechanism.

## Next Step

The next bounded experiment should change the representation objective, not
the actor, critic, reward head, or imagination schedule. Compare the current
local future target against task-neutral alternatives:

1. Cross-agent masked future prediction.
2. Shared event-token prediction from all agent views.
3. Paired cross-agent future prediction that preserves agent/action identity.
4. Shuffled-view and shuffled-action controls for every objective.

Promotion requires a held-out improvement that is specifically destroyed by
shuffling the aligned cross-agent information, followed by a short online
confirmation on at least one interaction and one coordination environment.

## Scope Limitation

The recurrent intervention is a causal interaction-memory probe feeding the
frozen representation predictor; it is not a full replacement of the frozen
temporal Transformer. The prior intervention measures teacher-forced raw KL
and does not propagate its adjusted latent samples through a 32-step rollout.
Those larger interventions are intentionally deferred because the cheaper
probes found no alignment-specific signal to justify them.
