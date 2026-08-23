# 3s_vs_4z outcome decomposition and frozen credit probes

This stage is diagnostic only. It does not modify the B0 actor, world model,
critic, replay, reward, or decentralized execution contract.

## Outcome decomposition

Three retained B0 checkpoints were evaluated for 32 deterministic battles with
the same held-out worker seeds and the Stage-0 dual outcome instrumentation.

| Seed | Wins | Timeout | Legacy | Corrected | Legacy gap | Enemy damage | Enemy deaths | Ally survivors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 123 | 9.38% | 84.38% | 19.67 | 15.95 | 3.72 | 634.90 | 1.625 | 1.844 |
| 223 | 0% | 53.12% | 11.98 | 8.45 | 3.53 | 353.09 | 0.188 | 0.594 |
| 323 | 0% | 50.00% | 11.45 | 8.58 | 2.87 | 356.64 | 0.375 | 0.531 |

The legacy signal is inflated by shield regeneration, but corrected damage is
substantial. Seeds 223 and 323 primarily damage shields and almost never finish
an enemy. Seed 123 does learn combat and preservation, but usually times out
after killing only one or two enemies. The failure is therefore not merely a
reward-accounting artifact.

## Frozen dataset

The three frozen checkpoints generated 96 sampled-policy battles each on held-
out collection seeds. Each transition records the frozen local JEPA categorical
state, local observation, legal joint action, and aligned 5/15-step corrected
damage, death, survival, timeout, and win labels.

- 288 battles;
- 49,153 team timesteps;
- 147,459 focal-agent examples;
- episode-disjoint 70/15/15 train/validation/test splits, stratified by source
  checkpoint;
- no learner parameter updates.

## Probe 1: local versus joint outcome information

Capacity-matched readouts use the same compact embedding of the frozen JEPA
categorical state. The local model sees only the focal state and action. The
joint model sees all three states and actions. A third evaluation shuffles only
the two peer actions on the held-out set.

| Outcome | Local | Joint | Joint with shuffled peer actions |
|---|---:|---:|---:|
| 15-step corrected return R2 | 0.645 | **0.713** | 0.684 |
| 15-step damage R2 | 0.699 | **0.776** | 0.745 |
| Ally death within 15, AUC | 0.758 | **0.836** | 0.815 |
| Enemy death within 15, AUC | 0.877 | **0.895** | 0.886 |
| Episode ally survivors R2 | 0.178 | **0.278** | 0.253 |
| Episode win AUC | 0.960 | **0.976** | 0.975 |

Joint state supplies meaningful outcome information unavailable locally.
Correct peer actions add a smaller but consistent increment for damage,
corrected return, and ally death. Peer actions add almost no unique episode-win
information in this observational dataset.

## Probe 2: all-action critic intervention

A frozen all-action critic was trained on the same episode-disjoint split. It
conditions on the joint JEPA states and the two peer actions, emits one value per
legal focal action, and is supervised only at the observed focal action against
15-step corrected return. Its held-out factual prediction achieved R2 = 0.728
and MAE = 0.299.

The decisive test used the seed-123 checkpoint for two matched sets of 96
deterministic battles. The privileged controller replaced one focal agent's
action per battle while preserving both peer actions. It changed 9.77% of focal
decisions.

| Outcome | Frozen actor | Focal intervention |
|---|---:|---:|
| Win rate | **13.54%** | 0% |
| Corrected return | **16.02** | 11.45 |
| Enemy damage | **628.37** | 477.77 |
| Enemy deaths | **1.719** | 0.333 |
| Ally survivors | **1.698** | 1.031 |
| Attack fraction | **12.87%** | 9.75% |
| Movement fraction | **62.16%** | 45.25% |
| No-op fraction | 24.61% | **43.82%** |

The factual prediction score did not translate into counterfactual action
quality. The action head learned observational correlations and preferred
actions associated with state occupancy rather than causal improvement. This
probe rejects direct offline argmax control and rejects treating a high factual
Q R2 as evidence of usable credit assignment.

## Decision

1. Keep B0 and matched separated REINFORCE as the clean optimization topology;
   do not promote PPO.
2. Peer information is useful for meaningful future outcomes, so a training-
   only joint outcome representation remains justified.
3. Do not connect the current all-action readout to the actor or call it a
   counterfactual critic.
4. Any next action-credit mechanism must obtain interventional support, for
   example through short action-conditioned JEPA imagination with uncertainty
   or explicit randomized focal-action data. Observational Q regression alone
   is ruled out.
