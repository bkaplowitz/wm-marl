# DreaMARL Experiment Ledger

## Scope and status

This document records the complete experiment history that is relevant to the
single-agent JEPA learner and its multi-agent extension. It was reconstructed on
2026-08-22 (Europe/London) from the Git history, 228 retained `launch.json`
manifests, local and pod JSONL metrics, frozen evaluation artifacts, diagnostic
reports, W&B run summaries, and the experiment decisions made during development.

The retained manifest census contains:

- 120 substantive MARL launches;
- 35 MARL dry runs, smoke tests, or compile probes;
- 64 substantive visual-DMC launches;
- 9 visual-DMC dry runs, smoke tests, or compile probes.

Copied report trees were deduplicated. Compilation probes and repeated dry runs
are grouped by purpose below; trained configurations are listed individually or
as matched seed groups. This is an experiment ledger, not a claim that every run
is suitable for a paper comparison.

### Result labels

- **Fixed eval**: deterministic latest-policy evaluation from a saved checkpoint,
  normally 20 episodes for Melting Pot and 10 or 20 for DMC. These are the most
  comparable results.
- **Online**: the mean of the final 20 collection episodes. It includes collection
  exploration and is very noisy on 1,000-step Melting Pot episodes.
- **Incomplete**: the run stopped before its declared budget. The last fixed
  checkpoint is reported when available.
- **Launch failure**: a manifest exists but training produced no metrics.
- **Diagnostic**: a frozen-replay or representation probe, not an RL score.

Unless a row explicitly says `team sum`, Melting Pot numbers are mean return per
agent. This distinction is important: the current evaluator logs both
`per_agent_return_mean` and `team_return_mean`.

## Executive result

The project produced one strong single-agent algorithm and one defensible MARL
baseline, but no successful MARL architectural addition yet.

1. The best single-agent model is the compact decoder-free CNN + causal
   Transformer JEPA configuration. At 500k steps over three seeds it reached
   `972.87` on Reacher Easy, `284.72` on Hopper Hop, `937.29` on Walker Walk,
   and `577.77` on Cheetah Run in fixed evaluation. It decisively exceeded the
   official DreamerV3 training curves on Reacher and Hopper, was close on Walker,
   and was worse on Cheetah.
2. The best MARL configuration is **B0 + exact recent replay**: shared local
   JEPA world model, local shared actor/critic, no peer state, strict decentralized
   execution, joint optimizer, and exponentially recent replay. On Externality
   it reached `47.02` and `46.64` for a two-seed mean of `46.83`, versus `18.53`
   for uniform B0. It also retained Rationalizable Coordination performance
   (`13.79` on the tested seed).
3. B1, B2, peer communication, local actor memory, teammate prediction,
   centralized critics, rollout JEPA, action grounding, latent-change prediction,
   lookahead, and the final factorized joint JEPA all failed to improve robust
   control. Several learned their auxiliary objective well, which proved that
   predictive accuracy alone was not being converted into useful actor/critic
   updates.
4. The simple B0 model is competitive on matrix coordination tasks, unstable and
   sub-SOTA on Externality, and weak on Coop Mining. The task boundary is thus
   not “single-agent JEPA cannot learn MARL”; it is sparse heterogeneous team
   credit and non-stationary coordination.
5. A major protocol issue was discovered late: historical manifests said
   `collective_reward=True`, but the Melting Pot wrapper only added a
   `COLLECTIVE_REWARD` observation. The adapter discarded it and trained on raw
   individual rewards. This does not affect Pure or Rationalizable Coordination,
   where rewards agree, but it is material for Externality and Coop Mining. The
   final F-JEPA experiment corrected the team objective, but its architecture
   still underperformed.

## Part I: best working single-agent RL model

### Exact evaluated configuration

The promoted model is the `dreamarl_visual_final_20260811` configuration. The
500k evidence corresponds to the launch contract at that date, not every later
MARL correctness change.

| Component | Evaluated setting |
| --- | --- |
| Observation | `64 x 64 x 3` RGB, no proprioception |
| Encoder | DreamerV3 CNN, base depth 64, multipliers `[2,3,4,4]`, kernel 5, SiLU, RMS normalization |
| Encoder output | 4096 |
| Temporal model | strict-causal Transformer, width 512, 2 layers, 8 heads, FF expansion 4, context 64 |
| Deterministic state | 8192 |
| Stochastic state | 32 categorical variables x 64 classes, 1% unimix |
| Posterior | observation-conditioned categorical posterior with full local history |
| Target encoder | EMA, source rate `0.01` / decay `0.99`, stop-gradient targets |
| Decoder | none |
| Posterior JEPA | normalized cosine loss, weight `2.0` |
| Dynamics JEPA | focal-action-conditioned cosine loss, weight `2.0` |
| Spatial JEPA | exactly 50% fixed-count masking, fill value 128, weight `1.0` |
| Anti-collapse | SIGReg, 17 knots, 256 projections, weight `0.05` |
| Reward/continuation | 255-bin symlog two-hot reward; binary continuation; weight `1.0` each |
| KL | dynamics `1.0`, representation `0.1`, one free nat |
| Actor and critic | 3 layers x 1024 units each; standard DreamerV3 imagination objective |
| Imagination | horizon 15, lambda `0.95`, actor entropy `3e-4` |
| Return normalization | 5th-95th percentile range, update rate `0.01` |
| Optimizer | one joint optimizer, Adam-style, LR `4e-5`, AGC `0.3`, 1,000-step warmup |
| Replay/training | uniform replay, batch 16 x length 64, train ratio 256, 16 envs |
| Evaluated replay carry | one-step recurrent replay context |
| Numeric mode | BF16 compute on A100 |

Later maintained code changed replay burn-in from 1 to 128 steps and corrected
long-horizon RoPE/cache semantics. Those are correctness improvements, but the
four-task 500k suite has not been rerun under that exact corrected replay
configuration. For `A=1`, changing SIGReg from pooled to per-agent is
mathematically equivalent up to floating-point ordering.

### Three-seed 500k results

Fixed-evaluation seed endpoints:

| Task | Seed 0 | Seed 1 | Seed 2 | Mean | Std across seeds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cheetah Run | 603.3 | 679.0 | 451.0 | 577.77 | 94.80 |
| Hopper Hop | 289.4 | 285.0 | 279.8 | 284.72 | 3.94 |
| Reacher Easy | 972.7 | 967.5 | 978.4 | 972.87 | 4.43 |
| Walker Walk | 939.3 | 934.4 | 938.2 | 937.29 | 2.11 |

Protocol-matched online-curve summary against the ten official visual-DMC
DreamerV3 runs:

| Task | DreaMARL AUC | DreamerV3 AUC | DreaMARL final 500k bin | DreamerV3 final 500k bin | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| Cheetah Run | 384.67 | 459.21 | 499.33 | 670.28 | DreamerV3 wins |
| Hopper Hop | 162.64 | 71.68 | 269.63 | 157.20 | DreaMARL wins clearly |
| Reacher Easy | 804.58 | 620.35 | 957.06 | 847.80 | DreaMARL wins clearly |
| Walker Walk | 751.44 | 793.24 | 883.87 | 942.76 | DreamerV3 wins slightly |

The fixed-eval means and training-bin numbers measure different things; they
should not be interchanged. The defensible single-agent claim is “competitive
with DreamerV3 across these four visual-DMC tasks, with large gains on Reacher
and Hopper,” not “universally better than DreamerV3.”

### 100k architecture and masking screen

The promotion screen used seed 0 with fixed evaluations every 20k:

| Task/configuration | Final fixed eval | Peak fixed eval | DreaMARL AUC | DreamerV3 100k mean | DreamerV3 AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reacher, CNN fixed-count | 721.7 | 837.9 | 555.7 | 376.2 | 279.3 |
| Hopper, CNN fixed-count | 98.38 | 98.38 | 34.8 | 10.1 | 1.2 |
| Walker, CNN fixed-count | 627.6 | 627.6 | 475.3 | 604.6 | 350.4 |
| Walker, CNN Bernoulli mask | 620.3 | 620.3 | -- | -- | -- |
| Walker, ViT Bernoulli mask | 460.9 | 460.9 | -- | -- | -- |
| Walker, ViT fixed-count mask | 679.1 | 679.1 | -- | -- | -- |
| Reacher, ViT multi-block | 403.4 | 403.4 | -- | -- | -- |
| Hopper, ViT multi-block | 85.04 | 85.04 | -- | -- | -- |

The ViT fixed-count result was best on Walker alone, but the CNN fixed-count
configuration won the cross-task/compute decision. It was much cheaper, much
better on Reacher, better on Hopper, and less sensitive to the mask recipe.

### Rejected single-agent visual variants and early gates

| Experiment | Result | Decision |
| --- | --- | --- |
| CNN without spatial JEPA | Reacher `831.4` at 123.9k; Walker `447.8` at 130.5k; both incomplete | Spatial prediction retained; no-spatial was inconsistent and Walker regressed |
| LeWorldModel-style large visual recipe | Reacher `149.6` and Walker `168.4` at about 115k of 300k | Too slow and much weaker than compact CNN; stopped |
| Faithful V-JEPA2.1 visual recipe | Reacher seed 0/1 stopped at 27.7k; online last-20 `73.19` / `100.75`; no fixed eval | Far too expensive for MBRL/MARL; queue aborted |
| Initial A=1 gate | Reacher and Cheetah first launches failed; repaired runs reached only 7.8k | Infrastructure gate, not a result |
| Early singleton gate | Reacher online `357.8` at 38.6k; Cheetah `48.65` at 38.8k | Demonstrated learning but superseded by the controlled screen |
| Current-code parity launch | Reacher 100k manifest only, no metrics | Launch failure; no result |

Only V-JEPA2.1 Reacher seeds 0/1 and LeWorldModel-style Reacher/Walker seed 0
actually accumulated substantive metrics. The planned Reacher/Walker/Hopper x
two-seed queue was stopped after the cost/performance screen; missing entries
must not be described as completed experiments.

## Part II: maintained MARL baseline

### B0 architecture

B0 places an explicit agent axis around the single-agent learner:

```text
[B,T,A,...] <-> [B*A,T,...]
```

It shares one encoder, local causal Transformer, actor, critic, and all heads
across agents. Each focal transition receives only that agent's observation,
previous stochastic state, and own previous action. Team starts remain grouped
for synchronized replay and imagination, but there are no peer tensors, no
communication, no agent IDs, no centralized critic, and no peer-conditioned
`_peer_residual()`. Execution is strict decentralized parameter sharing.

The best B0 training setup was:

- the compact single-agent JEPA representation stack;
- corrected 128-step loss-excluded Transformer replay burn-in, followed by 64
  optimized steps from a 192-step sampled sequence;
- batch 16 x length 64, train ratio 256, approximately 0.25 optimizer updates
  per environment step;
- one joint optimizer at `4e-5` for world model, actor, and critic;
- 15-step local imagination;
- replay capacity 50k for the 50k Melting Pot experiments;
- exact recent sampling weight `0.9998^age` for all model and behavior batches;
- 20-episode deterministic evaluation every 10k steps;
- approximately 142.52M parameters for the maintained B0 run.

### Important historical reward semantics

All B0, B1, B2, and most intermediate Melting Pot runs before F-JEPA trained
their reward head, critic, lambda returns, and actor on the raw vector of
individual rewards. The manifest field `meltingpot_reward_mode=collective` was
incorrect: DeepMind's `CollectiveRewardWrapper` adds a
`COLLECTIVE_REWARD=sum_i r_i` observation but does not replace `timestep.reward`,
and our adapter discarded that extra observation.

This creates no objective mismatch when every agent receives the same reward,
as in the tested Pure and Rationalizable matrix games. It matters in Externality
and Coop Mining:

- Externality rewarding transitions had heterogeneous agent rewards on roughly
  64-68% of replay events;
- Coop Mining seed 1 had 100% reward disagreement on rewarding transitions and
  94.16% of rewarding events paid only one agent;
- the agent causing a socially useful Externality blue-mushroom event receives
  zero individual reward while peers benefit.

The F-JEPA run was the first full run to train on the active-agent mean team
reward while preserving raw individual rewards for diagnostics. It still failed
as a complete architecture, so team-reward correction is necessary protocol
hygiene but is not by itself a solved credit-assignment mechanism.

## Part III: MARL experiment history

### Phase A: early infrastructure and negative controls, July 31-August 8

| Experiment | Budget/result | What it established |
| --- | --- | --- |
| Coin Game DreaMARL | 100k, 128-episode final eval `-0.0039` mean return/agent | Initial MARL system ran end to end but learned no useful Coin Game policy |
| Coin Game MAPPO seed 0 | 100k, trained eval `1.035`, random `0.289`, but curve gate failed | Baseline implementation could improve a seed, not stably |
| Coin Game MAPPO seed 1 | 100k, trained `-0.0508`, random `-0.0898`; gate failed | Strong seed dependence |
| SMAC `3m` DreaMARL seeds 0/1 | 100k each, final mean return `0.0` | Initial generic MARL path completely failed on SMAC |
| Initial Melting Pot Externality | 49.8k, online last-20 `40.59`, no fixed eval | Showed transient learning, but pre-fix and not comparable |
| Initial Pure Coordination | 49.3k, online `5.804`, no fixed eval | Matrix coordination signal appeared early |
| Initial Chicken/Rationalizable/Stag queues | manifests only, no metrics | Launch failures |
| Frozen representation interventions | Externality correct cosine-distance about `0.127`; Pure about `0.036`; null/shuffled actions and agents almost unchanged | Existing representation contained little usable peer/action-specific information |
| Focal action-assignment diagnostic, two WM seeds | horizon-1 cosine top-1 `0.913` / `0.936` over ten action candidates; win rate `0.935` / `0.952`; event win rate `0.939` / `0.876` | The local dynamics model was strongly conditioned on the focal action; missing own-action information was not the bottleneck |
| Cross-agent action probe, WM seed 0 | local accuracy `0.6800`, joint `0.6765`; mean NLL reduction `-0.00487` | Peer context made prediction slightly worse |
| Cross-agent action probe, WM seed 1 | local accuracy `0.6689`, joint `0.6654`; mean NLL reduction `-0.00713` | Negative result replicated |
| Joint-dynamics probe | joint context improved held-out NLL only about 0.7-1.0% at horizons 1-2, inconsistent by horizons 4-8; peer-action-only context did not help | A small joint signal existed, but it was weak and short-horizon |
| Team-outcome probe, two WM seeds | joint context reduced outcome MSE by `2.2%` / `3.4%` at h1, `9.3%` / `7.6%` at h8, and `24.9%` / `28.7%` at h32 | Other-agent state carried long-horizon team-outcome information even though next-action prediction barely improved |
| Team-value/local-memory probe | local test prediction beat the joint predictor; joint return RMSE was roughly `8.64-11.76` with strongly negative R2; full local-memory launch pairs failed | A naive pooled joint value representation generalized worse than local state |

### Phase B: early Melting Pot architecture ladder, August 2-15

These rows mostly predate deterministic fixed checkpoint evaluation. Their
online last-20 scores are historical signals, not valid ranking endpoints.

| Experiment | Task/seeds | Result | Verdict |
| --- | --- | ---: | --- |
| Shared context | Externality s0/s1 | `2.782` / `0.803` at about 9.6k, incomplete | Failed |
| Scaled 82M model | Externality s0/s1 | `1.73` / `1.70` at 38.4k, incomplete | Scaling did not solve learning |
| 82M with LR `4e-5` | Externality s0/s1 | `0.558` / `0.776` at 18.3k, incomplete | Failed |
| Early joint world model | Externality s0/s1 | `1.279` / `0.448` at 16.9k, incomplete | Failed |
| Exact CDP parity gate | Coop Mining s0 | online `6.658` at 36.3k, incomplete | Weak learning signal |
| Exact CDP parity gate | Externality s0 | online `5.588` at 49.7k | Weak |
| Stage-4 initial MARL | Coop Mining s0 | fixed `6.483` at 20k; stopped 24.0k | Some signal, weak |
| Stage-4 initial MARL | Externality s0 | fixed `2.54` at 20k; stopped 27.8k | Failed |
| Stage-1 explicit agent axis | Externality s0/s1 | online `17.68` / `21.63` at 50k | First replicated useful baseline signal |
| Stage-1 explicit agent axis | Pure Coordination s0 | online `6.786` | Competitive matrix behavior |
| Stage-3 peer feature | Coop Mining s0 | online `8.125` at 9k, incomplete | Too early to claim |
| Stage-3 peer feature | Externality s1 | online `0.597` at 10.4k, incomplete | Failed |
| Stage-3 peer feature | Pure Coordination s0 | online `7.311` | Good matrix score, no general benefit |
| Complete joint world model | Externality s0/s1 | online `17.63` / `4.46` | High variance, rejected |
| Local executable-state supervision | Externality s0/s1 | fixed `13.45` / `33.57`, mean `23.51` | Better seed 1 but unstable and below later recent B0 |
| Independent context-64 | Externality s0 | online `3.42` at 33.1k, incomplete | Failed |
| MARIE-like context-64 | Externality s1 | online `6.972` at 23.1k, incomplete | Failed |
| Recurrent MARIE-JEPA | Externality s1/s2 | online `1.45` / `0.60` at 1.93k | Runtime/learning failure |
| Restored core baseline | Coop Mining/Externality s0 | online `5.242` / `14.39` | Reference only |
| Joint-prior correction | Externality s0 | online `13.51` | No clear improvement |
| Joint-prior correction replication | Externality s1 | online `1.026` at 14.5k, incomplete | Unstable |
| Joint-prior correction | Pure Coordination s0 | online `6.781` | Neutral on matrix task |
| Fully joint recurrent transition | Externality/Pure s0 | `0.592` / `0.436` at about 5.2k | Failed |
| Joint-primary transition | Externality s0 | `0.0` at 1.44k | Failed immediately |
| Joint-action local JEPA | Externality s0 | `5.822` at 18.3k, incomplete | Failed |
| Joint-action local JEPA | Pure s0 | `0.238` at 39.6k, incomplete | Catastrophic regression |
| Gated peer residual | Externality s0/s1 | online `33.93` / `7.24` | One high seed, severe instability |
| Gated peer residual | Pure s0 | online `6.977` | Neutral/good matrix result |
| Gate-closed control | Externality s1 | online `5.86` | Suggested the seed-0 gated result was not robust |
| Peer attention | Externality s0 | online `5.05` at 37.8k, incomplete | Failed |
| First matched actor-memory screen (`imag_last=16`) | Externality control/treatment | online `2.208` at 25.0k / `1.982` at 21.3k, both incomplete | Inconclusive but negative; superseded by the corrected A/B |
| Corrected baseline v2 | Externality s0 | online `3.412` at 50k | Weak seed; exposed high variance |
| Corrected baseline | Coop Mining s0 | online `3.783` | Weak |

The early peer residual averaged peers' previous latent-action tokens at runtime.
It therefore violated the declared decentralized-execution contract. It was
removed from B0 rather than relabeled as a communication policy.

### Phase C: strict local actor belief and corrected B0, August 16-17

The local-actor experiment created a separate focal-history causal Transformer
belief so peer-conditioned world state could not leak into execution. It was a
genuine decentralized design, including replay-position restoration, FP32 RoPE,
and intervention testing.

| Variant | Seed 0 fixed | Seed 1 fixed | Status |
| --- | ---: | ---: | --- |
| Control: corrected local B0 path | `21.18` at 50k | peak `17.07` at 20k; `9.22` at 30k when stopped | Winner |
| Treatment: separate local actor Transformer/BPTT | `14.74` at 50k, peak `22.73` | `12.52` at 20k, stopped 26.2k | Worse; removed |

The treatment was correctly local, but it duplicated state estimation, had weak
real-history supervision, and made optimization harder. The experiment answered
the architecture question: strict decentralized execution should come from the
authoritative local world state in B0, not a second actor-only memory.

### Phase D: controlled B0, B1, and B2

All rows below use deterministic fixed evaluation.

| Variant | Externality seed 0 | Seed 1 | Two-seed final mean | Result |
| --- | ---: | ---: | ---: | --- |
| B0 uniform replay, 50k | `21.58` | `15.48` | `18.53` | Maintained architecture baseline; seed 1 peaked `53.45` then fell |
| B1 same-time centered set JEPA, 30k | `6.17` | `5.10` | `5.64` | Failed |
| B1 detached same-time set JEPA, 30k | `4.48` | `9.38` | `6.93` | Failed |
| B1 future K=1 joint-action JEPA, 30k | `10.66` | `13.40` | `12.03` | Best B1, still below B0 |
| B2 JEPA team-belief centralized critics, 30k | `9.55` | `5.28` | `7.42` | Failed |

B1 was a full training-only agent-axis JEPA: masked active agents, online/EMA set
encoders, slot prediction, Sinkhorn set matching, hidden-agent coverage, variance
and decorrelation terms, and future action conditioning. It reached auxiliary
cosines around `0.92-0.95`, so the module learned its representation objective.
It did not improve return.

B2 added a causal team-belief Transformer and centralized fast/slow critics while
keeping the actor decentralized. The critic could consume the belief, but return
fell further. These are implementation successes and performance failures; they
must not be called successful MARL mechanisms.

Rationalizable Coordination under uniform B0:

| Seed | Final fixed eval | Peak |
| --- | ---: | ---: |
| 0 | 13.42 | 13.66 |
| 1 | 12.31 | 12.33 |
| Mean | 12.87 | -- |

The quoted MATWM reference is `12.2`, so B0 is competitive on this task under
the available score convention.

Pure Coordination was not rerun with the final fixed-eval B0 launcher before the
current panel, but repeated compatible earlier B0-like runs scored `6.78-7.31`
online versus the quoted MATWM `6.8`. This is supportive, not as strong as the
Rationalizable fixed-eval result.

### Phase E: behavior optimization and replay

| Variant | Externality s0 | Externality s1 | Mean | Rationalizable s0/s1 | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| Uniform B0 control | 21.58 | 15.48 | 18.53 | 13.42 / 12.31 | Reference |
| Split WM/actor/critic optimizers | 22.05 | 11.61 | 16.83 | 13.02 / 12.59 | Neutral/worse; seed 1 actor became nearly deterministic early |
| Recent-WM/uniform-behavior dual routing | no metrics | no metrics | -- | -- | Both launches failed before training |
| Exact single-stream recent replay | 47.02 | 46.64 | **46.83** | 13.79 / not run | Only robust large improvement |
| Recent replay + train ratio 1024 | 10.06 | 9.13 | 9.60 | -- | Four-times update cadence was much slower and collapsed performance |

Exact recent replay improved the Externality endpoint by `+153%` over uniform
B0 and reduced final seed spread to `0.38`. Its fixed-eval trajectory was:

| Step | Seed 0 | Seed 1 | Two-seed mean |
| ---: | ---: | ---: | ---: |
| 10k | 0.70 | 1.64 | 1.17 |
| 20k | 12.86 | 26.21 | 19.54 |
| 30k | 24.53 | 14.39 | 19.46 |
| 40k | 24.87 | 9.59 | 17.23 |
| 50k | 47.02 | 46.64 | 46.83 |

This is still zigzag learning, but both seeds recovered to the same endpoint.
The 1024-ratio experiment disproved the hypothesis that the main problem was
simply too few optimizer updates. It produced roughly four times the updates,
ran about four times slower, and was much worse.

### Phase F: JEPA-specific mechanism sweep on recent B0

| Mechanism | Seeds/budget | Fixed result | Internal signal | Verdict |
| --- | --- | --- | --- | --- |
| Two-step rollout JEPA, scale 0.5 | s0 stopped 36.85k | `12.06` at 30k, peak `21.98` | Learned rollout targets | Worse/incomplete |
| Two-step rollout JEPA, scale 1.0 | s0 50k | `45.44` | Stable representation metrics | Tied/slightly worse than recent B0 `47.02` |
| Reward-gradient isolation | s0 stopped 20k | `2.54` | World reward gradients removed from representation | Catastrophic; reward gradients are essential |
| Causal teammate predictor consumed by actor+critic | s0/s1 stopped near 37k | `10.64` / `10.22` at 30k | About 54% teammate-action accuracy vs 12.5% chance | Predictor learned, consumer hurt control |
| Teammate predictor consumed by dynamics | s0 stopped 38.3k | `5.07` at 30k, peak `6.39` | Predictive head learned | Failed |
| Training-only true joint CTDE critic | s0 stopped 45.2k | `18.25` at 40k | Critic consumed true joint training state | Below recent B0; later instability |
| Action-grounding JEPA | s0/s1 50k | `55.29` / `1.42`, mean `28.36` | About 85% action accuracy, positive margin | Highest single seed but catastrophic cross-seed instability |
| Latent-change JEPA | s0 50k | `42.32` | Learned change target | No gain |
| JEPA lookahead teacher | s0/s1 stopped near 28.7k | `7.98` / `10.32` at 20k | Teacher entropy about 2.00 vs `ln(8)=2.079`; action score std only 0.002-0.019 | Teacher was nearly uniform and harmful |
| Final factorized joint JEPA + collective-mean objective | s0/s1 50k | `15.89` / `6.70`, mean `11.30` | joint JEPA cosine about 0.61-0.67; team-reward RMSE 0.01-0.02; critic EV 0.82-0.94 | Complete architecture, clear performance failure |

The final F-JEPA model was not a shortcut. It used a permutation-equivariant
agent-axis Transformer, focal and joint actions through AdaLN-Zero, FP32
attention, active-agent masking, a two-step stop-gradient rollout, collective
team reward, and an auxiliary individual reward path. Its strong supervised
metrics alongside poor return are decisive evidence that world-model fit and
critic explained variance are not sufficient for useful decentralized action
credit.

## Part IV: current seven-task B0 benchmark panel

The current panel deliberately reverted to the best performance architecture:
B0 + exact recent replay, commit `c15c122`, launched by `fcbd423`, two seeds,
50k steps, fixed 20-episode evaluation every 10k. It covers the seven tasks used
in the MATWM Melting Pot table.

Quoted MATWM reference values are:

| Task | MATWM quoted score |
| --- | ---: |
| Chicken in the Matrix: Arena | 21.5 |
| Coop Mining | 19.0 |
| Externality Mushrooms: Dense | 146.8 |
| Gift Refinements | 75.0 |
| Pure Coordination: Repeated | 6.8 |
| Rationalizable Coordination: Repeated | 12.2 |
| Stag Hunt in the Matrix: Arena | 7.2 |

These references are not yet proven protocol-identical. Available public
successor code uses `rewards_reduce: sum`, while our primary metric is per-agent
mean. For five-agent Externality, the recent-B0 `46.83` per-agent mean is
`234.15` as team sum. Therefore a raw comparison of `46.83` with `146.8` may be
a scale error. Both metrics must be published until MATWM's exact adapter and
aggregation are reproduced.

Completed current-panel result:

| Coop Mining step | Seed 0 | Seed 1 | Mean |
| ---: | ---: | ---: | ---: |
| 10k | 0.608 | 7.308 | 3.958 |
| 20k | 11.425 | 6.083 | 8.754 |
| 30k | 5.242 | 7.792 | 6.517 |
| 40k | 6.942 | 4.025 | 5.483 |
| 50k | 8.875 | 4.550 | **6.713** |

Coop Mining is a clear failure relative to the quoted `19.0`. Replay analysis
for seed 1 found 1,868 rewarding transitions: 1,759 iron, 70 gold, and 39 other.
Gold was only 3.75% of reward events; 94.16% of rewarding events paid one agent.
The chronological gold-event fraction rose from 0.7% to 7.9% and then fell to
6.6%. The model learns safe individual iron collection and only transiently
discovers coordinated gold collection.

As of the ledger snapshot, Gift Refinements was still running: seed 1 had a
`0.0` fixed eval at 10k and was at 12.3k; seed 0 was at 7.7k and had not reached
its first fixed checkpoint.
Chicken, Externality, Pure Coordination, Rationalizable Coordination, and Stag
Hunt were still queued in this fresh panel. Historical Externality and
Rationalizable results above remain the completed B0 evidence; pending panel
entries must not be reported as final results.

## Part V: correctness and protocol audit

The trusted B0/B1/B2 experiments were launched only after the following issues
were found and corrected or explicitly documented:

1. Transformer replay carry was reconstructed from one step while online policy
   used up to 64; replay now uses 128 loss-excluded burn-in steps.
2. Local actor replay initially failed to preserve absolute RoPE position;
   position is now part of replay carry and was tested mid-episode near step
   1,000.
3. BF16 RoPE lost adjacent-position resolution at long horizons; angle and
   trigonometry are computed in FP32.
4. The parallel Transformer path lacked the 64-step sliding context mask; it was
   added and checked beyond the context length.
5. Configured categorical 1% unimix was not passed into the action distribution;
   it was connected, restoring a probability floor.
6. Pooled SIGReg scaled linearly with agent count; it became per-agent then
   averaged, with inactive placeholders excluded.
7. Inactive imagined agents contaminated percentile return normalization; masks
   are applied before normalization updates.
8. Replay value validity was shifted by one timestep; value losses now use the
   matching validity indices.
9. `action_mask` was documented but unused; observed masks are enforced online
   and a supervised head supplies imagined availability.
10. Inactive agents previously advanced temporal caches; maintained semantics
    preserve carry while inactive and exclude inactive losses.
11. Inline evaluation could consume pending JAX parameter synchronization;
    controlled comparisons use fixed checkpoint evaluation.
12. Generic reward reporting obscured agent mean versus team sum; current
    evaluator records both.
13. `_peer_residual()` made policy state peer-conditioned despite a decentralized
    contract. It was removed from B0; focal-history intervention tests verify
    peer histories cannot change the focal online policy.
14. Melting Pot's Shimmy wrapper ignores `reset(seed=...)`. Construction-time
    Lab2D seed streams are controlled, but exact same-seed trajectory identity
    cannot be guaranteed and is documented as such.
15. The `collective_reward` setting did not change the training reward. This was
    finally corrected in F-JEPA, and raw individual/team metrics were separated.

Early results that predate these corrections are retained for historical
completeness but are not evidence about the final algorithm.

## Part VI: failed, aborted, and non-result launches

The following produced manifests but no substantive training result:

- first A=1 Reacher/Cheetah launches before the repaired singleton gate;
- initial Melting Pot Chicken, Rationalizable, Stag, and several “stable/final”
  relaunch directories;
- the first local-memory baseline/treatment and dual-memory screen;
- the parallel-Transformer 10k gate;
- current-code Reacher parity launch;
- recent-world/uniform-behavior split replay, both seeds;
- multiple joint-primary, joint-recurrent, peer-attention, action-cost,
  local-belief, gated-peer, joint-pool, and transition compile probes;
- V-JEPA2.1 Walker/Hopper and LeWorldModel-style second seeds that were queued
  but cancelled after the cost screen.

Thirty-five additional MARL and nine DMC dry/smoke/probe manifests were used to
validate shapes, compilation, action masks, checkpoint restore, replay carry,
and GPU memory. They are engineering checks, not experiments and have no score
to report.

## Part VII: JECC acceptance result

JECC v2 replaced the observational action-effect probe with the frozen B0
Transformer dynamics as its intervention operator. For every legal focal
action, it advanced the full local B0 carry, retained factual peer next states,
and predicted 5/15/32-step outcome embeddings and utility. The B0 actor and
execution path remained unchanged.

The exact acceptance fit used 5,000 optimizer updates on frozen seed-123 replay.
The final factual outcome cosine was `0.980`, yet the matched deterministic
96-battle intervention gate failed:

| Controller | Wins | Corrected return | Enemy deaths | Ally survivors | Timeout rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| B0 | 13/96 | 16.02 | 1.72 | 1.70 | 0.792 |
| Observational all-action critic | 0/96 | 11.37 | 0.50 | 1.14 | 0.865 |
| JECC v2 | 0/96 | 11.98 | 0.49 | 1.05 | 0.823 |

Thus B0-grounded one-step candidate states did not make the learned long-horizon
counterfactual ranking safe. Strong factual representation fit again failed to
imply useful intervention ordering. JECC v2 is retained for reproduction but
must not be used as the proposed method or launched for full RL training.
The controller changed 2,191 of 2,906 controllable focal decisions (`75.4%`),
so this was not a no-op collapse; its counterfactual ordering actively replaced
most B0 decisions with worse actions.

The production-shape engineering profile itself passed: one cached update took
`1.26 s`, peaked at `20,983 MiB` on an A100 80GB, and all four optimizer groups
had finite losses and gradient norms. Compute was therefore not the reason the
acceptance gate failed.
