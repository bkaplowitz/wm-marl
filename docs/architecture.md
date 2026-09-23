# Architecture and training flow

The actor is decentralized: each agent uses its own observation, previous action,
and history. Parameters are shared across agents. Joint attention and the
centralized critic are training-only components.

## Model components

| Component | Baseline |
| --- | --- |
| Observation encoder | Symlog vector inputs; three 1024-unit MLP layers |
| Target encoder | EMA copy, update rate 0.01 |
| Local history model | Two causal Transformer layers, width 512, eight heads, context 64 |
| Local deterministic state | 4096 features |
| Local stochastic state | 32 categorical variables, 64 classes each; latent unimix 0.01 |
| Local posterior | Observation embedding and local history → categorical distribution |
| Local prior | History → categorical distribution; trained by local KL |
| Joint predictor | Two agent-attention and 12 temporal-attention layers, width 256, context 16 |
| Joint outputs | Future observation embedding, reward, continuation, availability, liveness |
| Local availability head | One 512-unit layer; supplies imagined action masks |
| Actor | Three 512-unit layers; policy unimix 0 |
| Central critic | Two attention layers, width 256; two 256-unit value layers |
| Direct multi-step JEPA | Horizon-specific heads at 1, 2, 4, 8 |

There are no local reward or continuation heads. The joint availability head
still receives factual supervision, but PPO uses the local availability head
on each completed imagined state.

During imagination, the actor selects the current action from local latent
features. The joint predictor consumes the synchronized joint action and current
local states. It predicts an observation embedding; the **same local posterior**
used for real observations converts that prediction into a categorical state.
The local prior is therefore not the transition distribution used by joint PPO
imagination, although its KL and embedding objectives remain active in training.

## One learner update

1. Consume independently sampled world and behavior replay views. Reconstruct
   temporal histories using a 192-record prefix; optimize the following 64
   records in a batch of 16 sequences.
2. Update local and joint world-model parameter groups on the world view.
3. Reconstruct the behavior view with the updated world model and generate
   five-step joint imagined trajectories.
4. Freeze that trajectory batch, including realized action masks, old action
   probabilities, returns, and advantages.
5. Run five actor updates and five critic updates against the frozen batch.
6. Update target models and write refreshed world-view states back to replay.

`training/learner.py` owns this order. `training/behavior.py` prepares the PPO
batch and losses. Team-aware reconstruction and imagination are isolated in
`marl/replay.py` and `marl/imagination.py`.

## Supervision and gradient boundaries

- **Local representation:** posterior/prior KL, posterior and dynamics embedding
  prediction, local availability prediction, and SIGReg (`0.05`). The EMA
  embeddings used as JEPA targets are stopped.
- **Factual joint JEPA:** next-embedding cosine, posterior-interface consistency,
  joint outcomes, and posterior alignment (`0.05`). Local state inputs are
  detached in the baseline.
- **History gradient:** the optional profile adds factual embedding-prediction
  gradients to local history features. A frozen-joint pass uses the same dropout
  draw, preserving the ordinary joint-parameter update. EMA targets remain
  detached. See `training/joint.py`.
- **Direct multi-step prediction:** horizon-weighted embedding cosine, plus the
  legal-action counterfactual margin (`0.1`). These heads supervise learning;
  they do not generate the PPO trajectory. See `training/direct_jepa.py` and
  `training/multistep_jepa.py`.
- **Self-fed prediction:** recorded joint actions unroll the actual simulator;
  supervision at horizons 2, 4, and 5 uses eight sampled anchors and BPTT2.
  Local transition parameters are frozen on this path. Trajectory KL is retained
  at coefficient `0.1` inside the self-fed loss scale (`0.1`). See
  `training/self_fed.py`.
- **PPO:** actor and critic optimize stopped imagined features. Actor/critic
  gradients do not update the world model. Replay-value supervision is retained
  with weight `0.3`.

The local and joint model optimizers use learning rate `1e-4`. Actor and critic
use `3e-5`; PPO clipping is `0.2`, entropy is `0.003`, and lambda is `0.95`.
The discount is `1 - 1/333`, approximately `0.997`.

## Replay and startup

One FIFO store holds up to 250,000 records. Both training views independently
mix 50% uniform sequence starts and 50% exponential recency (`0.9998` per age
increment). Reporting uses its own uniform sampler and random stream.

The fixed4 stream samples the first world batch as soon as replay has an
eligible sequence start, then samples the first behavior batch when four starts
are eligible. It retains that pending pair until learning begins at step 5,000.
Each learner call selects the next pair **before** updating the model or writing
recurrent states back. All replay reads occur on the collection thread; there
is no background sampling race.

The 192-record replay prefix reconstructs temporal state. It is distinct from
extra world-only pretraining, which is disabled in the baseline.

## Source boundaries

`models/` and `world_model/` define parameterized computations.
`training/` defines objectives and optimization.
`marl/` handles the explicit team axis, joint rollout state, and centralized
critic context. The small runtime classes compose these responsibilities;
objective helpers are ordinary functions rather than additional agent classes.

Module names used for parameter creation are retained intentionally. Renaming
those names can change initialization and checkpoint compatibility even when a
mathematical layer looks equivalent. The empty carry slot alongside encoder and
dynamics state is retained for compatibility with the existing agent-state tree.
