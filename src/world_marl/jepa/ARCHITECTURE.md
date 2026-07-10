# JEPA Architecture

This note describes the current single-agent JEPA world-model architecture in
this repository. It is intended as a compact reference for the mainline method,
not a record of old ablations.

## Goal

The model learns action-conditioned dynamics in representation space:

\[
p(z_{t+1}, r_t, c_t \mid z_t, a_t)
\]

where:

- \(o_t\) is an environment observation;
- \(a_t\) is a continuous action;
- \(z_t = E_\theta(o_t)\) is a learned latent state;
- \(r_t\) is reward;
- \(c_t\) is continuation probability.

The model predicts future representations, rewards, and continuation. It does
not reconstruct observations or pixels.

## Components

The world model contains:

1. an observation encoder \(E_\theta\);
2. an action encoder;
3. a causal latent dynamics transformer;
4. a latent predictor;
5. a reward head;
6. a continuation head;
7. an actor head;
8. a critic head.

The observation encoder maps observations to latents:

\[
z_t = E_\theta(o_t)
\]

The actor and critic consume this same latent:

\[
a_t = \pi_\phi(z_t)
\]

\[
V_t = V_\psi(z_t)
\]

The dynamics model predicts from latent/action history:

\[
\hat z_{t+1},\ \hat r_t,\ \hat c_t
= M_\theta(z_{t-k:t}, a_{t-k:t})
\]

## Transformer Dynamics

The dynamics model is a causal transformer over latent/action history.

For each timestep, it forms a token from a projected latent and an encoded
action:

\[
x_t = W_z z_t + A_\theta(a_t)
\]

Position information is injected with rotary position embeddings (RoPE) inside
self-attention. RoPE rotates the query and key vectors as a function of timestep
before the attention dot product:

\[
q_t = R_t W_q x_t,\quad k_t = R_t W_k x_t
\]

No additive position embedding is added to the token stream.

The transformer uses causal attention. A token can attend only to previous
tokens in the configured context window. Attention across episode boundaries is
masked.

Each transformer block is pre-norm:

\[
h' = h + \mathrm{SelfAttention}(\mathrm{LN}(h))
\]

\[
h^{next} = h' + \mathrm{GEGLU}(\mathrm{LN}(h'))
\]

The final hidden state feeds the latent, reward, and continuation heads.

The latent transition is residual by default:

\[
\hat z_{t+1} = \mathrm{norm}(z_t + \Delta_\theta)
\]

For multi-step prediction, predicted latents are recursively appended back into
the context while replay actions provide the future action sequence.

## JEPA Loss

The JEPA target is the encoded next observation:

\[
z_{t+1}^{target} = E_\theta(o_{t+1})
\]

By default, the target branch is stopped:

\[
z_{t+1}^{target}
= \mathrm{stopgrad}(E_\theta(o_{t+1}))
\]

The latent prediction loss is cosine distance:

\[
L_{JEPA}
= 1 - \cos(\hat z_{t+1}, z_{t+1}^{target})
\]

Reward uses mean squared error. Continuation uses binary cross entropy. The
world-model loss is:

\[
L =
L_{JEPA}
+ \lambda_r L_r
+ \lambda_c L_c
+ \lambda_{reg} L_{SIGReg}
\]

SIGReg is the anti-collapse regularizer. It regularizes the latent distribution
without adding an observation decoder.

Episode boundaries are masked so the model is not trained to predict through
environment resets.

The current implementation uses one encoder. There is no EMA target encoder and
no observation decoder in the world-model loss. The default target-gradient
mode is stop-gradient; the same encoder defines both the current latent and the
future target latent.

## Diagnostic Decoder

Following the LeJEPA visualization recipe, an optional observation decoder can
be fit *after* world-model training (`decoder.py`, enabled with
`--decoder-train-steps`). It is an MLP probe from frozen latents back to
observations, optimized separately so no reconstruction gradient ever reaches
the encoder or dynamics. It is used only to render open-loop imagined rollouts
next to real held-out trajectories (`decoder_rollout_frames.png`,
`decoder_rollout_traces.png`); it plays no role in training or the pass/fail
gate.

Decoder batches mix the long replay 50/50 with the random-policy anchor
replay (via `sample_online_candidate_batch`, the same helper the online
refits use). After long online runs the ring buffer has evicted the offline
random data, but the rollout diagnostic still displays random-policy
validation windows — without the anchor mix the probe trains only on the
final actor distribution and reconstructs the display windows poorly, which
reads as (nonexistent) dynamics error.

## Control-Relevant Online Loss

Online refits keep the observation encoder frozen. The update changes the
action encoder, transformer, latent predictor, reward head, and continuation
head, while preserving the latent coordinate system consumed by the actor and
critic.

The optional control-value consistency loss makes the dynamics more useful for
policy improvement. The critic is used as a frozen teacher:

\[
\hat Q(z_t, a_t)
=
\hat r_t
+ \gamma \hat c_t V_\psi(\hat z_{t+1})
\]

\[
Q_{target}
=
r_t + \gamma c_t \mathrm{stopgrad}(V_\psi(E_\theta(o_{t+1})))
\]

\[
L_{control\_value}
=
\frac{1}{2}
\left(\hat Q(z_t,a_t) - Q_{target}\right)^2
\]

The value head is not updated by this loss. Gradients flow through the
transition, reward, and continuation predictions. This keeps the model
decoder-free while asking it to preserve information that matters for control.

## Offline Workflow

The basic single-agent workflow is:

1. collect random replay;
2. train the JEPA world model;
3. freeze the world model;
4. reset actor and critic heads;
5. warm the critic on replayed real-return targets;
6. train the actor and critic through imagined latent rollouts;
7. evaluate the actor in the real environment.

The actor objective backpropagates through latent imagination. The world model
parameters remain frozen during actor and critic training.

## Online Workflow

The online loop extends the offline workflow:

1. collect replay using the current actor;
2. hold out a recent-policy validation stream;
3. train a candidate world-model refit with the encoder frozen, using minibatches
   mixed from initial random anchor replay and the latest actor replay;
4. evaluate candidate checkpoints during refit;
5. keep the best checkpoint that improves recent-policy validation while keeping
   anchor validation within tolerance;
6. continue actor/critic training in the accepted world model.

Real data is retained even when a candidate world-model update is rejected.
The long replay keeps all collected data, while candidate refits use an explicit
anchor/recent sampling ratio so new actor data cannot silently overwrite the
random-replay coverage.

Let:

- \(B_A\) be the initial random anchor replay;
- \(B_{all}\) be the full replay containing all retained experience;
- \(B_R^{(i)}\) be the actor replay collected in online iteration \(i\);
- \(B_{val,A}\) be anchor validation replay;
- \(B_{val,R}^{(i)}\) be held-out recent-policy validation replay.

For candidate refits, each training minibatch is sampled as:

\[
B_{train}^{(i)}
= \rho B_A + (1-\rho)B_R^{(i)}
\]

where \(\rho\) is `online_anchor_batch_fraction` and defaults to \(0.5\).
The full replay \(B_{all}\) is still retained; the explicit mixture only
controls the candidate-refit training distribution.

Candidate checkpoints are evaluated during refit. For a candidate checkpoint
\(m\), define:

\[
\Delta_R = L_{val,R}^{old} - L_{val,R}^{m}
\]

\[
\Delta_A = L_{val,A}^{m} - L_{val,A}^{old}
\]

where lower validation loss is better. A checkpoint must improve recent-policy
validation and keep anchor degradation below tolerance:

\[
\Delta_R \ge \epsilon_R
\]

\[
\Delta_A \le \epsilon_A
\]

Among passing checkpoints, the selected candidate maximizes:

\[
S_m = \Delta_R - \alpha \max(\Delta_A, 0)
\]

where \(\alpha\) is `online_candidate_anchor_penalty`.

## Controls

The main comparisons are:

- `none`: normal action-conditioned world model;
- `no-action-world-model`: the world model receives zero actions;
- `shuffled-action-replay`: replay actions are shuffled before training;
- `frozen-random-world-model`: policy training uses an untrained world model.

These controls check whether policy improvement comes from action-conditioned
latent dynamics rather than actor drift or evaluation noise.

Controls are useful for confirmation runs, but they are not part of the fast
tuning loop. During tuning, the main comparison is the `none` agent across
architectural and training-cadence variants.

## Current Mainline

The current mainline is:

- single encoder;
- SIGReg regularization;
- stop-gradient JEPA targets by default;
- causal transformer dynamics with RoPE attention and GEGLU feed-forward blocks;
- direct latent-imagination actor training;
- optional symlog two-hot reward/value heads for Dreamer/STORM-style scale
  stabilization;
- optional batch-normalized imagined returns or value-baseline advantages for
  actor stabilization;
- frozen encoder during online world-model refits;
- explicit anchor/recent replay mixing during online refits;
- candidate refit gates on anchor and recent-policy validation;
- optional control-value consistency loss during online refits;
- conservative online actor updates with an action-change trust penalty;
- champion actor selection across online phases, so an online policy update is
  accepted only if real-environment evaluation does not regress beyond the
  configured tolerance.

When the online phase runs, the reported `policy_primary_improvement` is the
cumulative gain of the final champion over the pre-online offline policy
(`policy_online_total_improvement_vs_pre_online`), not the final cycle's
within-cycle delta — a converged run whose last cycles add nothing still
passes on its accumulated improvement (`policy_primary_improvement_key`
records which definition was used).

The current working configuration for vector DMC/Brax Reacher-style tasks is the
small-batch online-cadence setting:

- 16 parallel environments;
- 64 world-model sequences per update;
- 32 steps per sequence;
- 512 policy/imagination start states per actor update;
- 128 latent dimensions;
- 128 transformer hidden dimensions;
- 2 transformer layers;
- 4 attention heads;
- context window 4;
- model horizon 5;
- imagination horizon 15;
- 5 dynamics ensemble heads;
- 6 online actor-collection/refit/policy-improvement cycles for the mainline
  run, or 12 smaller `2048`-step online cycles for the DMC Reacher stability
  sweep;
- optional final champion evaluation with many episodes for reporting only.

This differs from the earlier high-throughput configuration with 512 parallel
environments and large world-model batches. The small-batch setup keeps the real
sample count closer to Dreamer-style control benchmarks and gives the policy
more frequent opportunities to steer the data distribution.

## Real Step Accounting

The training script reports real environment usage separately for:

- training replay;
- held-out validation replay;
- policy selection, evaluation, and confirmation episodes;
- the strict total across all real interactions.

For sample-efficiency comparisons, the most optimistic number is training replay
only. The strict number includes validation and policy evaluation interactions.
Imagined transitions, world-model updates, and actor/critic optimizer updates
are not counted as real environment steps.
