# JEPA Architecture

This note describes the current single-agent JEPA world-model architecture in
this repository. It is a reference for what the code does today, not a proposal
for future work.

## Objective

The model learns latent dynamics from real environment transitions:

\[
p(z_{t+1}, r_t, c_t \mid z_t, a_t)
\]

where:

- \(o_t\) is the environment observation.
- \(a_t\) is a continuous action.
- \(z_t = E_\theta(o_t)\) is the learned latent state.
- \(r_t\) is reward.
- \(c_t\) is continuation probability, or the probability that the episode has
  not terminated.

The model predicts the next latent representation, reward, and continuation. It
does not reconstruct observations or pixels.

## Model Components

The current model has five parts:

1. Observation encoder
2. Action encoder
3. Causal latent dynamics transformer
4. Prediction heads for next latent, reward, and continuation
5. Actor and critic heads

The observation encoder maps observations into latent states:

\[
z_t = E_\theta(o_t)
\]

The dynamics model consumes a short history of latent states and actions:

\[
(z_{t-k:t}, a_{t-k:t})
\]

The history is passed through a causal transformer with sinusoidal position
embeddings. The transformer output is used to predict:

\[
\hat{z}_{t+1}, \hat{r}_t, \hat{c}_t
= M_\theta(z_{t-k:t}, a_{t-k:t})
\]

The latent transition is residual by default:

\[
\hat{z}_{t+1} = \mathrm{norm}(z_t + \Delta_\theta)
\]

The actor and critic operate on latents:

\[
a_t = \pi_\phi(z_t^c)
\]

\[
V_t = V_\psi(z_t^c)
\]

Here \(z_t^c\) is the policy-facing control latent. In the default case,
\(z_t^c = z_t\). During alignment experiments, \(z_t^c = z_t Q\).

## Transformer Dynamics

The dynamics model is a causal transformer over latent/action history tokens.
For each timestep, the model builds one token by adding a projected latent and
an encoded action:

\[
x_t = W_z z_t + A_\theta(a_t)
\]

where \(A_\theta\) is either an action embedding for discrete actions or a small
MLP for continuous actions.

Sinusoidal position embeddings are added to these tokens:

\[
h_t^{(0)} = x_t + p_t
\]

The transformer uses a causal local attention mask. A timestep can only attend
to previous timesteps inside the configured context window, and attention is
masked across episode boundaries.

Each transformer block is pre-norm:

\[
h' = h + \mathrm{SelfAttention}(\mathrm{LN}(h))
\]

\[
h^{next} = h' + \mathrm{MLP}(\mathrm{LN}(h'))
\]

The MLP expands to `mlp_ratio * model_dim`, uses GELU, then projects back to
`model_dim`.

After the transformer stack, a final layer norm produces the dynamics hidden
state. The last hidden state is used by:

- the latent predictor;
- the reward head;
- the continuation head.

For multi-step prediction, the model recursively appends predicted latents back
into the context and consumes the corresponding future actions from replay.

## Training Loss

The JEPA target is the encoded next observation:

\[
z_{t+1}^{target} = E_\theta(o_{t+1})
\]

By default, gradients are stopped through the target branch:

\[
z_{t+1}^{target} = \mathrm{stopgrad}(E_\theta(o_{t+1}))
\]

The latent prediction loss is cosine distance:

\[
L_{JEPA} = 1 - \cos(\hat{z}_{t+1}, z_{t+1}^{target})
\]

Reward uses mean squared error. Continuation uses binary cross entropy.

The full world-model loss is:

\[
L =
L_{JEPA}
+ \lambda_r L_r
+ \lambda_c L_c
+ \lambda_{reg} L_{SIGReg}
\]

SIGReg is used to reduce representation collapse. It regularizes the latent
distribution without adding a decoder.

Episode boundaries are masked so prediction losses do not train across resets.

## Offline Policy Learning

The offline workflow is:

1. Collect replay from random actions.
2. Train the JEPA world model on this replay.
3. Reset actor and critic heads.
4. Freeze the world model.
5. Train the actor and critic through imagined latent rollouts.
6. Evaluate the actor in the real environment.

An imagined rollout starts from replay observations:

\[
o_t \rightarrow z_t
\]

Then the actor and world model are rolled forward in latent space:

\[
z_t
\xrightarrow{\pi_\phi}
a_t
\xrightarrow{M_\theta}
\hat{z}_{t+1}, \hat{r}_t, \hat{c}_t
\]

The direct actor objective maximizes predicted imagined return:

\[
G_t =
\sum_{k=0}^{H-1}
\gamma^k
\left(\prod_{j=0}^{k-1} \hat{c}_{t+j}\right)
\hat{r}_{t+k}
\]

Actor and critic updates are separated. The actor is trained through imagined
returns. The critic is trained on stopped imagined latents, so critic fitting
does not backpropagate through the actor via the imagined state path.

`candidate-distill` still exists as a diagnostic baseline. It samples candidate
actions in the model, selects high-scoring actions, and trains the actor toward
them. It is not the main policy-learning path.

## Online Updates

The online workflow extends the offline setup:

1. Use the current actor to collect new real replay.
2. Train a candidate world-model update on the replay buffer.
3. Evaluate the candidate on both anchor validation data and recent-policy
   validation data.
4. Accept or reject the candidate update.
5. Continue actor and critic training in the accepted model.

The current stable default freezes the observation encoder during online
world-model updates:

\[
E_\theta \text{ is fixed}
\]

The dynamics transformer, action encoder, predictor, reward head, and continue
head can still adapt:

\[
M_\theta, R_\theta, C_\theta \text{ are updated}
\]

This keeps the actor and critic input space stable while allowing the dynamics
and scalar prediction heads to adapt to new policy data.

## Candidate Update Gate

Online world-model updates are treated as candidate updates. A candidate is
checked on two validation distributions:

- anchor validation replay, usually from the original broader replay
  distribution;
- recent-policy validation replay, collected from the current actor but held out
  from the online training replay.

The candidate is accepted only if it improves recent-policy validation and does
not degrade anchor validation beyond a tolerance:

\[
L_{recent}^{new} < L_{recent}^{old} - \delta
\]

\[
L_{anchor}^{new} \le L_{anchor}^{old} + \epsilon
\]

If the gate rejects the candidate, the active world model remains unchanged.
The collected real data is still retained in replay.

## Control Latents

The code separates raw world latents from policy-facing control latents:

\[
z_t^w = E_\theta(o_t)
\]

\[
z_t^c = z_t^w Q
\]

The world model uses \(z_t^w\). The actor and critic use \(z_t^c\).

With \(Q = I\), control latents and world latents are identical. This is the
normal path when the encoder is frozen online.

The Procrustes alignment path is an ablation for online encoder updates. It
chooses an orthogonal matrix \(Q\) that maps new encoder latents close to the old
policy-facing latents on an anchor set:

\[
Q =
\arg\min_{Q^\top Q = I}
\| E_{new}(O_A)Q - E_{old}(O_A)Q_{old} \|_F^2
\]

This can correct simple rotations or reflections of the latent basis. It cannot
fix nonlinear representation changes, forgotten information, or model errors.
Recent experiments showed that freezing the encoder is currently the more stable
online path.

## Diagnostics

The main diagnostics are:

- JEPA prediction loss and open-loop latent prediction loss.
- Reward and continuation losses against constant baselines.
- SIGReg and collapse metrics: latent standard deviation, effective rank,
  covariance off-diagonal norm, and latent norms.
- Action-contrast metrics, which check whether the model actually uses actions.
- Policy return before and after imagined actor training.
- Online interface checks: raw latent drift, control latent drift, action drift,
  value drift, and policy return before and after world-model refits.
- Candidate-gate metrics on anchor and recent-policy validation data.

The standard controls are:

- `none`: normal action-conditioned world model.
- `no-action-world-model`: replaces model actions with zeros.
- `shuffled-action-replay`: breaks action-transition alignment in replay.
- `frozen-random-world-model`: skips world-model training.

The controls are used to check that policy improvement comes from learned
action-conditioned dynamics rather than evaluation noise or actor drift.

## Current Status

The current mainline is:

1. Train an action-conditioned JEPA world model.
2. Train a continuous actor and critic through latent imagination.
3. Use online actor replay for further data collection.
4. Keep the encoder frozen during online refits.
5. Gate candidate world-model updates before accepting them.

The implemented architecture is still single-agent and vector-observation based.
It does not yet include:

- image encoders or ViTs;
- ensemble uncertainty;
- uncertainty-penalized imagined rewards;
- adaptive imagination horizons;
- multi-agent centralized training / decentralized execution.

Those are future extensions, not part of the current implemented architecture.
