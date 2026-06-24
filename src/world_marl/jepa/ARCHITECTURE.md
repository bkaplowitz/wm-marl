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

Sinusoidal position embeddings are added:

\[
h_t^{(0)} = x_t + p_t
\]

The transformer uses causal attention. A token can attend only to previous
tokens in the configured context window. Attention across episode boundaries is
masked.

Each transformer block is pre-norm:

\[
h' = h + \mathrm{SelfAttention}(\mathrm{LN}(h))
\]

\[
h^{next} = h' + \mathrm{MLP}(\mathrm{LN}(h'))
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
3. train a candidate world-model refit with the encoder frozen;
4. accept the candidate only if recent-policy validation improves and anchor
   validation does not degrade beyond tolerance;
5. continue actor/critic training in the accepted world model.

Real data is retained even when a candidate world-model update is rejected.

## Controls

The main comparisons are:

- `none`: normal action-conditioned world model;
- `no-action-world-model`: the world model receives zero actions;
- `shuffled-action-replay`: replay actions are shuffled before training;
- `frozen-random-world-model`: policy training uses an untrained world model.

These controls check whether policy improvement comes from action-conditioned
latent dynamics rather than actor drift or evaluation noise.

## Current Mainline

The current mainline is:

- single encoder;
- SIGReg regularization;
- stop-gradient JEPA targets by default;
- causal transformer dynamics;
- direct latent-imagination actor training;
- frozen encoder during online world-model refits;
- candidate refit gates on anchor and recent-policy validation;
- optional control-value consistency loss during online refits.
