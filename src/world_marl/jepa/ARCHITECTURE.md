# JEPA Architecture

This note describes the current single-agent JEPA world-model architecture in
this repository. It is a snapshot of what the implementation does today.

## Goal

The model learns action-conditioned latent dynamics:

\[
p(z_{t+1}, r_t, c_t \mid z_t, a_t)
\]

where:

- \(o_t\) is an environment observation.
- \(a_t\) is a continuous action.
- \(z_t\) is a learned latent state.
- \(r_t\) is reward.
- \(c_t\) is continuation probability.

The model predicts future representations, rewards, and continuation. It does
not reconstruct observations or pixels.

## Latent Spaces

The architecture separates two latent spaces:

\[
z_t^w = E_\theta(o_t)
\]

\[
z_t^c = A(z_t^w)
\]

\(z^w\) is the raw world-model latent. The dynamics model learns and predicts in
this space.

\(z^c\) is the policy-facing control latent. The actor and critic consume this
space.

The control interface is currently affine:

\[
z_t^c = s(z_t^w R) + b
\]

where:

- \(R\) is a fitted rotation/reflection matrix;
- \(s\) is a scalar scale;
- \(b\) is a shift.

There are two active interface modes:

- identity: \(R = I,\ s = 1,\ b = 0\);
- Umeyama: fit \(R\), \(s\), and \(b\).

Procrustes alignment, which fits only \(R\), remains useful as a rigid diagnostic
baseline. It is not the preferred adaptive-encoder interface because current
experiments show that scale and shift matter.

The world model uses \(z^w\). The actor and critic use \(z^c\).

When an online refit updates the encoder, the interface can be fitted on anchor
observations \(O_A\). Umeyama solves the similarity-alignment problem:

\[
s, R, b =
\arg\min_{s,R,b}
\| s(E_{new}(O_A)R) + b - A_{old}(E_{old}(O_A)) \|_F^2
\]

The fitted interface is a policy-side adapter. It does not change the raw latent
dynamics model.

## Components

The current model has:

1. observation encoder;
2. action encoder;
3. causal latent dynamics transformer;
4. latent predictor;
5. reward head;
6. continuation head;
7. actor head;
8. critic head.

The observation encoder maps observations into raw world latents:

\[
z_t^w = E_\theta(o_t)
\]

The actor and critic read control latents:

\[
a_t = \pi_\phi(z_t^c)
\]

\[
V_t = V_\psi(z_t^c)
\]

The dynamics model predicts from raw latent/action history:

\[
\hat{z}_{t+1}^w,\ \hat{r}_t,\ \hat{c}_t
= M_\theta(z_{t-k:t}^w, a_{t-k:t})
\]

## Transformer Dynamics

The dynamics model is a causal transformer over latent/action history.

For each timestep, it forms a token from a projected latent and an encoded
action:

\[
x_t = W_z z_t^w + A_\theta(a_t)
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
\hat{z}_{t+1}^w = \mathrm{norm}(z_t^w + \Delta_\theta)
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
= 1 - \cos(\hat{z}_{t+1}^w, z_{t+1}^{target})
\]

Reward uses mean squared error. Continuation uses binary cross entropy.

The world-model loss is:

\[
L =
L_{JEPA}
+ \lambda_r L_r
+ \lambda_c L_c
+ \lambda_{reg} L_{SIGReg}
\]

SIGReg is used as the anti-collapse regularizer. It regularizes the latent
distribution without adding an observation decoder.

Episode boundaries are masked so the model is not trained to predict through
environment resets.

## Control-Interface Losses

During online refits, the encoder may change its raw latent coordinates. This
can break the actor and critic even if the world-model loss remains good.

The control-interface anchor penalizes movement in the policy-facing space:

\[
L_{anchor}
= 1 -
\cos(
A_{new}(E_{new}(o)),
\mathrm{stopgrad}(A_{old}(E_{old}(o)))
)
\]

This is different from anchoring raw latents. The anchor is about preserving the
controller's input coordinates, not freezing the world model's internal
representation.

Online candidate refits can also include a one-step control-coordinate
prediction loss:

\[
L_{control\_pred}
=
1 -
\cos(
A_{old}(\hat{z}_{t+1,new}^w),
\mathrm{stopgrad}(A_{old}(E_{old}(o_{t+1})))
)
\]

This trains the candidate to predict futures in the previous accepted
policy-facing coordinate system. This loss is applied during candidate training,
before any post-hoc Umeyama interface is fitted. The candidate is therefore not
trained only in its own self-defined latent coordinates.

## Offline Policy Learning

The offline workflow is:

1. collect random replay;
2. train the JEPA world model;
3. reset actor and critic heads;
4. freeze the world model;
5. train actor and critic through imagined latent rollouts;
6. evaluate the actor in the real environment.

An imagined rollout starts from real observations:

\[
o_t \rightarrow z_t^w \rightarrow z_t^c
\]

The actor chooses an action from \(z_t^c\):

\[
a_t = \pi_\phi(z_t^c)
\]

The world model advances the raw latent:

\[
\hat{z}_{t+1}^w,\ \hat{r}_t,\ \hat{c}_t
= M_\theta(z_t^w, a_t)
\]

The next actor input is:

\[
\hat{z}_{t+1}^c = A(\hat{z}_{t+1}^w)
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

## Online Updates

The online workflow extends the offline setup:

1. collect real replay with the current actor;
2. train a candidate world-model update;
3. optionally fit a control interface for the candidate;
4. evaluate the candidate on anchor and recent-policy validation data;
5. accept or reject the candidate;
6. continue actor and critic training in the accepted model.

The stable baseline freezes the observation encoder during online world-model
refits. The dynamics transformer, action encoder, predictor, reward head, and
continuation head still adapt.

Adaptive-encoder experiments unfreeze the encoder and use a fitted control
interface to preserve the actor/critic coordinate system.

## Candidate Update Gate

Candidate world-model updates are checked on two validation distributions:

- anchor validation replay from the broader historical distribution;
- recent-policy validation replay collected from the current actor and held out
  from training.

The standard gate accepts a candidate only if recent-policy validation improves
and anchor validation does not degrade beyond a tolerance:

\[
L_{recent}^{new} < L_{recent}^{old} - \delta
\]

\[
L_{anchor}^{new} \le L_{anchor}^{old} + \epsilon
\]

The gate can also use a fixed-coordinate prediction loss. This gate metric is
computed after any candidate interface has been fitted. It compares the
candidate's prediction in the candidate control space against the old accepted
target in the old accepted control space:

\[
L_{control\_pred}
=
1 -
\cos(
A_{new}(\hat{z}_{t+1,new}^w),
\mathrm{stopgrad}(A_{old}(E_{old}(o_{t+1})))
)
\]

This asks whether the new model predicts futures that remain meaningful to the
existing controller, not only whether it predicts well in its own newly chosen
latent coordinates.

If the gate rejects the candidate, the active world model is left unchanged. The
new real data remains in replay.

## Conservative Imagination

The model can optionally use an ensemble of prediction heads:

\[
M_\theta^{(i)}(z_t^w, a_t)
\rightarrow
\hat{z}_{t+1}^{w,i}, \hat{r}_t^{(i)}, \hat{c}_t^{(i)}
\]

The heads share the encoder, action encoder, and transformer trunk. Each head has
its own latent predictor, reward head, and continuation head.

During imagination, the rollout uses the ensemble mean transition:

\[
\bar{z}_{t+1}^w = \frac{1}{K}\sum_i \hat{z}_{t+1}^{w,i}
\]

\[
\bar{r}_t = \frac{1}{K}\sum_i \hat{r}_t^{(i)}
\]

\[
\bar{c}_t = \frac{1}{K}\sum_i \sigma(\hat{c}_t^{(i)})
\]

The ensemble also produces an uncertainty score. The latent part is spherical
disagreement:

\[
u_z =
1 -
\left\|
\frac{1}{K}
\sum_i
\frac{\hat{z}_{t+1}^{w,i}}{\|\hat{z}_{t+1}^{w,i}\|}
\right\|_2^2
\]

Reward and continuation variance can also contribute:

\[
u_t = \alpha_z u_z + \alpha_r u_r + \alpha_c u_c
\]

When enabled, the actor optimizes conservative imagined reward:

\[
\tilde{r}_t = \bar{r}_t - \lambda_u u_t
\]

The rollout can stop trusting a trajectory when transition uncertainty or
cumulative uncertainty crosses configured thresholds.

By default the ensemble size is one and the uncertainty penalty is zero. In that
case this path reduces to single-model latent imagination.

## Diagnostics

The main diagnostics are:

- JEPA prediction loss;
- open-loop latent prediction loss;
- fixed-coordinate control prediction loss;
- train-time control prediction loss during candidate refits;
- reward and continuation losses;
- SIGReg and collapse metrics;
- action-contrast metrics;
- real policy return before and after imagined actor training;
- raw latent drift after online refits;
- control latent drift after online refits;
- actor action drift after online refits;
- value drift after online refits;
- candidate acceptance rate on anchor/recent validation.

The standard controls are:

- normal action-conditioned world model;
- no-action world model;
- shuffled-action replay;
- frozen-random world model.

These controls check whether policy improvement comes from learned
action-conditioned dynamics rather than actor drift, evaluation noise, or model
shortcuts.

## Current Status

The current mainline supports:

1. action-conditioned JEPA world-model training;
2. direct actor/critic training through latent imagination;
3. online actor replay;
4. candidate world-model refits;
5. frozen-encoder online refits as the stable baseline;
6. adaptive-encoder experiments with Umeyama control-interface alignment;
7. control-interface anchor and control-coordinate prediction losses;
8. control-coordinate candidate gates;
9. optional ensemble disagreement for conservative imagination.

The implementation is still single-agent and vector-observation based. It does
not yet include image encoders, ViTs, multi-agent CTDE, or a large benchmark
sweep.
