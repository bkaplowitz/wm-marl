# JEPA World-Model Architecture

This note describes the current single-agent JEPA model-based RL architecture and
the reasoning behind it. It is meant to keep the research object clear as the
code evolves.

## Goal

The goal is to learn a world model that supports policy improvement in latent
space.

The model should learn:

\[
p(z_{t+1}, r_t, c_t \mid z_t, a_t)
\]

where:

- \(o_t\) is the real environment observation.
- \(a_t\) is the continuous action.
- \(z_t\) is a learned latent representation of \(o_t\).
- \(r_t\) is predicted reward.
- \(c_t\) is predicted continuation probability, meaning not terminated.

The policy is then improved by rolling forward imagined latent trajectories,
not by reconstructing observations or pixels.

## Current Model

The current model has four conceptual parts:

1. Observation encoder
2. Action-conditioned latent dynamics
3. Reward and continue prediction heads
4. Actor and critic operating on latents

The observation encoder maps real observations into a latent state:

\[
z_t^w = E_\theta(o_t)
\]

The superscript \(w\) means "world latent." This is the latent representation
used by the world model itself.

The action-conditioned world model predicts the next latent, reward, and
continuation:

\[
\hat{z}_{t+1}^w, \hat{r}_t, \hat{c}_t =
M_\theta(z_t^w, a_t)
\]

The JEPA prediction objective compares the predicted next latent to the encoded
future observation:

\[
z_{t+1}^{target} = \mathrm{stopgrad}(E_\theta(o_{t+1}))
\]

\[
L_{JEPA} = 1 - \cos(\hat{z}_{t+1}^w, z_{t+1}^{target})
\]

The model also predicts reward and continuation:

\[
L_{model}
= L_{JEPA}
+ \lambda_r L_r
+ \lambda_c L_c
+ \lambda_{reg} L_{SIGReg}
\]

SIGReg is used to keep the latent representation from collapsing.

## What Makes This JEPA-Like

The model predicts future representations, not observations.

There is no decoder trained to reconstruct the raw observation:

\[
\hat{o}_{t+1} \notin \text{training objective}
\]

The central prediction problem is representation-space prediction:

\[
z_t, a_t \rightarrow z_{t+1}
\]

This is why the learned latent space has to be good for both prediction and
control.

## Policy Learning

The actor and critic operate in latent space.

The actor maps a latent to an action:

\[
a_t = \pi_\phi(z_t^c)
\]

The critic maps a latent to a value estimate:

\[
V_\psi(z_t^c)
\]

The superscript \(c\) means "control latent." This is the policy-facing latent
representation consumed by the actor and critic.

Policy improvement uses imagined latent rollouts:

\[
z_t^w
\xrightarrow{\pi}
a_t
\xrightarrow{M}
\hat{z}_{t+1}^w, \hat{r}_t, \hat{c}_t
\]

The actor is optimized to increase predicted imagined return:

\[
G_t = \sum_{k=0}^{H-1}
\gamma^k
\left(\prod_{j=0}^{k-1} \hat{c}_{t+j}\right)
\hat{r}_{t+k}
\]

The current main path is direct latent-imagination actor optimization. Candidate
distillation exists only as a diagnostic/planning baseline, not the main
algorithmic claim.

## The Online Problem

The offline setting is:

1. Collect random replay.
2. Train the world model.
3. Freeze the world model.
4. Train actor and critic inside imagined latent rollouts.
5. Evaluate the actor in the real environment.

This already tests whether the learned latent model can support policy
improvement.

The online setting is harder:

1. Use the improved actor to collect new real replay.
2. Refit or update the world model on the expanded replay.
3. Continue policy improvement inside the updated world model.

Three problems appear online:

1. Coverage shift: the actor visits states not well covered by random replay.
2. Model exploitation: the actor can exploit errors in the learned model.
3. Latent-coordinate drift: the world model can change its latent coordinate
   system while the actor and critic still expect the old one.

The third problem is especially important for JEPA-style models.

## Latent Gauge Drift

The JEPA objective is largely invariant to orthogonal rotations of the latent
space.

Suppose the encoder and predictor both rotate their latents by an orthogonal
matrix \(R\):

\[
E'_\theta(o) = E_\theta(o)R
\]

\[
M'_\theta(zR, a) = M_\theta(z, a)R
\]

Cosine prediction can remain almost unchanged:

\[
\cos(\hat{z}R, zR) = \cos(\hat{z}, z)
\]

SIGReg is also compatible with such rotations because an isotropic latent
distribution does not prefer one orthogonal basis over another.

The world model may therefore remain good while the policy input changes:

\[
\pi_\phi(zR) \neq \pi_\phi(z)
\]

This is the latent gauge problem: the representation can change coordinates
without changing the world-model loss, but the actor and critic are not
coordinate-invariant.

## World Latents And Control Latents

To separate prediction from control, we now distinguish two spaces:

\[
z_t^w = E_\theta(o_t)
\]

\[
z_t^c = z_t^w Q
\]

where:

- \(z_t^w\) is the raw world latent used by dynamics.
- \(z_t^c\) is the policy-facing control latent used by actor and critic.
- \(Q\) is an orthogonal alignment matrix.

The world model uses:

\[
M_\theta(z_t^w, a_t)
\]

The actor and critic use:

\[
\pi_\phi(z_t^c), \quad V_\psi(z_t^c)
\]

When \(Q = I\), control latents are just raw world latents. This is the old
behavior.

## Orthogonal Control-Latent Alignment

After an online world-model refit, the encoder may change from \(E_{old}\) to
\(E_{new}\). The actor and critic were trained on the old control latents:

\[
z_{old}^c = E_{old}(o)Q_{old}
\]

We choose a new alignment \(Q_{new}\) so that the new encoder produces control
latents close to the old policy-facing latents on a fixed anchor set:

\[
Q_{new}
=
\arg\min_Q
\left\|
E_{new}(O_A)Q - E_{old}(O_A)Q_{old}
\right\|_F^2
\]

subject to:

\[
Q^\top Q = I
\]

This is the orthogonal Procrustes problem.

Let:

\[
A = E_{new}(O_A)
\]

\[
B = E_{old}(O_A)Q_{old}
\]

Compute:

\[
A^\top B = U S V^\top
\]

Then:

\[
Q_{new} = U V^\top
\]

The updated policy input becomes:

\[
z_{new}^c = E_{new}(o)Q_{new}
\]

while the dynamics still use:

\[
z_{new}^w = E_{new}(o)
\]

This lets the world model improve its internal representation while the actor
and critic keep receiving approximately stable coordinates.

## What Alignment Can And Cannot Fix

Orthogonal alignment can fix harmless rotations or reflections of the latent
space.

It cannot fix:

- nonlinear deformation of the representation,
- loss of task-relevant information,
- model errors in newly visited regions,
- actor exploitation of inaccurate imagined rollouts.

That is why we track:

\[
\text{raw latent drift}
\]

\[
\text{control latent drift}
\]

\[
\text{Procrustes residual}
\]

\[
\text{policy action drift}
\]

\[
\text{value drift}
\]

If raw latent drift is high but control latent drift is low, alignment is doing
its job.

If both raw and control latent drift are high, the representation changed too
much for a single orthogonal map to repair.

## Current Online Mainline

The current online mainline is a two-phase training scheme:

1. Initial world-model fit:

   \[
   E_\theta, M_\theta, R_\theta, C_\theta
   \quad \text{are all trained.}
   \]

2. Online world-model refits:

   \[
   E_\theta
   \quad \text{is frozen.}
   \]

   \[
   M_\theta, R_\theta, C_\theta
   \quad \text{continue adapting.}
   \]

Here \(R_\theta\) is the reward head and \(C_\theta\) is the continue head.

This is not intended as the final architecture. It is the current stable
backbone because the first online drift experiment showed that updating the
encoder during online refits can damage the policy-facing latent space.

The observed pattern was:

- No stabilization: large raw/control latent drift, large action drift, and a
  large immediate post-refit policy drop.
- Cosine anchoring: reduced drift, but did not remove the post-refit policy
  drop.
- Procrustes alignment: reduced control drift, action drift, and value drift,
  but did not remove the post-refit policy drop.
- Encoder freezing: removed latent/interface drift and avoided the immediate
  post-refit policy drop.

So the current practical rule is:

\[
\text{initial fit learns the representation;}
\quad
\text{online refits preserve the representation.}
\]

This keeps the actor/critic input space stable while the dynamics and scalar
prediction heads adapt to newly collected policy data.

## Stabilization Ablations

The stabilization ablations remain useful diagnostics:

1. No alignment

   \[
   Q = I
   \]

2. Encoder frozen during online refit

   \[
   E_{new} = E_{old}
   \]

3. Cosine latent anchoring

   \[
   L_{anchor}
   =
   1 - \cos(E_{new}(o), \mathrm{stopgrad}(E_{old}(o)Q_{old}))
   \]

4. Procrustes control-latent alignment

   \[
   z_{new}^c = E_{new}(o)Q_{new}
   \]

The purpose of these ablations is to determine how much online degradation is
caused by policy-facing latent drift versus deeper model or coverage problems.

## Candidate World-Model Updates

Online world-model updates are treated as proposals.

The active agent should not be overwritten simply because a new model finished
training.

A candidate model is evaluated on:

1. Anchor validation replay
2. Recent-policy validation replay

The intended acceptance logic is:

\[
L_{recent}^{new} < L_{recent}^{old} - \delta
\]

and:

\[
L_{anchor}^{new} \leq L_{anchor}^{old} + \epsilon
\]

This asks for plasticity on recent policy data without unacceptable forgetting
on the anchor distribution.

## Current Claim

The current system supports this claim:

> A JEPA-style latent world model can be trained from real replay and then used
> to improve a continuous-control actor through imagined latent rollouts.

The stronger online claim is still under construction:

> Online JEPA model-based RL can keep improving when the actor collects new data,
> if world-model updates preserve a stable control latent and are accepted only
> when they improve recent-policy prediction without excessive forgetting.

The current online result supports a narrower intermediate claim:

> Freezing the encoder during online refits gives a stable control interface for
> continued latent-imagination actor training.

## Next Architecture Step

The next major step is uncertainty-gated imagination.

The planned model will use an ensemble of world-model heads:

\[
M^{(i)}_\theta(z_t^w, a_t)
\rightarrow
\hat{z}_{t+1}^{w,i}, \hat{r}_t^{i}, \hat{c}_t^{i}
\]

The ensemble disagreement estimates model uncertainty:

\[
u_z
=
1 -
\left\|
\frac{1}{K}
\sum_{i=1}^K
\frac{\hat{z}_{t+1}^{w,i}}
{\|\hat{z}_{t+1}^{w,i}\|}
\right\|_2^2
\]

Reward and continue uncertainty can be added:

\[
u_t = \alpha_z u_z + \alpha_r u_r + \alpha_c u_c
\]

The actor should optimize conservative imagined reward:

\[
\tilde{r}_t = \bar{r}_t - \lambda_u u_t
\]

and imagination should shorten or stop when uncertainty is too high:

\[
u_t > \tau
\quad \text{or} \quad
\sum_{j=0}^{t} u_j > B
\]

This should reduce model exploitation by discouraging the actor from optimizing
trajectories where the learned model is unsupported by real data.

## Bigger Picture

The intended research direction is not standard Dreamer reproduction.

The architecture is moving toward:

- JEPA-style representation prediction instead of observation reconstruction,
- latent-space policy improvement,
- stable policy-facing control latents across online model updates,
- candidate-gated world-model refits,
- uncertainty-gated imagined rollouts,
- eventual extension from single-agent control to CTDE-style MARL.

For MARL, the natural extension is:

\[
z_t^{joint} = E(o_t^1, \ldots, o_t^N)
\]

\[
p(z_{t+1}^{joint}, r_t, c_t
\mid
z_t^{joint}, a_t^1, \ldots, a_t^N)
\]

with decentralized actors consuming agent-specific policy-facing latents and a
centralized world model trained over joint dynamics.
