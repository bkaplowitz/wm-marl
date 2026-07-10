# Genie2 Continuous JAX Architecture

Genie 2 is the primary architecture target for this branch:
[Genie 2: A large-scale foundation world model](https://deepmind.google/blog/genie-2-a-large-scale-foundation-world-model/).

This is not a faithful public Genie 1 clone. Public Genie 1 and its
VQ/MaskGIT-style tokenizer-dynamics stack are kept as an ablation target only:
[Genie: Generative Interactive Environments](https://arxiv.org/abs/2402.15391).

Jasmine and Jafar remain implementation references, not the model name or
primary specification. When implementation details are borrowed, cite the
relevant source in the module or experiment metadata:

- [Jasmine paper](https://arxiv.org/abs/2510.27002)
- [Jasmine repository](https://github.com/p-doom/jasmine)
- [Jafar repository](https://github.com/FLAIROx/jafar)

## Public Source Boundary

The public Genie 2 disclosure states that Genie 2 is an autoregressive latent diffusion world model.
Video frames pass through an autoencoder, latent frames are processed by a large
transformer dynamics model with a causal mask, inference samples frame by frame
from past latent frames plus user actions, and classifier-free guidance improves
action controllability.

Genie 3 is a capability target, not a complete public architecture. Public
material describes real-time 720p, 24fps interaction, stronger long-horizon
consistency, frame-by-frame world generation conditioned on world description and
user actions, and promptable world events. It does not disclose enough detail to
implement an exact clone. This package should therefore implement a transparent
Genie-2-style continuous latent model and track Genie-3-style capabilities as
diagnostics and later variants.

## Main Model Contract

The primary contract is continuous latent world modeling:

```text
observations -> continuous latent autoencoder -> latent frames z_t
observation or latent transitions -> continuous LAM -> latent actions u_t
latent history + u_t + optional prompt/event conditioning
    -> causal transformer dynamics
    -> diffusion or flow denoising objective for next latent frame/chunk
sampled next latent -> decoder -> next observation
latent history/action -> reward head and continue head
```

The continuous latent autoencoder is the first-stage compression model. It may be
a compact convolutional autoencoder, VAE-style latent autoencoder, or MAE-style
encoder/decoder, but it must produce continuous latent frames rather than a
discrete token codebook in the primary arm.

The continuous LAM infers continuous latent actions from adjacent observations or
latent frames. The baseline trains a variational posterior together with a
transition reconstructor that predicts the next latent from the previous latent
and inferred action; a scaled Gaussian KL regularizes the action bottleneck. LAM
infers continuous latent actions for unlabeled video-style data; where labeled
environment actions exist, the labeled actions supervise alignment and the
latent-to-real-action bridge.

The dynamics model is a causal transformer over latent-frame history, latent
actions, optional prompt embeddings, and optional world-event conditioning. Its
head predicts the diffusion noise, velocity, or flow target needed to sample the
next latent frame or a short latent chunk. The baseline uses linear conditional
flow matching from Gaussian noise to the next latent, classifier-free action
dropout during training, and guided Euler integration during rollout.

## VLA-Style Control

The policy interface should be compatible with a VLA-like controller:

```text
observation/history + optional task text -> latent action u_t
latent action u_t -> dynamics rollout inside the learned world
latent action u_t -> bridge -> real environment action when labels exist
```

The VLA-style actor may output continuous latent actions directly, distribution
parameters, or short action chunks. It is separate from the dynamics model and
does not turn real environment actions into the main conditioning signal.

The baseline actor and value model are trained from reward- and continue-weighted
rollouts inside the learned latent dynamics. During real-environment evaluation,
each current observation is encoded, the trained actor chooses a latent action,
and only then does the calibrated bridge map that latent action to an environment
action.

## Sampling

Interactive inference is autoregressive:

```text
past decoded/encoded frames + chosen latent action -> denoise next latent
next latent -> decode observation
append observation/latent to history
repeat
```

Classifier-free guidance is part of the main plan. During training, randomly
drop action and prompt conditioning in the dynamics objective. During sampling,
combine conditioned and unconditioned predictions to improve controllability.

## Real-Environment Action Bridge

Real environment control is secondary. Labeled trajectories train a small
latent-to-real-action bridge:

```text
labeled trajectories -> continuous LAM latent action + real action label
latent action -> empirical, linear, MLP, or distributional real-action mapping
policy(observation) -> latent action -> bridge -> real environment action
```

The bridge supports real-environment evaluation and calibration. It does not
replace the continuous latent action interface inside the learned simulator.

## Reward, Continue, and Evaluation Heads

Reward and continue heads are added for RL evaluation:

```text
latent history/action or next latent -> reward
latent history/action or next latent -> continue probability
```

These heads are trained from real replay labels and evaluated separately from
the visual generation objective. They do not replace the latent diffusion/flow
dynamics objective.

## Ablations

VQ/MaskGIT is ablation-only. The public Genie 1 path can be implemented as
`genie1_maskgit_ablation` with a VQ-VAE video tokenizer, discrete LAM codebook,
and next-frame token dynamics. That ablation should never silently replace the
continuous Genie-2-style primary arm.

LeWM/LeJEPA innovations are ablations only. Candidate ablations include
SIGReg-style regularization, control-value consistency, validation gates,
anchor/recent replay mixing, and ensembles, excluding the energy dependency
unless explicitly scoped in a later branch.
