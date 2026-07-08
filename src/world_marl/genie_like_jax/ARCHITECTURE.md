# Genie-Like JAX Architecture

This architecture is taken directly from the public Genie paper:
[Genie: Generative Interactive Environments](https://arxiv.org/abs/2402.15391).

The model name in this repository is `genie_like_jax`. Jasmine and Jafar are
implementation references, not the model name or primary specification. When an
implementation choice is borrowed from either source, the module and experiment
metadata must cite the relevant source:

- [Jasmine paper](https://arxiv.org/abs/2510.27002)
- [Jasmine repository](https://github.com/p-doom/jasmine)
- [Jafar repository](https://github.com/FLAIROx/jafar)

## Main Model Contract

The primary package contract follows the public Genie paper:

```text
video observations -> VQ-VAE video tokenizer -> discrete frame tokens
video observations -> LAM VQ codebook -> latent_action_id
frame token history + latent_action_id -> MaskGIT dynamics -> next-frame tokens
next-frame tokens -> tokenizer decoder -> next observation
next observation or latent state -> reward/continue heads -> RL evaluation labels
```

LAM produces discrete latent action codes from video or observation sequences
during training. At interactive inference, the user or learned policy chooses
latent action codes directly. The VQ-VAE video tokenizer is primary for the
faithful Genie-like baseline, and the dynamics predicts next-frame tokens
conditioned on prior frame tokens plus latent action codes.

This is intentionally not a real-action-conditioned dynamics model. The main
model does not train `p(next_observation | observation_history, real_action)`.

## Genie 3 Target

Genie 3 is a capability target, not a complete public architecture. The public
Google DeepMind material describes real-time interactive generation, stronger
long-horizon consistency, promptable world events, and higher-fidelity
navigation, but it does not publish enough architectural detail to implement an
exact Genie 3 clone. This branch therefore keeps public Genie and
Jasmine-compatible mechanics as the faithful baseline, then adds Genie-3-inspired
variants only when their mechanisms are explicit in public papers or local
experiments.

## Modern Variants

Direct next-observation generation is a modern variant, not the faithful public
Genie baseline. It can be explored through a diffusion or continuous-latent
dynamics arm, especially where Jasmine-style diffusion or newer video-generation
methods outperform MaskGIT token prediction for this repository's environments.
Jasmine's MaskGIT and causal baselines use a discrete VQ-VAE tokenizer, while
its diffusion baseline uses an MAE tokenizer; those paths should remain
separate experiment arms.

Codebook-free variants must be named explicitly, for example
`genie_like_jax_diffusion` or `genie_like_jax_continuous_latent`, and compared
against the VQ tokenizer plus VQ latent-action baseline rather than silently
replacing it.

## Real-Environment Action Bridge

Real environment control is a secondary bridge for evaluation. Following Appendix
E of the public Genie paper, labeled trajectories can train a small
latent-to-real-action mapping:

```text
labeled trajectories -> LAM latent_action_id + real_action_label
latent_action_id -> empirical or learned real-action distribution
policy(observation) -> latent_action_id -> bridge -> real_env_action
```

The bridge is used for calibration, real-environment evaluation, and ablations.
It is not the conditioning signal for the main dynamics model.

## RL Heads and Ablations

Reward and continue heads are added for RL evaluation. They do not replace the
next-frame token prediction and decoded-observation generation objectives.

LeWM/LeJEPA innovations are ablations only. They can be added after the public
baseline is working, excluding the energy dependency unless explicitly scoped in
a later branch.
