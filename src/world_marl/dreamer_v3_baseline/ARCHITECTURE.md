# DreamerV3 Baseline Architecture

This architecture is taken directly from the DreamerV3 paper:
[DreamerV3: Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104).

The package is a faithful JAX research baseline. It does not include later
architectural additions, alternate representation-learning objectives, or
control-specific ablations. Those belong outside this baseline.

## Model Contract

The baseline contract is:

```text
observation_t -> encoder -> posterior RSSM state
posterior/prior RSSM state -> decoder -> reconstructed observation_t
posterior/prior RSSM state -> reward head -> reward_t
posterior/prior RSSM state -> continue head -> continue_t
imagined RSSM state -> actor -> action_t
imagined RSSM state -> critic -> value_t
```

The world model consists of an encoder, recurrent state-space model, decoder,
reward head, and continue head. The policy side consists of an actor and critic
trained from imagined rollouts through the learned latent dynamics.

## RSSM State

The RSSM state is the standard DreamerV3 state:

- Deterministic recurrent state updated by a GRU-style transition.
- Categorical stochastic latent state.
- Posterior transition conditioned on the encoded current observation.
- Prior transition conditioned on the previous state and action.

The posterior is used for representation learning on replay sequences. The prior
is used for open-loop prediction and imagined actor-critic rollouts.

## Online Training

For real JAX-native adapters, the environment scan carries the current actor/RSSM
policy state, circular replay, optimizer state, replay-ratio state, and PRNG key.
Each arrival is filtered through the posterior, the actor chooses the outgoing
action, and the transition plus replay-context latent is inserted before scheduled
joint model/actor/critic updates. The default train ratio is 32 replayed transitions
per environment transition. Training starts only when replay can supply a full
batch and follows the author's one-update first eligible step behavior.

The synthetic CLI path trains on a fixed generated sequence solely as a component
test. It is not reported as an online DreamerV3 benchmark.

## Losses

World-model training optimizes the paper's reconstruction, reward, continue,
dynamics, and representation objectives:

- Observation reconstruction from posterior states.
- Reward prediction from latent states.
- Continue prediction for non-terminal continuation probability.
- KL balancing between posterior and prior categorical latents.
- Free bits/free nats to avoid over-regularizing useful latent information.
- Symlog and two-hot treatment for reward/value targets where applicable.

Reset and terminal masks must prevent leakage across episode boundaries.

## Actor-Critic Training

Actor and critic training use imagined rollouts from the learned RSSM prior:

```text
posterior replay state -> imagine with actor and RSSM prior -> imagined rewards,
continues, values -> actor and critic objectives
```

Evaluation returns are measured in the real environment. Imagined returns,
world-model losses, finite checks, and reconstruction diagnostics are supporting
metrics, not substitutes for real-environment evaluation.

### Real-environment and pixel evaluation

`dmc:<domain>/<task>` uses MuJoCo Playground and MJX for accelerator-resident
collection and evaluation. This path currently exposes the task's vector
observation, so it validates real control integration but is not evidence for a
pixel-based Dreamer benchmark.

`dmc-pixels:<domain>/<task>` uses the Python `dm_control` pixel wrapper. It does
not implement the repository's required JAX scan collection contract and is
therefore rejected by the training CLI instead of falling back to a host loop.
It must not be reported as an accelerator-resident or completed visual
evaluation. A genuine visual benchmark requires a JAX-compatible renderer and
the same interaction/update budget across model arms.

`pixels:pointmass` remains a synthetic CNN and tensor-plumbing fixture. Its
results must not be presented or aggregated as DMC evaluation evidence. See the
[official DMC repository](https://github.com/google-deepmind/dm_control),
[pixel wrapper](https://github.com/google-deepmind/dm_control/blob/main/dm_control/suite/wrappers/pixels.py),
and [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground).
