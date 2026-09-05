# Structural comparison with DMAWM and MATWM

Audit date: 2026-09-04. These are design hypotheses for controlled experiments,
not claims that the repaired MA-JEPA already outperforms either baseline.

## Primary evidence

DMAWM was inspected from its official repository at commit
`2acaaeb82805b55d275e3ce08ac8f713ec9afbb2`. Its
[world-model implementation](https://github.com/DiXue98/DMAWM/blob/2acaaeb82805b55d275e3ce08ac8f713ec9afbb2/world_model.py)
does all of the following:

- Trains the joint stochastic prior against the local factorized posterior
  distribution (lines 342–356), and reconstructs observations (372–380).
- Computes critic targets from real replay rewards (440–461), using current
  versus stored behavior action probabilities to form cumulative joint-agent
  importance weights (464–487).
- Sends the real value-prediction loss through the critic into world-model
  features, with importance weights capped at 4 (489–511), then separately
  updates the critic on detached features (515–530).

Its
[actor implementation](https://github.com/DiXue98/DMAWM/blob/2acaaeb82805b55d275e3ce08ac8f713ec9afbb2/actor_critic/actor.py)
uses repeated whole-batch clipped PPO with frozen old log probabilities
(115–159), closely resembling this branch. Missing minibatching is therefore
not a demonstrated explanation for a performance gap.

The [MATWM paper](https://arxiv.org/html/2506.18537v1) uses observation
reconstruction, predicted teammate-action distributions in the actor state,
recent replay for world training and uniform replay for actor roots. Equation
11 uses a log-probability policy-gradient objective with percentile return
scaling, plus EMA critic regularization. Thus the strongest cited baseline
does not establish an inherent PPO ceiling advantage. The official code URL
printed in the paper returned 404 during this audit, so code-level MATWM
claims cannot be verified here.

DMAWM's ratio is not the per-action PPO ratio. It sums log-probability
differences over the agents and accumulates them backward over the replay
suffix, resetting at terminal/truncation boundaries, before exponentiating
([`build_cumulative_log_probs`, lines 202–212](https://github.com/DiXue98/DMAWM/blob/2acaaeb82805b55d275e3ce08ac8f713ec9afbb2/utils/tools.py#L202-L212)).
This is a cumulative joint-trajectory suffix importance weight. The code caps
the resulting value-loss multiplier at 4; it does not make arbitrary old
replay equivalent to an on-policy target. The current MA-JEPA replay-value
term is instead explicitly an uncorrected auxiliary critic target. These
must not be described as the same estimator.

## Pinned DMAWM run and evaluation settings

These are repository settings, not an inference about unpublished runs.
The [training configuration](https://github.com/DiXue98/DMAWM/blob/2acaaeb82805b55d275e3ce08ac8f713ec9afbb2/configs/trainer_configs/dreamer.yaml#L15-L40)
specifies 16 rollout environments, 5,000 prefill steps, train ratio 128,
gamma 0.99, GAE lambda 0.95, and five PPO epochs. Its
[evaluation defaults](https://github.com/DiXue98/DMAWM/blob/2acaaeb82805b55d275e3ce08ac8f713ec9afbb2/configs/trainer_configs/dreamer.yaml#L62-L67)
use four evaluation environments and 32 episodes every 10,000 steps.

The [2s_vs_1sc script](https://github.com/DiXue98/DMAWM/blob/2acaaeb82805b55d275e3ce08ac8f713ec9afbb2/scripts/dreamer/smac_2s_vs_1sc.sh)
runs seeds 0–4, requests 1,005,000 environment steps, and overrides imagination
to four steps and replay capacity to 250,000. The 2c_vs_64zg, 3s5z, and MMM2
scripts request the same step budget. The
[corridor script](https://github.com/DiXue98/DMAWM/blob/2acaaeb82805b55d275e3ce08ac8f713ec9afbb2/scripts/dreamer/smac_corridor.sh)
requests 405,000 steps. These scripts enable shared actors, shared critics,
and the critic transformer. A 50,000-step MA-JEPA result cannot be compared
directly with one of these final-budget DMAWM results; compare curves at the
same transition budget as well as final performance at the published budget.

## What the current JEPA implementation does and does not establish

The current encoder receives gradients from local reward, continuation,
availability, posterior JEPA, local dynamics JEPA, and distributional
regularization. This is more than an unsupervised embedding objective. It
still does not establish that two states with different *long-term action
values* remain distinguishable.

Both `CentralAttentionCritic` and `JointObservationJEPA._mix` explicitly stop
gradients through their input local states. Restoring real-replay critic
targets grounds the critic but does not provide DMAWM's real-value gradient
to the learned local representation. Joint reward prediction also cannot
repair information discarded by the local representation through that path.
This is a structural distinction, not a recommendation to remove every
gradient stop.

The joint simulator produces a single next observation embedding and passes
it through the executable local posterior. For partially observed events,
the expected embedding may fail to represent any coherent successor even
when cosine error is small. Sampling the subsequent categorical posterior
does not prove that it recovers the correct conditional distribution.
DMAWM instead supervises the joint latent distribution directly. JEPA can
retain its decoder-free design while making the executable successor
distribution an explicit validation target.

The direct H1/H2/H4/H8 JEPA prediction head and the recurrent simulator solve
different problems. Good direct-horizon errors do not demonstrate that the
one-step simulator remains accurate when its outputs are fed back into the
local posterior and the joint temporal cache. Under the current profile,
`rollout_steps=1` and zero multistep anchors disable the available recurrent
two-step training. The new self-fed report measures this discrepancy; it
does not train it away.

## Distinguish new tests from existing unsuccessful experiments

- `f194171` (consumer loss) sends a predicted embedding through a frozen local
  posterior and trains forward KL to the factual posterior. It explicitly
  freezes the local posterior/representation. This is consumer-distribution
  alignment, not real-return supervision of the representation.
- `d409b61` (terminal outcome, subsequently reverted) adds a terminal outcome
  classifier and potential-based actor reward shaping. Its central critic
  still stops gradients through local states. It does not test a real-value
  representation gradient either.
- `914e09e` (candidate Q1 actor) reads detached candidate teammate plans into
  a policy adapter. This changes the actor readout while retaining stopped
  representation inputs.

These branch histories prohibit treating consumer KL, terminal shaping, or
candidate action readouts as untried remedies. Code differences alone also
do not establish why a past treatment failed; its experiment results must be
considered before another related run.

## Diagnostic gates before larger architecture changes

1. **Control sufficiency.** Fit a frozen-feature probe to held-out real
   short-horizon returns and task-relevant observable quantities (health,
   cooldown, visible enemy range). Compare posterior features, encoder
   embeddings, and raw local observations under the same train/test split.
   If the frozen features cannot recover an observable decision variable,
   more PPO passes cannot recreate it. A task-targeted auxiliary predictive
   head is a narrower test than reintroducing a full decoder.
2. **Executable rollout consistency.** Compare the new recurrent self-fed
   reward/mask/liveness/posterior metrics with direct-horizon JEPA metrics.
   If recurrent errors deteriorate rapidly despite strong direct prediction,
   train a short unrolled version of the *same* simulator from factual roots,
   with stopped boundary state and factual action sequences. Measure
   improvement in held-out self-fed errors before increasing policy horizon.
3. **Action-value ranking.** On factual roots, compare predicted action
   effects with short real-environment branches or a held-out intervention
   set. Embedding spread, action-margin loss, and low mean reward error do not
   prove that the model ranks useful actions correctly.
4. **Real-value gradient, if indicated.** A controlled auxiliary head may
   send real-return gradients to local encoder/posterior features while
   leaving actor PPO and the centralized critic's normal stop-gradient
   boundary intact. It must use fresh/current-policy or corrected replay
   targets and be compared against the same critic-only target. Monitor
   representation drift and held-out control probes, not only value loss.

The first repaired run should establish real win-rate and self-fed model
behavior before stacking these hypotheses. None of these mechanisms,
individually or together, guarantees superiority across the task suites.
