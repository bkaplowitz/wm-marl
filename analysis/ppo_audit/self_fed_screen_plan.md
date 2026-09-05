> Superseded by user steering on 2026-09-05 before any arm was launched. Retained as a prepared protocol, not experimental results. See prefill_recurrent_ema_protocol.md.

# Draft: recurrent world-model training and shorter PPO imagination

Status: prepared, not launched. Anchor selection requires the current screen's
four completed 2s3z final128 outcomes. The recurrent common scale 0.1 remains
provisional pending production gradient-magnitude and GPU time/memory checks.
Existing correction runs retain their original 50k/final128 protocol.

## Questions and fixed allocation

Does fitting the world model at recurrently generated inputs improve real
control? Separately, does reducing imagined PPO horizon from five to two help
when multi-step simulation error is large? These are independent interventions;
this screen does not test combining recurrence with horizon two.

| Slot | Treatment | Task | Seed | Source |
| --- | --- | --- | ---: | --- |
| 0 | Recurrent loss, PPO H5 | 2s3z | 0 | New optional implementation |
| 1 | Corrected baseline, PPO H2 | 2s3z | 0 | Prior corrected implementation |
| 2 | Recurrent loss, PPO H5 | 2s3z | 1 | New optional implementation |
| 3 | Corrected baseline, PPO H2 | 2s3z | 1 | Prior corrected implementation |
| 4 | Recurrent loss, PPO H5 | 3m | 0 | New optional implementation |
| 5 | Corrected baseline, PPO H5 | 2s3z | 123 | Prior corrected implementation |

All six share the final-selected factual critic anchor setting. Existing
factual teacher-forced losses remain intact in recurrent arms. Every arm has
consumer KL disabled. The H2 controls change only PPO imagination horizon from
the corresponding corrected H5 baseline. Seed 123 extends baseline training-seed
replication; it is not an additional recurrent or H2 replicate.

## Source and objective contract

The prior corrected package is commit
`2663ae57ab80482ad201f9c3bdaa9fe2c732dbe9`, package SHA256
`9db9598e0da7cced1eab43036845b9ff593bb9f61713720f4654460a5c49b4e7`.
The optional implementation is commit
`d623d4d34fcbebb8e345a1f038f71a988940235b`, package SHA256
`a06ed84235620d433a2030189e7017d05393d32a5b7247325cb37a046d64e7bc`.
The launcher executes controls from the prior package, with no new self-fed
configuration fields, and recurrent arms from the new package. Production gate
results may require a new implementation snapshot; update these pins and rerun
validation rather than changing a source directory after launch.

Candidate recurrent configuration is `agent.marl.ctde.self_fed` with enabled
true, horizons `[2, 4, 5]`, anchors 8, common scale 0.1, and consumer_kl_scale 0.
Scale 0.1 is a candidate, not a validated production setting. The common scale
multiplies existing embedding, interface, reward, continuation, action-mask, and
alive relative loss weights; record all resolved weights as well as the scale.
PPO horizon stays five for these recurrent arms.

The new objective uses recorded joint actions, samples the frozen local
posterior, and detaches state between recurrent steps. Gradients reach the
current joint model and prediction heads; the auxiliary does not update actor,
encoder, local dynamics, critic, or teammate predictors. This tests fitting at
self-fed inputs, not full backpropagation through the imagined sequence. Boundary
masks preserve valid terminal arrivals and later team rewards after a unit dies.

The optional-source tests verify exact full-state initialization and PPO-active
training parity among absent configuration, default-off, and enabled scale-zero
modes. Those tests alone are within-source comparisons. Actual deployment also
requires a source-bound record of executed prior-versus-new CPU parity, with
`parity_kind=cross_source_executed_train`, passed status, both package hashes,
and the test command. That paired CPU check passed: 624 initialization tensors and 922 full active-PPO
update/output tensors were identical byte-for-byte, including paths, shapes,
and dtypes; see `parity_d623d4d/parity.json`.
Production-shaped H5 loss/gradient magnitude, GPU memory, and update time remain
required before accepting the recurrent scale and source for training.

## Anchor selection and fixed evaluation

Use all four current corrected 2s3z final128 outcomes: scales 0 and 0.3, seeds
0 and 1. Select the higher mean final win rate, then mean final return for an
exact win-rate tie; retain scale 0.3 if both are tied. This is a development
selection rule, not an independent benchmark estimate. Do not select from an
early curve peak or independently by task/seed. Preserve both settings' results.

Preserve base hyperparameters: one collection environment, 50,000 joint
transitions including prefill, PPO start 5,000, world-model start 0, 1,000-update
world optimizer warmups, train ratio 128, actor 3×1024, actor/critic LR 3e-5,
collection unimix 0.05, and fixed entropy coefficient 0.01. Only slots 1 and 3
change PPO imagination horizon to two. Do not bundle critic EMA, replay age,
capacity, optimizer, or entropy-schedule changes into these comparisons.

Use SC2 4.10/Base75689 and the same SMAC maps. Evaluate 32 greedy episodes every
5,000 transitions at seed offset 50,000; evaluate the exact 50k checkpoint on a
separate fixed 128 episodes at offset 100,000 with four evaluation environments.
Report all ten curve points and final128 separately. No best-checkpoint selection
or omission of failed runs. Record additional update/GPU/wall time from recurrence.

## Diagnostics and limitations

Fresh frozen diagnostics show H5 action-mask false positives increasing from
14.04% teacher-forced to 39.86% self-fed, liveness error from 0.0178 to 0.0716,
and positive self-fed return-MSE excess with a positive confidence interval.
These motivate recurrent fitting and the H2 control. Use the paired frozen
artifacts to retain the exact sample/horizon scope of those screening numbers.
After training, repeat matched-root teacher-forced/self-fed checks, including
reward/continuation calibration and executable availability/peer-action errors.
Pair critic values with real discounted stochastic-policy outcomes at the same
states; imagined means versus whole-episode returns are not paired calibration.

Historical consumer-posterior experiments reduced posterior KL while achieving
poor real returns, but used posterior weight 4 and removed or sharply reduced
embedding/interface losses. Their budgets and exit evidence also differ; see
`historical_consumer_findings.md`. They do not isolate small added KL on self-fed
states. Nevertheless, consumer consistency is still a hypothesis and remains
optional, with no immediate KL arms in this screen.

Recorded-action recurrence does not prove counterfactual accuracy for actions
absent from replay or remove behavior-policy action mismatch. Better latent error
alone does not establish better control. Report each seed's final win rate,
return, timeout rate, curve area, and compute; use training seeds as replication.

Original PPO controls cover two training seeds, with preserved seed 0 from an
earlier runtime whose SC2 build was not recoverable. Current-host seed 1 gives
the cleaner runtime match. Two new seeds and one 3m regression run do not establish
robustness or superiority over DMAWM/MATWM; use `benchmark_targets.md` for that
claim. If the selected anchor becomes zero, the available corrected 3m reference
uses 0.3, so that comparison changes both anchor and recurrence and requires a
matched 3m control before a causal claim.

## Launcher and evidence preservation

`scripts/run_ppo_self_fed_screen.py` reuses the tested correction runner's child
ownership, GPU-idle waiting, fingerprints, W&B logging, checkpoint check, and
fixed final128 helpers. It requires the optional source and prior baseline
source/hash, explicit self-fed scale, completed baseline-results root, and
source-bound disabled-parity record for an actual launch. A provisional
`--validate-only --candidate-anchor-scale 0.3` resolves configurations without
selecting the experimental anchor or starting training. It cannot launch with
that provisional flag. CPU config resolution has passed all six candidate
profiles; see `self_fed_launcher_validation.json`.

Use new immutable source/output directories and a distinct W&B group with unique
train/final IDs. Start local metric/config mirroring before training and archive
each completed checkpoint/config/raw replay independently, with verified SHA256.
Copy later diagnostics separately. Preserve source data throughout. Do not
commit live mirrors, mutable plots, or large checkpoints. Await final selection,
production gate, and the parent's coordinated GPU allocation before launch.
