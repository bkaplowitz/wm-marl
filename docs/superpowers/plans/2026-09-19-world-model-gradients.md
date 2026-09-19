# World-model gradient experiments implementation plan

**Goal:** independently test critic-value supervision and joint-prediction gradients into the existing local world model.

**Base:** fetched all remotes on 2026-09-19. `1d840321fb732e7cb930f471f3655d47a33eec70` includes Osaze's latest model commit `362f728` and launcher. Branch: `feat/world-model-gradient-experiments`.

## Design and constraints

- `agent.world_model_gradients.critic_value_scale: 0.0` adds an auxiliary real-replay value loss to the world-model update when positive. It reuses the central critic with live input derivatives; the world optimizer still excludes critic and actor parameters. No separate value head. Normal imagined PPO and its replay critic objective are unchanged.
- Its targets use recorded rewards, the detached slow critic on factual states, and bounded V-trace joint-action correction (rho=c=1, existing replay lambda and discount). Reuse `training/factual_value.py` from commit `47af44e`. Record actual collection probabilities after legal masking and collection mixing, aligned with the action leaving the current state. Targets and importance ratios have no gradients. This is an auxiliary representation loss, not an unbiased policy-value estimator and not backpropagation through imagined rewards.
- `agent.world_model_gradients.joint_prediction: false` controls local gradients from existing joint world-model objectives: factual source features, dense categorical alignment, and self-fed local transitions/categorical encoding. Existing joint loss weights stay unchanged, including outcome and action-margin heads sharing the joint representation.
- All teachers and EMA targets remain stopped. Self-fed roots/cache stay detached and existing BPTT chunk boundaries remain. Within each chunk, enabling joint gradients also enables local feedback into subsequent joint predictions. The factual replay path trains the observation encoder; the self-fed path trains local dynamics after detached roots.
- Both options default off, with no extra collection fields or auxiliary calls when disabled. Preserve actor objectives, decentralized execution, RNG sequence in the disabled path, and existing optimizer ownership.
- Reuse current YAML/dotted overrides, `majepa-train`, and campaign launcher. No new launcher, dependencies, cloud jobs, commits, or pushes. Preserve pre-existing `.gitignore` and `todo.md` edits.

## Implementation and verification

- [x] Repair relevant stale training tests (removed two-action imports/options) while preserving current single-action PPO, warmup, death and reset coverage. Add failing option/gradient checks before implementation.
- [x] Add default config, CLI/run-spec options, validation and recorded launch metadata. Check independent combinations and invalid/nonfinite scales.
- [x] Restore factual V-trace utility; test action/reward timing, terminals, truncations, resets, absent slots, joint ratios and detached targets. Add optional behavior probability collection/replay fields and the central-critic input-gradient keyword. Add world auxiliary and scalar metrics to the existing learner loss.
- [x] Parameterize existing joint source barrier and frozen categorical/transition helpers. Test off/on gradients for encoder, temporal dynamics and categorical head, with no teacher gradients and unchanged forward values.
- [x] Run JIT training for all four combinations, verify finite updates and group ownership, verify actor/critic PPO never modifies WM, and preserve disabled-path behavior.
- [x] Add `experiments/world-model-gradients.json`: baseline, critic (scale 0.1), joint, both; seeds 0/1/2; SMAC 2s3z difficulty 7; current `localmask_reference` settings held fixed. Resolve with existing campaign dry-run. Placement IDs remain explicit placeholders until a cloud launch is requested.
- [x] Document exact scope and limitations; independently review gradients and experiment configuration, fix findings, run focused tests and check diff.

## Test interface repairs

Tests referring to unavailable two-action and truncated-geometric wrapper APIs
were migrated to the maintained single-action and exponential-recency replay
interfaces. Replacement replay checks exercise finite geometric sampling,
seeded uniform draws, FIFO eviction, invalid inputs, independent replay RNGs,
and storage alignment for actions and the new collection-probability metadata.
The final gate is the full pytest suite, not only the feature tests.

## Experiment interpretation

Compare matched seeds using final greedy evaluation and learning curves, plus local/joint gradient norms, auxiliary value loss and V-trace effective sample size. Scale 0.1 is an initial treatment choice, not a tuned result. The joint flag enables all existing joint-objective gradients into local parameters, including existing action-margin gradients; it does not establish that every objective helps. Three seeds support a preliminary comparison only. No performance improvement is assumed.

## Completed validation

- Main focused run: 78 tests passed (world gradients, factual targets, alignment, full learner, configuration, campaign, queue and tracking).
- Restored current PPO tests plus the stricter all-key campaign comparison: 12 passed.
- Collection-probability and updated evaluation/configuration checks: 10 passed.
- Final per-objective categorical/temporal gradient and actual PPO-only ownership checks: 3 passed.
- Both-off source regression: identical parameters, optimizer states, recurrent state, replay output and metrics across all 806 leaves compared with a pre-edit snapshot, seed7701 initialization and seed7702 update.
- Ruff lint passed for all changed Python files; checked formatted test/new utility/CLI files; `git diff --check` passed.
- Campaign dry-run resolved all 12 runs and each combination. No cloud launch was performed.
- Two independent final reviewers found no remaining blockers after fixing standalone evaluation flag propagation. Additional checks covered actual mixed collection probabilities, all-key campaign isolation, categorical-head gradients and PPO-only ownership.

The replay collection failure is resolved: all 10 tests of the maintained replay APIs pass. The full suite passes without exclusions: **113 passed in 215.57 seconds**. Repository-wide `ruff check src tests` and `git diff --check` pass. Both source distribution and wheel build successfully, and packaged gradient code/configuration match the working tree. No replay selector was reintroduced.

The feature remains uncommitted on the requested new branch. Existing user `.gitignore` and `todo.md` changes were preserved. No claims about real SMAC performance or full-size GPU memory/runtime follow from these CPU checks.
