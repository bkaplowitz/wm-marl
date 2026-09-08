# Pinning the reference and tracing the Stalker-map regression

The maintained `ma_jepa` profile now selects the completed `am1-bernoulli`
development configuration: one environment, 5k prefill within 50k transitions,
train ratio 128, BPTT2, self-fed posterior alignment, 50% uniform world replay,
H5 imagination, Bernoulli imagined availability, full-copy critic targets,
encoder EMA .01, and no factual V-trace or representation-value auxiliary.
World optimizer warmups are zero; report RNG is isolated. The public launch
manifest and default curve-evaluation settings have been updated to describe
this configuration. The frozen sources of existing experiments are unchanged.

Validation resolved the public launch command and compared all 312 configuration
keys with the saved successful reference. The only differences are log path,
logging sinks/filter, and memory preallocation; all learning and environment
settings match. Explicit `--eval-interval 0` still disables curve evaluations.
The modified Python files parse successfully. No new performance run was used
to validate a configuration-only promotion.

This reference scored 42/49/40% on 2s3z and 90/85/95% on 3s_vs_3z, but 0/0/0%
on 3s_vs_4z. It is a development reference with demonstrated strengths, not a
claim of a universal winner. The threshold-availability results and all other
experiment records remain available.

The new factual-value treatments finished seed-0 final100 at 9% / 21% on 2s3z
and 0% / 10% on 3s_vs_4z (critic-only / critic plus representation). They have
not displaced the reference. Their queued second seeds remain deferred.

## Recovered historical results

Fresh W&B retrieval recovered the previously missing REINFORCE 3s_vs_4z
evaluations. Each is a fixed 128-episode final evaluation of a 50k run; these are
not training-curve peaks.

| REINFORCE configuration | Seed 0 | Seed 1 | Seed 123 |
| --- | ---: | ---: | ---: |
| c29, source folder 85db360 | 43/128 (33.6%) | 0/128 | 92/128 (71.9%) |
| c33, source folder 05e98f8, legal collection mix .05 | 35/128 (27.3%) | 0/128 | 0/128 |

W&B references:
- c29: https://wandb.ai/osaze-obahor/dreamarl-c29-workshop-four-seed/runs/fc29w3s4zs0
- c29: https://wandb.ai/osaze-obahor/dreamarl-c29-workshop-four-seed/runs/fc29w3s4zs1
- c29: https://wandb.ai/osaze-obahor/dreamarl-c29-workshop-four-seed/runs/fc29w3s4zs123
- c33: https://wandb.ai/osaze-obahor/dreamarl-c33-legal-unimix-variance/runs/fc33l53sv4zs0
- c33: https://wandb.ai/osaze-obahor/dreamarl-c33-legal-unimix-variance/runs/fc33l53sv4zs1
- c33: https://wandb.ai/osaze-obahor/dreamarl-c33-legal-unimix-variance/runs/fc33l53sv4zs123

The old model had the capability, but also substantial seed variance. More
importantly, degradation was visible in later REINFORCE runs, before the PPO
migration. Thus a claim that PPO alone removed this capability is unsupported.
These historical runs use seeds 0/1/123, whereas the current candidate uses
0/1/2; do not call those seed sets a fully paired comparison.

The older runs expose empty W&B API configs. Their downloaded
`wandb-metadata.json` files retain the actual launch arguments. Those arguments
explicitly verify H15, actor LR 1e-5, entropy .0003, train ratio 256, and 50k
training transitions. Defaults not present in the arguments are reconstructed
from the Git revisions named in their source directories, rather than claimed
to come from a recovered complete runtime config. The associated source commits
exist locally. Raw retrieval and provenance are saved under
`analysis/ppo_audit/decision_20260908/reinforce_comparison`.

## The regression predates PPO

Between Git revisions 85db360 and 05e98f8, the executable source diff is only
46 inserted and three removed lines across five files. The intervening commits
are:

- `778fab1`: mark a SMAC episode-limit transition terminal, ending its value
  bootstrap. Previously, an episode-limit transition was last but nonterminal.
- `05e98f8`: add uniform exploration over legal actions during real collection.
  c33 explicitly sets this extra mixture to .05; c29 predates the feature.

The c33 runs retain H15 and the same explicit actor learning-rate/entropy/ratio
settings. The two code changes are not independently identified by this
comparison. Both changed the trajectories and/or targets, and finite seed
variation remains. Nonetheless, these are specific, localized suspects supported
by the historical sequence, unlike an unspecified missing representation loss.

The recovered c30 timeout-change-only 3s_vs_4z run is seed 1, which was already
zero in c29. It therefore does not isolate why the previously successful seed
123 failed later. A finite-horizon target and a continuing-task bootstrap also
represent different assumptions. Keep the current benchmark convention fixed
in parameter comparisons and audit its targets separately; do not silently
change episode semantics to recover a score.

## What changed again during the PPO-era configuration

| Setting | Successful c29 launch / source | Current reference |
| --- | --- | --- |
| Actor objective | REINFORCE | PPO |
| Imagined steps before the final value bootstrap | 15 | 5 |
| Actor LR | 1e-5 | 3e-5 |
| Actor passes per learner batch | 1 | 5 |
| World-model train ratio | 256 | 128 |
| Extra legal collection mixture | Absent | .05 |
| Actor entropy coefficient | .0003 | .01 |
| Advantage scale | Running return percentiles | Batch standard deviation with centering |
| Slow-value prediction regularizer | 1 | 0 |

At batch size 16 and length 64, the configured steady-state world update rate
changed from .25 to .125 per real transition. Actor updates changed from .25 to
.625 per transition, despite less world-model training. These count optimizer
calls, not equivalent gradient sizes: Adam, advantage scaling, reuse of samples,
the world update order, and the prefill/warmup schedules differ. It would be
incorrect to multiply those factors and claim an exact effective-LR change.

H5 can still learn long-term values through its bootstrap; it is not a five-step
real planning limit. But it relies on the critic earlier. H15 explicitly models
more of an attack/retreat sequence, while exposing more compounding model error.
The old successes make this an evidence-based parameter comparison, not proof
that a longer horizon will improve the revised model.

## Next direct comparisons

Keep the pinned representation, replay, rewards, termination convention, critic
target copying, batch sizes, and training budget fixed. The first comparisons
I recommend are existing parameter overrides:

| Candidate | Change from the reference | Question |
| --- | --- | --- |
| Collection | `agent.collection_unimix: 0.0` | Does the extra 5% uniform collection component disrupt learning useful sequences? |
| Horizon | `agent.imag_length: 15` | Does the shorter imagined rollout rely too heavily on early value bootstraps? |
| Combination | Both changes above | Do the longer rollout and cleaner collection interact? |
| Actor LR | `agent.ppo.actor_lr: 1e-5` | Is the current actor moving too quickly relative to model learning? |

Zero extra collection mixing keeps categorical policy sampling and the actor's
existing .01 distribution unimix. It does not turn collection into greedy action
selection. Longer imagination retains the existing H2/H4/H5 recurrent objective
for this isolated comparison; it does not silently add a new loss or change its
weights. Check rollout reliability and actual wins together when interpreting it.

Use the same candidate settings across 3s_vs_4z and 2s3z and check retention on
3s_vs_3z. An improvement confined to one selected seed or map is insufficient.
The completed reference runs supply comparison evidence; no replacement control
or new training sweep has been launched during this review.

Entropy/advantage rescaling is lower priority. The earlier BPTT2-only 2s3z
screen scored base 25%, actor-one-epoch 16%, entropy .0003 15%, percentile
advantages 0%, and percentile plus entropy .0003 7% (one seed each). Those
results predate alignment and replay mixing, so they are not final answers for
the current base, but they argue against blindly copying all legacy settings.
The coefficients also operate on different advantage scales. Increasing train
ratio is another possible follow-up, but changing it also increases actor and
critic update counts unless explicitly separated.

The key correction to our reasoning is to trace the actual configuration and
source history before blaming PPO or adding more objectives. The history gives
specific parameter and boundary changes to test; it does not yet identify one
guaranteed fix.
