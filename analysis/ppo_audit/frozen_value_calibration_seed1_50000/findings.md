# Frozen real-policy value calibration: seed 1, 50,000 steps

Completed 2026-09-05. The correction sharply reduces the critic's post-death
errors, but surviving agents' values still overestimate actual episode returns.
This supports testing a more stable value target as a hypothesis; it does not
establish that target EMA will improve policy learning.

## Protocol and provenance

Original `92014f1` and corrected anchor `2663ae5` each used their own exact saved
configuration, final 50,000-step checkpoint, and source package. The calibrated
critic consumes the same current local posterior returned by the canonical
decentralized actor. Both ran 32 fresh `eval_sample` episodes, environment seed
270001 (`eval_seed=70001`, worker offset 200000), diagnostic seed 9001, one
environment, and discount `332/333`. Collection unimix was excluded, matching
the learned stochastic policy. Evaluation parameters and counters were identical
before and after both runs.

The value target is the discounted shared team reward from the **next** arrival
through the adapter-defined episode end, retaining teammate rewards after focal
agent death. The maintained adapter labels timeouts terminal: original had 11
timeouts and corrected had 2. Consequently these are returns for the maintained
capped task, not hypothetical returns beyond a timeout. No terminal value is
used as a bootstrap. H5/H15 comparisons instead use actual rewards plus the same
frozen critic at the actual successor, with zero terminal bootstrap; these are
Bellman consistency checks, not finite-return prediction errors.

| Item | Original | Corrected with factual value anchor |
|---|---|---|
| Training run | `original_base-2s3z-seed1` | `team_return_anchor-2s3z-seed1` |
| Source | `92014f1` | `2663ae5` |
| Checkpoint | `20260905T084909F063960` | `20260905T085005F524901` |
| Preserved state count, including reset/terminal | 2,635 | 2,114 |
| Evaluation duration | 122.98 s | 118.31 s |
| Peak process RSS | 6.36 GiB | 6.39 GiB |
| Completion UTC | 08:57:52 | 09:01:58 |

Both ran sequentially on assigned GPU 0 after predecessor final128 evaluation,
supervisor exit, and idle-device checks. GPU 0 was released after completion.
The checkpoint copies remain hardlinked in isolated remote snapshots; no source
training files were altered. This directory preserves raw compressed episodes,
configs, per-file source/checkpoint hashes, run manifests, and execution logs.
The exact calibration script is `d433a7e` SHA256
`e880e606b62e2fea8a01a6584a5844deb584d1ccf6d16cc41e110566f753444d`.

## Fresh policy outcomes

| Outcome | Original | Corrected |
|---|---:|---:|
| Wins | 0/32 | 1/32 |
| Win probability, Wilson 95% | 0–10.7% | 0.6–15.7% |
| Mean undiscounted return | 11.027 [10.509, 11.526] | 12.559 [11.736, 13.426] |
| Mean discounted initial return | 10.225 [9.769, 10.654] | 11.672 [10.939, 12.443] |
| Mean episode steps | 81.34 | 65.06 |

The return difference is +1.532, with a conditional episode-bootstrap interval
[0.582, 2.504]. The small win sample does not establish a reliable win-rate gain.
These fresh episodes are separate from the completed final128 screen, which
reported original 1/128 and corrected anchor 9/128.

## Critic against actual remaining team return

Bias is predicted value minus Monte Carlo return. Brackets are 95% percentile
intervals from 4,000 whole-episode bootstrap draws; all agent slots and times
within each selected episode remain together. Metrics weight the observed
state-agent pairs, not episodes equally. The live and slow heads predict exactly
the same values in both runs because both saved configs use `slowvalue.rate=1.0`.

| Slice | Original bias [95%] | Corrected bias [95%] | Original RMSE | Corrected RMSE |
|---|---:|---:|---:|---:|
| Episode initial | +1.814 [1.380, 2.273] | +2.674 [1.907, 3.405] | 2.255 | 3.437 |
| Alive states | +0.778 [0.428, 1.151] | +2.178 [1.632, 2.714] | 1.880 | 3.179 |
| Dead states | +5.677 [4.863, 6.714] | +0.423 [0.285, 0.570] | 6.874 | 0.918 |
| First post-death state | +7.207 [6.596, 7.798] | +0.860 [0.586, 1.117] | 8.083 | 1.459 |

Alive-state counts are 7,784 original and 6,810 corrected; dead-state counts are
5,231 and 3,600. These are correlated observations from only 32 episodes per
policy. First post-death counts are 126 and 124. The original death mask did not
train values for the same continuing team objective; the dead-state comparison
tests the desired cooperative objective and is not a claim about its old loss.

Corrected initial value is 14.347 against actual discounted return 11.672.
Its live-state optimism remains substantial despite the improved dead-state
behavior. The corrected-minus-original alive-state bias interval is
[+0.735, +2.033]; its dead-state bias interval is [-6.300, -4.428]. These
cross-policy contrasts describe different visited state distributions and cannot
isolate the causal effect of the factual anchor from the other corrections.

## Short-horizon consistency versus full returns

| Alive-state comparison | Original bias | Corrected bias | Original RMSE [95%] | Corrected RMSE [95%] |
|---|---:|---:|---:|---:|
| H5 real Bellman target | -0.362 | +0.328 | 2.609 [2.408, 2.807] | 1.379 [1.242, 1.528] |
| H15 real Bellman target | -0.894 | +1.206 | 3.873 [3.538, 4.222] | 2.243 [1.895, 2.624] |
| Full Monte Carlo return | +0.778 | +2.178 | 1.880 [1.616, 2.156] | 3.179 [2.717, 3.634] |

The corrected critic is more consistent over actual short trajectories while
still optimistic about full remaining return. A bootstrap target can retain
systematic future value error, so smaller Bellman residuals do not establish
calibration. A slower target is a reasonable isolated next test, alongside the
separate recurrent-model treatment; this result does not prove that EMA fixes
the remaining simulator errors or that either intervention will raise wins.

## Evidence and reproduction

- [Original raw records](original/result32/episodes.npz),
  [summary](original/result32/summary.json), and
  [provenance](original/result32/provenance.json).
- [Corrected raw records](corrected/result32/episodes.npz),
  [summary](corrected/result32/summary.json), and
  [provenance](corrected/result32/provenance.json).
- [All clustered estimates](clustered_calibration.json).

Both archive hashes were verified locally. Recomputing every original summary
count, prediction/target mean, bias, and RMSE from raw episodes reproduced the
saved summaries within numerical tolerance. An additional normalization check
confirmed that duplicating correlated agent slots does not narrow the
episode-cluster intervals. Ruff and format checks passed for the summarizer.

From the repository root:

```sh
python analysis/ppo_audit/summarize_frozen_value_calibration.py \
  --original analysis/ppo_audit/frozen_value_calibration_seed1_50000/original/result32 \
  --corrected analysis/ppo_audit/frozen_value_calibration_seed1_50000/corrected/result32 \
  --output analysis/ppo_audit/frozen_value_calibration_seed1_50000/clustered_calibration.json
```

Uncertainty is conditional on these frozen checkpoints and this evaluation
stream. Cross-policy intervals independently resample whole episodes, since the
policies generate different trajectories despite a shared evaluation seed.
These intervals omit training-seed variation and are not an algorithm-level
causal claim. The analysis deliberately stops at this completed comparison.
