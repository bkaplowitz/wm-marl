# Factual value treatments and Stalker-map evidence

Prepared on 6 September 2026 on `ma-jepa-ppo`. These are optional local candidates,
not changes to the frozen BPTT2 matrix. No performance experiment has been launched
for either candidate, and no performance improvement is established.

## DMAWM and the map hypothesis

DMAWM commit `2acaaeb82805b55d275e3ce08ac8f713ec9afbb2`, `world_model.py:440–530`,
uses target values on recorded posterior states, real replay rewards, bounded
behavior importance weighting, and value gradients into world-model features.
Its code establishes that these mechanisms exist; it does not establish that the
authors added them after encountering our same failures.

SMAC identifies the Stalkers-versus-Zealots family as kiting tasks. `2s3z` has
mixed Stalkers and Zealots on both teams, adding role coordination. Shared unit
mechanics make a common improvement plausible. However, the successful
`2s_vs_1sc` runs also use Stalkers, and the observed failure behaviors differ.

The frozen final100 outcomes show:

| Map | Seed | Win | Timeout | Attack fraction | Move fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3s_vs_3z | 0 | 15% | 32% | .237 | .351 |
| 3s_vs_3z | 1 | 75% | 7% | .322 | .388 |
| 3s_vs_3z | 2 | 9% | 26% | .250 | .338 |
| 3s_vs_4z | 0 | 0% | 45% | .143 | .347 |
| 3s_vs_4z | 1 | 1% | 25% | .175 | .436 |
| 3s_vs_4z | 2 | 0% | 55% | .142 | .461 |
| 2s3z | 0 | 24% | 0% | .552 | .136 |
| 2s3z | 1 | 26% | 1% | .610 | .131 |
| 2s3z | 2 | 26% | 6% | .544 | .126 |

Action fractions include dead-agent no-ops. They cannot be interpreted as active
agent attack probabilities or per-unit-type behavior on heterogeneous maps.
`3s_vs_4z` also has legacy-minus-corrected diagnostic return gaps of 2.76, 1.53,
and 1.73, alongside substantial enemy shield regeneration. This is worth tracking
but is not proof of deliberate reward exploitation; the reward protocol remains
unchanged for comparison with the matrix and SMAC baselines.

The general hypothesis is improved valuation of delayed consequences and team
trades. It does not imply a single map-specific action rule or reward bonus.

### Revision after historical REINFORCE context

The user reports that `3s_vs_3z` and `3s_vs_4z` previously worked with REINFORCE,
while `2s3z` remained a partial success. Exact old run IDs, budgets, and seed
aggregates for the first two maps were not recovered from the cached audit, so
their historical performance is user-provided context, not a new numerical
comparison. Recovered c32 `2s3z` final128 scores at 50k are 52.34%, 34.38%, and
35.16% (mean 40.63%), versus 24%, 26%, and 26% in the current final100 matrix.
The configurations and seed sets differ.

Source inspection at `c86451e^` confirms that the pre-PPO branch already used
imagined-root bootstraps in replay value training (`training/learner.py:634–674`)
and stopped critic input features (`models/ctde.py:468–469`). Thus the two proposed
mechanisms address limitations shared with that REINFORCE implementation; their
absence alone is not an explanation for a PPO-era regression. Interaction with
the newer optimization remains possible.

The PPO migration also replaced percentile-return scaling with batch-standardized
advantages, increased the nominal entropy coefficient from .0003 to .01 in the
maintained defaults, introduced five actor/critic epochs and new optimizer
settings, and removed the explicit slow-critic prediction regularizer. These
source-default differences are not confirmed settings of the user's earlier
successful Stalker runs. Entropy coefficients are not directly comparable across
the changed advantage scales, and changing the EMA rate does not restore the
removed regularization loss.

Prioritize a controlled actor-learning comparison alongside factual targets.
Keep world/critic update counts, replay, BPTT2, collection, evaluation, and other
settings fixed when testing actor update intensity or the legacy actor objective.
Simply setting `ppo.epochs=1` also reduces critic updates and would conflate the
two effects. The representation auxiliary remains a separate hypothesis for
improving long-term control, not a demonstrated restoration of lost capability.

## Implemented options

Apply one optional config profile **after** `smac_vector ma_jepa`, retaining the
existing BPTT2, prefill, learning-rate, collection, budget, and evaluation overrides.
These overlays do not enable recurrence by themselves.

| Profile | Factual critic target | Representation auxiliary |
| --- | --- | --- |
| No extra profile | Existing imagined-root bootstrap | None |
| `factual_value` | Recorded successor target values with bounded joint V-trace | None |
| `factual_value_representation` | Same factual target | Local value head, scale .1 |

The representation scale is a starting candidate, not a tuned or demonstrated
optimum. Both options default off in the maintained profile. They require **fresh
replay**, because old replay has no stored collection probabilities. Missing
probabilities fail explicitly; no synthetic unit-ratio fallback is used.

### Target and probability contract

- The collector records `behavior_logprob[t]` for the actual action `a[t]` from
  the actual legal policy including collection mixing. This metadata never enters
  the encoder's observation space.
- Reconstructed current-policy log probabilities use recorded legal masks.
  Per-agent log ratios are summed across acting team members; dead/absent slots
  contribute an identity factor. No collection mixing is added to the target
  learned policy.
- Transition t receives real `reward[t+1]`. Target values come from factual
  successor features, never an imagined root return. Rho and c default to 1,
  clipped in log space; the existing replay lambda .95 is retained.
- A lambda V-trace correction replaces DMAWM's exact cumulative occupancy-weight
  formulation. This is an explicitly different, bounded off-policy design. It is
  not claimed to be unbiased when latent representations are changing.
- Terminal rewards are retained, truncations bootstrap once, resets and absent
  roster slots cannot bridge traces, and individual death does not cut team return.
- Effective sample size and clipping metrics are logged under the existing PPO
  critic namespace so ineffective replay weighting can be detected.

### Representation contract

The extra local value head is part of the world optimizer. Its inputs are the
current encoder/history features, and its targets are stopped factual V-trace
returns. It supplies a gradient to local representation modules. The central
critic and PPO actor keep their normal detached-feature paths. The auxiliary
head is not used at execution and adds no centralized information to the actor.

The real replay critic loss remains .3. Source-world losses and the actor's
imagined PPO objective are unchanged. The representation option changes compute
cost and may change representation drift; performance must be measured.

## Initial checks and next comparison

Focused checks cover source-action/arrival-reward alignment, clipped joint ratios,
dead-team return continuation, terminal/reset/truncation boundaries, independence
from imagined root returns, actual mixture probabilities, stopped targets, and
nonzero feature gradients through the auxiliary.

The integration check runs the complete tiny learner with BPTT2 on CPU, including
world and PPO updates for both options. Validation passed: 24 focused return,
replay-value, and PPO checks, plus both complete BPTT2 learner cases. Ruff and
`git diff --check` also passed. These checks establish implementation correctness
within their coverage, not a performance benefit. Fresh-policy final evaluations on real maps
are still required. Compare both treatments with the reference across `2s3z`,
`3s_vs_3z`, and `3s_vs_4z`, with `MMM` or `3m` as a retention check. Use multiple
seeds and fixed final100 scores rather than selecting a best checkpoint or seed.

Primary references:
- DMAWM: https://github.com/DiXue98/DMAWM/blob/2acaaeb82805b55d275e3ce08ac8f713ec9afbb2/world_model.py#L440-L530
- SMAC scenarios: https://github.com/oxwhirl/smac/blob/master/docs/smac.md
- V-trace: https://arxiv.org/abs/1802.01561
