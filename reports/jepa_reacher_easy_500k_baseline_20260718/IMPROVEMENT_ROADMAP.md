# JEPA Improvement Roadmap

This roadmap starts from the frozen five-seed baseline and the cleaned
canonical implementation. It deliberately avoids broad sweeps, task-specific
rules, and simultaneous bundles of unrelated changes.

The ordering is important: correctness and protocol clarity come before
capacity scaling or objective changes.

## Goal 1: Correct the replay boundary contract

### Problem

The current replay field `dones` is used both to stop sequence propagation and
to zero Bellman continuation. DMC truncations can require the first behavior
without the second.

### Change

Represent three meanings explicitly:

```text
is_last
is_terminal
cut
```

Use:

- `is_last OR cut` for sequence, attention, and target-crossing masks;
- `is_terminal` for continuation supervision and real lambda returns;
- `cut` only for collector-imposed resets.

### Why first

This is a general correctness fix. Critic, continuation, replay, and schedule
experiments are hard to interpret while one target carries incompatible
semantics.

### Verification

1. Unit-test terminal, truncation, and forced-reset transitions separately.
2. Prove legacy replay/checkpoint loading has an explicit migration path.
3. Run short DMC pilots and verify continuation targets at time limits.
4. Run paired 200k Reacher/easy seeds 1 and 2.

### Promotion criterion

No new instability, improved or equal 200k mean and lower-tail return, and no
task-specific condition.

## Goal 2: Make the existing schedule budget-relative

### Problem

The 500k schedule uses five absolute transition boundaries. Shorter presets
inherit those values and therefore execute a different curriculum.

### Change

Express the current schedule as progress through the declared learning budget:

| Behavior | 500k-equivalent progress |
| --- | ---: |
| End recent WM replay | 10% |
| Change actor cadence | 10% |
| Freeze encoder | about 20% |
| Expand value clip | about 30-50% |
| Enable reset-aligned starts | about 40% |

The 500k resolved manifest must preserve the current phase boundaries exactly.
Only 100k and 200k behavior changes initially.

### Why second

This improves the "one algorithm across budgets" claim without adding a
mechanism or changing the successful 500k reference.

### Verification

1. Manifest test for exact 500k equivalence.
2. 100k and 200k paired runs against the old absolute schedule.
3. Compare return at budget, area under the learning curve, failure rate, and
   seed spread.

### Promotion criterion

Better short-budget performance with identical 500k resolution and fewer
public schedule knobs.

## Goal 3: Smooth the actor standard-deviation bound

### Problem

Hard clipping `log_std` creates zero-gradient regions outside the configured
range.

### Change

Map an unconstrained actor output smoothly into
`[log(0.1), log(1.0)]`. Preserve:

- actor size;
- action distribution family;
- exploration range;
- entropy coefficient;
- deterministic evaluation semantics.

### Verification

1. Distribution and gradient unit tests at extreme logits.
2. Resume paired seeds from a pre-100k snapshot if parameter migration is
   defined; otherwise run fresh 200k pilots.
3. Track action saturation, `log_std` boundary occupancy, actor KL, and return.

### Promotion criterion

Equal or better mean and lower tail with fewer saturated or gradient-dead
standard deviations.

## Goal 4: Simplify the critic stabilization stack

### Problem

EMA targets, slow-value regularization, and real-replay critic loss may partly
duplicate one another.

### Experiment

After Goal 1, run one knockout at a time:

1. canonical stack;
2. remove slow-value regularization;
3. only if needed, test a reduced real-replay coefficient.

Do not remove the EMA critic first; it supplies the actor baseline and lambda
bootstrap as well as critic targets.

### Verification

Use identical seeds and initial replay. Compare:

- real-replay value calibration;
- target drift;
- actor update KL;
- return at 100k and 200k;
- failure rate and CVaR10.

### Promotion criterion

Remove a term only if performance and lower-tail stability are preserved. A
tie favors the simpler objective.

## Goal 5: Isolate the five schedule rules

This is a simplification study, not a sweep. Use branch-from-snapshot tests
when a rule activates late enough that earlier training is identical.

Recommended order:

1. reset-aligned starts;
2. value-clip expansion;
3. encoder freeze;
4. recent replay;
5. actor cadence.

The actor cadence is tested last because development evidence for late policy
stability is strongest. Recent replay is tested from fresh runs because it
changes early world-model training. Late rules may use exact phase-boundary
snapshots to avoid repeating unchanged computation.

### Promotion criterion

Every retained schedule rule must improve at least one predeclared metric
without materially harming the others. A rule with no measurable contribution
is removed.

## Goal 6: Improve sample efficiency without adding rules

Only after Goals 1-5 should update allocation be reconsidered.

The first candidate is update redistribution, not more random bootstrap data
or a larger network:

- keep the 5,120-transition reset-rich bootstrap;
- keep one WM update per new real transition;
- let critic updates lead actor updates early;
- preserve or reduce the total actor-update count.

This asks whether better value calibration can produce earlier policy gains
without increasing real data or architectural complexity.

### Metrics

- return at 50k, 100k, 150k, 200k, and 500k;
- area under return-versus-training-steps;
- final deterministic mean;
- seed-level and episode-level lower tails;
- wall-clock and update counts.

## Goal 7: Scale only when harder tasks show a capacity limit

Do not enlarge the Reacher model merely because some seeds fail. The current
network already demonstrates near-solved behavior.

Model scaling becomes justified when multiple harder tasks show:

- persistent underfit in held-out latent/reward prediction;
- actor and critic optimization remain stable;
- additional data and update redistribution do not close the gap;
- the same bottleneck appears across seeds.

If scaling is needed, change one dimension at a time and report parameter and
compute scaling explicitly.

## Validation Funnel

Every algorithmic change follows the same funnel:

1. static checks and unit tests;
2. exact equivalence for unaffected paths;
3. cheap paired diagnostic or snapshot continuation;
4. fresh 200k runs on two predeclared seeds;
5. fresh 500k five-seed Reacher/easy confirmation;
6. at least three additional DMC tasks;
7. comparison against the fixed in-repo DreamerV3 protocol.

No candidate is promoted from a best seed, a selected checkpoint, or an
evaluation-driven training decision.

## Immediate Priority

The next implementation branch should contain only Goal 1. It is the clearest
general correctness improvement and establishes a trustworthy base for every
later simplification or performance claim.
