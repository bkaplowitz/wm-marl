# SMAC Reproducibility Protocol

## Environment

DreaMARL-CTDE uses SMAC v1 with StarCraft II 4.10. The pinned SMAC revision is:

```text
d6aab33f76abc3849c50463a8592a84f59a5ef84
```

The environment profile fixes:

- difficulty `7`;
- continuing episodes;
- native shared SMAC reward for learning;
- fixed agent roster slots;
- legal-action masks at collection and in imagination;
- explicit controllable liveness for dead units.

Dead agents remain present in the fixed roster and may select only no-op. The
action mask controls execution, while liveness remains a prediction target.

## Installation

Initialize the pinned sources and create the project runtime:

```bash
git submodule update --init --recursive
uv sync --python 3.11 --extra dev --extra smac --extra cuda12
```

Omit `--extra cuda12` for CPU-only environment checks. Obtain the Linux
StarCraft II 4.10 package separately and expose it to SMAC:

```bash
export SC2PATH=/path/to/StarCraftII
```

Before a training run, verify that the selected Python can import SMAC, find
StarCraft II, reset each requested map, report three allied agents, and perform
one legal environment step. The repository does not bundle or relicense the
StarCraft II assets.

## Maps

The initial comparison panel uses:

| Map | Allied agents | Purpose |
| --- | ---: | --- |
| `3m` | 3 | Basic homogeneous coordination and policy discovery |
| `3s_vs_4z` | 3 | Asymmetric hard-map coordination, target selection, and survival |

Budgets are counted in real environment transitions. Agent transitions are
reported separately and equal three times the environment-transition count on
both maps.

## Algorithms

The public CTDE configurations are selected directly:

```bash
SC2PATH=/path/to/StarCraftII \
uv run dreamarl-train-dreamarl \
  --python .venv/bin/python \
  --task smac_3m \
  --num-agents 3 \
  --algorithm ctde-one-step \
  --seed 0 \
  --total-env-steps 100000 \
  --eval-interval 5000 \
  --eval-episodes 32 \
  --eval-envs 1 \
  --eval-seed-offset 50000
```

Replace `smac_3m` with `smac_3s_vs_4z` for the hard map. Replace
`ctde-one-step` with `ctde-two-step` for the bounded self-fed treatment. All
other launch arguments must remain matched in a controlled comparison.

Use `--algorithm local` with the same three-agent SMAC task for the independent
local-model control.

## Periodic fixed evaluation

Every 5,000 environment transitions, the current policy is evaluated for 32
battles. Evaluation is deterministic, uses one environment worker, does not add
experience to replay, does not select a checkpoint, and preserves training
policy state.

For training seed `s`, the evaluation worker index begins at `50,000`. SMAC
therefore receives held-out seed `s + 50,000`; with more than one evaluation
worker, worker `k` receives `s + 50,000 + k`. The same offset is reused at every
evaluation point so learning curves compare policies on a fixed battle stream.

An explicit final evaluation of the latest complete checkpoint can be run with:

```bash
SC2PATH=/path/to/StarCraftII \
uv run dreamarl-eval-dreamarl runs/dreamarl/<experiment> \
  --episodes 32 \
  --envs 1 \
  --eval-seed 50000 \
  --policy-mode deterministic
```

For training seed `s`, set `--eval-seed` to `s + 50000` to match the periodic
protocol.

## Reported outcomes

The behavioral decision metrics are:

- battle wins and deterministic win rate;
- enemy deaths, survivors, and damage;
- ally deaths and survival;
- timeout frequency;
- native legacy SMAC return;
- corrected combat return and its damage/death components;
- no-op, stop, move, attack, target-selection, and target-switch counts.

The native benchmark reward remains the optimization target. Corrected damage,
death, survival, and action metrics are diagnostics and do not shape reward.
World-model cosine, posterior-interface error, critic calibration, and auxiliary
head losses verify that learning is active, but they do not substitute for
control outcomes.

## Comparison requirements

A result table must state:

- map and StarCraft II version;
- algorithm name and CTDE version;
- real environment-step budget;
- training and evaluation seeds;
- number of completed seeds;
- fixed-evaluation episode count and interval;
- whether the reported curve is periodic fixed evaluation or training return.

Do not compare a selected peak, incomplete run, or partial external reproduction
with a completed fixed-evaluation aggregate.
