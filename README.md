# DreaMARL

DreaMARL is a first-party, decoder-free visual model-based reinforcement
learning implementation. Its locked single-agent configuration combines a
categorical stochastic state, a causal Transformer, EMA-target
joint-embedding prediction, and DreamerV3 actor-critic semantics. The maintained
MARL path preserves exact single-agent behavior at `A=1`. For `A>1`, it adds an
explicit team axis, parameter sharing, and synchronized independent local
imagination. B0 execution is strictly decentralized: each actor, critic, and
world-model transition receives only the focal agent's observation/action
history. B1 adds a training-only agent-axis JEPA. B2 retains the observation-local
actor and transition model but trains fast and slow centralized critics from the
focal local state plus an explicit JEPA-derived active-team belief.

The maintained performance baseline is B0. B1 and B2 are retained as
reproducible negative experimental stages: their representation and information
paths were validated, but neither improved control on the two-seed Externality
Mushrooms evaluation, and B2 underperformed B1 and B0. New mechanisms must branch
from B0 unless new evidence overturns that result.

The current architecture is documented in
[`docs/dreamarl/ARCHITECTURE.md`](docs/dreamarl/ARCHITECTURE.md). The empirical
single-agent lock-in is recorded in
[`docs/dreamarl/VISUAL_LOCKIN.md`](docs/dreamarl/VISUAL_LOCKIN.md).

## Repository Layout

```text
src/dreamarl/             maintained first-party algorithm
src/dreamarl/marl/        minimal shared-local agent-axis core
src/dreamarl/ablations/   retained scientific controls
src/dreamarl/baselines/   launch and evaluation adapters only
external/dreamerv3/       pinned official DreamerV3 source
external/dreamer-cdp/     pinned official Dreamer-CDP source
external/nedreamer/       pinned official NE-Dreamer source
external/marie/           pinned paper-era official MARIE source
configs/dreamerv3/        benchmark protocol manifests
tests/                    maintained algorithm and benchmark tests
```

The external repositories are immutable benchmark references. DreaMARL does
not import their model implementations. It uses the pinned DreamerV3 checkout
for Embodied infrastructure and as a numerical reference in parity tests.
Detailed attribution and source revisions are in
[`docs/dreamarl/PROVENANCE.md`](docs/dreamarl/PROVENANCE.md).

## Setup

```bash
git submodule update --init --recursive
uv sync --python 3.11 --extra dev --extra meltingpot
uv run dreamarl-setup-dreamerv3 --accelerator cuda12
```

Use `--accelerator cpu` for a local installation smoke test. Dreamer-CDP and
NE-Dreamer use separate environments and are installed only when their
benchmarks are needed:

```bash
uv run dreamarl-setup-dreamer-cdp --accelerator cuda12
uv run dreamarl-setup-nedreamer
```

## Train And Evaluate

Run the locked model on visual DMC:

```bash
uv run dreamarl-train-dreamarl \
  --task dmc_walker_walk \
  --num-agents 1 \
  --seed 0 \
  --total-env-steps 500000 \
  --wandb-project dreamarl \
  --wandb-entity YOUR_ENTITY
```

Run DreaMARL on Melting Pot:

```bash
uv run dreamarl-train-dreamarl \
  --task meltingpot_externality_mushrooms__dense \
  --num-agents 5 \
  --seed 0 \
  --total-env-steps 50000 \
  --wandb-project dreamarl \
  --wandb-entity YOUR_ENTITY
```

Select the first agent-axis JEPA stage with `--marl-stage b1`. This adds the
whole-agent-masked prediction of the complete EMA team-slot representation at
the current timestep and predicts the next EMA team representation from the
masked current team plus the aligned joint replay action. The B1 input is
stop-gradient, preserving the local single-agent world model. Balanced matching against mean-centered, agent-relative
EMA content, explicit hidden-agent coverage, and slot anti-collapse regularization anchor the learned team
coordinates. The EMA team teacher is training-only; B0's decentralized actor
and local imagination graph are retained.

Select `--marl-stage b2` for strict centralized-training/decentralized-execution
control. At every replay and imagined state, B2 predicts each active agent's EMA
observation embedding from its causal local world state, summarizes the complete
team into eight 256-wide slots, and concatenates the flattened 2048-wide belief
with the focal 10240-wide state for both value heads. The actor still receives
only the focal local state. Critic gradients stop at both inputs; the team belief
is trained directly against the full-team EMA slots and by B1's aligned
joint-action transition objective.

Melting Pot seeds are supplied when constructing the underlying Lab2D
substrate. Shimmy 2.0.1 explicitly ignores `reset(seed)`, and the pinned Lab2D
backend can produce different observations for two environments given the same
construction seed and identical actions. A seed therefore controls the Lab2D
seed stream but does not guarantee a bitwise-identical trajectory. Manifests
record this distinction; reproducible comparisons still require multiple runs
and retained evaluation artifacts.

Fixed evaluation restores the latest complete checkpoint, performs no
checkpoint search, and does not add transitions to training replay:

```bash
uv run dreamarl-eval-dreamarl runs/dreamarl/<experiment> \
  --episodes 20 \
  --eval-seed 10000
```

Generate protocol-matched aggregate plots from completed runs:

```bash
uv run dreamarl-plot-dreamarl-paper \
  runs/dreamarl \
  --output-dir runs/paper_plots
```

## Official Baselines

The benchmark commands execute the pinned upstream implementations as isolated
processes. Their launch manifests record source revisions, seeds, budgets, and
normalized artifacts.

```bash
uv run dreamarl-train-dmc-dreamerv3 --help
uv run dreamarl-train-dmc-dreamer-cdp --help
uv run dreamarl-train-dmc-nedreamer --help
MARIE_PYTHON=/path/to/python3.10 uv run dreamarl-train-marie --help
```

MARIE uses an isolated legacy environment and StarCraft II 4.10. Its exact
source, runtime, SMAC protocol, and promotion gate are documented in
[`docs/dreamarl/MARIE_REPRODUCTION.md`](docs/dreamarl/MARIE_REPRODUCTION.md).

## Tests

Algorithm tests need the pinned DreamerV3 source because it provides Embodied
and the numerical reference modules:

```bash
PYTHONPATH=external/dreamerv3:src \
  .venv/bin/python -m pytest -q
```

The maintained tests cover public configuration, explicit agent-axis
transformations, exact `A=1` parity, shared-local multi-agent updates, causal and
recurrent Transformer equivalence, representation losses, actor-critic
training, environment semantics, fixed evaluation, artifact normalization, and
isolated benchmark launchers. One-off diagnostics and removed mechanisms are
not part of this branch.
