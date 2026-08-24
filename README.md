# DreaMARL

DreaMARL is a decoder-free, joint-embedding predictive world model for
reinforcement learning. The local model combines a categorical latent state,
a causal Transformer, EMA-target prediction, and Dreamer-style latent
imagination. It learns observation representations without reconstructing
pixels.

The same local model is used in three public configurations:

- **DreaMARL** is the single-agent algorithm.
- **Independent DreaMARL** shares the local model and actor across agents while
  keeping every transition, value, and action observation-local.
- **DreaMARL-CTDE** adds a training-only joint JEPA simulator and centralized
  attention critic. The deployed actor remains strictly local.

DreaMARL-CTDE has two controlled predictive objectives. The **one-step** model
learns factual joint transitions. The **two-step** model retains that objective
and adds bounded self-fed two-step supervision. These names correspond to
manifest versions `1.1` and `2`. Older experiment manifests may use the labels
`B0`, `B1`, or `B2`; those are historical experiment identifiers, not the
public architecture names.

The implementation and current empirical evidence are documented in:

- [Architecture](docs/architecture.md)
- [Single-agent visual results](docs/results/single_agent.md)
- [Provenance and attribution](docs/provenance.md)
- [SMAC protocol](docs/reproducibility/smac.md)
- [MARIE reproduction notes](docs/reproducibility/marie.md)

## Architecture at a glance

```text
decentralized execution

local observation_i -> local encoder/posterior_i -> shared actor -> action_i

centralized training only

synchronized local states + joint action
                    |
                    v
       joint action-conditioned JEPA
                    |
       predicted next local embedding_i
                    |
       existing local posterior interface
                    |
           next executable state_i

synchronized stopped local states -> attention critic -> value_i
```

The joint simulator predicts the representation of what each agent will
observe next; it does not manufacture a privileged actor state. During real
execution, policy synchronization contains only the local encoder, local
dynamics, and actor. Neither the joint simulator nor centralized critic is
available to the actor.

## Repository layout

```text
src/dreamarl/              first-party algorithm
src/dreamarl/marl/         reversible agent-axis and CTDE integration
src/dreamarl/models/       local and joint predictive modules
src/dreamarl/training/     replay, objectives, imagination, and optimization
src/dreamarl/envs/         DMC and SMAC adapters
src/dreamarl/ablations/    isolated scientific controls
src/dreamarl/baselines/    launch and artifact adapters for comparisons
external/dreamerv3/        pinned DreamerV3 reference and runtime
external/marie/            pinned MARIE reference implementation
external/dreamer-cdp/      pinned Dreamer-CDP comparison implementation
external/nedreamer/        pinned NE-Dreamer comparison implementation
docs/                      architecture, results, and reproducibility notes
tests/                     algorithm, environment, and launcher tests
```

External repositories are pinned comparison sources. DreaMARL does not import
their learned model implementations. The DreamerV3 checkout supplies the
Embodied runtime and numerical references used by parity tests.

## Installation

Clone the repository with its pinned references and create the main Python
environment:

```bash
git submodule update --init --recursive
uv sync --python 3.11 --extra dev --extra dmc --extra smac --extra cuda12
```

Omit `--extra cuda12` for a CPU-only installation smoke test. SMAC experiments
also require StarCraft II 4.10, with `SC2PATH` pointing to its installation.
The pinned SMAC source is installed by the `smac` extra. DreamerV3, MARIE,
Dreamer-CDP, and NE-Dreamer use isolated comparison environments that are only
created when their runs are needed. For example:

```bash
uv run dreamarl-setup-dreamerv3 --accelerator cuda12
```

## Single-agent training

Train the maintained visual model on DMC:

```bash
uv run dreamarl-train-dreamarl \
  --python .venv/bin/python \
  --task dmc_walker_walk \
  --num-agents 1 \
  --algorithm local \
  --seed 0 \
  --total-env-steps 500000 \
  --wandb-project dreamarl \
  --wandb-entity YOUR_ENTITY
```

The maintained visual configuration uses 64 x 64 RGB observations, the compact
convolutional encoder, two-layer causal Transformer, categorical stochastic
state, posterior and dynamics JEPA losses, 50% fixed-count spatial masking,
and SIGReg.

## Independent multi-agent control

The independent control applies the shared local learner independently to each
agent while preserving synchronized team trajectories:

```bash
SC2PATH=/path/to/StarCraftII \
uv run dreamarl-train-dreamarl \
  --python .venv/bin/python \
  --task smac_3m \
  --num-agents 3 \
  --algorithm local \
  --seed 0 \
  --total-env-steps 100000
```

The actor, critic, and local transition receive only the focal agent's history.
This configuration is the architectural control for DreaMARL-CTDE.

## DreaMARL-CTDE on SMAC

The one-step configuration is selected with `--algorithm ctde-one-step`:

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

The matched two-step treatment changes only the rollout objective:

```bash
SC2PATH=/path/to/StarCraftII \
uv run dreamarl-train-dreamarl \
  --python .venv/bin/python \
  --task smac_3m \
  --num-agents 3 \
  --algorithm ctde-two-step \
  --seed 0 \
  --total-env-steps 100000 \
  --eval-interval 5000 \
  --eval-episodes 32 \
  --eval-envs 1 \
  --eval-seed-offset 50000
```

Both variants train from scratch. They share the local execution model,
centralized critic, optimizer topology, replay protocol, and one-step joint
loss. The two-step variant self-feeds only a bounded set of valid replay
anchors and stops gradients across the first predicted transition.

The actor-stability treatment is selected with `--algorithm ctde-pcr`. It is
the one-step CTDE model with recent replay and a one-update-delayed functional
policy KL on an independent replay reference batch. The reference latents and
legality mask are stopped, so this term updates only the decentralized actor.

## Evaluation

Fixed evaluation restores the latest complete checkpoint, does not search for
the best checkpoint, and never writes evaluation experience to training replay:

```bash
uv run dreamarl-eval-dreamarl runs/dreamarl/<experiment> \
  --episodes 32 \
  --envs 1 \
  --eval-seed 50000 \
  --policy-mode deterministic
```

SMAC reports win rate, wins, enemy deaths, ally survival, timeout frequency,
legacy benchmark reward, and corrected combat diagnostics. Win rate and battle
outcomes—not predictive cosine alone—determine whether a control change is
useful.

Generate aggregate single-agent plots from completed artifacts with:

```bash
uv run dreamarl-plot-dreamarl-paper \
  runs/dreamarl \
  --output-dir runs/paper_plots
```

## Comparison tooling

The following commands execute pinned upstream implementations as isolated
processes and normalize their artifacts; they are not alternate DreaMARL model
families:

```bash
uv run dreamarl-train-dmc-dreamerv3 --help
uv run dreamarl-train-dmc-dreamer-cdp --help
uv run dreamarl-train-dmc-nedreamer --help
MARIE_PYTHON=/path/to/python3.10 uv run dreamarl-train-marie --help
```

Exact revisions and license boundaries are listed in
[provenance.md](docs/provenance.md) and [NOTICE.md](NOTICE.md).

## Tests

The main test command includes the pinned DreamerV3 source because it supplies
Embodied and reference modules:

```bash
PYTHONPATH=external/dreamerv3:src \
  .venv/bin/python -m pytest -q
```

The maintained tests cover single-agent parity, explicit agent-axis layouts,
strict policy information boundaries, replay burn-in, one-step and two-step
joint prediction, centralized-critic inputs, action masks and liveness, SMAC
semantics, fixed evaluation, and isolated comparison launchers.

## License and citation

DreaMARL is released under the MIT License. See [LICENSE](LICENSE),
[NOTICE.md](NOTICE.md), and [CITATION.cff](CITATION.cff).
