# DreaMARL

This branch contains one supported algorithm: the final DreaMARL multi-agent
architecture for SMAC. There are no alternate algorithm profiles or ablation
entry points in this branch.

DreaMARL is a decoder-free, joint-embedding predictive world model using:

- per-agent categorical latent dynamics and causal Transformers;
- stopped EMA representation targets instead of observation reconstruction;
- centralized joint-action-conditioned JEPA training;
- TBv2 teammate-action belief;
- identity-preserving future peer-plan attention;
- a centralized attention critic;
- strictly decentralized shared-parameter actors.

The complete architecture and handoff are documented in
[docs/final_dreamarl.md](docs/final_dreamarl.md).

## Install

```bash
git submodule update --init --recursive
uv sync --python 3.11 --extra dev --extra smac --extra cuda12
```

SMAC requires StarCraft II 4.10 and `SC2PATH` pointing to its installation.

## Train

The launcher has no architecture selector; it always launches final DreaMARL:

```bash
SC2PATH=/path/to/StarCraftII \
uv run dreamarl-train-dreamarl \
  --python .venv/bin/python \
  --task smac_8m \
  --num-agents 8 \
  --seed 234 \
  --total-env-steps 50000 \
  --eval-interval 1000 \
  --eval-episodes 16 \
  --eval-envs 4 \
  --eval-seed-offset 50000 \
  --wandb-project YOUR_PROJECT \
  --wandb-entity YOUR_ENTITY
```

Internally this resolves only:

```text
smac_vector + dreamarl_final
```

The canonical settings are replay x4, actor/critic optimizer start at step
3,000, actor width 512, actor LR `1e-5`, entropy scale `6e-4`, balanced legal
action masking, action-counterfactual scale 0, and multi-step horizons
`[1, 2, 4, 8]`.

## Evaluate

Evaluate the latest complete checkpoint without checkpoint selection:

```bash
uv run dreamarl-eval-dreamarl runs/dreamarl/<experiment> \
  --episodes 128 \
  --envs 4 \
  --eval-seed 100000 \
  --policy-mode deterministic
```

The evaluator rejects manifests produced by historical algorithm branches.

## Repository layout

```text
src/dreamarl/agent.py             local latent world model and behavior
src/dreamarl/marl/                agent-axis and centralized training
src/dreamarl/models/              predictive and role-aware modules
src/dreamarl/training/            losses, imagination, and optimization
src/dreamarl/envs/smac.py         SMAC environment boundary
src/dreamarl/configs.yaml         final model plus debug profile
docs/final_dreamarl.md            architecture and continuation handoff
tests/                             semantic and integration gates
```

## Tests

```bash
PYTHONPATH=external/dreamerv3:src \
  .venv/bin/python -m pytest -q
```

The essential gates cover exact final-profile resolution, strict decentralized
execution, peer-plan causality and identity handling, optimizer ownership,
warmstart state freezing, replay alignment, action masks, SMAC semantics, and
fixed held-out evaluation.

## License

DreaMARL is released under the MIT [LICENSE](LICENSE). Pinned external sources
retain their own licenses; see [NOTICE.md](NOTICE.md) and
[docs/provenance.md](docs/provenance.md).
