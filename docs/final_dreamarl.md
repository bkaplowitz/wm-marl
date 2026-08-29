# Final DreaMARL handoff

## Supported entry point

Use the public algorithm name `final-dreamarl`. It resolves to exactly three
configuration profiles:

```text
smac_vector + dreamarl_final
```

It fixes the validated architecture directly: balanced legal-action masking,
TBv2, identity-aware multi-step JEPA with action scale 0, replay x4, 3k
actor/critic warmstart, actor width 512, actor LR `1e-5`, and entropy scale
`6e-4`. No alternate algorithm profiles are included.

## Architecture

### Executable path

Each agent executes from its own observation/action history:

```text
local observation history
  -> shared encoder and categorical latent dynamics
  -> TBv2 prediction of current teammate actions q0
  -> stopped teammate-belief actor adapter
  -> shared 512-wide actor
  -> local legal action
```

Parameters are shared, recurrent state is per agent, and no global state or
peer observation is available at execution. TBv2 predicts peer behavior from
the focal agent's local state; this preserves decentralized execution.

Policy synchronization contains:

```text
enc, dyn, pol, ctde_teammate_belief, ctde_teammate_actor
```

The centralized critic, joint simulator, future-plan decoder, and multi-step
predictor are training-only.

### Centralized world learning

The joint simulator consumes synchronized stopped local posterior states,
aligned joint actions, roster/liveness masks, and causal replay history. It
predicts each agent's next stopped EMA local observation embedding, reward,
continuation, legal actions, and liveness.

The multi-step JEPA heads predict horizons `[1, 2, 4, 8]`. A causal shared GRU
predicts each peer's future action marginals from the focal local root and the
focal planned action prefix. Those future plans condition the multi-step JEPA
loss. Peer identity is preserved by shared projection plus identity-aware
attention rather than mean pooling. The future-plan path never receives peer
future actions as forward inputs; factual peer actions are stopped labels only.

The action-counterfactual JEPA loss is disabled (`scale = 0`). This is the
validated action-scale-0 configuration.

### Critic and optimization

The centralized attention critic sees synchronized stopped local posterior
states during training. Actor execution remains decentralized.

| Setting | Value |
| --- | ---: |
| Replay sequence | 192 burn-in + 64 optimized |
| Batch | 16 sequences |
| Replay sampling | recent world / uniform behavior dual view |
| Replay capacity | 250,000 |
| Train ratio | 1024 |
| First eligible learner step | approximately environment step 1,279 |
| Actor/critic optimizer start | environment step 3,000 |
| Local/joint world LR | `4e-5` |
| Actor LR | `1e-5` |
| Actor width/layers | 512 / 3 |
| Value width/layers | 1024 / 3 |
| Imagination horizon | 15 |
| Action entropy scale | `6e-4` |
| Joint token width | 256 |
| Joint temporal layers | 12 |
| Peer-plan attention | width 256, 4 heads |

The warmstart freezes actor/critic parameters, optimizer moments, optimizer
steps, behavior normalizers, and slow critic until step 3,000. Local world,
joint world, slow encoder, TBv2, future-plan, and JEPA learning continue.

## Reproduction

```bash
git submodule update --init --recursive
uv sync --python 3.11 --extra dev --extra smac --extra cuda12

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
  --eval-seed-offset 50000
```

Evaluate the latest complete checkpoint without checkpoint selection:

```bash
uv run dreamarl-eval-dreamarl runs/dreamarl/<experiment> \
  --episodes 128 \
  --envs 4 \
  --eval-seed 100000 \
  --policy-mode deterministic
```

## Evidence and known limitation

The configuration is strong on homogeneous SMAC maps such as `3m` and `8m`.
The unresolved limitation is heterogeneous execution, especially `2s3z`.
Matched held-out diagnostics localize that failure to target allocation and
temporal commitment rather than basic world-model fit: the failing policy
switches legal targets too often, achieves lower focus-fire agreement, and
selects the weakest legal enemy less reliably than the winning baseline.

The next continuation should therefore investigate a learned,
role-conditioned commitment/readout in the actor. It should not replace the
world model, add a VQ-VAE by default, or infer that larger actor/critic networks
are required. Any such continuation should preserve exact initialization,
strict decentralized execution, legal-action masking, and the fixed held-out
evaluation protocol.

## Development boundary

This is a single-algorithm handoff. New work should begin from this branch and
change one mechanism at a time.
