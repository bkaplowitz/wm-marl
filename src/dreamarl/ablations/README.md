# DreaMARL Ablations

This package contains controls that are intentionally excluded from canonical
DreaMARL: the DreamerV3 RSSM and decoder, compact ViTs, V-JEPA 2.1 and
LeWorldModel representation recipes, alternative mask topologies, online/MSE
targets, and partial JEPA objectives.

The production command has no architecture switches. Run a control explicitly:

```bash
python -m dreamarl.ablations.train \
  --task dmc_walker_walk \
  --num-agents 1 \
  --temporal-model rssm \
  --world-model-objective reconstruction
```

Ablation launches use their own manifest and evaluator. They share
environment, replay, actor-critic, and logging infrastructure with the canonical
algorithm so comparisons remain controlled.

The paper-recipe launchers are explicit:

```bash
python -m dreamarl.ablations.train \
  --task dmc_reacher_easy --num-agents 1 \
  --representation-recipe vjepa21

python -m dreamarl.ablations.train \
  --task dmc_reacher_easy --num-agents 1 \
  --representation-recipe leworldmodel
```

`vjepa21` uses the official 256px ViT-B/16 dimensions, four hierarchical
outputs, 3-axis RoPE, 8-small/2-large masks, tubelet-pair-consistent 16-frame
masks, minimum-count truncation, fixed 0.99925 EMA, and masked plus
distance-weighted visible L1. The causal controller necessarily applies the
visual Transformer frame by frame; it does not introduce V-JEPA's
bidirectional video attention into online action selection.

`leworldmodel` uses the official unmasked 224px ViT-Tiny/14 dimensions and CLS
output, differentiable online MSE target, and per-timestep SIGReg with 1024
projections at weight 0.09. The DreaMARL causal dynamics and actor-critic stay
fixed; these runs are controlled integrations of representation recipes, not
reproductions of the papers' datasets or control/planning systems.
