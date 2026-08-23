"""Validation for combinations exposed only by the ablation launcher."""


def validate(config) -> None:
    objective = str(config.objective)
    target = str(config.embedding_target)
    loss = str(config.embedding_loss)
    spatial = bool(config.spatial_jepa.enabled)
    topology = str(config.spatial_jepa.topology)
    if objective not in {"reconstruction", "embedding"}:
        raise ValueError(f"unknown objective: {objective}")
    if target not in {"ema", "online"}:
        raise ValueError(f"unknown embedding target: {target}")
    if loss not in {"cosine", "mse"}:
        raise ValueError(f"unknown embedding loss: {loss}")
    if target == "online" and spatial:
        raise ValueError("spatial prediction requires an EMA target")
    if topology not in {"bernoulli", "fixed_count", "multiblock"}:
        raise ValueError(f"unknown mask topology: {topology}")
