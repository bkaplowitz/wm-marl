from dreamarl.ablations.eval import _model_arguments
from dreamarl.ablations.backend import world_model_backend


def test_ablation_evaluation_reconstructs_model_overrides() -> None:
    manifest = {
        "world_model": "parallel_transformer",
        "world_model_objective": "embedding",
        "embedding_target": "online",
        "embedding_loss": "mse",
        "posterior_jepa": True,
        "dynamics_jepa": True,
        "spatial_jepa": True,
        "spatial_mask_ratio": 0.5,
        "spatial_mask_topology": "multiblock",
        "visual_encoder": "vit",
        "sigreg": True,
    }
    arguments = _model_arguments(manifest)

    def value(flag):
        return arguments[arguments.index(flag) + 1]

    assert value("--agent.embedding_target") == "online"
    assert value("--agent.embedding_loss") == "mse"
    assert value("--agent.spatial_jepa.topology") == "multiblock"
    assert value("--agent.enc.typ") == "vit"


def test_rssm_control_remains_available() -> None:
    assert world_model_backend("rssm").name == "rssm"
