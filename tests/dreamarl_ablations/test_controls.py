from __future__ import annotations

import json
from pathlib import Path

import elements
import numpy as np
import ruamel.yaml as yaml

from dreamarl.marl.axes import TeamAxis
from dreamarl.ablations.algorithm import AblationAlgorithm
from dreamarl.ablations.config import AblationRunSpec
from dreamarl.ablations.contracts import verify_run_contract
from dreamarl.envs.single_agent import SingletonAgentEnv
from dreamarl.launcher import run_training
from dreamarl.main import _load_configs
from dreamarl.runtime import algorithm_root, repository_root
from dreamarl.training.common import predict
from dreamarl.marl.spaces import remove_agent_axis


def _spec(tmp_path: Path, **updates) -> AblationRunSpec:
    values = {
        "experiment_dir": tmp_path / "run",
        "task": "meltingpot_externality_mushrooms__dense",
        "seed": 7,
        "train_steps": 50_000,
        "num_agents": 5,
        "platform": "cpu",
        "python": Path("/usr/bin/python3"),
        "save_every_seconds": 1_800,
        "wandb_project": "world-marl",
        "wandb_entity": "osaze-obahor",
    }
    values.update(updates)
    return AblationRunSpec(**values)


def test_run_spec_exposes_one_maintained_algorithm(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    manifest = spec.to_dict()
    contract = verify_run_contract(spec)
    assert spec.configs == ["meltingpot_vision"]
    assert manifest["world_model"] == "parallel_transformer"
    assert manifest["replay_sampling"] == "uniform"
    assert manifest["algorithm_components"] == [
        "DreamerV3 convolutional encoder",
        "history-conditioned observation posterior",
        "causal Transformer temporal dynamics",
        (
            "decoder-free posterior, action-conditioned dynamics, and "
            "masked-spatial EMA-target cosine joint-embedding prediction"
        ),
        "SIGReg embedding anti-collapse regularization",
        "one-step recurrent replay context",
        "uniform replay",
    ]
    assert contract["policy_peer_access"] is False
    assert contract["execution"] == "decentralized"


def test_single_and_multi_agent_runs_share_one_contract(tmp_path: Path) -> None:
    singleton = verify_run_contract(_spec(tmp_path, num_agents=1))
    multi = verify_run_contract(_spec(tmp_path, num_agents=7))
    ignored = {"num_agents"}
    assert {key: value for key, value in singleton.items() if key not in ignored} == {
        key: value for key, value in multi.items() if key not in ignored
    }
    assert singleton["single_agent_status"] == (
        "same architecture and schedule with an identity agent-axis fold"
    )


def test_temporal_control_preserves_official_dreamerv3_configuration() -> None:
    loader = yaml.YAML(typ="safe")
    configs = _load_configs(algorithm_root() / "ablations" / "configs.yaml")
    maintained = configs["defaults"]["agent"]
    reference = loader.load(
        (
            repository_root() / "external" / "dreamerv3" / "dreamerv3" / "configs.yaml"
        ).read_text(encoding="utf-8")
    )["defaults"]["agent"]

    maintained = dict(maintained)
    reference = dict(reference)
    assert maintained.pop("ablation") is True
    assert maintained.pop("num_agents") == 1
    assert maintained.pop("objective") == "embedding"
    assert maintained.pop("embedding_target") == "ema"
    assert maintained.pop("embedding_loss") == "cosine"
    assert maintained.pop("posterior_jepa") is True
    assert maintained.pop("dynamics_jepa") is True
    assert maintained.pop("spatial_jepa") == {
        "enabled": True,
        "mask_ratio": 0.5,
        "topology": "fixed_count",
        "fill_value": 128,
    }
    assert maintained.pop("sigreg") == {
        "enabled": True,
        "knots": 17,
        "num_proj": 256,
        "aggregation": "pooled",
    }
    assert maintained.pop("target_encoder") == {"rate": 0.01, "every": 1}
    maintained_scales = dict(maintained["loss_scales"])
    assert maintained_scales.pop("posterior_jepa") == 2.0
    assert maintained_scales.pop("dynamics_jepa") == 2.0
    assert maintained_scales.pop("spatial_jepa") == 1.0
    assert maintained_scales.pop("sigreg") == 0.05
    maintained["loss_scales"] = maintained_scales
    maintained_dyn = maintained.pop("dyn")
    reference_dyn = reference.pop("dyn")
    maintained_enc = maintained.pop("enc")
    reference_enc = reference.pop("enc")
    assert maintained == reference

    assert maintained_enc["typ"] == reference_enc["typ"] == "simple"
    assert maintained_enc["simple"] == reference_enc["simple"]
    assert maintained_enc["vit"] == {
        "patch": 8,
        "model": 256,
        "layers": 6,
        "heads": 8,
        "ffup": 4,
        "token_dim": 64,
        "units": 1024,
        "act": "silu",
        "norm": "rms",
        "winit": "trunc_normal_in",
        "symlog": True,
    }

    assert maintained_dyn["typ"] == "parallel_transformer"
    assert maintained_dyn["rssm"] == reference_dyn["rssm"]
    transformer = maintained_dyn["parallel_transformer"]
    rssm = reference_dyn["rssm"]
    for key, value in rssm.items():
        assert transformer[key] == value
    assert (transformer["model"], transformer["layers"], transformer["heads"]) == (
        512,
        2,
        8,
    )
    assert transformer["context"] == 64
    assert transformer["posterior_context"] == "history"


def test_visual_dmc_profile_matches_official_dreamerv3_protocol() -> None:
    loader = yaml.YAML(typ="safe")
    maintained_configs = loader.load(
        (algorithm_root() / "configs.yaml").read_text(encoding="utf-8")
    )
    reference_configs = loader.load(
        (
            repository_root() / "external" / "dreamerv3" / "dreamerv3" / "configs.yaml"
        ).read_text(encoding="utf-8")
    )
    maintained = elements.Config(maintained_configs["defaults"]).update(
        maintained_configs["dmc_vision"]
    )
    reference = elements.Config(reference_configs["defaults"]).update(
        reference_configs["dmc_vision"]
    )

    for key in ("steps", "envs", "train_ratio"):
        assert maintained.run[key] == reference.run[key]
    maintained_dmc = dict(maintained.env.dmc)
    assert maintained_dmc.pop("use_seed") is True
    reference_dmc = dict(reference.env.dmc)
    reference_dmc.pop("use_seed", None)
    assert maintained_dmc == reference_dmc


def test_temporal_backend_is_an_explicit_isolated_override(tmp_path: Path) -> None:
    rssm = _spec(tmp_path, temporal_model="rssm")
    transformer = _spec(
        tmp_path,
        temporal_model="parallel_transformer",
        posterior_context="history",
    )
    rssm_typ = rssm.command.index("--agent.dyn.typ")
    transformer_typ = transformer.command.index("--agent.dyn.typ")
    assert rssm.command[rssm_typ + 1] == "rssm"
    assert transformer.command[transformer_typ + 1] == "parallel_transformer"
    assert "--agent.dyn.parallel_transformer.posterior_context" not in rssm.command
    posterior = transformer.command.index(
        "--agent.dyn.parallel_transformer.posterior_context"
    )
    assert transformer.command[posterior + 1] == "history"
    assert rssm.to_dict()["algorithm_components"][1:3] == [
        "history-conditioned observation posterior",
        "block-GRU RSSM temporal dynamics",
    ]


def test_encoder_and_mask_topology_are_explicit_isolated_overrides(
    tmp_path: Path,
) -> None:
    spec = _spec(
        tmp_path,
        visual_encoder="vit",
        spatial_mask_topology="multiblock",
    )

    def value(flag: str) -> str:
        return spec.command[spec.command.index(flag) + 1]

    assert value("--agent.enc.typ") == "vit"
    assert value("--agent.spatial_jepa.topology") == "multiblock"
    assert spec.to_dict()["visual_encoder"] == "vit"
    assert spec.to_dict()["spatial_mask_topology"] == "multiblock"


def test_faithful_vjepa_variant_records_grid_and_resolution(tmp_path: Path) -> None:
    spec = _spec(
        tmp_path,
        task="dmc_reacher_easy",
        num_agents=1,
        visual_encoder="vjepa",
        spatial_mask_topology="vjepa_multiblock",
    )

    def values(flag: str) -> list[str]:
        index = spec.command.index(flag)
        return spec.command[index + 1 : index + 3]

    assert values("--env.dmc.size") == ["224", "224"]
    assert spec.to_dict()["visual_encoder"] == "vjepa"
    assert spec.to_dict()["visual_resolution"] == 224
    recipe = spec.to_dict()["spatial_mask_recipe"]
    assert recipe["grid"] == [14, 14]
    assert recipe["groups"][0]["blocks"] == 8
    assert recipe["groups"][1]["scale"] == 0.7


def test_world_model_objective_and_posterior_jepa_are_explicit(
    tmp_path: Path,
) -> None:
    maintained = _spec(tmp_path)
    control = _spec(
        tmp_path,
        world_model_objective="reconstruction",
        posterior_jepa=False,
        dynamics_jepa=False,
        spatial_jepa=False,
        sigreg=False,
    )
    hybrid = _spec(
        tmp_path,
        world_model_objective="reconstruction",
        posterior_jepa=True,
        dynamics_jepa=False,
        spatial_jepa=False,
        sigreg=False,
    )
    predictive = _spec(
        tmp_path,
        world_model_objective="reconstruction",
        posterior_jepa=True,
        dynamics_jepa=True,
        spatial_jepa=False,
        sigreg=False,
    )
    objective = maintained.command.index("--agent.objective")
    feature = maintained.command.index("--agent.posterior_jepa")
    assert maintained.command[objective + 1] == "embedding"
    assert maintained.command[feature + 1] == "True"
    assert verify_run_contract(maintained)["decoder_role"].startswith("absent")
    assert verify_run_contract(maintained)["spatial_jepa"] is True
    spatial = maintained.command.index("--agent.spatial_jepa.enabled")
    assert maintained.command[spatial + 1] == "True"
    objective = control.command.index("--agent.objective")
    feature = control.command.index("--agent.posterior_jepa")
    assert control.command[objective + 1] == "reconstruction"
    assert control.command[feature + 1] == "False"
    feature = hybrid.command.index("--agent.posterior_jepa")
    assert hybrid.command[feature + 1] == "True"
    assert verify_run_contract(hybrid)["posterior_jepa"] is True
    assert verify_run_contract(predictive)["dynamics_jepa"] is True
    regularized = _spec(
        tmp_path,
        world_model_objective="reconstruction",
        posterior_jepa=False,
        dynamics_jepa=False,
        spatial_jepa=False,
        sigreg=True,
    )
    assert verify_run_contract(regularized)["sigreg"] is True
    feature = regularized.command.index("--agent.sigreg.enabled")
    assert regularized.command[feature + 1] == "True"
    assert "reconstruction" in verify_run_contract(control)["decoder_role"]


def test_leworldmodel_style_recipe_is_explicit_and_decoder_free(
    tmp_path: Path,
) -> None:
    spec = _spec(
        tmp_path,
        world_model_objective="embedding",
        embedding_target="online",
        embedding_loss="mse",
        posterior_jepa=False,
        dynamics_jepa=True,
        spatial_jepa=False,
        sigreg=True,
        sigreg_scale=0.1,
        sigreg_num_proj=1024,
        sigreg_aggregation="per_timestep",
    )
    contract = verify_run_contract(spec)

    def value(flag: str) -> str:
        return spec.command[spec.command.index(flag) + 1]

    assert value("--agent.embedding_target") == "online"
    assert value("--agent.embedding_loss") == "mse"
    assert value("--agent.posterior_jepa") == "False"
    assert value("--agent.dynamics_jepa") == "True"
    assert value("--agent.spatial_jepa.enabled") == "False"
    assert value("--agent.sigreg.num_proj") == "1024"
    assert value("--agent.sigreg.aggregation") == "per_timestep"
    assert contract["decoder_role"] == (
        "absent; visual targets are full-gradient online embeddings"
    )
    assert "full-gradient online-target MSE" in spec._objective_description


def test_jepa_agent_does_not_construct_a_decoder() -> None:
    resolved = _load_configs(algorithm_root() / "ablations" / "configs.yaml")
    config = elements.Config(resolved["defaults"])
    config = config.update(resolved["debug"])
    config = config.update({"jax.precompile": False})

    obs_space = {
        "image": elements.Space(np.uint8, (1, 64, 64, 3), 0, 256),
        "reward": elements.Space(np.float32, (1,)),
        "is_first": elements.Space(bool, ()),
        "is_last": elements.Space(bool, ()),
        "is_terminal": elements.Space(bool, ()),
    }
    act_space = {
        "action": elements.Space(np.float32, (1, 2), -1.0, 1.0),
    }

    def make_agent(
        objective,
        posterior_jepa=False,
        dynamics_jepa=False,
        spatial_jepa=False,
        sigreg=False,
        embedding_target="ema",
        embedding_loss="cosine",
    ):
        agent_config = elements.Config(
            **config.agent.update(
                objective=objective,
                embedding_target=embedding_target,
                embedding_loss=embedding_loss,
                posterior_jepa=posterior_jepa,
                dynamics_jepa=dynamics_jepa,
                spatial_jepa={
                    **dict(config.agent.spatial_jepa),
                    "enabled": spatial_jepa,
                },
                sigreg={**dict(config.agent.sigreg), "enabled": sigreg},
            ),
            logdir="/tmp/dreamarl-test",
            seed=0,
            jax=config.jax,
            batch_size=config.batch_size,
            batch_length=config.batch_length,
            replay_context=config.replay_context,
            report_length=config.report_length,
            replica=0,
            replicas=1,
        )
        model = object.__new__(AblationAlgorithm)
        AblationAlgorithm.__init__(model, obs_space, act_space, agent_config)
        return model

    embedding = make_agent("embedding", posterior_jepa=True)
    assert embedding.dec is None
    assert embedding.target_enc is not None
    assert embedding.slowenc is not None
    assert all(module is not embedding.target_enc for module in embedding.modules)

    leworldmodel_style = make_agent(
        "embedding",
        dynamics_jepa=True,
        sigreg=True,
        embedding_target="online",
        embedding_loss="mse",
    )
    assert leworldmodel_style.dec is None
    assert leworldmodel_style.target_enc is None
    assert leworldmodel_style.slowenc is None
    assert "dynamics_jepa" in leworldmodel_style.scales
    assert "sigreg" in leworldmodel_style.scales
    assert "posterior_jepa" not in leworldmodel_style.scales
    assert "spatial_jepa" not in leworldmodel_style.scales

    control = make_agent("reconstruction")
    assert control.dec is not None
    assert control.target_enc is None
    assert control.slowenc is None
    assert control.dec in control.modules

    hybrid = make_agent("reconstruction", posterior_jepa=True)
    assert hybrid.dec is not None
    assert hybrid.target_enc is not None
    assert hybrid.slowenc is not None
    assert hybrid.dec in hybrid.modules
    assert all(module is not hybrid.target_enc for module in hybrid.modules)
    assert "posterior_jepa" in hybrid.scales
    assert "image" in hybrid.scales

    predictive = make_agent(
        "reconstruction",
        posterior_jepa=True,
        dynamics_jepa=True,
    )
    assert predictive.dec is not None
    assert predictive.target_enc is not None
    assert "posterior_jepa" in predictive.scales
    assert "dynamics_jepa" in predictive.scales
    assert "image" in predictive.scales

    spatial = make_agent(
        "embedding",
        posterior_jepa=True,
        dynamics_jepa=True,
        spatial_jepa=True,
        sigreg=True,
    )
    assert spatial.dec is None
    assert "posterior_jepa" in spatial.scales
    assert "dynamics_jepa" in spatial.scales
    assert "spatial_jepa" in spatial.scales
    assert "sigreg" in spatial.scales
    assert "image" not in spatial.scales


def test_curve_evaluation_is_explicit_and_uses_held_out_workers(tmp_path: Path) -> None:
    baseline = _spec(tmp_path)
    measured = _spec(
        tmp_path,
        curve_eval_interval=10_000,
        curve_eval_episodes=20,
        curve_eval_seed_offset=50_000,
        curve_eval_policy_mode="stochastic",
    )
    assert "--run.curve_eval_interval" not in baseline.command
    interval = measured.command.index("--run.curve_eval_interval")
    episodes = measured.command.index("--run.curve_eval_eps")
    offset = measured.command.index("--run.curve_eval_seed_offset")
    mode = measured.command.index("--run.curve_eval_policy_mode")
    assert measured.command[interval + 1] == "10000"
    assert measured.command[episodes + 1] == "20"
    assert measured.command[offset + 1] == "50000"
    assert measured.command[mode + 1] == "eval_sample"


def test_visual_dmc_uses_the_same_singleton_algorithm(tmp_path: Path) -> None:
    spec = _spec(tmp_path, task="dmc_reacher_easy", num_agents=1)
    assert spec.configs == ["dmc_vision"]
    assert (
        spec.to_dict()["algorithm_components"]
        == _spec(tmp_path, num_agents=1).to_dict()["algorithm_components"]
    )


def test_visual_dmc_rejects_non_singleton_agent_count(tmp_path: Path) -> None:
    spec = _spec(tmp_path, task="dmc_reacher_easy", num_agents=2)
    with np.testing.assert_raises_regex(ValueError, "num_agents=1"):
        _ = spec.configs


def test_singleton_environment_adds_only_the_agent_axis() -> None:
    class FakeEnv:
        obs_space = {
            "image": elements.Space(np.uint8, (8, 8, 3), 0, 256),
            "reward": elements.Space(np.float32, ()),
            "is_first": elements.Space(bool, ()),
            "is_last": elements.Space(bool, ()),
            "is_terminal": elements.Space(bool, ()),
        }
        act_space = {
            "action": elements.Space(np.float32, (2,), -1.0, 1.0),
            "reset": elements.Space(bool, ()),
        }

        def step(self, action):
            np.testing.assert_array_equal(
                action["action"], np.asarray([0.25, -0.5], np.float32)
            )
            assert np.asarray(action["reset"]).shape == ()
            return {
                "image": np.zeros((8, 8, 3), np.uint8),
                "reward": np.float32(1.5),
                "is_first": np.bool_(False),
                "is_last": np.bool_(False),
                "is_terminal": np.bool_(False),
            }

        def close(self):
            return None

    env = SingletonAgentEnv(FakeEnv())
    assert env.num_agents == 1
    assert env.obs_space["image"].shape == (1, 8, 8, 3)
    assert env.obs_space["reward"].shape == (1,)
    assert env.obs_space["is_first"].shape == ()
    assert env.act_space["action"].shape == (1, 2)
    observation = env.step(
        {
            "action": np.asarray([[0.25, -0.5]], np.float32),
            "reset": np.bool_(False),
        }
    )
    assert observation["image"].shape == (1, 8, 8, 3)
    np.testing.assert_array_equal(observation["reward"], np.asarray([1.5]))
    assert np.asarray(observation["is_first"]).shape == ()


def test_agent_axis_round_trips() -> None:
    axis = TeamAxis(3)
    policy = np.arange(2 * 3 * 5).reshape(2, 3, 5)
    replay = np.arange(2 * 4 * 3 * 5).reshape(2, 4, 3, 5)
    np.testing.assert_array_equal(axis.unfold_batch(axis.fold_batch(policy)), policy)
    np.testing.assert_array_equal(
        axis.unfold_sequence(axis.fold_sequence(replay)), replay
    )


def test_shared_agent_space_rejects_heterogeneous_bounds() -> None:
    space = elements.Space(
        np.float32,
        (2, 1),
        np.asarray([[-1.0], [-2.0]], np.float32),
        np.asarray([[1.0], [1.0]], np.float32),
    )
    with np.testing.assert_raises_regex(ValueError, "heterogeneous"):
        remove_agent_axis("action", space, 2)


def test_distribution_prediction_is_deterministic() -> None:
    class Distribution:
        def pred(self):
            return np.asarray([2, 1], np.int32)

    result = predict({"action": Distribution()})
    np.testing.assert_array_equal(result["action"], np.asarray([2, 1], np.int32))


def test_global_boundaries_broadcast_without_value_changes() -> None:
    axis = TeamAxis(3)
    policy = np.arange(4, dtype=np.float32)
    replay = np.arange(12, dtype=np.float32).reshape(4, 3)
    np.testing.assert_array_equal(
        axis.broadcast_batch(policy).reshape(4, 3),
        np.repeat(policy[:, None], 3, axis=1),
    )
    np.testing.assert_array_equal(
        axis.unfold_sequence(axis.broadcast_sequence(replay)),
        np.repeat(replay[:, :, None], 3, axis=2),
    )


def test_replay_writeback_keeps_stepid_global_and_state_per_agent() -> None:
    batch, length, agents = 2, 4, 3
    axis = TeamAxis(agents)
    stepid = np.arange(batch * length * 20).reshape(batch, length, 20)
    folded_stepid = axis.broadcast_sequence(stepid)
    folded_state = np.arange(batch * agents * length * 5).reshape(
        batch * agents, length, 5
    )
    updates = axis.unfold_replay_updates(
        {"stepid": folded_stepid, "dyn/deter": folded_state}
    )
    np.testing.assert_array_equal(updates["stepid"], stepid)
    assert updates["dyn/deter"].shape == (batch, length, agents, 5)


def test_dry_run_records_current_contract(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    assert run_training(spec, dry_run=True) == 0
    manifest = json.loads(
        (spec.experiment_dir / "launch.json").read_text(encoding="utf-8")
    )
    assert manifest["implementation"] == "first-party decoder-free DreaMARL"
    assert manifest["world_model"] == "parallel_transformer"
    assert manifest["configs"] == ["meltingpot_vision", "ablation_components"]
