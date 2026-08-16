"""Scientific alternative assembly and Embodied runtime contract.

The model components are assembled here. Policy execution, learning, replay
adaptation, reporting, and optimization live in the ``training`` package.
"""

import elements
import embodied.jax
import jax
import jax.numpy as jnp
import numpy as np

from ..marl.core import TeamAxisAdapter
from ..marl.axes import ENVIRONMENT_FIELDS, MODEL_EXCLUDED_FIELDS, TeamAxis
from ..marl.spaces import add_agent_axis, remove_agent_axis
from ..training.learner import LearnerMixin
from ..training.optimization import OptimizationMixin
from ..training.policy import PolicyMixin
from ..training.replay import ReplayMixin
from ..training.reporting import ReportingMixin
from ..world_model import world_model_backend


class AblationAlgorithm(
    TeamAxisAdapter,
    PolicyMixin,
    LearnerMixin,
    ReportingMixin,
    ReplayMixin,
    OptimizationMixin,
    embodied.jax.Agent,
):
    banner = [
        r"---  ___                           __   ______ ---",
        r"--- |   \ _ _ ___ __ _ _ __  ___ _ \ \ / /__ / ---",
        r"--- | |) | '_/ -_) _` | '  \/ -_) '/\ V / |_ \ ---",
        r"--- |___/|_| \___\__,_|_|_|_\___|_|  \_/ |___/ ---",
    ]

    def __init__(self, obs_space, act_space, config):
        self.team = TeamAxis(int(config.num_agents))
        self.num_agents = self.team.size
        self.joint_obs_space = obs_space
        self.joint_act_space = act_space
        self.obs_space = {
            key: (
                space
                if key in ENVIRONMENT_FIELDS
                else remove_agent_axis(key, space, self.num_agents)
            )
            for key, space in obs_space.items()
        }
        self.act_space = {
            key: remove_agent_axis(key, space, self.num_agents)
            for key, space in act_space.items()
        }
        self.config = config
        self.objective = str(config.objective)
        self.posterior_jepa = bool(config.posterior_jepa)
        self.dynamics_jepa = bool(config.dynamics_jepa)
        self.spatial_jepa = bool(config.spatial_jepa.enabled)
        self.sigreg = bool(config.sigreg.enabled)
        self.embedding_target = str(config.embedding_target)
        self.embedding_loss = str(config.embedding_loss)
        canonical = (
            config.dyn.typ == "parallel_transformer"
            and config.enc.typ == "simple"
            and self.objective == "embedding"
            and self.embedding_target == "ema"
            and self.embedding_loss == "cosine"
            and self.posterior_jepa
            and self.dynamics_jepa
            and self.spatial_jepa
            and self.sigreg
            and config.spatial_jepa.topology == "fixed_count"
        )
        if not canonical:
            from .validation import validate

            validate(config)
            from .backend import world_model_backend as ablation_backend

            self.world_model = ablation_backend(config.dyn.typ)
        else:
            self.world_model = world_model_backend()

        enc_space = {
            key: value
            for key, value in self.obs_space.items()
            if key not in MODEL_EXCLUDED_FIELDS
        }
        dec_space = dict(enc_space)
        self.enc = self.world_model.encoder(config.enc.typ)(
            enc_space, **config.enc[config.enc.typ], name="enc"
        )
        self.enc_output_dim = self.enc.calculate_encoder_output_dim()
        if self.spatial_jepa and not self.enc.imgkeys:
            raise ValueError("spatial JEPA requires at least one image observation")
        self.spatial_predictor = None
        if str(config.spatial_jepa.topology) == "vjepa_multiblock":
            from . import visual as ablation_visual

            if config.enc.typ != "vjepa":
                raise ValueError("vjepa_multiblock requires the 224px V-JEPA encoder")
            if self.objective != "embedding" or self.embedding_target != "ema":
                raise ValueError(
                    "vjepa_multiblock requires decoder-free EMA-target training"
                )
            grid_height, grid_width, token_dim = self.enc.image_grid_shape()
            self.spatial_predictor = ablation_visual.SpatialTokenPredictor(
                grid=(grid_height, grid_width),
                input_dim=token_dim,
                name="spatial_predictor",
            )

        self.dec_space = dec_space
        if self.objective == "reconstruction":
            self.dec = self.world_model.decoder(config.dec.typ)(
                dec_space, **config.dec[config.dec.typ], name="dec"
            )
        else:
            self.dec = None

        if self.embedding_target == "ema" and (
            self.posterior_jepa or self.dynamics_jepa or self.spatial_jepa
        ):
            self.target_enc = self.world_model.encoder(config.enc.typ)(
                enc_space, **config.enc[config.enc.typ], name="target_enc"
            )
            self.slowenc = embodied.jax.SlowModel(
                self.target_enc,
                source=self.enc,
                **config.target_encoder,
            )
        else:
            self.target_enc = None
            self.slowenc = None

        self.dyn = self.world_model.dynamics_model(config.dyn.typ)(
            self.act_space,
            self.enc_output_dim,
            **config.dyn[config.dyn.typ],
            name="dyn",
        )
        self.feat2tensor = self.world_model.feature_tensor

        scalar = elements.Space(np.float32, ())
        binary = elements.Space(bool, (), 0, 2)
        self.rew = embodied.jax.MLPHead(scalar, **config.rewhead, name="rew")
        self.con = embodied.jax.MLPHead(binary, **config.conhead, name="con")

        discrete, continuous = config.policy_dist_disc, config.policy_dist_cont
        outputs = {
            key: discrete if space.discrete else continuous
            for key, space in self.act_space.items()
        }
        self.pol = embodied.jax.MLPHead(
            self.act_space, outputs, **config.policy, name="pol"
        )

        self.val = embodied.jax.MLPHead(scalar, **config.value, name="val")
        self.slowval = embodied.jax.SlowModel(
            embodied.jax.MLPHead(scalar, **config.value, name="slowval"),
            source=self.val,
            **config.slowvalue,
        )
        self.retnorm = embodied.jax.Normalize(**config.retnorm, name="retnorm")
        self.valnorm = embodied.jax.Normalize(**config.valnorm, name="valnorm")
        self.advnorm = embodied.jax.Normalize(**config.advnorm, name="advnorm")

        self.modules = [self.dyn, self.enc, self.rew, self.con, self.pol, self.val]
        if self.dec is not None:
            self.modules.insert(2, self.dec)
        if self.spatial_predictor is not None:
            self.modules.insert(2, self.spatial_predictor)
        self.opt = embodied.jax.Optimizer(
            self.modules,
            self._build_optimizer(config),
            summary_depth=1,
            name="opt",
        )

        self.scales = self.config.loss_scales.copy()
        reconstruction_scale = self.scales.pop("rec")
        posterior_jepa_scale = self.scales.pop("posterior_jepa")
        dynamics_jepa_scale = self.scales.pop("dynamics_jepa")
        spatial_jepa_scale = self.scales.pop("spatial_jepa")
        sigreg_scale = self.scales.pop("sigreg")
        if self.objective == "reconstruction":
            self.scales.update({key: reconstruction_scale for key in self.dec_space})
        if self.posterior_jepa:
            self.scales["posterior_jepa"] = posterior_jepa_scale
        if self.dynamics_jepa:
            self.scales["dynamics_jepa"] = dynamics_jepa_scale
        if self.spatial_jepa:
            self.scales["spatial_jepa"] = spatial_jepa_scale
        if self.sigreg:
            self.scales["sigreg"] = sigreg_scale

    @property
    def policy_keys(self):
        return "^(enc|dyn|dec|pol)/"

    def critic(self, features, bdims, *, slow=False):
        value_head = self.slowval if slow else self.val
        inputs = self.feat2tensor(features) if isinstance(features, dict) else features
        return value_head(inputs, bdims)

    @property
    def ext_space(self):
        spaces = {
            "consec": elements.Space(np.int32),
            "stepid": elements.Space(np.uint8, 20),
        }
        if self.config.replay_context:
            entries = dict(enc=self.enc.entry_space, dyn=self.dyn.entry_space)
            if self.dec is not None:
                entries["dec"] = self.dec.entry_space
            spaces.update(elements.tree.flatdict(entries))
        return {
            key: (
                space
                if key in ("consec", "stepid")
                else add_agent_axis(space, self.num_agents)
            )
            for key, space in spaces.items()
        }

    def init_policy(self, batch_size):
        return self.team.unfold_tree_batch(
            self._init_local(batch_size * self.num_agents)
        )

    def _init_local(self, batch_size):
        def zeros(space):
            return jnp.zeros((batch_size, *space.shape), space.dtype)

        return (
            self.enc.initial(batch_size),
            self.dyn.initial(batch_size),
            self.dec.initial(batch_size) if self.dec is not None else {},
            jax.tree.map(zeros, self.act_space),
        )

    def init_train(self, batch_size):
        return self.init_policy(batch_size)

    def init_report(self, batch_size):
        return self.init_policy(batch_size)
