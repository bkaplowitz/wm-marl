"""Locked local decoder-free DreaMARL learner."""

import elements
import embodied.jax
import jax
import jax.numpy as jnp
import numpy as np

from .marl.axes import MODEL_EXCLUDED_FIELDS
from .training.learner import LearnerMixin
from .training.optimization import OptimizationMixin
from .training.policy import PolicyMixin
from .training.replay import ReplayMixin
from .training.reporting import ReportingMixin
from .world_model import world_model_backend


class Agent(
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
        self.obs_space = obs_space
        self.act_space = act_space
        self.config = config
        self.world_model = world_model_backend()
        self.objective = "embedding"
        self.embedding_target = "ema"
        self.embedding_loss = "cosine"
        self.posterior_jepa = True
        self.dynamics_jepa = True
        self.spatial_jepa = True
        self.sigreg = True
        self.spatial_predictor = None
        self.dec = None

        enc_space = {
            key: value
            for key, value in self.obs_space.items()
            if key not in MODEL_EXCLUDED_FIELDS
        }
        self.enc = self.world_model.encoder("simple")(
            enc_space, **config.enc.simple, name="enc"
        )
        self.enc_output_dim = self.enc.calculate_encoder_output_dim()
        self.target_enc = self.world_model.encoder("simple")(
            enc_space, **config.enc.simple, name="target_enc"
        )
        self.slowenc = embodied.jax.SlowModel(
            self.target_enc, source=self.enc, **config.target_encoder
        )
        self.dyn = self.world_model.dynamics_model("parallel_transformer")(
            self.act_space,
            self.enc_output_dim,
            team_size=int(config.num_agents),
            **config.dyn.parallel_transformer,
            name="dyn",
        )
        self.feat2tensor = self.world_model.feature_tensor
        scalar = elements.Space(np.float32, ())
        binary = elements.Space(bool, (), 0, 2)
        self.rew = embodied.jax.MLPHead(scalar, **config.rewhead, name="rew")
        self.con = embodied.jax.MLPHead(binary, **config.conhead, name="con")
        outputs = {
            key: config.policy_dist_disc if space.discrete else config.policy_dist_cont
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
        self.modules.extend(self.additional_modules())
        self.opt = embodied.jax.Optimizer(
            self.modules,
            self._build_optimizer(config),
            summary_depth=1,
            name="opt",
        )
        self.scales = config.loss_scales.copy()

    def additional_modules(self):
        """Return algorithm modules trained by the shared optimizer."""

        return []

    @property
    def policy_keys(self):
        return "^(enc|dyn|pol)/"

    @property
    def ext_space(self):
        spaces = {
            "consec": elements.Space(np.int32),
            "stepid": elements.Space(np.uint8, 20),
        }
        if self.config.replay_context:
            spaces.update(
                elements.tree.flatdict(
                    {
                        "enc": self.enc.entry_space,
                        "dyn": self.dynamics_replay_entry_space(),
                    }
                )
            )
        return spaces

    def init_policy(self, batch_size):
        def zeros(space):
            return jnp.zeros((batch_size, *space.shape), space.dtype)

        return (
            self.enc.initial(batch_size),
            self.dyn.initial(batch_size),
            {},
            jax.tree.map(zeros, self.act_space),
        )

    def init_train(self, batch_size):
        return self.init_policy(batch_size)

    def init_report(self, batch_size):
        return self.init_policy(batch_size)

    def report_rows(self, batch_size):
        return min(batch_size, 6)

    def critic(self, features, bdims, *, slow=False):
        """Evaluate the locked local value model."""
        value_head = self.slowval if slow else self.val
        inputs = self.feat2tensor(features) if isinstance(features, dict) else features
        return value_head(inputs, bdims)
