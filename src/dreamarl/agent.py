"""Locked local decoder-free DreaMARL learner."""

import elements
import embodied.jax
import jax
import jax.numpy as jnp
import numpy as np

from .marl.axes import MODEL_EXCLUDED_FIELDS
from .models.heads import MLPHead
from .models.normalize import Normalize
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
        self.spatial_jepa = False
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
        self.spatial_jepa = bool(config.spatial_jepa.enabled) and bool(self.enc.imgkeys)
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
            **config.dyn.parallel_transformer,
            name="dyn",
        )
        required_burnin = int(config.dyn.parallel_transformer.context) * int(
            config.dyn.parallel_transformer.layers
        )
        if 0 < int(config.replay_context) < required_burnin:
            raise ValueError(
                "Transformer replay_context must be zero or at least "
                f"context * layers ({required_burnin}), got "
                f"{config.replay_context}"
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
        self.pol = MLPHead(self.act_space, outputs, **config.policy, name="pol")
        self.action_mask_key = self._action_mask_key()
        if self.action_mask_key is not None:
            mask_space = self.obs_space["action_mask"]
            maskhead = getattr(config, "maskhead", config.conhead)
            self.actmask = embodied.jax.MLPHead(mask_space, **maskhead, name="actmask")
        else:
            self.actmask = None
        self.val, self.slowval = self._make_value_models(scalar, config)
        self.retnorm = Normalize(**config.retnorm, name="retnorm")
        self.valnorm = Normalize(**config.valnorm, name="valnorm")
        self.advnorm = Normalize(**config.advnorm, name="advnorm")

        if int(config.num_agents) == 1:
            # Preserve the confirmed competitive single-agent construction and
            # optimizer path byte-for-byte.
            self.modules = [
                self.dyn,
                self.enc,
                self.rew,
                self.con,
                self.pol,
                self.val,
            ]
            if self.actmask is not None:
                self.modules.append(self.actmask)
            self.modules.extend(self.additional_modules())
            self.opt = embodied.jax.Optimizer(
                self.modules,
                self._build_optimizer(config),
                summary_depth=1,
                name="opt",
            )
        else:
            additional_modules = self.additional_modules()
            # Keep the historical differentiation target order and isolate the
            # MARL change to optimizer ownership and hyperparameters.
            self.modules = [
                self.dyn,
                self.enc,
                self.rew,
                self.con,
                self.pol,
                self.val,
            ]
            if self.actmask is not None:
                self.modules.append(self.actmask)
            self.modules.extend(additional_modules)
            world_modules = [self.dyn, self.enc, self.rew, self.con]
            if self.actmask is not None:
                world_modules.append(self.actmask)
            world_modules.extend(additional_modules)
            self.opt = self._build_grouped_optimizer(
                self.modules,
                world_modules,
                [self.pol],
                [self.val],
            )
        self.scales = config.loss_scales.copy()
        if self.actmask is not None:
            self.scales["action_mask"] = float(
                getattr(
                    config,
                    "action_mask_scale",
                    self.scales.get("action_mask", 1.0),
                )
            )

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
        if str(getattr(self.config, "replay_sampling", "uniform")) != "uniform":
            spaces["replay_sample_role"] = elements.Space(np.int8)
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

    def _make_value_models(self, scalar, config):
        """Construct the maintained fast and slow value models."""

        value = embodied.jax.MLPHead(scalar, **config.value, name="val")
        slowvalue = embodied.jax.SlowModel(
            embodied.jax.MLPHead(scalar, **config.value, name="slowval"),
            source=value,
            **config.slowvalue,
        )
        return value, slowvalue

    def critic(self, features, bdims, *, slow=False, context=None):
        """Evaluate the maintained value model."""
        value_head = self.slowval if slow else self.val
        inputs = self.feat2tensor(features) if isinstance(features, dict) else features
        if context is not None:
            inputs = jnp.concatenate([inputs, context], axis=-1)
        return value_head(inputs, bdims)

    def _action_mask_key(self):
        if "action_mask" not in self.obs_space:
            return None
        discrete = [key for key, space in self.act_space.items() if space.discrete]
        if len(self.act_space) != 1 or len(discrete) != 1:
            raise ValueError(
                "the action_mask contract currently requires exactly one "
                "discrete action"
            )
        key = discrete[0]
        classes = int(np.asarray(self.act_space[key].classes).reshape(-1)[0])
        if self.obs_space["action_mask"].shape != (classes,):
            raise ValueError(
                "action_mask must have one boolean entry per categorical action"
            )
        return key
