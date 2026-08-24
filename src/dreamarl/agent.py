"""Local decoder-free DreaMARL learner."""

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
        self.sigreg = True
        self.dec = None

        enc_space = {
            key: value
            for key, value in self.obs_space.items()
            if key not in MODEL_EXCLUDED_FIELDS
        }
        self.enc = self.world_model.encoder("simple")(
            enc_space, **config.enc.simple, name="enc"
        )
        self.spatial_jepa = bool(self.enc.imgkeys)
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
        churn = getattr(config, "actor_churn", None)
        self.actor_churn_enabled = bool(getattr(churn, "enabled", False))
        if self.actor_churn_enabled:
            self.churn_pol = MLPHead(
                self.act_space,
                outputs,
                **config.policy,
                name="churn_pol",
            )
            self.churn_teacher = embodied.jax.SlowModel(
                self.churn_pol,
                source=self.pol,
                rate=1.0,
                every=1,
            )
        else:
            self.churn_pol = None
            self.churn_teacher = None
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

        additional_modules = self.additional_modules()
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
        ctde_modules = tuple(getattr(self, "ctde_modules", ()))
        ctde_module_ids = {id(module) for module in ctde_modules}
        if not ctde_module_ids.issubset({id(module) for module in additional_modules}):
            raise ValueError("CTDE optimizer modules must be additional modules")
        world_modules = [self.dyn, self.enc, self.rew, self.con]
        if self.actmask is not None:
            world_modules.append(self.actmask)
        world_modules.extend(
            module for module in additional_modules if id(module) not in ctde_module_ids
        )

        if ctde_modules:
            self.opt = self._build_ctde_optimizer(
                self.modules,
                world_modules,
                list(ctde_modules),
                [self.pol],
                [self.val],
            )
        else:
            # Preserve the confirmed competitive local construction and optimizer
            # path exactly for singleton and parameter-shared multi-agent runs.
            self.opt = embodied.jax.Optimizer(
                self.modules,
                self._build_optimizer(config),
                summary_depth=1,
                name="opt",
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
        return self._local_initial(batch_size)

    def _local_initial(self, batch_size):
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
