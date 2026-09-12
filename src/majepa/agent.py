"""Local decoder-free MA-JEPA learner."""

import elements
import embodied.jax
import jax
import jax.numpy as jnp
import numpy as np

from .marl.axes import MODEL_EXCLUDED_FIELDS
from .models.heads import MLPHead
from .training.learner import LearnerMixin
from .training.optimization import OptimizationMixin
from .training.policy import PolicyMixin
from .training.replay import ReplayMixin
from .training.reporting import ReportingMixin
from .models.encoder import Encoder
from .world_model import ParallelTransformerDynamics, feature_tensor


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
        if str(config.replay_sampling) != "recent_world_uniform_behavior":
            raise ValueError("MA-JEPA requires independent world and behavior replay")
        self.ppo_start_step = int(getattr(config, "ppo_start_step", 0))
        if self.ppo_start_step < 0:
            raise ValueError("ppo_start_step must be nonnegative")
        if int(config.ppo.epochs) < 1:
            raise ValueError("PPO requires at least one optimization epoch")
        self.ppo_actor_epochs = int(config.ppo.get("actor_epochs", 0)) or int(
            config.ppo.epochs
        )
        self.ppo_critic_epochs = int(config.ppo.get("critic_epochs", 0)) or int(
            config.ppo.epochs
        )
        if min(self.ppo_actor_epochs, self.ppo_critic_epochs) < 1:
            raise ValueError("Actor and critic epoch counts must be positive")
        if not 0.0 < float(config.ppo.clip_epsilon) < 1.0:
            raise ValueError("PPO clip_epsilon must be in (0, 1)")
        if float(config.ppo.entropy_coefficient) < 0.0:
            raise ValueError("PPO entropy_coefficient must be nonnegative")
        if float(config.ppo.replay_value_scale) < 0.0:
            raise ValueError("PPO replay_value_scale must be nonnegative")
        if not 0.0 <= float(config.ppo.replay_value_lam) <= 1.0:
            raise ValueError("PPO replay_value_lam must be in [0, 1]")
        if float(config.ppo.replay_value_scale) and int(config.imag_last) == 1:
            raise ValueError(
                "replay value learning requires at least two imagination roots"
            )
        enc_space = {
            key: value
            for key, value in self.obs_space.items()
            if key not in MODEL_EXCLUDED_FIELDS
        }
        self.enc = Encoder(enc_space, **config.enc.simple, name="enc")
        self.enc_output_dim = self.enc.calculate_encoder_output_dim()
        self.target_enc = Encoder(enc_space, **config.enc.simple, name="target_enc")
        self.slowenc = embodied.jax.SlowModel(
            self.target_enc, source=self.enc, **config.target_encoder
        )
        self.dyn = ParallelTransformerDynamics(
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
        self.feat2tensor = feature_tensor
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
        self.opt = self._build_ctde_optimizer(
            self.modules,
            [self.dyn, self.enc, self.rew, self.con, self.actmask],
            list(self.ctde_modules),
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
        if self.ppo_start_step:
            # Runtime-only control input. It is injected after replay sampling,
            # so it never becomes replay content or changes sampled sequences.
            spaces["_environment_step"] = elements.Space(np.int32)
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
