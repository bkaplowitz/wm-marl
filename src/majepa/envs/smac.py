"""SMAC-v1 environment boundary for protocol-matched MARL evaluation."""

from __future__ import annotations

import functools

import elements
import embodied
import numpy as np


class SMACEnv(embodied.Env):
    """Expose one homogeneous SMAC map through MA-JEPA's agent-axis contract."""

    def __init__(
        self,
        map_name: str,
        *,
        seed: int | None = None,
        difficulty: str = "7",
        continuing_episode: bool = True,
        sc2_env=None,
    ):
        if sc2_env is None:
            from smac.env import StarCraft2Env

            kwargs = {
                "map_name": map_name,
                "difficulty": str(difficulty),
                "continuing_episode": bool(continuing_episode),
            }
            if seed is not None:
                kwargs["seed"] = int(seed)
            sc2_env = StarCraft2Env(**kwargs)
        self._env = sc2_env
        info = dict(self._env.get_env_info())
        self.num_agents = int(info["n_agents"])
        self._obs_dim = int(info["obs_shape"])
        self._action_dim = int(info["n_actions"])
        self._num_no_attack = int(
            getattr(
                self._env,
                "n_actions_no_attack",
                self._action_dim - int(info.get("n_enemies", 0)),
            )
        )
        self._num_enemies = int(
            info.get("n_enemies", self._action_dim - self._num_no_attack)
        )
        if min(self.num_agents, self._obs_dim, self._action_dim) < 1:
            raise ValueError(f"invalid SMAC environment info: {info}")
        self._needs_reset = True
        self._last_attack_targets = np.full(self.num_agents, -1, np.int32)

    @functools.cached_property
    def obs_space(self):
        return {
            "observation": elements.Space(np.float32, (self.num_agents, self._obs_dim)),
            "reward": elements.Space(np.float32, (self.num_agents,)),
            "agent_present": elements.Space(bool, (self.num_agents,)),
            "agent_alive": elements.Space(bool, (self.num_agents,)),
            "controllable_alive": elements.Space(bool, (self.num_agents,)),
            "action_mask": elements.Space(bool, (self.num_agents, self._action_dim)),
            "is_first": elements.Space(bool, ()),
            "is_last": elements.Space(bool, ()),
            "is_terminal": elements.Space(bool, ()),
            "log/battle_won": elements.Space(np.float32, ()),
            "log/dead_allies": elements.Space(np.float32, ()),
            "log/dead_enemies": elements.Space(np.float32, ()),
            "log/timeout": elements.Space(np.float32, ()),
            "log/legacy_reward": elements.Space(np.float32, ()),
            "log/corrected_reward": elements.Space(np.float32, ()),
            "log/enemy_damage": elements.Space(np.float32, ()),
            "log/enemy_health_damage": elements.Space(np.float32, ()),
            "log/enemy_shield_damage": elements.Space(np.float32, ()),
            "log/enemy_shield_regen": elements.Space(np.float32, ()),
            "log/enemy_deaths_step": elements.Space(np.float32, ()),
            "log/ally_deaths_step": elements.Space(np.float32, ()),
            "log/ally_survivors": elements.Space(np.float32, ()),
            "log/enemy_survivors": elements.Space(np.float32, ()),
            "log/action_noop_count": elements.Space(np.float32, ()),
            "log/action_stop_count": elements.Space(np.float32, ()),
            "log/action_move_count": elements.Space(np.float32, ()),
            "log/action_attack_count": elements.Space(np.float32, ()),
            "log/action_target_switch_count": elements.Space(np.float32, ()),
            **{
                f"log/attack_target_{index}_count": elements.Space(np.float32, ())
                for index in range(self._num_enemies)
            },
        }

    @functools.cached_property
    def act_space(self):
        return {
            "action": elements.Space(np.int32, (self.num_agents,), 0, self._action_dim),
            "reset": elements.Space(bool, (), 0, 2),
        }

    def step(self, action):
        if bool(np.asarray(action["reset"])) or self._needs_reset:
            return self._reset()
        actions = np.asarray(action["action"], np.int32)
        if actions.shape != (self.num_agents,):
            raise ValueError(
                f"expected actions shaped {(self.num_agents,)}, got {actions.shape}"
            )
        mask = self._action_mask()
        if np.any(actions < 0) or np.any(actions >= self._action_dim):
            raise ValueError(f"SMAC action outside [0, {self._action_dim}): {actions}")
        invalid = ~mask[np.arange(self.num_agents), actions]
        if invalid.any():
            raise ValueError(
                "SMAC policy selected unavailable actions for agents "
                f"{np.flatnonzero(invalid).tolist()}: {actions.tolist()}"
            )
        reward, terminated, info = self._env.step(actions.tolist())
        info = dict(info or {})
        self._needs_reset = bool(terminated)
        truncated = bool(info.get("episode_limit", False))
        diagnostics = self._combat_diagnostics(actions, float(reward), info)
        return self._observation(
            reward=float(reward),
            is_first=False,
            is_last=self._needs_reset,
            is_terminal=self._needs_reset and not truncated,
            info=info,
            diagnostics=diagnostics,
        )

    def close(self):
        return self._env.close()

    def _reset(self):
        self._env.reset()
        self._needs_reset = False
        self._last_attack_targets.fill(-1)
        return self._observation(
            reward=0.0,
            is_first=True,
            is_last=False,
            is_terminal=False,
            info={},
            diagnostics=self._empty_diagnostics(),
        )

    def _observation(
        self, *, reward, is_first, is_last, is_terminal, info, diagnostics
    ):
        observations = np.asarray(self._env.get_obs(), np.float32)
        expected = (self.num_agents, self._obs_dim)
        if observations.shape != expected:
            raise ValueError(
                f"SMAC observations must be shaped {expected}, got {observations.shape}"
            )
        action_mask = self._action_mask()
        # SMAC uses a fixed agent roster for the complete episode. Dead units
        # remain zero-observation slots whose mask permits only no-op. Keeping
        # those slots active matches MARIE's fixed-shape learner contract; the
        # legal-action mask, rather than liveness masking, controls behavior.
        alive = np.ones((self.num_agents,), bool)
        controllable_alive = action_mask[:, 1:].any(axis=-1)
        rewards = np.full((self.num_agents,), reward, np.float32)
        return {
            "observation": observations,
            "reward": rewards,
            "agent_present": np.ones((self.num_agents,), bool),
            "agent_alive": alive,
            "controllable_alive": controllable_alive,
            "action_mask": action_mask,
            "is_first": np.bool_(is_first),
            "is_last": np.bool_(is_last),
            "is_terminal": np.bool_(is_terminal),
            "log/battle_won": np.float32(bool(info.get("battle_won", False))),
            "log/dead_allies": np.float32(info.get("dead_allies", 0)),
            "log/dead_enemies": np.float32(info.get("dead_enemies", 0)),
            **{key: np.float32(value) for key, value in diagnostics.items()},
        }

    def _empty_diagnostics(self):
        diagnostics = {
            "log/timeout": 0.0,
            "log/legacy_reward": 0.0,
            "log/corrected_reward": 0.0,
            "log/enemy_damage": 0.0,
            "log/enemy_health_damage": 0.0,
            "log/enemy_shield_damage": 0.0,
            "log/enemy_shield_regen": 0.0,
            "log/enemy_deaths_step": 0.0,
            "log/ally_deaths_step": 0.0,
            "log/ally_survivors": float(self.num_agents),
            "log/enemy_survivors": float(self._num_enemies),
            "log/action_noop_count": 0.0,
            "log/action_stop_count": 0.0,
            "log/action_move_count": 0.0,
            "log/action_attack_count": 0.0,
            "log/action_target_switch_count": 0.0,
        }
        diagnostics.update(
            {
                f"log/attack_target_{index}_count": 0.0
                for index in range(self._num_enemies)
            }
        )
        return diagnostics

    def _combat_diagnostics(self, actions, legacy_reward, info):
        diagnostics = self._empty_diagnostics()
        diagnostics["log/timeout"] = float(info.get("episode_limit", False))
        diagnostics["log/legacy_reward"] = legacy_reward

        actions = np.asarray(actions, np.int32)
        diagnostics["log/action_noop_count"] = float(np.sum(actions == 0))
        diagnostics["log/action_stop_count"] = float(np.sum(actions == 1))
        diagnostics["log/action_move_count"] = float(
            np.sum((actions >= 2) & (actions < self._num_no_attack))
        )
        attack = actions >= self._num_no_attack
        targets = actions - self._num_no_attack
        diagnostics["log/action_attack_count"] = float(attack.sum())
        switches = attack & (self._last_attack_targets >= 0)
        switches &= targets != self._last_attack_targets
        diagnostics["log/action_target_switch_count"] = float(switches.sum())
        self._last_attack_targets[attack] = targets[attack]
        for index in range(self._num_enemies):
            diagnostics[f"log/attack_target_{index}_count"] = float(
                np.sum(attack & (targets == index))
            )

        previous_enemies = getattr(self._env, "previous_enemy_units", None)
        current_enemies = getattr(self._env, "enemies", None)
        previous_allies = getattr(self._env, "previous_ally_units", None)
        current_allies = getattr(self._env, "agents", None)
        enemy_stats = self._unit_changes(previous_enemies, current_enemies)
        ally_stats = self._unit_changes(previous_allies, current_allies)
        if enemy_stats is not None:
            damage, health_damage, shield_damage, shield_regen, deaths, survivors = (
                enemy_stats
            )
            diagnostics["log/enemy_health_damage"] = health_damage
            diagnostics["log/enemy_shield_damage"] = shield_damage
            diagnostics["log/enemy_shield_regen"] = shield_regen
            diagnostics["log/enemy_damage"] = damage
            diagnostics["log/enemy_deaths_step"] = deaths
            diagnostics["log/enemy_survivors"] = survivors
        if ally_stats is not None:
            diagnostics["log/ally_deaths_step"] = ally_stats[4]
            diagnostics["log/ally_survivors"] = ally_stats[5]

        corrected = diagnostics["log/enemy_damage"]
        corrected += (
            float(getattr(self._env, "reward_death_value", 0.0))
            * diagnostics["log/enemy_deaths_step"]
        )
        corrected += float(getattr(self._env, "reward_win", 0.0)) * float(
            info.get("battle_won", False)
        )
        if bool(getattr(self._env, "reward_sparse", False)):
            corrected = legacy_reward
        elif bool(getattr(self._env, "reward_scale", False)):
            denominator = float(getattr(self._env, "max_reward", 1.0)) / float(
                getattr(self._env, "reward_scale_rate", 1.0)
            )
            corrected /= max(denominator, np.finfo(np.float32).eps)
        diagnostics["log/corrected_reward"] = corrected
        return diagnostics

    @staticmethod
    def _unit_changes(previous, current):
        if previous is None or current is None:
            return None
        previous = dict(previous)
        current = dict(current)
        if previous.keys() != current.keys():
            return None
        health_damage = 0.0
        shield_damage = 0.0
        shield_regen = 0.0
        damage = 0.0
        deaths = 0
        survivors = 0
        for index in previous:
            old = previous[index]
            new = current[index]
            old_health = float(getattr(old, "health", 0.0))
            new_health = float(getattr(new, "health", 0.0))
            old_shield = float(getattr(old, "shield", 0.0))
            new_shield = float(getattr(new, "shield", 0.0))
            damage += max(
                old_health + old_shield - new_health - new_shield,
                0.0,
            )
            health_damage += max(old_health - new_health, 0.0)
            shield_damage += max(old_shield - new_shield, 0.0)
            shield_regen += max(new_shield - old_shield, 0.0)
            deaths += int(old_health > 0.0 and new_health <= 0.0)
            survivors += int(new_health > 0.0)
        return damage, health_damage, shield_damage, shield_regen, deaths, survivors

    def _action_mask(self):
        mask = np.asarray(self._env.get_avail_actions(), bool)
        expected = (self.num_agents, self._action_dim)
        if mask.shape != expected:
            raise ValueError(
                f"SMAC action mask must be shaped {expected}, got {mask.shape}"
            )
        if not mask.any(axis=-1).all():
            raise ValueError("every SMAC agent must expose at least one legal action")
        return mask
