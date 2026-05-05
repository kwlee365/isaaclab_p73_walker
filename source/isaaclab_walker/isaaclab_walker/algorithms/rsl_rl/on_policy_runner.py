# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# Copyright (c) 2025, Jaeyong Shin (jasonshin0537@snu.ac.kr).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom RSL-RL runner for P73 that is compatible with rsl-rl-lib>=3.0.0."""

from __future__ import annotations

import warnings

from rsl_rl.runners import OnPolicyRunner
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent, resolve_rnd_config, resolve_symmetry_config

# Import custom PPO implementation and expose it under the name "PPO" so that
# the upstream runner logic (`eval(self.alg_cfg["class_name"])`) resolves to this implementation.
from .ppo import P73PPO as PPO

# PPOFuture / ActorCriticFuture (TOCABI-style contact-force framework port)
# Note:
# - These are imported here so that `eval(class_name)` can resolve them from this module's globals
#   in rsl-rl-lib variants that call our custom runner's construction logic.
from .ppo_future import PPOFuture  # noqa: F401
from .ac_future import ActorCriticAdaptationFuture  # noqa: F401


class P73OnPolicyRunner(OnPolicyRunner):
    """On-policy runner that swaps the PPO algorithm with `P73PPO` (supports bound loss)."""

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        super().__init__(env, train_cfg, log_dir, device)
        print("[INFO] P73OnPolicyRunner: Using P73PPO instead of base PPO")

    def _construct_algorithm(self, obs) -> PPO:  # type: ignore[override]
        """Construct the actor-critic algorithm (mirrors upstream runner for rsl-rl-lib>=3.0.0)."""
        # resolve RND config
        self.alg_cfg = resolve_rnd_config(self.alg_cfg, obs, self.cfg["obs_groups"], self.env)
        # resolve symmetry config
        self.alg_cfg = resolve_symmetry_config(self.alg_cfg, self.env)

        # resolve deprecated normalization config
        if self.cfg.get("empirical_normalization") is not None:
            warnings.warn(
                "The `empirical_normalization` parameter is deprecated. Please set `actor_obs_normalization` and "
                "`critic_obs_normalization` as part of the `policy` configuration instead.",
                DeprecationWarning,
            )
            if self.policy_cfg.get("actor_obs_normalization") is None:
                self.policy_cfg["actor_obs_normalization"] = self.cfg["empirical_normalization"]
            if self.policy_cfg.get("critic_obs_normalization") is None:
                self.policy_cfg["critic_obs_normalization"] = self.cfg["empirical_normalization"]

        # initialize the actor-critic
        actor_critic_class = eval(self.policy_cfg.pop("class_name"))
        actor_critic: ActorCritic | ActorCriticRecurrent = actor_critic_class(
            obs, self.cfg["obs_groups"], self.env.num_actions, **self.policy_cfg
        ).to(self.device)

        # initialize the algorithm (eval will resolve "PPO" to our imported alias above)
        alg_class = eval(self.alg_cfg.pop("class_name"))
        alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg)

        # initialize rollout storage
        alg.init_storage("rl", self.env.num_envs, self.num_steps_per_env, obs, [self.env.num_actions])
        return alg