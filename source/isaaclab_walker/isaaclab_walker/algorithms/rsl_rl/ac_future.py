# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# Copyright (c) 2025, Jaeyong Shin (jasonshin0537@snu.ac.kr).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Actor-Critic with future-target latent learning (P73 / IsaacLab 5.1).

Latent(latent_dim) layout (prefix-supervised):
  - vel3: [v_x, v_y, yaw_rate]                               (dims 0:3)
  - foot_force6: [L(Fx,Fy,Fz), R(Fx,Fy,Fz)]                  (dims 3:9)
  - future(future_dim): matched to target_encoder(O(t+1))     (dims 9:9+future_dim)

Any remaining tail dims (latent_dim - (9 + future_dim)) are left unconstrained by auxiliary losses
and can be freely used by the actor.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from rsl_rl.networks import EmpiricalNormalization
from rsl_rl.utils import resolve_nn_activation


class ActorCriticAdaptationFuture(nn.Module):
    """Actor-Critic network with encoder + target-encoder (no decoder).

    Compatible with rsl_rl PPO interface:
      - act(), act_inference(), evaluate()
      - get_actions_log_prob()
      - entropy, action_mean, action_std

    PPOFuture-specific helpers:
      - get_latent(), split_latent(), encode_target()
    """

    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims=(256, 256, 256),
        critic_hidden_dims=(256, 256, 256),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        # Encoder (history -> latent)
        encoder_hidden_dims=(512, 512, 256),
        latent_dim: int = 64,
        num_single_obs: int = -1,
        # Target encoder (O(t+1) -> future)
        target_obs_dim: int = 0,
        target_encoder_hidden_dims=(128,),
        future_dim: int = 30,
        policy_obs_scale_single: list | None = None,
        critic_obs_scale: list | None = None,
        target_obs_scale: list | None = None,
        **kwargs,
    ):
        # Ignore unknown kwargs to stay robust across cfg versions.
        if kwargs:
            print(
                "ActorCriticAdaptationFuture.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        # Latent layout conventions used by PPOFuture auxiliary losses:
        #   latent = [vel3, foot6, future(future_dim), optional_extra...]
        self.vel_dim = 3
        self.foot_dim = 6
        min_latent_dim = self.vel_dim + self.foot_dim + int(future_dim)
        if int(latent_dim) < min_latent_dim:
            raise ValueError(
                f"latent_dim must be >= {min_latent_dim} (= vel3 + foot6 + future_dim). "
                f"Got latent_dim={latent_dim}, future_dim={future_dim}."
            )

        self.obs_groups = obs_groups
        self.num_actor_obs = 0
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCriticAdaptationFuture module only supports 1D observations."
            self.num_actor_obs += obs[obs_group].shape[-1]
        self.num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCriticAdaptationFuture module only supports 1D observations."
            self.num_critic_obs += obs[obs_group].shape[-1]
        self.num_actions = int(num_actions)
        self.latent_dim = int(latent_dim)
        self.future_dim = int(future_dim)

        # Infer single-frame dim if not provided (common when actor obs is history-flattened).
        if int(num_single_obs) <= 0:
            raise ValueError(
                "num_single_obs must be set explicitly for ActorCriticAdaptationFuture. "
                "It should equal the policy single-frame observation dimension."
            )
        self.num_single_obs = int(num_single_obs)

        if int(target_obs_dim) <= 0:
            raise ValueError("target_obs_dim must be > 0 for PPOFuture.")
        self.target_obs_dim = int(target_obs_dim)

        activation_fn = resolve_nn_activation(activation)

        # ========== Encoder: obs_history -> latent ==========
        enc_layers: list[nn.Module] = [nn.Linear(self.num_actor_obs, int(encoder_hidden_dims[0])), activation_fn]
        for i in range(len(encoder_hidden_dims) - 1):
            enc_layers += [nn.Linear(int(encoder_hidden_dims[i]), int(encoder_hidden_dims[i + 1])), activation_fn]
        enc_layers += [nn.Linear(int(encoder_hidden_dims[-1]), self.latent_dim)]
        self.encoder = nn.Sequential(*enc_layers)
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(self.num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()

        # ========== Target Encoder: target_obs -> future ==========
        te_layers: list[nn.Module] = [nn.Linear(self.target_obs_dim, int(target_encoder_hidden_dims[0])), activation_fn]
        for i in range(len(target_encoder_hidden_dims) - 1):
            te_layers += [
                nn.Linear(int(target_encoder_hidden_dims[i]), int(target_encoder_hidden_dims[i + 1])),
                activation_fn,
            ]
        te_layers += [nn.Linear(int(target_encoder_hidden_dims[-1]), self.future_dim)]
        self.target_encoder = nn.Sequential(*te_layers)

        # ========== Actor: current_obs + latent -> actions ==========
        actor_input_dim = self.num_single_obs + self.latent_dim
        act_layers: list[nn.Module] = [nn.Linear(actor_input_dim, int(actor_hidden_dims[0])), activation_fn]
        for i in range(len(actor_hidden_dims) - 1):
            act_layers += [nn.Linear(int(actor_hidden_dims[i]), int(actor_hidden_dims[i + 1])), activation_fn]
        act_layers += [nn.Linear(int(actor_hidden_dims[-1]), self.num_actions)]
        self.actor = nn.Sequential(*act_layers)

        # ========== Critic: critic_obs -> value ==========
        crit_layers: list[nn.Module] = [nn.Linear(self.num_critic_obs, int(critic_hidden_dims[0])), activation_fn]
        for i in range(len(critic_hidden_dims) - 1):
            crit_layers += [nn.Linear(int(critic_hidden_dims[i]), int(critic_hidden_dims[i + 1])), activation_fn]
        crit_layers += [nn.Linear(int(critic_hidden_dims[-1]), 1)]
        self.critic = nn.Sequential(*crit_layers)
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(self.num_critic_obs)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()

        # ========== Per-term obs scale buffers ==========
        history_length = self.num_actor_obs // self.num_single_obs
        if self.num_actor_obs != history_length * self.num_single_obs:
            raise ValueError(
                f"num_actor_obs ({self.num_actor_obs}) must be an integer multiple of "
                f"num_single_obs ({self.num_single_obs})."
            )

        if policy_obs_scale_single is None:
            policy_scale_single_t = torch.ones(self.num_single_obs)
        else:
            policy_scale_single_t = torch.as_tensor(policy_obs_scale_single, dtype=torch.float32)
            if policy_scale_single_t.numel() != self.num_single_obs:
                raise ValueError(
                    f"policy_obs_scale_single must have length num_single_obs={self.num_single_obs}, "
                    f"got {policy_scale_single_t.numel()}."
                )
        policy_scale_full = policy_scale_single_t.repeat(history_length)
        assert policy_scale_full.numel() == self.num_actor_obs
        self.register_buffer("policy_obs_scale", policy_scale_full)

        if critic_obs_scale is None:
            critic_scale_t = torch.ones(self.num_critic_obs)
        else:
            critic_scale_t = torch.as_tensor(critic_obs_scale, dtype=torch.float32)
            if critic_scale_t.numel() != self.num_critic_obs:
                raise ValueError(
                    f"critic_obs_scale must have length num_critic_obs={self.num_critic_obs}, "
                    f"got {critic_scale_t.numel()}."
                )
        self.register_buffer("critic_obs_scale", critic_scale_t)

        if target_obs_scale is None:
            target_scale_t = torch.ones(self.target_obs_dim)
        else:
            target_scale_t = torch.as_tensor(target_obs_scale, dtype=torch.float32)
            if target_scale_t.numel() != self.target_obs_dim:
                raise ValueError(
                    f"target_obs_scale must have length target_obs_dim={self.target_obs_dim}, "
                    f"got {target_scale_t.numel()}."
                )
        self.register_buffer("target_obs_scale", target_scale_t)

        # ========== Action Noise ==========
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(self.num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(self.num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}")

        self.distribution: Normal | None = None
        Normal.set_default_validate_args(False)

    # ---------------- API used by rsl_rl ----------------
    def reset(self, dones=None):
        return

    def update_normalization(self, obs):
        if self.actor_obs_normalization:
            actor_obs = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            critic_obs = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)

    @property
    def action_mean(self):
        return self.distribution.mean  # type: ignore[union-attr]

    @property
    def action_std(self):
        return self.distribution.stddev  # type: ignore[union-attr]

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)  # type: ignore[union-attr]

    def get_actor_obs(self, obs):
        if isinstance(obs, torch.Tensor):
            return obs
        obs_list = []
        for obs_group in self.obs_groups["policy"]:
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1)

    def get_critic_obs(self, obs):
        if isinstance(obs, torch.Tensor):
            return obs
        obs_list = []
        for obs_group in self.obs_groups["critic"]:
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1)

    def update_distribution(self, obs):
        observations = self.get_actor_obs(obs)
        observations = self.actor_obs_normalizer(observations)
        observations = observations * self.policy_obs_scale
        latent = self.encoder(observations)
        current_obs = observations[:, -self.num_single_obs :]
        actor_input = torch.cat((current_obs, latent), dim=-1)
        mean = self.actor(actor_input)

        if self.noise_std_type == "scalar":
            std = F.softplus(self.std).expand_as(mean) + 1.0e-6
        else:
            std = torch.exp(self.log_std).expand_as(mean)  # type: ignore[attr-defined]
            std = torch.nan_to_num(std, nan=1.0e-6, posinf=1.0e2, neginf=1.0e-6).clamp_min(1.0e-6)
        self.distribution = Normal(mean, std)

    def act(self, obs, **kwargs):
        self.update_distribution(obs)
        return self.distribution.sample()  # type: ignore[union-attr]

    def act_inference(self, obs):
        observations = self.get_actor_obs(obs)
        observations = self.actor_obs_normalizer(observations)
        observations = observations * self.policy_obs_scale
        latent = self.encoder(observations)
        current_obs = observations[:, -self.num_single_obs :]
        actor_input = torch.cat((current_obs, latent), dim=-1)
        return self.actor(actor_input)

    def get_actions_log_prob(self, actions: torch.Tensor):
        return self.distribution.log_prob(actions).sum(dim=-1)  # type: ignore[union-attr]

    def evaluate(self, obs, **kwargs):
        critic_observations = self.get_critic_obs(obs)
        critic_observations = self.critic_obs_normalizer(critic_observations)
        critic_observations = critic_observations * self.critic_obs_scale
        return self.critic(critic_observations)

    # ---------------- helpers for PPOFuture ----------------
    def get_latent(self, obs) -> torch.Tensor:
        observations = self.get_actor_obs(obs)
        observations = self.actor_obs_normalizer(observations)
        observations = observations * self.policy_obs_scale
        return self.encoder(observations)

    def split_latent(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split latent into (vel3, foot6, future(future_dim))."""
        vel3 = latent[:, 0 : self.vel_dim]
        foot6 = latent[:, self.vel_dim : self.vel_dim + self.foot_dim]
        fut_start = self.vel_dim + self.foot_dim
        fut_end = fut_start + self.future_dim
        future = latent[:, fut_start:fut_end]
        return vel3, foot6, future

    def encode_target(self, target_obs: torch.Tensor) -> torch.Tensor:
        """Encode target observation O(t+1) -> future(future_dim)."""
        if target_obs.shape[-1] != self.target_obs_dim:
            raise ValueError(f"target_obs last dim must be {self.target_obs_dim}. Got: {target_obs.shape}")
        target_obs = target_obs * self.target_obs_scale
        return self.target_encoder(target_obs)

