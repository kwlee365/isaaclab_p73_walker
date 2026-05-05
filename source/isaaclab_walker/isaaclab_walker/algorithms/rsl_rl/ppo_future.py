# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# Copyright (c) 2025, Jaeyong Shin (jasonshin0537@snu.ac.kr).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom PPO with future-target latent losses (P73 / IsaacLab 5.1).

Adds three auxiliary losses:
  - L_vel:    MSE(latent_vel3, gt_vel3)                    [from critic_obs prefix at t]
  - L_foot:   MSE(latent_foot6, gt_foot_force6)            [from critic_obs prefix at t]
  - L_future: MSE(latent_future, target_encoder(O(t+1)))    [from stored target_obs_next]
"""

from __future__ import annotations

from typing import Any, cast
import os

import torch

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.storage.rollout_storage import RolloutStorage


class RolloutStorageFuture(RolloutStorage):
    """Rollout storage extended with target observations for O(t+1)."""

    class Transition(RolloutStorage.Transition):
        def __init__(self):
            super().__init__()
            self.target_observations_next = None

    def __init__(self, *args, target_obs_shape=None, **kwargs):
        super().__init__(*args, **kwargs)
        if target_obs_shape is None:
            raise ValueError("target_obs_shape must be provided.")
        self.target_observations_next = torch.zeros(
            self.num_transitions_per_env, self.num_envs, *target_obs_shape, device=self.device
        )

    def add_transitions(self, transition: Transition):
        super().add_transitions(transition)
        # `super().add_transitions` increments self.step, so we write at (step-1)
        if transition.target_observations_next is None:
            raise ValueError(
                "transition.target_observations_next is None. Did you forget to set it in process_env_step()?"
            )
        self.target_observations_next[self.step - 1].copy_(transition.target_observations_next)


class PPOFuture(PPO):
    """PPO with additional latent supervision and future-matching losses."""

    def __init__(
        self,
        policy,
        *args,
        w_vel: float = 1.0,
        w_foot: float = 1.0,
        w_future: float = 1.0,
        target_obs_dim: int = 0,
        **kwargs,
    ):
        super().__init__(policy, *args, **kwargs)
        self.w_vel = float(w_vel)
        self.w_foot = float(w_foot)
        self.w_future = float(w_future)
        self.target_obs_dim = int(target_obs_dim)
        if self.target_obs_dim <= 0:
            raise ValueError(f"target_obs_dim must be > 0 for PPOFuture. Got: {self.target_obs_dim}")

        # Override transition type so we can stash O(t+1).
        self.transition = RolloutStorageFuture.Transition()

    # ---------------- debug helpers ----------------
    def _debug_enabled(self) -> bool:
        return os.environ.get("DEBUG_FUTURE", "0") == "1"

    def init_storage(self, training_type, num_envs, num_transitions_per_env, obs, actions_shape):
        # Create base storage, but extended with target_next
        self.storage = RolloutStorageFuture(
            training_type,
            num_envs,
            num_transitions_per_env,
            obs,
            actions_shape,
            device=self.device,
            target_obs_shape=[self.target_obs_dim],
        )

    def process_env_step(self, obs, rewards, dones, extras):
        # Store target observation for next step (O(t+1))
        target_next = obs.get("target", None)
        if target_next is None:
            raise KeyError(
                "obs['target'] missing. Ensure env provides a 'target' observation group."
            )
        self.transition.target_observations_next = target_next.to(self.device).detach()
        super().process_env_step(obs, rewards, dones, extras)

    def update(self):  # noqa: C901
        """Override PPO.update() to add auxiliary losses.

        We follow the same basic structure as upstream PPO, but compute latent supervision losses
        on the *original* (non-augmented) samples to avoid coupling the aux targets to symmetry augmentation.
        """
        if self.policy.is_recurrent:
            raise NotImplementedError("PPOFuture only supports non-recurrent policies for now.")

        mse = torch.nn.MSELoss()

        storage = cast(RolloutStorageFuture, self.storage)
        policy: Any = self.policy

        # Flatten stored tensors (same base ordering for all modalities)
        observations = storage.observations.flatten(0, 1)
        target_next_all = storage.target_observations_next.flatten(0, 1)

        actions = storage.actions.flatten(0, 1)
        values = storage.values.flatten(0, 1)
        returns = storage.returns.flatten(0, 1)
        old_actions_log_prob = storage.actions_log_prob.flatten(0, 1)
        advantages = storage.advantages.flatten(0, 1)
        old_mu = storage.mu.flatten(0, 1)
        old_sigma = storage.sigma.flatten(0, 1)

        batch_size = storage.num_envs * storage.num_transitions_per_env
        mini_batch_size = batch_size // self.num_mini_batches

        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_vel_loss = 0.0
        mean_foot_loss = 0.0
        mean_future_loss = 0.0
        mean_symmetry_loss = 0.0 if self.symmetry else None

        if self.symmetry and self.symmetry["use_data_augmentation"]:
            raise NotImplementedError(
                "PPOFuture currently supports symmetry mirror-loss only. "
                "Set use_data_augmentation=False."
            )

        # iterate over epochs/minibatches with a shared index for obs/critic/target_next
        for _epoch in range(self.num_learning_epochs):
            indices = torch.randperm(self.num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)
            for i in range(self.num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                obs_batch = observations[batch_idx]
                target_next_batch = target_next_all[batch_idx]

                actions_batch = actions[batch_idx]
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]

                # recompute distribution/value
                policy.act(obs_batch)
                actions_log_prob_batch = policy.get_actions_log_prob(actions_batch)
                value_batch = policy.evaluate(obs_batch)

                mu_batch = policy.action_mean
                sigma_batch = policy.action_std
                entropy_batch = policy.entropy

                if self.desired_kl is not None and self.schedule == "adaptive":
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                            + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                            / (2.0 * torch.square(sigma_batch))
                            - 0.5,
                            axis=-1,
                        )
                        kl_mean = torch.mean(kl)
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        for param_group in self.optimizer.param_groups:
                            param_group["lr"] = self.learning_rate

                # surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                )
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # value loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                        -self.clip_param, self.clip_param
                    )
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

                # ---------- Symmetry mirror loss (mirror-only path) ----------
                if self.symmetry:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    # Symmetry function expects flattened policy observation tensor [B, actor_obs_dim].
                    # In PPOFuture, obs_batch can be a TensorDict with multiple groups.
                    obs_for_sym = policy.get_actor_obs(obs_batch)
                    if obs_for_sym.ndim == 1:
                        obs_for_sym = obs_for_sym.unsqueeze(0)
                    obs_mirrored_batch, _ = data_augmentation_func(
                        obs=obs_for_sym, actions=None, env=self.symmetry["_env"]
                    )
                    original_batch_size = obs_for_sym.shape[0]

                    # Policy mean on [orig; mirrored_obs]
                    mean_actions_batch = policy.act_inference(obs_mirrored_batch.detach().clone())
                    action_mean_orig = mean_actions_batch[:original_batch_size]

                    # Mirror target for action means: [orig; mirrored_action(orig)]
                    _, actions_mean_symm_batch = data_augmentation_func(
                        obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                    )

                    symmetry_loss = mse(
                        mean_actions_batch[original_batch_size:],
                        actions_mean_symm_batch.detach()[original_batch_size:],
                    )
                    if self.symmetry["use_mirror_loss"]:
                        loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                    else:
                        symmetry_loss = symmetry_loss.detach()

                # ---------- Auxiliary losses ----------
                latent = policy.get_latent(obs_batch)
                vel3_pred, foot6_pred, future_pred = policy.split_latent(latent)

                critic_obs_batch = policy.get_critic_obs(obs_batch)
                vel3_gt = critic_obs_batch[:, 0:3].detach()
                foot6_gt = critic_obs_batch[:, 3:9].detach()

                vel_loss = mse(vel3_pred, vel3_gt)
                foot_loss = mse(foot6_pred, foot6_gt)

                future_tgt = policy.encode_target(target_next_batch).detach()
                future_loss = mse(future_pred, future_tgt)

                loss = loss + self.w_vel * vel_loss + self.w_foot * foot_loss + self.w_future * future_loss

                # gradients
                self.optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # logging accum
                mean_value_loss += float(value_loss.item())
                mean_surrogate_loss += float(surrogate_loss.item())
                mean_entropy += float(entropy_batch.mean().item())
                mean_vel_loss += float(vel_loss.item())
                mean_foot_loss += float(foot_loss.item())
                mean_future_loss += float(future_loss.item())
                if mean_symmetry_loss is not None:
                    mean_symmetry_loss += float(symmetry_loss.item())

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_vel_loss /= num_updates
        mean_foot_loss /= num_updates
        mean_future_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        self.storage.clear()

        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "aux/vel": mean_vel_loss,
            "aux/foot": mean_foot_loss,
            "aux/future": mean_future_loss,
        }
        if self.symmetry:
            loss_dict["symmetry"] = float(mean_symmetry_loss)  # type: ignore[arg-type]
        return loss_dict

