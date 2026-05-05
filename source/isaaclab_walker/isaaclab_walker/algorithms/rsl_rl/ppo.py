# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# Copyright (c) 2025, Jaeyong Shin (jasonshin0537@snu.ac.kr).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic

class P73PPO(PPO):
    """Custom PPO implementation for P73 robot.
    
    This class inherits from the standard RSL-RL PPO and can be customized
    to override default behaviors.
    """
    def __init__(
        self,
        policy: ActorCritic,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 0.001,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        device: str = "cpu",
        normalize_advantage_per_mini_batch: bool = False,
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
        # Bound loss parameters
        bound_loss_cfg: dict | None = None,
    ) -> None:
        # rsl-rl-lib>=3.0.0 signature does NOT include `storage`; storage is initialized via `init_storage()`.
        super().__init__(
            policy,
            num_learning_epochs=num_learning_epochs,
            num_mini_batches=num_mini_batches,
            clip_param=clip_param,
            gamma=gamma,
            lam=lam,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            use_clipped_value_loss=use_clipped_value_loss,
            schedule=schedule,
            desired_kl=desired_kl,
            device=device,
            normalize_advantage_per_mini_batch=normalize_advantage_per_mini_batch,
            rnd_cfg=rnd_cfg,
            symmetry_cfg=symmetry_cfg,
            multi_gpu_cfg=multi_gpu_cfg,
        )
        print("[INFO] Using custom P73PPO Algorithm (bound loss enabled)" if bound_loss_cfg else "[INFO] Using custom P73PPO Algorithm")
        self.bound_loss_cfg = bound_loss_cfg

    def update(self) -> dict[str, float]:
        # Start from upstream PPO.update() implementation (rsl-rl-lib>=3.0.0) and add optional bound loss.
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0

        # -- RND loss
        mean_rnd_loss = 0.0 if self.rnd else None
        # -- Symmetry loss
        mean_symmetry_loss = 0.0 if self.symmetry else None
        # -- Bound loss
        mean_bound_loss = 0.0 if self.bound_loss_cfg else None

        # generator for mini batches
        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:
            # number of augmentations per sample
            num_aug = 1
            # original batch size
            original_batch_size = obs_batch.batch_size[0]

            # normalize advantages per mini-batch if configured
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # symmetry data augmentation
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch,
                    actions=actions_batch,
                    env=self.symmetry["_env"],
                )
                num_aug = int(obs_batch.batch_size[0] / original_batch_size)
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            # Recompute actions log prob and entropy for current batch
            if self.policy.is_recurrent:
                self.policy.act(obs_batch, masks=masks_batch, hidden_state=hid_states_batch[0])
                actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
                value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_state=hid_states_batch[1])
            else:
                self.policy.act(obs_batch)
                actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
                value_batch = self.policy.evaluate(obs_batch)

            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            # KL + adaptive LR
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

                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # Symmetry loss (mirror loss)
            if self.symmetry:
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(obs=obs_batch, actions=None, env=self.symmetry["_env"])
                    num_aug = int(obs_batch.shape[0] / original_batch_size)

                mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(obs=None, actions=action_mean_orig, env=self.symmetry["_env"])

                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                )
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # RND loss
            if self.rnd:
                with torch.no_grad():
                    rnd_state_batch = self.rnd.get_rnd_state(obs_batch[:original_batch_size])
                    rnd_state_batch = self.rnd.state_normalizer(rnd_state_batch)
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            # Bound loss (optional)
            if self.bound_loss_cfg:
                bound_loss_coef = float(self.bound_loss_cfg.get("bound_loss_coef", 0.0))
                soft_bound = float(self.bound_loss_cfg.get("bound_range", 1.0))
                bound_loss = self.bound_loss(mu_batch, soft_bound)
                loss += bound_loss_coef * bound_loss

            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            if self.rnd:
                self.rnd_optimizer.zero_grad()  # type: ignore[union-attr]
                rnd_loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            # logging accumulators
            mean_value_loss += float(value_loss.item())
            mean_surrogate_loss += float(surrogate_loss.item())
            mean_entropy += float(entropy_batch.mean().item())
            if mean_rnd_loss is not None:
                mean_rnd_loss += float(rnd_loss.item())
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += float(symmetry_loss.item())
            if mean_bound_loss is not None:
                mean_bound_loss += float(bound_loss.item())

        # average over updates
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        if mean_bound_loss is not None:
            mean_bound_loss /= num_updates

        self.storage.clear()

        loss_dict: dict[str, float] = {"value": mean_value_loss, "surrogate": mean_surrogate_loss, "entropy": mean_entropy}
        if self.rnd:
            loss_dict["rnd"] = float(mean_rnd_loss)  # type: ignore[arg-type]
        if self.symmetry:
            loss_dict["symmetry"] = float(mean_symmetry_loss)  # type: ignore[arg-type]
        if self.bound_loss_cfg:
            loss_dict["bound"] = float(mean_bound_loss)  # type: ignore[arg-type]
        return loss_dict

    def bound_loss(self, mu, soft_bound):
        mu_loss_high = torch.clamp_min(mu - soft_bound, 0.0)**2
        mu_loss_low = torch.clamp_max(mu + soft_bound, 0.0)**2
        b_loss = (mu_loss_low + mu_loss_high).sum()
        return b_loss