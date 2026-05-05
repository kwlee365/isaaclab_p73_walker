# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# Copyright (c) 2025, Jaeyong Shin (jasonshin0537@snu.ac.kr).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom event functions for P73 tasks (IsaacLab 5.1).

This file adds a startup physics material randomization term that also caches
per-environment (static_friction, dynamic_friction) so it can be used as a
2D observation term (TargetCfg).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast
import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class randomize_rigid_body_material_and_cache(ManagerTermBase):  # noqa: N801
    """Startup physics material randomization with a 2D cache for Target observation.

    Wraps IsaacLab's `randomize_rigid_body_material` event term but additionally stores:
      env._physics_material_sd: (num_envs, 2) = [static_friction, dynamic_friction]

    Cached values are the mean over shapes in the selected asset for each environment.
    """

    def __init__(self, cfg, env: "ManagerBasedEnv"):
        super().__init__(cfg=cfg, env=env)
        # defer import to avoid heavy dependency import at module import time
        from isaaclab.envs.mdp.events import randomize_rigid_body_material as _Base

        self._base = _Base(cast(Any, cfg), env)
        self._env = env

        # initialize cache so observation is always defined
        self._env.__dict__["_physics_material_sd"] = torch.zeros(env.num_envs, 2, device=env.device, dtype=torch.float32)

    def __call__(
        self,
        env: "ManagerBasedEnv",
        env_ids: torch.Tensor | None,
        static_friction_range,
        dynamic_friction_range,
        restitution_range,
        num_buckets,
        asset_cfg,
        make_consistent: bool = False,
    ):
        # Apply base randomization
        self._base(
            env=env,
            env_ids=env_ids,
            static_friction_range=static_friction_range,
            dynamic_friction_range=dynamic_friction_range,
            restitution_range=restitution_range,
            num_buckets=num_buckets,
            asset_cfg=asset_cfg,
            make_consistent=make_consistent,
        )

        # Read back and cache mean(static,dynamic) per env.
        # get_material_properties(): (num_envs, total_shapes, 3) on CPU
        materials = self._base.asset.root_physx_view.get_material_properties()
        if env_ids is None:
            env_ids_cpu = torch.arange(env.scene.num_envs, device="cpu")
        else:
            env_ids_cpu = env_ids.cpu()

        sd_mean_cpu = materials[env_ids_cpu, :, :2].mean(dim=1)  # (num_env_ids, 2)

        cache = self._env.__dict__.get("_physics_material_sd", None)
        if not isinstance(cache, torch.Tensor) or cache.shape != (env.num_envs, 2):
            cache = torch.zeros(env.num_envs, 2, device=env.device, dtype=torch.float32)
            self._env.__dict__["_physics_material_sd"] = cache

        cache[env_ids_cpu.to(device=env.device, dtype=torch.long)] = sd_mean_cpu.to(device=env.device, dtype=torch.float32)


def push_by_persistent_force(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    force_range: dict[str, tuple[float, float]],
    duration_s: tuple[float, float] = (0.5, 2.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_name: str = "base_link",
):
    """Apply a persistent external force on the base link for a random duration.

    Unlike push_by_setting_velocity (instantaneous), this applies a constant force
    over multiple simulation steps, making it estimable by the DWM encoder.

    The force and remaining duration are cached in env.__dict__ so that:
    1. The force is applied every step via permanent_wrench_composer.
    2. An observation term can read the current force for TargetCfg supervision.

    Args:
        force_range: Dict with keys "x", "y", "z" mapping to (min, max) in Newtons.
        duration_s: (min, max) duration in seconds for how long the force persists.
        asset_cfg: Scene entity configuration for the robot.
        body_name: Name of the body to apply force on.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    dt = env.step_dt

    # --- initialize buffers on first call ---
    key_force = "_persistent_push_force"
    key_remaining = "_persistent_push_remaining"
    if key_force not in env.__dict__:
        env.__dict__[key_force] = torch.zeros(env.num_envs, 3, device=env.device, dtype=torch.float32)
        env.__dict__[key_remaining] = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    force_buf: torch.Tensor = env.__dict__[key_force]
    remaining_buf: torch.Tensor = env.__dict__[key_remaining]

    # --- sample new force and duration for triggered envs ---
    n = len(env_ids)
    range_list = [force_range.get(k, (0.0, 0.0)) for k in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device=env.device, dtype=torch.float32)  # (3, 2)
    new_force = torch.rand(n, 3, device=env.device) * (ranges[:, 1] - ranges[:, 0]) + ranges[:, 0]

    dur_min, dur_max = duration_s
    dur_steps = (torch.rand(n, device=env.device) * (dur_max - dur_min) + dur_min) / dt
    dur_steps = dur_steps.to(torch.long).clamp(min=1)

    force_buf[env_ids] = new_force
    remaining_buf[env_ids] = dur_steps

    env.__dict__[key_force] = force_buf
    env.__dict__[key_remaining] = remaining_buf


def _step_persistent_push_force(
    env: "ManagerBasedEnv",
    asset_cfg_name: str = "robot",
    body_name: str = "base_link",
):
    """Step-level helper: decrement timer, apply/clear force via wrench composer.

    Called automatically from reward terms that read disturbance state.
    Ensures exactly one update per env step via a step-count guard.
    """
    key_force = "_persistent_push_force"
    key_remaining = "_persistent_push_remaining"
    key_last_step = "_persistent_push_last_step"
    if key_force not in env.__dict__:
        return

    # guard: only update once per step
    current_step = env.common_step_counter
    last_step = env.__dict__.get(key_last_step, -1)
    if current_step == last_step:
        return
    env.__dict__[key_last_step] = current_step

    asset: Articulation = env.scene[asset_cfg_name]
    force_buf: torch.Tensor = env.__dict__[key_force]
    remaining_buf: torch.Tensor = env.__dict__[key_remaining]

    # clear force for envs whose timer expired
    expired = remaining_buf <= 0
    force_buf[expired] = 0.0

    # decrement remaining for active envs
    active_mask = remaining_buf > 0
    remaining_buf[active_mask] -= 1

    env.__dict__[key_force] = force_buf
    env.__dict__[key_remaining] = remaining_buf

    # find the body index
    body_ids, _ = asset.find_bodies(body_name, preserve_order=True)
    if len(body_ids) != 1:
        raise RuntimeError(f"Expected exactly one body for {body_name!r}. Got {body_ids}")
    body_id = body_ids[0]

    # apply force via permanent_wrench_composer
    forces = force_buf.unsqueeze(1)  # (N, 1, 3)
    torques = torch.zeros_like(forces)
    body_ids_t = torch.tensor([body_id], device=env.device, dtype=torch.int32)

    asset.permanent_wrench_composer.set_forces_and_torques(
        forces=forces,
        torques=torques,
        body_ids=body_ids_t,
        is_global=True,
    )

