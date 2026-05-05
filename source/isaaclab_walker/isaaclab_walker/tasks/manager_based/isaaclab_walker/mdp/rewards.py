# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# Copyright (c) 2025, Jaeyong Shin (jasonshin0537@snu.ac.kr).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    Always active regardless of command velocity (no zero-command gate).
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.05
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]),
        dim=1,
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


"""Custom rewards and penalties"""


def different_step_times(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize different step times between the two feet.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the penalty
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    penalty = 2 * (torch.std(last_contact_time, dim=1) + torch.std(last_air_time, dim=1))
    # no penalty for zero command
    penalty *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.05
    return penalty


def different_air_contact_times(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize huge difference between air and contact times.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the penalty
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    air_contact_time_abs_diff = torch.abs(last_air_time - last_contact_time)
    penalty = torch.sum(air_contact_time_abs_diff, dim=1)
    # no penalty for zero command
    penalty *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.05
    return penalty


def feet_ground_parallel(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.2,
) -> torch.Tensor:
    """Penalize feet orientation deviation from being parallel to the ground.

    This function ensures that the feet are oriented parallel to the ground when in contact.
    The penalty is computed using the deviation of gravity direction expressed in each foot frame.

    Args:
        env: The RL environment instance.
        sensor_cfg: Configuration for contact sensor to detect foot contact.
        asset_cfg: Configuration for the robot asset (feet bodies).
        threshold: Contact force threshold to determine if foot is in contact.

    Returns:
        Penalty tensor of shape (num_envs,).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]

    # Check if feet are in contact with the ground (history-based robust contact)
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
        > threshold
    )  # (num_envs, num_feet)

    # Feet orientation in world frame
    feet_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids]  # (num_envs, num_feet, 4)

    # Gravity vector in world (broadcast to feet)
    gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=env.device, dtype=torch.float32)
    gravity_vec = gravity_vec.unsqueeze(0).unsqueeze(0).expand(feet_quat_w.shape[0], feet_quat_w.shape[1], -1)

    # Express gravity in feet frame: if foot is parallel to ground, xy components should be ~0
    feet_gravity_local = quat_apply_inverse(feet_quat_w, gravity_vec)
    feet_orientation_error = torch.sum(torch.square(feet_gravity_local[:, :, :2]), dim=2)  # (num_envs, num_feet)

    # Only apply penalty when feet are in contact
    penalty = torch.sum(feet_orientation_error * contacts, dim=1)  # (num_envs,)
    return penalty


def feet_ground_parallel_swing(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize feet orientation deviation from being parallel to the ground
    during the swing (no-contact) phase.

    Mirror of `feet_ground_parallel` but the contact mask is inverted: the
    penalty is summed only over feet that are NOT in contact. Use a smaller
    weight than the contact-phase term — swing-phase foot rotation is more
    natural near touchdown / takeoff and an over-aggressive penalty conflicts
    with ankle torque limits.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]

    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
        > threshold
    )  # (num_envs, num_feet)
    swing = (~contacts).float()

    feet_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids]
    gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=env.device, dtype=torch.float32)
    gravity_vec = gravity_vec.unsqueeze(0).unsqueeze(0).expand(feet_quat_w.shape[0], feet_quat_w.shape[1], -1)
    feet_gravity_local = quat_apply_inverse(feet_quat_w, gravity_vec)
    feet_orientation_error = torch.sum(torch.square(feet_gravity_local[:, :, :2]), dim=2)

    penalty = torch.sum(feet_orientation_error * swing, dim=1)
    return penalty


def feet_parallel(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.2,
) -> torch.Tensor:
    """Penalize variance in feet yaw orientation when both feet are in contact."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]

    # Contact detection
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
        > threshold
    )  # (num_envs, num_feet)

    # Feet orientations
    feet_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids]  # (num_envs, num_feet, 4)

    # Compute yaw for each foot quaternion
    yaw_angles = []
    for i in range(feet_quat_w.shape[1]):
        foot_quat = feet_quat_w[:, i, :]
        yaw = torch.atan2(
            2.0 * (foot_quat[:, 3] * foot_quat[:, 2] + foot_quat[:, 0] * foot_quat[:, 1]),
            1.0 - 2.0 * (foot_quat[:, 1] ** 2 + foot_quat[:, 2] ** 2),
        )
        yaw_angles.append(yaw)
    feet_yaw = torch.stack(yaw_angles, dim=1)  # (num_envs, num_feet)

    yaw_variance = torch.var(feet_yaw, dim=1)  # (num_envs,)
    all_feet_contact = torch.all(contacts, dim=1)  # (num_envs,)
    return yaw_variance * all_feet_contact.float()


def feet_stumble(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold_ratio: float = 3.0,
) -> torch.Tensor:
    """Penalize feet stumbling on obstacles (horizontal force exceeding vertical force).

    Returns:
        (num_envs,) binary penalty (1.0 if any foot stumbles).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]  # (num_envs, num_feet, 3)

    force_xy = torch.norm(contact_forces[:, :, :2], dim=-1)
    force_z = torch.abs(contact_forces[:, :, 2])

    stumble_condition = force_xy > (threshold_ratio * force_z)
    return torch.any(stumble_condition, dim=1).float()


def contact_momentum(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize impact momentum during foot landing for soft contact.

    Computes: sum_feet |v_z * F_z|
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]

    feet_vel_w = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :]  # (num_envs, num_feet, 3)
    feet_vel_z = feet_vel_w[:, :, 2]  # (num_envs, num_feet)

    contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]  # (num_envs, num_feet, 3)
    contact_force_z = contact_forces[:, :, 2]  # (num_envs, num_feet)

    momentum = torch.abs(feet_vel_z * contact_force_z)  # (num_envs, num_feet)
    return torch.sum(momentum, dim=1)


def joint_torques_limit_soft_l2(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    soft_ratio: float = 0.8,
) -> torch.Tensor:
    """Penalize applied torques that exceed ``soft_ratio * joint_effort_limit``.

    Per-joint normalized quadratic overage:
        overage_i = max(0, |τ_i| / τ_max_i - soft_ratio)
        reward    = - Σ_i overage_i²

    Below the soft threshold the term contributes zero, so normal torque usage is
    unpenalized. The normalization by ``τ_max`` makes each joint contribute on the
    same scale regardless of motor capacity.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    tau = asset.data.applied_torque[:, asset_cfg.joint_ids]
    tau_max = asset.data.joint_effort_limits[:, asset_cfg.joint_ids]
    norm = tau.abs() / tau_max.clamp(min=1e-6)
    over = (norm - soft_ratio).clamp(min=0.0)
    return torch.sum(over * over, dim=1)


def action_rate_l2_selective(
    env: "ManagerBasedRLEnv",
    action_indices: list[int],
) -> torch.Tensor:
    """Per-joint action-rate L2 penalty restricted to the given action indices.

    Enables per-joint-group weighting of ``|a_t - a_{t-1}|²``. Real-robot data
    shows |Δa| varies ~14× across joints (Knee p99≈0.14, HipRoll p99≈0.018),
    so splitting by group lets jerky joints carry a heavier weight without
    over-regularizing already-quiet ones.
    """
    am = env.action_manager
    idx = torch.as_tensor(action_indices, device=am.action.device, dtype=torch.long)
    diff = am.action.index_select(1, idx) - am.prev_action.index_select(1, idx)
    return torch.sum(diff * diff, dim=1)


def action_accel_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the second-order difference of the actions using L2 squared kernel.

    This corresponds to penalizing rapid changes in the action-rate (i.e., action "acceleration" in discrete time):
        Δ²a_t = a_t - 2 a_{t-1} + a_{t-2}
    """
    # Isaac Lab versions differ in what action history is stored by ActionManager.
    # - Some versions expose `prev_prev_action` (a_{t-2}), enabling true second-order difference.
    # - If it is missing, fall back to first-order action difference (equivalent to action-rate penalty).
    am = env.action_manager
    a_t = am.action
    a_t_1 = getattr(am, "prev_action", None)
    a_t_2 = getattr(am, "prev_prev_action", None)

    if a_t_1 is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    if a_t_2 is None:
        # fallback: Δa_t = a_t - a_{t-1}
        return torch.sum(torch.square(a_t - a_t_1), dim=1)

    return torch.sum(torch.square(a_t - 2.0 * a_t_1 + a_t_2), dim=1)


def ref_feet_yaw_mismatch_l2(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    *,
    reference_body_name: str = "Head_Link",
    reference_cfg: SceneEntityCfg | None = None,
    threshold: float = 0.2,
) -> torch.Tensor:
    """Penalize yaw(heading) mismatch between a reference body and each foot (L2 on wrapped angle).

    This term is useful when the policy learns feet that are parallel to each other,
    but the pelvis/base orientation drifts relative to the upper-body (e.g., head heading).

    Gating:
        Per-foot contact gating: each foot's penalty is applied only when that foot is in contact.

    Args:
        env: RL environment.
        sensor_cfg: Contact sensor configuration resolving feet contact bodies.
        asset_cfg: Robot articulation configuration resolving feet bodies for orientation.
        reference_body_name: Body name/regex to resolve the reference body (e.g., "Head_Link" or "Pelvis_Link").
        reference_cfg: Optional SceneEntityCfg to pre-resolve exactly one reference body id.
        threshold: Contact force threshold (N).

    Returns:
        (num_envs,) penalty.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]

    # Per-foot contacts: (num_envs, num_feet)
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > threshold
    )

    # Resolve reference body id (must be exactly one).
    ref_id: int
    if reference_cfg is not None and not isinstance(reference_cfg.body_ids, slice):
        ref_ids = reference_cfg.body_ids
        if len(ref_ids) != 1:
            raise RuntimeError(f"reference_cfg must resolve exactly one body id. Got body_ids={ref_ids}.")
        ref_id = int(ref_ids[0])
    else:
        ref_ids, ref_names = asset.find_bodies(reference_body_name, preserve_order=True)
        if len(ref_ids) != 1:
            raise RuntimeError(
                f"Expected exactly one body match for reference_body_name={reference_body_name!r}. "
                f"Got ids={ref_ids}, names={ref_names}."
            )
        ref_id = int(ref_ids[0])

    # Body quaternions in world: (num_envs, num_feet, 4) and (num_envs, 4)
    feet_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids]
    ref_quat_w = asset.data.body_quat_w[:, ref_id, :]

    # Yaw extraction consistent with existing `feet_parallel` implementation.
    # Quaternion ordering assumed by the formula: (x, y, z, w).
    feet_yaw = torch.atan2(
        2.0 * (feet_quat_w[..., 3] * feet_quat_w[..., 2] + feet_quat_w[..., 0] * feet_quat_w[..., 1]),
        1.0 - 2.0 * (feet_quat_w[..., 1] ** 2 + feet_quat_w[..., 2] ** 2),
    )  # (num_envs, num_feet)
    ref_yaw = torch.atan2(
        2.0 * (ref_quat_w[..., 3] * ref_quat_w[..., 2] + ref_quat_w[..., 0] * ref_quat_w[..., 1]),
        1.0 - 2.0 * (ref_quat_w[..., 1] ** 2 + ref_quat_w[..., 2] ** 2),
    )  # (num_envs,)

    # Wrapped yaw difference: [-pi, pi]
    dyaw = feet_yaw - ref_yaw.unsqueeze(1)
    dyaw_wrapped = torch.atan2(torch.sin(dyaw), torch.cos(dyaw))

    # L2 penalty, per-foot contact gated
    penalty_per_foot = torch.square(dyaw_wrapped) * contacts.float()
    return torch.sum(penalty_per_foot, dim=1)


def feet_yaw_symmetry_about_base_l2(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    *,
    reference_body_name: str = "base_link",
    threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize asymmetric foot yaw relative to the base heading.

    Computes the midpoint of left-foot yaw and right-foot yaw, then penalizes
    the angular difference between that midpoint and the base (reference) yaw.
    When both feet are symmetrically splayed about the base heading, the penalty is zero.

    Requires exactly 2 feet in asset_cfg.body_ids (L then R, preserve_order=True).
    Contact-gated: penalty is applied only when both feet are in contact.

    Args:
        env: RL environment.
        sensor_cfg: Contact sensor config for feet (must resolve 2 bodies).
        asset_cfg: Robot config for feet bodies (must resolve 2 bodies, L then R).
        reference_body_name: Body name for the base/pelvis heading reference.
        threshold: Contact force threshold (N).

    Returns:
        (num_envs,) penalty.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]

    # Contact gating: both feet must be in contact
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > threshold
    )  # (num_envs, 2)
    both_contact = contacts[:, 0] & contacts[:, 1]  # (num_envs,)

    # Resolve reference body
    ref_ids, _ = asset.find_bodies(reference_body_name, preserve_order=True)
    ref_id = int(ref_ids[0])

    # Quaternions: Isaac Sim uses (w, x, y, z) but internal storage is (x, y, z, w)
    feet_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids]  # (num_envs, 2, 4)
    ref_quat_w = asset.data.body_quat_w[:, ref_id, :]  # (num_envs, 4)

    # Extract yaw from quaternion (x, y, z, w ordering)
    def _yaw(q):
        return torch.atan2(
            2.0 * (q[..., 3] * q[..., 2] + q[..., 0] * q[..., 1]),
            1.0 - 2.0 * (q[..., 1] ** 2 + q[..., 2] ** 2),
        )

    left_yaw = _yaw(feet_quat_w[:, 0])   # (num_envs,)
    right_yaw = _yaw(feet_quat_w[:, 1])   # (num_envs,)
    base_yaw = _yaw(ref_quat_w)           # (num_envs,)

    # Midpoint of left and right foot yaw (circular mean of 2 angles)
    # Use atan2(sin, cos) to handle wrapping correctly
    mid_yaw = torch.atan2(
        torch.sin(left_yaw - base_yaw) + torch.sin(right_yaw - base_yaw),
        torch.cos(left_yaw - base_yaw) + torch.cos(right_yaw - base_yaw),
    )  # angle of midpoint relative to base_yaw; if symmetric, this is ~0

    penalty = torch.square(mid_yaw) * both_contact.float()
    return penalty


def _tocabi_compute_cmd_and_body_vel(
    env: ManagerBasedRLEnv,
    *,
    asset: Articulation,
    command_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute command and body planar velocity magnitudes used across TOCABI rewards.

    - cmd_vel = ||cmd_xy|| + |cmd_yaw|
    - body_vel = ||root_lin_vel_b_xy||
    """
    cmd = env.command_manager.get_command(command_name)  # (num_envs, 3) [vel_x, vel_y, ang_z]
    cmd_vel = torch.norm(cmd[:, :2], dim=1) + torch.abs(cmd[:, 2])  # (num_envs,)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)  # (num_envs,)
    return cmd_vel, body_vel


def _tocabi_compute_feet_yaw_rel_and_stance_metrics(
    env: ManagerBasedRLEnv,
    *,
    asset: Articulation,
    sensor_cfg: SceneEntityCfg,
    contact_threshold: float,
    stance_width_m: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute yaw_rel(foot vs base) and simple stance metrics in yaw-only base frame.

    Frames:
    - yaw_rel is computed in a gravity-aligned base frame (yaw-only base frame):
      yaw_rel = wrap_to_pi(yaw_foot_w - yaw_root_w) for each foot.
    - stance metrics use feet positions relative to root_pos_w, rotated by -yaw_root_w (Rz only).

    Returns:
        yaw_rel: (num_envs, 2) yaw(foot) - yaw(base) wrapped to [-pi, pi].
        x_mid: (num_envs,) midpoint x in yaw-only base frame.
        y_mid: (num_envs,) midpoint y in yaw-only base frame.
        y_sep: (num_envs,) absolute lateral separation in yaw-only base frame.
        x_sep: (num_envs,) absolute longitudinal separation in yaw-only base frame.
        both_in_contact: (num_envs,) True if both feet have Fz > contact_threshold.
    """
    body_ids = sensor_cfg.body_ids
    if body_ids is None:
        raise RuntimeError("SceneEntityCfg.body_ids is None. Ensure sensor_cfg resolves body_ids at runtime.")
    assert body_ids is not None

    # --- contacts ---
    contact_sensor = cast(ContactSensor, env.scene.sensors[sensor_cfg.name])
    net_forces_w = contact_sensor.data.net_forces_w
    if net_forces_w is None:
        raise RuntimeError("ContactSensor net_forces_w buffer is None.")
    contact_fz = net_forces_w[:, body_ids, 2]  # (num_envs, 2)
    feet_in_contact = contact_fz > contact_threshold
    both_in_contact = torch.all(feet_in_contact, dim=1)  # (num_envs,)

    # --- yaw difference in yaw-only base frame ---
    root_quat_w = asset.data.root_quat_w  # (num_envs, 4) in (w, x, y, z)
    _, _, yaw_base = math_utils.euler_xyz_from_quat(root_quat_w)  # (num_envs,)

    feet_quat_w = asset.data.body_quat_w[:, body_ids]  # (num_envs, 2, 4)
    _, _, yaw_feet_flat = math_utils.euler_xyz_from_quat(feet_quat_w.reshape(-1, 4))
    yaw_feet = yaw_feet_flat.view(feet_quat_w.shape[0], feet_quat_w.shape[1])  # (num_envs, 2)
    yaw_rel = math_utils.wrap_to_pi(yaw_feet - yaw_base.unsqueeze(1))  # (num_envs, 2)

    # --- stance error in yaw-only base frame ---
    root_pos_w_xy = asset.data.root_pos_w[:, :2]  # (num_envs, 2)
    feet_pos_w_xy = asset.data.body_pos_w[:, body_ids, :2]  # (num_envs, 2, 2)
    rel_xy = feet_pos_w_xy - root_pos_w_xy.unsqueeze(1)  # (num_envs, 2, 2)

    # rotate by -yaw_base: [x'; y'] = Rz(-yaw) [x; y]
    cy = torch.cos(-yaw_base).unsqueeze(1)
    sy = torch.sin(-yaw_base).unsqueeze(1)
    x = rel_xy[..., 0]
    y = rel_xy[..., 1]
    x_r = cy * x - sy * y
    y_r = sy * x + cy * y

    x_mid = 0.5 * (x_r[:, 0] + x_r[:, 1])
    y_mid = 0.5 * (y_r[:, 0] + y_r[:, 1])
    y_sep = torch.abs(y_r[:, 0] - y_r[:, 1])
    x_sep = torch.abs(x_r[:, 0] - x_r[:, 1])

    return yaw_rel, x_mid, y_mid, y_sep, x_sep, both_in_contact

def feet_air_time_biped(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    threshold: float,
    velocity_threshold: float = 0.1,
    *,
    contact_threshold: float = 5.0,
    yaw_threshold_deg: float = 10.0,
    pos_threshold_m: float = 0.01,
    stance_width_m: float = 0.20,
    x_sep_target_m: float = 0.0,
    stop_cmd_vel_max: float = 0.05,
    # Schedule gate (shared with contact_schedule / swing_clearance) for grace wind-down and push recovery.
    cmd_zero_max: float = 1.0e-3,
    grace_steps: int = 0,
    push_suppress_threshold: float = 0.0,
    push_activates: bool = False,
) -> torch.Tensor:
    """Reward air time when walking, reward contact time when standing (biped version).

    This is a humanoid-adapted version of Boston Dynamics Spot's air_time_reward.
    Unlike the quadruped version which hardcodes 4 feet, this works with bipeds (2 feet).

    Key difference from basic feet_air_time:
    - Walking (v > threshold): reward air time (encourages stepping)
    - Standing (v < threshold): reward contact time (discourages in-place stepping!)

    This prevents the common reward hacking where robots step in place to gain air time reward.

    Args:
        env: The learning environment.
        command_name: Name of the base velocity command.
        sensor_cfg: Scene entity configuration for contact sensors.
        asset_cfg: Scene entity configuration for the robot asset.
        threshold: Minimum air time to receive reward (mode_time in Spot).
        velocity_threshold: Velocity below which robot is considered standing.

    Returns:
        Reward tensor of shape (num_envs,).
    """
    # Extract sensors and asset
    # NOTE: `env.scene.sensors[...]` is typed as a generic SensorBase in IsaacLab.
    # We cast here to satisfy the type-checker; runtime object is a ContactSensor.
    contact_sensor = cast(ContactSensor, env.scene.sensors[sensor_cfg.name])

    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")

    body_ids = sensor_cfg.body_ids
    if body_ids is None:
        raise RuntimeError("SceneEntityCfg.body_ids is None. Ensure sensor_cfg resolves body_ids at runtime.")
    assert body_ids is not None

    # Get current air and contact times for feet
    air_time_all = contact_sensor.data.current_air_time
    contact_time_all = contact_sensor.data.current_contact_time
    if air_time_all is None or contact_time_all is None:
        raise RuntimeError("ContactSensor air/contact time buffers are None. Ensure track_air_time is enabled.")
    current_air_time = air_time_all[:, body_ids]  # (num_envs, num_feet)
    current_contact_time = contact_time_all[:, body_ids]  # (num_envs, num_feet)

    # Compute max time (either air or contact)
    t_max = torch.max(current_air_time, current_contact_time)  # (num_envs, num_feet)
    t_min = torch.clip(t_max, max=threshold)  # (num_envs, num_feet)

    # Standing reward: prefer contact over air (negative if in air)
    # This is the key to preventing in-place stepping!
    stance_reward = torch.clip(current_contact_time - current_air_time, -threshold, threshold)  # (num_envs, num_feet)

    # Determine if walking or standing (per-env), using a simple stop-alignment switch.
    # Then expand to per-foot.
    is_walking_like_env, _ = tocabi_should_walk_stop_align(
        env,
        command_name=command_name,
        sensor_cfg=sensor_cfg,
        asset_cfg=asset_cfg,
        velocity_threshold=velocity_threshold,
        contact_threshold=contact_threshold,
        stop_cmd_vel_max=stop_cmd_vel_max,
        yaw_threshold_deg=yaw_threshold_deg,
        pos_threshold_m=pos_threshold_m,
        stance_width_m=stance_width_m,
        x_sep_target_m=x_sep_target_m,
    )
    # Force walking branch while schedule is active (grace wind-down / push recovery),
    # so a sudden cmd→0 still produces a few more forced steps before stance lock-in.
    schedule_active = _tocabi_is_schedule_active_from_cmd(
        env,
        command_name=command_name,
        cmd_zero_max=cmd_zero_max,
        grace_steps=grace_steps,
        push_suppress_threshold=push_suppress_threshold,
        push_activates=push_activates,
    )
    is_walking_like_env = is_walking_like_env | schedule_active
    num_feet = current_air_time.shape[1]
    is_walking = is_walking_like_env.unsqueeze(1).expand(-1, num_feet)  # (num_envs, num_feet)

    # Select reward based on walking/standing
    reward = torch.where(
        is_walking,
        torch.where(t_max < threshold, t_min, 0.0),  # Walking: reward air time
        stance_reward,  # Standing: reward contact time (penalize air time!)
    )  # (num_envs, num_feet)

    # Sum over all feet
    return torch.sum(reward, dim=1)  # (num_envs,)

def low_speed_feet_alignment_penalty(
    env: ManagerBasedRLEnv,
    *,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    # Activation gate (user intent: cmd==0 and base speed is very low)
    cmd_zero_max: float = 1.0e-3,
    body_vel_max: float = 0.05,
    # Velocity source for the gate (optional): if set, use this body's planar (xy) linear speed instead of root/base.
    # Example: vel_body_name="Head_Link" to gate by head linear velocity.
    vel_body_name: str | None = None,
    # Alignment thresholds (same semantics as tocabi_should_walk_stop_align)
    yaw_threshold_deg: float = 10.0,
    pos_threshold_m: float = 0.01,
    stance_width_m: float = 0.20,
    x_sep_target_m: float = 0.0,
    contact_threshold: float = 5.0,
    # Optional scaling
    yaw_scale: float = 1.0,
    pos_scale: float = 1.0,
    require_contact: bool = False,
) -> torch.Tensor:
    """Continuous penalty for feet misalignment in low-speed stop condition.

    Motivation:
        When the commanded velocity is (near) zero and the base is already slow,
        we want to *force* the policy to finish a clean feet alignment (yaw + stance),
        so that standing-mode rewards (e.g., `feet_air_time_biped` stance branch) can take over.

    Gate (activation):
        - cmd_vel = ||cmd_xy|| + |cmd_wz| <= cmd_zero_max
        - gate_vel = ||root_lin_vel_b_xy|| <= body_vel_max  (default)
          or gate_vel = ||body_lin_vel_w_xy(vel_body_name)|| <= body_vel_max  (if vel_body_name is provided)

    Penalty (continuous, thresholded):
        - yaw_err: sum over feet of relu(|yaw_rel| - yaw_th)
        - pos_err: relu(mid_err - pos_threshold_m) + relu(width_err - pos_threshold_m)
                   + relu(x_sep_err - pos_threshold_m)
          where x_sep_err = ||x_L - x_R| - x_sep_target_m| (longitudinal stance lock).

    Returns:
        Non-negative penalty tensor of shape (num_envs,). Use a negative weight in the reward config.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # --- gate: cmd == 0 AND base is already slow ---
    cmd_vel, body_vel = _tocabi_compute_cmd_and_body_vel(env, asset=asset, command_name=command_name)
    if vel_body_name is not None:
        vel_ids, vel_names = asset.find_bodies(vel_body_name, preserve_order=True)
        if len(vel_ids) != 1:
            raise RuntimeError(
                f"Expected exactly one body match for vel_body_name={vel_body_name!r}. "
                f"Got ids={vel_ids}, names={vel_names}."
            )
        vel_id = vel_ids[0]
        # Use world-frame linear velocity for the selected body and gate by planar speed.
        body_vel = torch.linalg.norm(asset.data.body_lin_vel_w[:, vel_id, :2], dim=1)
    is_low_speed = torch.logical_and(cmd_vel <= cmd_zero_max, body_vel <= body_vel_max)

    # --- alignment metrics (yaw-only base frame) ---
    yaw_rel, x_mid, y_mid, y_sep, x_sep, both_in_contact = _tocabi_compute_feet_yaw_rel_and_stance_metrics(
        env,
        asset=asset,
        sensor_cfg=sensor_cfg,
        contact_threshold=contact_threshold,
        stance_width_m=stance_width_m,
    )

    if require_contact:
        is_low_speed = torch.logical_and(is_low_speed, both_in_contact)

    yaw_th = (yaw_threshold_deg * torch.pi) / 180.0
    yaw_excess = torch.clamp(torch.abs(yaw_rel) - yaw_th, min=0.0)  # (num_envs, 2)
    yaw_err = torch.sum(yaw_excess, dim=1)  # (num_envs,)

    mid_err = torch.sqrt(x_mid * x_mid + y_mid * y_mid)  # (num_envs,)
    width_err = torch.abs(y_sep - stance_width_m)  # (num_envs,)
    x_sep_err = torch.abs(x_sep - x_sep_target_m)  # (num_envs,)
    pos_err = (
        torch.clamp(mid_err - pos_threshold_m, min=0.0)
        + torch.clamp(width_err - pos_threshold_m, min=0.0)
        + torch.clamp(x_sep_err - pos_threshold_m, min=0.0)
    )

    penalty = yaw_scale * yaw_err + pos_scale * pos_err
    return torch.where(is_low_speed, penalty, torch.zeros_like(penalty))

def low_speed_double_support_penalty(
    env: ManagerBasedRLEnv,
    *,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    # Activation gate: same semantics as low_speed_feet_alignment_penalty
    cmd_zero_max: float = 1.0e-3,
    body_vel_max: float = 0.05,
    vel_body_name: str | None = None,
    # Contact detection
    contact_threshold: float = 5.0,
) -> torch.Tensor:
    """Penalize not having *both* feet in contact during low-speed stop condition.

    This complements `single_foot_standing_penalty` by also penalizing the "both feet in air" case
    under a strict stop gate (cmd near zero AND body velocity already low).

    Gate (activation):
        - cmd_vel = ||cmd_xy|| + |cmd_wz| <= cmd_zero_max
        - gate_vel = ||root_lin_vel_b_xy|| <= body_vel_max (default)
          or gate_vel = ||body_lin_vel_w_xy(vel_body_name)|| <= body_vel_max (if vel_body_name is provided)

    Penalty:
        - returns 1.0 when active AND (NOT both feet have Fz > contact_threshold), else 0.0

    Returns:
        Non-negative penalty tensor of shape (num_envs,). Use a negative weight in the reward config.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # --- gate: cmd == 0 AND body is already slow ---
    cmd_vel, body_vel = _tocabi_compute_cmd_and_body_vel(env, asset=asset, command_name=command_name)
    if vel_body_name is not None:
        vel_ids, vel_names = asset.find_bodies(vel_body_name, preserve_order=True)
        if len(vel_ids) != 1:
            raise RuntimeError(
                f"Expected exactly one body match for vel_body_name={vel_body_name!r}. "
                f"Got ids={vel_ids}, names={vel_names}."
            )
        vel_id = vel_ids[0]
        body_vel = torch.linalg.norm(asset.data.body_lin_vel_w[:, vel_id, :2], dim=1)
    is_low_speed = torch.logical_and(cmd_vel <= cmd_zero_max, body_vel <= body_vel_max)

    # --- contacts ---
    contact_sensor = cast(ContactSensor, env.scene.sensors[sensor_cfg.name])
    body_ids = sensor_cfg.body_ids
    if body_ids is None:
        raise RuntimeError("SceneEntityCfg.body_ids is None. Ensure sensor_cfg resolves body_ids at runtime.")
    net_forces_w = contact_sensor.data.net_forces_w
    if net_forces_w is None:
        raise RuntimeError("ContactSensor net_forces_w buffer is None.")
    contact_fz = net_forces_w[:, body_ids, 2]  # (num_envs, num_feet)
    feet_in_contact = contact_fz > contact_threshold
    both_in_contact = torch.all(feet_in_contact, dim=1)  # (num_envs,)

    penalty = torch.where(torch.logical_and(is_low_speed, ~both_in_contact), torch.ones_like(cmd_vel), torch.zeros_like(cmd_vel))
    return penalty

def feet_yaw_align_cmd_gated_l2(
    env: "ManagerBasedRLEnv",
    *,
    command_name: str,
    wz_threshold: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize feet yaw misalignment (L2) regardless of contact, but gate it during turning.

    This term is intended to keep both feet's yaw headings aligned even during walking (including swing/airborne),
    while avoiding harming turning behaviors. We disable the penalty when the commanded yaw-rate is positive:
        gate = 1 if cmd_wz <= wz_threshold else 0
    With the default wz_threshold=0.0, this means: disable as soon as cmd_wz > 0.

    Penalty:
        (wrap_to_pi(yaw_left - yaw_right))^2 * gate

    Args:
        env: The learning environment.
        command_name: Base velocity command name (contains [vx, vy, wz]).
        wz_threshold: Yaw-rate threshold (rad/s). The penalty is disabled when cmd_wz > wz_threshold.
        asset_cfg: Scene entity configuration for robot feet bodies (must resolve exactly 2 feet bodies).

    Returns:
        Penalty tensor (num_envs,): yaw misalignment penalty when cmd_wz <= wz_threshold else 0.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # Resolve commanded yaw-rate
    cmd = env.command_manager.get_command(command_name)  # (num_envs, 3) [vx, vy, wz]
    gate = (cmd[:, 2] <= wz_threshold).to(dtype=torch.float32)

    # Feet orientations in world (quaternion in (w, x, y, z)).
    feet_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids, :]  # (num_envs, num_feet, 4)
    if feet_quat_w.shape[1] != 2:
        # Keep safe: if config resolves an unexpected number of feet bodies, do nothing.
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    _, _, yaw_left = math_utils.euler_xyz_from_quat(feet_quat_w[:, 0, :])
    _, _, yaw_right = math_utils.euler_xyz_from_quat(feet_quat_w[:, 1, :])

    dyaw = math_utils.wrap_to_pi(yaw_left - yaw_right)
    penalty = torch.square(dyaw)

    return penalty * gate

def feet_clearance_height(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_height: float = 0.05
) -> torch.Tensor:
    """Penalize feet not reaching target clearance height when moving horizontally.
    
    수식: penalty = Σ_{feet} (p_z^{target} - p_z^i)^2 * v_{xy}^i
    
    이 리워드는 로봇이 걸을 때 발을 충분히 들어 올리도록 유도합니다:
    - p_z^i: 발의 현재 높이 (world frame Z coordinate)
    - p_z^{target}: 목표 지면 이격 높이 (target clearance height)
    - v_{xy}^i: 발의 수평 속도 (XY plane velocity magnitude)
    
    발이 목표 높이보다 낮을 때, 수평으로 움직이면 페널티를 줍니다.
    이는 발 끌림(foot scuffing) 현상을 방지하고 안정적인 보행을 유도합니다.
    
    Args:
        env: RL environment instance
        asset_cfg: Configuration for feet bodies (use body_names to specify feet)
        target_height: Target clearance height above ground (meters, default: 0.05m = 5cm)
        
    Returns:
        torch.Tensor: (num_envs,) Feet clearance penalty
        
    Note:
        - 가중치 권장: -0.25
        - "발을 옆으로 움직이려면, 먼저 발을 목표 높이만큼 들어라"
        - 발이 공중에 있을 때만 작동 (지면 접촉 시에는 페널티 없음)
    """
    # Extract the used quantities
    asset = env.scene[asset_cfg.name]
    
    # 발의 현재 위치 (world frame) - (num_envs, num_feet, 3)
    feet_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    
    # 발의 현재 높이 (Z coordinate) - (num_envs, num_feet)
    feet_height = feet_pos_w[:, :, 2]
    
    # 발의 선속도 (world frame) - (num_envs, num_feet, 3)
    feet_vel_w = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :]
    
    # 발의 수평 속도 (XY plane) - (num_envs, num_feet)
    # v_xy = sqrt(v_x^2 + v_y^2)
    feet_vel_xy = torch.norm(feet_vel_w[:, :, :2], dim=-1)
    
    # 높이 오차: (target - current)^2
    # 발이 목표 높이보다 낮으면 양수 오차 발생
    height_error = target_height - feet_height  # (num_envs, num_feet)
    height_error_squared = torch.square(torch.clamp(height_error, min=0.0))  # Only penalize if below target
    
    # 페널티: 높이 오차^2 × 수평 속도
    # 수평으로 빠르게 움직일수록, 낮은 발에 대한 페널티가 커짐
    feet_penalty = height_error_squared * feet_vel_xy  # (num_envs, num_feet)
    
    # 모든 발의 페널티 합산
    total_penalty = torch.sum(feet_penalty, dim=1)  # (num_envs,)
    
    return total_penalty

def feet_clearance_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    std: float,
    tanh_mult: float,
    velocity_threshold: float = 0.1,
) -> torch.Tensor:
    """Reward high foot clearance during swing phase, only when robot should be walking.

    This is a humanoid-adapted version of Spot's foot_clearance_reward with velocity conditioning
    to prevent in-place foot lifting. The reward is ONLY active when commanded to walk.

    Key differences from Spot's version:
    - Walking (v > threshold): reward high clearance during swing
    - Standing (v < threshold): NO reward (prevents in-place stepping!)

    Args:
        env: The learning environment.
        command_name: Name of the base velocity command.
        asset_cfg: Scene entity configuration for feet bodies.
        target_height: Target clearance height (m).
        std: Standard deviation for Gaussian kernel.
        tanh_mult: Multiplier for velocity sensitivity (higher = only fast swings rewarded).
        velocity_threshold: Velocity below which reward is disabled.

    Returns:
        Reward tensor of shape (num_envs,).
    """
    # Extract asset
    asset: Articulation = env.scene[asset_cfg.name]

    # Check if robot should be walking
    # Include both linear (x, y) and angular (z) velocity commands
    # For in-place rotation, robot should lift feet for clearance
    cmd = env.command_manager.get_command(command_name)  # (num_envs, 3) [vel_x, vel_y, ang_z]
    cmd_vel = torch.norm(cmd[:, :2], dim=1) + torch.abs(cmd[:, 2])  # (num_envs,) linear + angular
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)  # (num_envs,)

    # Robot is walking if either cmd or body velocity > threshold
    is_walking = torch.logical_or(
        cmd_vel > 0.0,
        body_vel > velocity_threshold
    )  # (num_envs,)

    # Compute foot clearance reward (Spot's original logic)
    foot_z_target_error = torch.square(
        asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height
    )  # (num_envs, num_feet)

    foot_velocity_tanh = torch.tanh(
        tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    )  # (num_envs, num_feet)

    reward_per_foot = foot_z_target_error * foot_velocity_tanh  # (num_envs, num_feet)
    clearance_reward = torch.exp(-torch.sum(reward_per_foot, dim=1) / std)  # (num_envs,)

    # Only give reward when walking, zero when standing
    final_reward = torch.where(is_walking, clearance_reward, torch.zeros_like(clearance_reward))

    return final_reward

def tocabi_should_walk_stop_align(
    env: ManagerBasedRLEnv,
    *,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    velocity_threshold: float = 0.1,
    contact_threshold: float = 5.0,
    # stop-intent gate
    stop_cmd_vel_max: float = 0.05,
    # yaw/position thresholds (user intent: compare each foot to base yaw)
    yaw_threshold_deg: float = 10.0,
    pos_threshold_m: float = 0.01,
    stance_width_m: float = 0.20,
    # Longitudinal foot separation target (parallel-stance lock-in).
    # |x_L - x_R| should fall within `pos_threshold_m` of this value at stop.
    # Default 0.0 = both feet on the same x-line (no staggered stance).
    x_sep_target_m: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a simple walking/standing switch specialized for stop alignment.

    Behavior:
    - Normal walking: (cmd_vel > 0) OR (body_vel > velocity_threshold) => walk
    - Stop alignment: when cmd_vel < stop_cmd_vel_max, if yaw/position error is large => walk

    Returns:
        should_walk: (num_envs,) bool.
        is_aligned: (num_envs,) bool, only meaningful under stop intent.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd_vel, body_vel = _tocabi_compute_cmd_and_body_vel(env, asset=asset, command_name=command_name)

    # --- compute yaw/stance metrics (yaw-only base frame) ---
    yaw_rel, x_mid, y_mid, y_sep, x_sep, both_in_contact = _tocabi_compute_feet_yaw_rel_and_stance_metrics(
        env,
        asset=asset,
        sensor_cfg=sensor_cfg,
        contact_threshold=contact_threshold,
        stance_width_m=stance_width_m,
    )

    # normal walking condition:
    # NOTE: Using (cmd_vel > 0.0) makes almost all nonzero commands count as "walking",
    # which can unintentionally keep the env in walking-like mode and disable standing-gated terms.
    # Use the same threshold for command magnitude as for body velocity.
    walking_by_velocity = torch.logical_or(cmd_vel > velocity_threshold, body_vel > velocity_threshold)

    # stop-intent alignment switch
    stop_intent = cmd_vel < stop_cmd_vel_max
    yaw_th = (yaw_threshold_deg * torch.pi) / 180.0
    yaw_bad = torch.any(torch.abs(yaw_rel) > yaw_th, dim=1)  # any foot deviates from base yaw

    mid_err = torch.sqrt(x_mid * x_mid + y_mid * y_mid)
    width_err = torch.abs(y_sep - stance_width_m)
    x_sep_err = torch.abs(x_sep - x_sep_target_m)
    pos_bad = mid_err > pos_threshold_m
    pos_bad = torch.logical_or(pos_bad, width_err > pos_threshold_m)
    pos_bad = torch.logical_or(pos_bad, x_sep_err > pos_threshold_m)

    align_bad = torch.logical_or(yaw_bad, pos_bad)
    is_aligned = torch.logical_and(~align_bad, both_in_contact)

    should_walk = torch.logical_or(walking_by_velocity, torch.logical_and(stop_intent, align_bad))
    return should_walk, is_aligned


def _tocabi_phase01_from_episode_steps(env: "ManagerBasedRLEnv", *, period_steps: int) -> torch.Tensor:
    """Compute gait phase in [0, 1) from env's episode step counter.

    Args:
        period_steps: Gait cycle length in environment steps (after decimation).

    Returns:
        phase01: (num_envs,) float in [0,1).
    """
    if period_steps <= 0:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    step_count = getattr(env, "episode_length_buf", None)
    if not isinstance(step_count, torch.Tensor):
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    return (step_count % int(period_steps)).to(torch.float32) / float(period_steps)


def _tocabi_is_schedule_active_from_cmd(
    env: "ManagerBasedRLEnv",
    *,
    command_name: str,
    cmd_zero_max: float,
    grace_steps: int = 0,
    push_suppress_threshold: float = 0.0,
    push_activates: bool = False,
    asset_cfg_name: str = "robot",
    # --- optional misalignment trigger (stop-intent AND align_bad) ---
    misalign_activates: bool = False,
    align_sensor_cfg: SceneEntityCfg | None = None,
    align_asset_cfg: SceneEntityCfg | None = None,
    stop_cmd_vel_max: float = 0.05,
    yaw_threshold_deg: float = 10.0,
    pos_threshold_m: float = 0.01,
    stance_width_m: float = 0.20,
    x_sep_target_m: float = 0.0,
    align_contact_threshold: float = 5.0,
) -> torch.Tensor:
    """Return (num_envs,) bool: schedule active based on command, push, and (optionally) feet-misalign state.

    Two push modes controlled by `push_activates`:
      - False (suppress): schedule OFF during push. Use for swing_clearance
        so the robot isn't penalized for low feet during recovery.
      - True (activate): schedule ON during push even if vel=0. Use for
        contact_schedule so the robot is rewarded for stepping to recover balance.

    With `misalign_activates=True` and `align_sensor_cfg`/`align_asset_cfg` provided,
    the schedule ALSO activates when (cmd ~ 0) AND feet alignment is bad
    (yaw or stance deviation beyond thresholds). This forces the walking branch
    while the robot is still trying to settle into a parallel stance, so it keeps
    stepping until both feet align — instead of locking in a misaligned stance.

    With grace_steps > 0, the schedule stays active for that many steps after
    the trigger disappears. Grace countdown is frozen during push.
    """
    # update persistent push force (once per step, guarded internally)
    from .events import _step_persistent_push_force
    _step_persistent_push_force(env, asset_cfg_name=asset_cfg_name)

    cmd = env.command_manager.get_command(command_name)  # (num_envs, 3)
    cmd_vel = torch.linalg.norm(cmd[:, :2], dim=1) + torch.abs(cmd[:, 2])
    cmd_active = cmd_vel > cmd_zero_max

    # --- detect push ---
    pushed = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    if push_suppress_threshold > 0.0:
        force_buf = env.__dict__.get("_persistent_push_force", None)
        if force_buf is not None:
            force_mag = torch.linalg.norm(force_buf, dim=1)  # (N,)
            pushed = force_mag > push_suppress_threshold

    if push_activates:
        # push ACTIVATES schedule: cmd OR pushed
        trigger = cmd_active | pushed
    else:
        # push SUPPRESSES schedule: cmd AND NOT pushed
        trigger = cmd_active & ~pushed

    # --- optional misalignment trigger ---
    # When commanded to stop but feet aren't in the desired parallel stance,
    # treat the env as "still walking" so contact-schedule / swing-clearance
    # keep rewarding stepping until alignment is reached.
    if misalign_activates and align_sensor_cfg is not None and align_asset_cfg is not None:
        asset: Articulation = env.scene[align_asset_cfg.name]
        yaw_rel, x_mid, y_mid, y_sep, x_sep, _ = _tocabi_compute_feet_yaw_rel_and_stance_metrics(
            env,
            asset=asset,
            sensor_cfg=align_sensor_cfg,
            contact_threshold=align_contact_threshold,
            stance_width_m=stance_width_m,
        )
        yaw_th = (yaw_threshold_deg * torch.pi) / 180.0
        yaw_bad = torch.any(torch.abs(yaw_rel) > yaw_th, dim=1)
        mid_err = torch.sqrt(x_mid * x_mid + y_mid * y_mid)
        width_err = torch.abs(y_sep - stance_width_m)
        x_sep_err = torch.abs(x_sep - x_sep_target_m)
        pos_bad = mid_err > pos_threshold_m
        pos_bad = torch.logical_or(pos_bad, width_err > pos_threshold_m)
        pos_bad = torch.logical_or(pos_bad, x_sep_err > pos_threshold_m)
        align_bad = torch.logical_or(yaw_bad, pos_bad)
        stop_intent = cmd_vel < stop_cmd_vel_max
        misalign_trigger = torch.logical_and(stop_intent, align_bad)
        # During push we still want push_activates / push_suppress semantics to
        # win: don't add extra activation while pushed (keeps suppress-mode safe).
        misalign_trigger = misalign_trigger & ~pushed
        trigger = trigger | misalign_trigger

    if grace_steps <= 0:
        return trigger

    # unique key per mode to avoid collision between activate/suppress callers
    grace_key = "_schedule_grace_countdown_act" if push_activates else "_schedule_grace_countdown_sup"
    if grace_key not in env.__dict__:
        env.__dict__[grace_key] = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    countdown: torch.Tensor = env.__dict__[grace_key]

    # reset on episode start
    ep_start = env.episode_length_buf == 0
    countdown[ep_start] = 0

    # trigger active → reset countdown to full grace period
    countdown[trigger] = grace_steps
    # trigger inactive & not pushed → decrement countdown
    inactive = ~trigger & ~pushed & (countdown > 0)
    countdown[inactive] -= 1
    # push active & not trigger → freeze countdown (don't decrement)
    # so grace resumes after push ends

    env.__dict__[grace_key] = countdown
    return trigger | (~pushed & (countdown > 0))


def _tocabi_desired_contact_biped_ds(
    phase01: torch.Tensor,
    *,
    ds_ratio: float,
) -> torch.Tensor:
    """Desired contact schedule for a biped with explicit double-support windows.

    Schedule (phase01 in [0,1)):
    - DS around the two transitions at phase=0 and phase=0.5.
    - SSP1: Left stance, Right swing
    - SSP2: Right stance, Left swing

    We interpret `ds_ratio` as the *total* DS fraction per full cycle (two DS windows combined).
    Each transition gets a DS window of length ds_ratio/2, centered on the transition.

    Returns:
        desired: (num_envs, 2) float in {0,1} for [L, R].
    """
    ds_ratio_c = float(max(0.0, min(ds_ratio, 0.49)))
    h = ds_ratio_c / 4.0  # because: 2 transitions × (2h) = ds_ratio

    ds_mask = (phase01 < h) | (phase01 > (1.0 - h)) | (torch.abs(phase01 - 0.5) < h)
    left_stance_mask = (phase01 >= h) & (phase01 < (0.5 - h))
    right_stance_mask = (phase01 >= (0.5 + h)) & (phase01 <= (1.0 - h))

    desired = torch.zeros(phase01.shape[0], 2, device=phase01.device, dtype=torch.float32)
    desired[ds_mask, :] = 1.0
    desired[left_stance_mask, 0] = 1.0
    desired[left_stance_mask, 1] = 0.0
    desired[right_stance_mask, 0] = 0.0
    desired[right_stance_mask, 1] = 1.0
    return desired


def contact_schedule_reward_biped_ds(
    env: "ManagerBasedRLEnv",
    *,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    period_steps: int,
    ds_ratio: float = 0.2,
    contact_threshold: float = 5.0,
    cmd_zero_max: float = 1.0e-3,
    grace_steps: int = 0,
    push_suppress_threshold: float = 0.0,
    push_activates: bool = False,
    # --- optional misalignment trigger ---
    misalign_activates: bool = False,
    asset_cfg: SceneEntityCfg | None = None,
    stop_cmd_vel_max: float = 0.05,
    yaw_threshold_deg: float = 10.0,
    pos_threshold_m: float = 0.01,
    stance_width_m: float = 0.20,
    x_sep_target_m: float = 0.0,
) -> torch.Tensor:
    """Reward matching a phase-based biped contact schedule (with DS), active only when walking.

    - standing(cmd~0): OFF (reward=0) to avoid in-place marching.
    - walking(cmd>0): enforce alternating contacts with explicit double-support windows.
    - pushed(|force| > threshold): behavior depends on push_activates.
    - misaligned at stop (`misalign_activates=True`): schedule stays ON while
      feet aren't in the desired parallel stance, so the robot keeps stepping
      until alignment is reached.

    Args:
        command_name: Name of command term (e.g., "base_velocity").
        sensor_cfg: Contact sensor selecting exactly two feet bodies in [L, R] order.
        period_steps: Gait cycle length in env steps.
        ds_ratio: Total double-support fraction per cycle (0~0.49).
        contact_threshold: Threshold on Fz (N) to decide contact.
        cmd_zero_max: Command magnitude below which schedule is disabled.
        grace_steps: Steps to keep schedule active after command drops to zero.
        push_suppress_threshold: Force magnitude (N) above which push is detected.
        push_activates: If True, push activates schedule (for recovery stepping).
            If False, push suppresses schedule.
        misalign_activates: If True (and asset_cfg given), activate schedule when
            cmd~0 but feet alignment is bad.
        asset_cfg: Robot asset cfg (required for misalign_activates).
        stop_cmd_vel_max: Command magnitude below which we consider stop intent
            (used by misalign trigger).
        yaw_threshold_deg, pos_threshold_m, stance_width_m, x_sep_target_m:
            Same alignment thresholds as `feet_air_time_biped` /
            `tocabi_should_walk_stop_align`.

    Returns:
        (num_envs,) reward in [0,1] (approximately).
    """
    is_active = _tocabi_is_schedule_active_from_cmd(
        env,
        command_name=command_name,
        cmd_zero_max=cmd_zero_max,
        grace_steps=grace_steps,
        push_suppress_threshold=push_suppress_threshold,
        push_activates=push_activates,
        misalign_activates=misalign_activates,
        align_sensor_cfg=sensor_cfg if misalign_activates else None,
        align_asset_cfg=asset_cfg if misalign_activates else None,
        stop_cmd_vel_max=stop_cmd_vel_max,
        yaw_threshold_deg=yaw_threshold_deg,
        pos_threshold_m=pos_threshold_m,
        stance_width_m=stance_width_m,
        x_sep_target_m=x_sep_target_m,
        align_contact_threshold=contact_threshold,
    )
    if not torch.any(is_active):
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    phase01 = _tocabi_phase01_from_episode_steps(env, period_steps=period_steps)
    desired = _tocabi_desired_contact_biped_ds(phase01, ds_ratio=ds_ratio)  # (N,2) [L,R]

    contact_sensor = cast(ContactSensor, env.scene.sensors[sensor_cfg.name])
    body_ids = sensor_cfg.body_ids
    if body_ids is None:
        raise RuntimeError("sensor_cfg.body_ids is None. Ensure sensor_cfg resolves body_ids at runtime.")
    if isinstance(body_ids, slice):
        raise RuntimeError("sensor_cfg.body_ids resolved to slice(None). Please pass explicit foot body_names.")
    if len(body_ids) != 2:
        raise RuntimeError(f"Expected exactly 2 feet in sensor_cfg. Got body_ids={body_ids}.")

    net_forces_w = contact_sensor.data.net_forces_w
    if net_forces_w is None:
        raise RuntimeError("ContactSensor net_forces_w buffer is None.")
    fz = net_forces_w[:, body_ids, 2]  # (N,2)
    actual = (fz > contact_threshold).to(torch.float32)  # (N,2)

    match = 1.0 - torch.mean(torch.abs(actual - desired), dim=1)
    match = torch.clamp(match, 0.0, 1.0)
    return torch.where(is_active, match, torch.zeros_like(match))


def swing_clearance_min_profile_penalty(
    env: "ManagerBasedRLEnv",
    *,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    period_steps: int,
    ds_ratio: float = 0.2,
    clearance_height: float = 0.15,
    profile_offset: float = 0.0,
    contact_threshold: float = 5.0,
    cmd_zero_max: float = 1.0e-3,
    grace_steps: int = 0,
    push_suppress_threshold: float = 0.0,
    push_activates: bool = False,
    rise_end: float = 0.20,
    plateau_end: float = 0.75,
    descent_end: float = 0.90,
    soft_ratio: float = 0.30,
    # --- optional misalignment trigger ---
    misalign_activates: bool = False,
    stop_cmd_vel_max: float = 0.05,
    yaw_threshold_deg: float = 10.0,
    pos_threshold_m: float = 0.01,
    stance_width_m: float = 0.20,
    x_sep_target_m: float = 0.0,
) -> torch.Tensor:
    """Penalty if swing-foot clearance is below a reference profile (phase-based), active only when walking.

    Ground reference (no terrain query):
      Use per-foot stance height `stance_z` captured when the foot is in contact.
      clearance := z_now - stance_z

    Reference profile (human-gait-inspired envelope with soft landing):
      Plateau-with-soft-landing envelope over phi_local ∈ [0,1]:
        [0,           rise_end    ]: linear rise 0 -> h
        [rise_end,    plateau_end ]: plateau at h (covers bimodal MTC dip of real gait)
        [plateau_end, descent_end ]: linear descent h -> h_soft = soft_ratio * h
        [descent_end, 1.0         ]: cosine ease-out h_soft -> 0 (soft landing)

    With `misalign_activates=True`, the schedule additionally activates when the
    robot is commanded to stop but feet are not aligned (parallel stance). This
    keeps the swing-clearance penalty on while the robot is still stepping to
    align — preventing it from "freezing low" mid-correction.
    """
    is_active = _tocabi_is_schedule_active_from_cmd(
        env,
        command_name=command_name,
        cmd_zero_max=cmd_zero_max,
        grace_steps=grace_steps,
        push_suppress_threshold=push_suppress_threshold,
        push_activates=push_activates,
        misalign_activates=misalign_activates,
        align_sensor_cfg=sensor_cfg if misalign_activates else None,
        align_asset_cfg=asset_cfg if misalign_activates else None,
        stop_cmd_vel_max=stop_cmd_vel_max,
        yaw_threshold_deg=yaw_threshold_deg,
        pos_threshold_m=pos_threshold_m,
        stance_width_m=stance_width_m,
        x_sep_target_m=x_sep_target_m,
        align_contact_threshold=contact_threshold,
    )
    if not torch.any(is_active):
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    phase01 = _tocabi_phase01_from_episode_steps(env, period_steps=period_steps)  # (N,)
    desired = _tocabi_desired_contact_biped_ds(phase01, ds_ratio=ds_ratio)  # (N,2) [L,R]
    swing_mask = desired < 0.5  # (N,2)

    asset: Articulation = env.scene[asset_cfg.name]
    if asset_cfg.body_ids is None:
        raise RuntimeError("asset_cfg.body_ids is None. Ensure asset_cfg resolves body_ids at runtime.")
    if isinstance(asset_cfg.body_ids, slice):
        raise RuntimeError("asset_cfg.body_ids resolved to slice(None). Please pass explicit feet body_names.")
    if len(asset_cfg.body_ids) != 2:
        raise RuntimeError(f"Expected exactly 2 feet in asset_cfg. Got body_ids={asset_cfg.body_ids}.")
    feet_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2].to(torch.float32)  # (N,2)

    contact_sensor = cast(ContactSensor, env.scene.sensors[sensor_cfg.name])
    if sensor_cfg.body_ids is None:
        raise RuntimeError("sensor_cfg.body_ids is None. Ensure sensor_cfg resolves body_ids at runtime.")
    if isinstance(sensor_cfg.body_ids, slice):
        raise RuntimeError("sensor_cfg.body_ids resolved to slice(None). Please pass explicit feet body_names.")
    if len(sensor_cfg.body_ids) != 2:
        raise RuntimeError(f"Expected exactly 2 feet in sensor_cfg. Got body_ids={sensor_cfg.body_ids}.")
    net_forces_w = contact_sensor.data.net_forces_w
    if net_forces_w is None:
        raise RuntimeError("ContactSensor net_forces_w buffer is None.")
    fz = net_forces_w[:, sensor_cfg.body_ids, 2]  # (N,2)
    is_contact = fz > contact_threshold

    stance_key = "_p73_stance_z_buffer"
    if stance_key not in env.__dict__:
        env.__dict__[stance_key] = feet_z.detach().clone()
    stance_z: torch.Tensor = env.__dict__[stance_key]
    stance_z = torch.where(is_contact, feet_z, stance_z)
    env.__dict__[stance_key] = stance_z

    clearance = feet_z - stance_z  # (N,2)

    ds_ratio_c = float(max(0.0, min(ds_ratio, 0.49)))
    h = ds_ratio_c / 4.0
    denom = max(1.0e-6, 0.5 - 2.0 * h)

    # Right swings during SSP1: phase in [h, 0.5-h)
    phi_r = torch.clamp((phase01 - h) / denom, 0.0, 1.0)
    # Left swings during SSP2: phase in [0.5+h, 1-h)
    phi_l = torch.clamp((phase01 - (0.5 + h)) / denom, 0.0, 1.0)
    phi_local = torch.stack([phi_l, phi_r], dim=1)  # (N,2) [L,R]

    # Human-gait-inspired envelope with soft landing.
    h_base = float(clearance_height)
    r1 = float(rise_end)
    r2 = float(plateau_end)
    r3 = float(descent_end)
    h_soft = h_base * float(soft_ratio)
    eps = 1.0e-6
    p = phi_local
    rise = h_base * (p / max(r1, eps))
    plateau = torch.full_like(p, h_base)
    descent = h_base + (h_soft - h_base) * ((p - r2) / max(r3 - r2, eps))
    t = torch.clamp((p - r3) / max(1.0 - r3, eps), 0.0, 1.0)
    landing = 0.5 * h_soft * (1.0 + torch.cos(torch.pi * t))
    z_ref = torch.where(
        p < r1, rise,
        torch.where(
            p < r2, plateau,
            torch.where(p < r3, descent, landing),
        ),
    )
    z_ref = torch.clamp(z_ref - float(profile_offset), min=0.0)  # (N,2)
    per_foot = torch.clamp(z_ref - clearance, min=0.0) * swing_mask.to(torch.float32)
    penalty = torch.mean(per_foot, dim=1)
    return torch.where(is_active, penalty, torch.zeros_like(penalty))


def contact_force_limit_soft_penalty(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    robot_mass: float = 100.0,
    safety_factor: float = 1.2,
    std: float = 200.0,
) -> torch.Tensor:
    """Soft penalty when contact force exceeds safety_factor * mg.

    Uses a Gaussian kernel so that small exceedances produce near-zero penalty while
    large exceedances saturate at 1.0 per foot:
        threshold = safety_factor * robot_mass * g
        excess    = max(0, |F| - threshold)
        penalty_i = 1 - exp(-excess^2 / std^2)

    Returns:
        (num_envs,) sum of per-foot penalties.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    current_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    force_magnitudes = torch.norm(current_forces, dim=-1)  # (num_envs, num_feet)
    threshold = safety_factor * robot_mass * 9.81
    excess_forces = torch.clamp(force_magnitudes - threshold, min=0.0)
    foot_penalties = torch.exp(-(excess_forces ** 2) / (std ** 2))
    foot_penalties = torch.where(excess_forces > 0, 1.0 - foot_penalties, torch.zeros_like(foot_penalties))
    return torch.sum(foot_penalties, dim=1)


def bio_mimetic_soft_hard_constraint(
    env: "ManagerBasedRLEnv",
    *,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_offset: float | torch.Tensor = 0.0,
    deadband: float = 0.0,
    stiffness: float = 10.0,
    exp_clip: float = 50.0,
    # Optional asymmetric parameters (direction-dependent). If provided, these override symmetric deadband/stiffness.
    deadband_pos: float | None = None,
    deadband_neg: float | None = None,
    stiffness_pos: float | None = None,
    stiffness_neg: float | None = None,
) -> torch.Tensor:
    """Bio-mimetic soft-hard constraint penalty for a set of joints (TOCABI custom reward).

    Template usage:
        - Configure joint subset with `asset_cfg` (must resolve `asset_cfg.joint_ids`).
        - Target is: q_target = q_default + target_offset
        - Only 3 core parameters per joint group:
            Target (via target_offset), Deadband, Stiffness

    Returns:
        Penalty tensor of shape (num_envs,). Non-negative (>= 0).
        Use a negative weight in reward config to apply it as a penalty term.
    """
    if exp_clip <= 0.0:
        raise ValueError(f"exp_clip must be > 0. Got {exp_clip}.")

    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    if joint_ids is None:
        raise RuntimeError("SceneEntityCfg.joint_ids is None. Ensure asset_cfg resolves joint_ids at runtime.")

    q = asset.data.joint_pos[:, joint_ids]
    q_default = asset.data.default_joint_pos[:, joint_ids]

    # Target: default + offset (offset can be scalar or broadcastable tensor).
    offset_t = torch.as_tensor(target_offset, device=q.device, dtype=q.dtype)
    q_target = q_default + offset_t

    # Backward-compatible default: symmetric penalty.
    # If any asymmetric param is provided, require all 4 and use the asymmetric form.
    if (
        deadband_pos is not None
        or deadband_neg is not None
        or stiffness_pos is not None
        or stiffness_neg is not None
    ):
        if deadband_pos is None or deadband_neg is None or stiffness_pos is None or stiffness_neg is None:
            raise ValueError(
                "Asymmetric penalty requires all of: deadband_pos, deadband_neg, stiffness_pos, stiffness_neg."
            )
        if float(deadband_pos) < 0.0 or float(deadband_neg) < 0.0:
            raise ValueError(
                f"deadband_pos/deadband_neg must be >= 0. Got deadband_pos={deadband_pos}, deadband_neg={deadband_neg}."
            )
        if float(stiffness_pos) <= 0.0 or float(stiffness_neg) <= 0.0:
            raise ValueError(
                f"stiffness_pos/stiffness_neg must be > 0. Got stiffness_pos={stiffness_pos}, stiffness_neg={stiffness_neg}."
            )

        # Asymmetric (direction-dependent):
        # e = q - q_target
        # v_pos = max(0,  e - deadband_pos), v_neg = max(0, -e - deadband_neg)
        # penalty = exp(stiffness_pos*v_pos)-1 + exp(stiffness_neg*v_neg)-1
        e = q - q_target
        v_pos = torch.clamp(e - float(deadband_pos), min=0.0)
        v_neg = torch.clamp((-e) - float(deadband_neg), min=0.0)

        x_pos = torch.clamp(float(stiffness_pos) * v_pos, max=float(exp_clip))
        x_neg = torch.clamp(float(stiffness_neg) * v_neg, max=float(exp_clip))
        per_joint_penalty = torch.expm1(x_pos) + torch.expm1(x_neg)
    else:
        if deadband < 0.0:
            raise ValueError(f"deadband must be >= 0. Got {deadband}.")
        if stiffness <= 0.0:
            raise ValueError(f"stiffness must be > 0. Got {stiffness}.")

        # Symmetric (|q - q_target|):
        # violation = max(0, |q-q_target| - deadband)
        # penalty   = exp(stiffness*violation) - 1
        error = torch.abs(q - q_target)
        violation = torch.clamp(error - float(deadband), min=0.0)
        x = torch.clamp(float(stiffness) * violation, max=float(exp_clip))
        per_joint_penalty = torch.expm1(x)
    return torch.sum(per_joint_penalty, dim=1)


def bio_mimetic_soft_hard_constraint_conditional(
    env: "ManagerBasedRLEnv",
    *,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
    command_index: int = 2,
    command_threshold: float = 0.1,
    base_vel_index: int = 2,
    base_vel_threshold: float = 0.07,
    target_offset: float | torch.Tensor = 0.0,
    deadband_pos_active: float = 0.35,
    deadband_neg_active: float = 0.35,
    deadband_pos_inactive: float = 0.0,
    deadband_neg_inactive: float = 0.0,
    stiffness_pos: float = 1.5,
    stiffness_neg: float = 1.5,
    exp_clip: float = 50.0,
) -> torch.Tensor:
    """Bio-mimetic soft-hard constraint with command/velocity-conditional asymmetric deadband.

    Active (deadband relaxed) when EITHER:
      1. |command[command_index]| > command_threshold  (명령이 켜져 있을 때)
      2. |base_vel[base_vel_index]| > base_vel_threshold  (외력 등으로 실제 속도가 발생했을 때)

    Args:
        command_index: Index into the command vector. 0=vx, 1=vy, 2=ang_z.
        command_threshold: Absolute threshold for the command to be considered "active".
        base_vel_index: Index into base velocity. 0=vx, 1=vy for lin_vel; 2=ang_z for ang_vel.
        base_vel_threshold: Actual base velocity threshold to activate deadband.
        deadband_pos_active: Positive deadband when active.
        deadband_neg_active: Negative deadband when active.
        deadband_pos_inactive: Positive deadband when inactive.
        deadband_neg_inactive: Negative deadband when inactive.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    if joint_ids is None:
        raise RuntimeError("SceneEntityCfg.joint_ids is None.")

    # Condition 1: command active
    cmd = env.command_manager.get_command(command_name)  # (num_envs, 3)
    cmd_active = torch.abs(cmd[:, command_index]) > command_threshold

    # Condition 2: actual base velocity exceeds threshold (외력, 미끄러짐 등)
    if base_vel_index < 2:
        # linear velocity (x or y) in base frame
        actual_vel = asset.data.root_lin_vel_b[:, base_vel_index]
    else:
        # angular velocity (z) in world frame
        actual_vel = asset.data.root_ang_vel_w[:, 2]
    vel_active = torch.abs(actual_vel) > base_vel_threshold

    # Either condition → activate deadband
    active = cmd_active | vel_active  # (num_envs,)

    # Per-env asymmetric deadbands: (num_envs, 1)
    db_pos = torch.where(active, deadband_pos_active, deadband_pos_inactive).unsqueeze(1)
    db_neg = torch.where(active, deadband_neg_active, deadband_neg_inactive).unsqueeze(1)

    q = asset.data.joint_pos[:, joint_ids]
    q_default = asset.data.default_joint_pos[:, joint_ids]
    offset_t = torch.as_tensor(target_offset, device=q.device, dtype=q.dtype)
    q_target = q_default + offset_t

    e = q - q_target
    v_pos = torch.clamp(e - db_pos, min=0.0)
    v_neg = torch.clamp((-e) - db_neg, min=0.0)

    x_pos = torch.clamp(stiffness_pos * v_pos, max=exp_clip)
    x_neg = torch.clamp(stiffness_neg * v_neg, max=exp_clip)
    per_joint_penalty = torch.expm1(x_pos) + torch.expm1(x_neg)

    return torch.sum(per_joint_penalty, dim=1)


def feet_height_symmetry_penalty(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    velocity_threshold: float = 0.1,
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize asymmetric maximum swing height between feet (prevents limping gait).

    Tracks the maximum height each foot reaches during its swing phase and penalizes
    the difference between left and right foot max heights. Only active when walking.

    Args:
        asset_cfg: Robot asset config with foot body names (e.g. ".*_Foot_Link").
        sensor_cfg: Contact sensor config with the same foot body names.
        command_name: Base velocity command name.
        velocity_threshold: Linear+angular cmd norm or body vel above which "walking" activates.
        contact_threshold: Contact force (N) above which foot is considered in stance.

    Returns:
        Penalty tensor (num_envs,): |max_z_left - max_z_right| when walking, else 0.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor = cast(ContactSensor, env.scene.sensors[sensor_cfg.name])

    # Foot heights in world frame (N, 2)
    feet_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    if feet_height.shape[1] != 2:
        raise RuntimeError(f"feet_height_symmetry_penalty expects 2 feet, got {feet_height.shape[1]}")

    # Per-env max swing height buffer (reset at stance entry)
    buf_key = "_p73_max_foot_height_buffer"
    if buf_key not in env.__dict__:
        env.__dict__[buf_key] = torch.zeros(env.num_envs, 2, device=env.device, dtype=torch.float32)
    max_height_buffer: torch.Tensor = env.__dict__[buf_key]

    # Contact forces z-axis → stance detection
    net_forces_w = contact_sensor.data.net_forces_w
    contact_forces = net_forces_w[:, sensor_cfg.body_ids, 2]  # (N, 2)
    is_stance = contact_forces > contact_threshold

    # Accumulate max height; reset when foot enters stance
    max_height_buffer = torch.maximum(feet_height, max_height_buffer)
    max_height_buffer[is_stance] = feet_height[is_stance]
    env.__dict__[buf_key] = max_height_buffer

    # Penalty = |max_left - max_right|
    height_diff = torch.abs(max_height_buffer[:, 0] - max_height_buffer[:, 1])

    # Gate: only penalize when walking
    cmd = env.command_manager.get_command(command_name)
    cmd_vel = torch.norm(cmd[:, :2], dim=1) + torch.abs(cmd[:, 2])
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    is_walking = torch.logical_or(cmd_vel > velocity_threshold, body_vel > velocity_threshold)

    return torch.where(is_walking, height_diff, torch.zeros_like(height_diff))


def air_time_variance_penalty_gated(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    velocity_threshold: float = 0.12,
    *,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize variance in air/contact time between feet, only when walking.

    Computes var(last_air_time) + var(last_contact_time) across the two feet
    and zeros it out when the robot is standing/slow.

    Requires ContactSensor with track_air_time=True.

    Args:
        sensor_cfg: Contact sensor config with foot body names.
        command_name: Base velocity command name.
        velocity_threshold: Walking gate threshold (m/s equivalent norm).
        asset_cfg: Robot asset config (for body velocity check).

    Returns:
        Penalty tensor (num_envs,): variance penalty when walking, else 0.
    """
    contact_sensor = cast(ContactSensor, env.scene.sensors[sensor_cfg.name])
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")

    body_ids = sensor_cfg.body_ids
    last_air_time = contact_sensor.data.last_air_time[:, body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, body_ids]

    penalty = torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )

    # Gate: only penalize when walking
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    cmd_vel = torch.norm(cmd[:, :2], dim=1) + torch.abs(cmd[:, 2])
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    is_walking = torch.logical_or(cmd_vel > velocity_threshold, body_vel > velocity_threshold)

    return torch.where(is_walking, penalty, torch.zeros_like(penalty))


def spring_compliance_pos_match_reward(
    env: "ManagerBasedRLEnv",
    *,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    head_body_name: str = "base_link",
    k_spring: float = 8000.0,
    contact_threshold: float = 5.0,
    robot_mass: float = 100.0,
    baseline_margin: float = 1.2,
    gravity: float = 9.81,
    command_name: str = "base_velocity",
    velocity_threshold: float = 0.05,
    cmd_vel_eps: float = 1.0e-3,
) -> torch.Tensor:
    """Spring-like compliance penalty (pos-match) for impact mitigation.

    On first contact per foot, captures the foot z (relative to a reference body, default: base_link)
    as z_d. While in contact, allows compliant deviation proportional to excess vertical force:
        z_target = z_d + max(Fz - Fz0, 0) / K
        cost     = (z - z_target)^2

    Fz0 is a baseline deadzone (nominal mg split over contacting feet) to avoid penalizing
    normal support during double support.

    Gate: active only when walking (cmd_vel >= cmd_vel_eps OR body_vel > velocity_threshold).

    Args:
        sensor_cfg: Contact sensor config (must resolve to exactly 2 foot body_ids).
        asset_cfg: Robot asset config.
        head_body_name: Reference body for relative z (e.g. "base_link" or "Pelvis_Link").
        k_spring: Virtual spring stiffness K (N/m).
        contact_threshold: Fz (N) above which foot is in contact.
        robot_mass: Robot mass (kg) for baseline Fz0.
        baseline_margin: Multiplier on mg baseline (>= 1.0 recommended).
        gravity: Gravity magnitude (m/s^2).
        command_name: Base velocity command name.
        velocity_threshold: Body velocity above which walking gate activates.
        cmd_vel_eps: Treat cmd_vel < cmd_vel_eps as zero.

    Returns:
        Cost tensor (num_envs,), non-negative. Use negative weight in reward config.
    """
    if k_spring <= 0.0:
        raise ValueError(f"k_spring must be > 0. Got {k_spring}.")
    if robot_mass <= 0.0:
        raise ValueError(f"robot_mass must be > 0. Got {robot_mass}.")
    if baseline_margin < 0.0:
        raise ValueError(f"baseline_margin must be >= 0. Got {baseline_margin}.")
    if gravity <= 0.0:
        raise ValueError(f"gravity must be > 0. Got {gravity}.")

    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor = cast(ContactSensor, env.scene.sensors[sensor_cfg.name])

    body_ids = sensor_cfg.body_ids
    if body_ids is None:
        raise RuntimeError("SceneEntityCfg.body_ids is None. Ensure sensor_cfg resolves body_ids at runtime.")

    # Contact force (world z)
    net_forces_w = contact_sensor.data.net_forces_w
    if net_forces_w is None:
        raise RuntimeError("ContactSensor net_forces_w buffer is None.")
    fz = net_forces_w[:, body_ids, 2]  # (N, 2)
    fz_pos = torch.clamp(fz, min=0.0)
    in_contact = fz_pos > contact_threshold  # (N, 2)

    # Walking gate
    cmd_vel, body_vel = _tocabi_compute_cmd_and_body_vel(env, asset=asset, command_name=command_name)
    should_walk = torch.logical_or(cmd_vel >= cmd_vel_eps, body_vel > velocity_threshold)

    # Foot relative z w.r.t. reference body
    ref_ids, _ = asset.find_bodies(head_body_name, preserve_order=True)
    if len(ref_ids) != 1:
        raise RuntimeError(f"Expected exactly one body match for head_body_name={head_body_name!r}.")
    ref_id = ref_ids[0]
    ref_pos_w = asset.data.body_pos_w[:, ref_id, :]           # (N, 3)
    feet_pos_w = asset.data.body_pos_w[:, body_ids, :]         # (N, 2, 3)
    rel_z = feet_pos_w[..., 2] - ref_pos_w[:, 2].unsqueeze(1) # (N, 2)

    # Persistent z_d buffer (captured at first contact per foot, per env)
    zd_key = "_p73_spring_pos_match_zd"
    valid_key = "_p73_spring_pos_match_zd_valid"
    if zd_key not in env.__dict__ or env.__dict__[zd_key].shape != rel_z.shape:
        env.__dict__[zd_key] = rel_z.detach().clone()
        env.__dict__[valid_key] = torch.zeros_like(rel_z, dtype=torch.bool)

    z_d: torch.Tensor = env.__dict__[zd_key]
    z_d_valid: torch.Tensor = env.__dict__[valid_key]

    # First contact detection
    if contact_sensor.cfg.track_air_time:
        first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, body_ids]
    else:
        prev_key = "_p73_spring_pos_match_prev_contact"
        if prev_key not in env.__dict__ or env.__dict__[prev_key].shape != in_contact.shape:
            env.__dict__[prev_key] = torch.zeros_like(in_contact, dtype=torch.bool)
        prev_in_contact: torch.Tensor = env.__dict__[prev_key]
        first_contact = torch.logical_and(in_contact, ~prev_in_contact)
        env.__dict__[prev_key] = in_contact.detach().clone()

    # Capture z_d at first contact
    z_d = torch.where(first_contact, rel_z, z_d)
    z_d_valid = torch.logical_or(z_d_valid, first_contact)
    env.__dict__[zd_key] = z_d
    env.__dict__[valid_key] = z_d_valid

    z_d_eff = torch.where(z_d_valid, z_d, rel_z)

    # Baseline deadzone: nominal mg split over contacting feet
    n_contact = torch.clamp(in_contact.sum(dim=1), min=1).to(fz_pos.dtype)  # (N,)
    fz0 = (baseline_margin * robot_mass * gravity / n_contact).unsqueeze(1)  # (N, 1)
    fz_excess = torch.clamp(fz_pos - fz0, min=0.0)

    # Spring compliance target and squared cost
    z_target = z_d_eff + fz_excess / k_spring
    err = rel_z - z_target
    per_foot = torch.square(err)
    gate = torch.logical_and(in_contact, should_walk.unsqueeze(1))
    per_foot = torch.where(gate, per_foot, torch.zeros_like(per_foot))

    return torch.mean(per_foot, dim=1)


def stand_still_joint_deviation_l1(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.06,
    velocity_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    standing_pose: dict[str, float] | None = None,
) -> torch.Tensor:
    """Penalize joint deviations when the robot should be standing still.

    Unlike the upstream version:
      - Checks actual base velocity (not just command)
      - Supports a custom standing pose instead of default_joint_pos

    The penalty is only active when BOTH:
      - the velocity command is small (norm < command_threshold)
      - the actual base linear velocity is small (norm < velocity_threshold)
    """
    command = env.command_manager.get_command(command_name)
    cmd_small = torch.norm(command[:, :2], dim=1) < command_threshold

    asset: Articulation = env.scene[asset_cfg.name]
    base_vel = asset.data.root_lin_vel_b[:, :2]
    vel_small = torch.norm(base_vel, dim=1) < velocity_threshold

    gate = (cmd_small & vel_small).float()

    if standing_pose is not None:
        cache_key = "_stand_still_standing_target"
        if cache_key not in env.__dict__:
            target = asset.data.default_joint_pos.clone()
            for joint_name, value in standing_pose.items():
                joint_ids, _ = asset.find_joints(joint_name)
                if joint_ids:
                    target[:, joint_ids[0]] = value
            env.__dict__[cache_key] = target
        target = env.__dict__[cache_key]
        deviation = torch.sum(torch.abs(asset.data.joint_pos - target), dim=1)
    else:
        deviation = torch.sum(
            torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1
        )

    return deviation * gate


def base_height_below_target_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    # One-sided L2: returns 0 when h >= target, (target-h)^2 when h < target.
    # Lets weight be increased aggressively without punishing upward overshoots,
    # and brief balance-recovery dips only incur penalty proportional to depth.
    asset: Articulation = env.scene[asset_cfg.name]
    h = asset.data.root_pos_w[:, 2]
    deficit = torch.clamp(target_height - h, min=0.0)
    return torch.square(deficit)


def base_height_target_exp(
    env: ManagerBasedRLEnv,
    target_height: float,
    std: float = 0.03,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    # exp(-((h - target)/std)^2) in (0, 1]. Use with POSITIVE weight.
    # Gradient vanishes when far from target -> a falling robot gets no
    # incentive to "fall further to minimize |penalty|" (pure-L2 failure mode).
    asset: Articulation = env.scene[asset_cfg.name]
    h = asset.data.root_pos_w[:, 2]
    return torch.exp(-torch.square((h - target_height) / std))
