from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
from isaaclab.sensors import Camera, Imu, RayCaster, RayCasterCamera, TiledCamera
import math

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

from isaaclab.envs.mdp import *

def gait_phase(
    env: "ManagerBasedRLEnv",
    *,
    period_steps: int,
    command_name: str = "base_velocity",
    cmd_zero_max: float = 1.0e-3,
) -> torch.Tensor:
    """Gait phase in [0, 1) computed from env's episode step counter.

    Design choice:
    - When the commanded velocity is (near) zero, we treat the env as *standing* and output phase=0.
      This prevents the policy from learning in-place marching tied to phase when standing.

    Notes:
    - Isaac Lab's ManagerBasedRLEnv maintains `episode_length_buf` (step counter since reset).
    - `period_steps` is in *environment steps* (i.e., after decimation).
    """
    if period_steps <= 0:
        return torch.zeros(env.num_envs, 1, device=env.device, dtype=torch.float32)

    # Standing gate: cmd ~ 0 => phase=0
    try:
        cmd = env.command_manager.get_command(command_name)  # (num_envs, 3)
        cmd_vel = torch.linalg.norm(cmd[:, :2], dim=1) + torch.abs(cmd[:, 2])  # (num_envs,)
        is_standing = cmd_vel <= cmd_zero_max
    except Exception:
        # If command manager isn't available (edge case), fall back to always-on phase.
        is_standing = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    step_count = getattr(env, "episode_length_buf", None)
    if not isinstance(step_count, torch.Tensor):
        return torch.zeros(env.num_envs, 1, device=env.device, dtype=torch.float32)

    phase = (step_count % int(period_steps)).to(torch.float32) / float(period_steps)
    phase = phase.view(-1, 1)
    phase = torch.where(is_standing.view(-1, 1), torch.zeros_like(phase), phase)
    return phase


def gait_phase_sin(
    env: "ManagerBasedRLEnv",
    *,
    period_steps: int,
    command_name: str = "base_velocity",
    cmd_zero_max: float = 1.0e-3,
) -> torch.Tensor:
    """Sin(2π * gait_phase)."""
    phase = gait_phase(env, period_steps=period_steps, command_name=command_name, cmd_zero_max=cmd_zero_max)
    return torch.sin(2.0 * torch.pi * phase)


def gait_phase_cos(
    env: "ManagerBasedRLEnv",
    *,
    period_steps: int,
    command_name: str = "base_velocity",
    cmd_zero_max: float = 1.0e-3,
) -> torch.Tensor:
    """Cos(2π * gait_phase)."""
    phase = gait_phase(env, period_steps=period_steps, command_name=command_name, cmd_zero_max=cmd_zero_max)
    return torch.cos(2.0 * torch.pi * phase)


def joint_pos_ordered(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint positions of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # need to reorder joint_ids to the order of the joint_names
    joint_ids_reordered = [asset.data.joint_names.index(joint_name) for joint_name in asset_cfg.joint_names]
    return asset.data.joint_pos[:, joint_ids_reordered]

def joint_vel_ordered(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """The joint velocities of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their velocities returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids_reordered = [asset.data.joint_names.index(joint_name) for joint_name in asset_cfg.joint_names]
    return asset.data.joint_vel[:, joint_ids_reordered]

def joint_pos_ordered_rel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint positions of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # need to reorder joint_ids to the order of the joint_names
    joint_ids_reordered = [asset.data.joint_names.index(joint_name) for joint_name in asset_cfg.joint_names]
    return asset.data.joint_pos[:, joint_ids_reordered] - asset.data.default_joint_pos[:, joint_ids_reordered]

def last_processed_action(env: ManagerBasedEnv, action_name: str | None = None) -> torch.Tensor:
    """The last input action to the environment.

    The name of the action term for which the action is required. If None, the
    entire action tensor is returned.
    """
    if action_name is None:
        return env.action_manager.action
    else:
        return env.action_manager.get_term(action_name).processed_actions


def base_vel_xy_yawrate(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Base velocity supervision target: [v_x, v_y, yaw_rate] in body frame (3D).

    This is used as gt_vel3 prefix in PPOFuture critic observations.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    lin_vel_b = asset.data.root_lin_vel_b  # (num_envs, 3)
    ang_vel_b = asset.data.root_ang_vel_b  # (num_envs, 3)
    return torch.stack((lin_vel_b[:, 0], lin_vel_b[:, 1], ang_vel_b[:, 2]), dim=-1)


def physics_material_sd(env: ManagerBasedEnv) -> torch.Tensor:
    """Return cached physics material properties (static, dynamic friction) as 2D.

    The values are cached on the environment by a startup event term
    (`randomize_rigid_body_material_and_cache`).
    If the cache is missing, returns zeros.
    """
    buf = env.__dict__.get("_physics_material_sd", None)
    if isinstance(buf, torch.Tensor) and buf.shape[-1] == 2:
        return buf
    return torch.zeros(env.num_envs, 2, device=env.device, dtype=torch.float32)


def feet_contact_forces_l_r(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Left/Right foot contact forces in base gravity-aligned frame (yaw-only) as 6D.

    Output layout: [L(Fx,Fy,Fz), R(Fx,Fy,Fz)] where each force is expressed in a yaw-only
    gravity-aligned base frame (roll/pitch removed, yaw kept).
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    forces_w = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]  # (num_envs, 2, 3)

    asset: Articulation = env.scene["robot"]
    root_quat_w = asset.data.root_quat_w  # (num_envs, 4)
    base_rpy = math_utils.euler_xyz_from_quat(root_quat_w)
    yaw_quat = math_utils.quat_from_euler_xyz(
        torch.zeros_like(base_rpy[0]),
        torch.zeros_like(base_rpy[1]),
        base_rpy[2],
    )  # (num_envs, 4)

    # Flatten feet dimension and repeat quats for vectorized transform
    forces_w_flat = forces_w.reshape(env.num_envs * forces_w.shape[1], 3)
    yaw_quat_flat = yaw_quat.repeat_interleave(forces_w.shape[1], dim=0)
    forces_ga_flat = math_utils.quat_apply_inverse(yaw_quat_flat, forces_w_flat)
    forces_ga = forces_ga_flat.reshape(env.num_envs, forces_w.shape[1] * 3)
    return forces_ga


def base_link_mass_delta(
    env: ManagerBasedEnv,
    *,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_name: str = "base_link",
) -> torch.Tensor:
    """Return (current_mass - default_mass) of a body as 1D.

    Intended for predicting startup randomized base mass (e.g., add_base_mass).
    Uses PhysX readback for current masses and IsaacLab-parsed defaults for default mass.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_ids, body_names = asset.find_bodies(body_name, preserve_order=True)
    if len(body_ids) != 1:
        raise RuntimeError(f"Expected exactly one body match for {body_name!r}. Got: {body_names}")
    body_id = int(body_ids[0])

    # PhysX readback is on CPU
    masses_cpu = asset.root_physx_view.get_masses()  # (num_envs, num_bodies)
    masses = masses_cpu.to(device=env.device, dtype=torch.float32)
    default_mass = asset.data.default_mass[:, body_id].to(device=env.device, dtype=torch.float32)
    delta = masses[:, body_id] - default_mass
    return delta.unsqueeze(-1)


def base_link_com_offset(
    env: ManagerBasedEnv,
    *,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_name: str = "base_link",
) -> torch.Tensor:
    """Return body COM offset w.r.t. a cached reference as 3D (x,y,z).

    IsaacLab parses default mass/inertia, but does not expose a `default_com` tensor.
    Therefore, we cache the *first* observed COM as a reference and return offsets relative to it.

    Intended for predicting startup randomized base COM (e.g., base_com event).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_ids, body_names = asset.find_bodies(body_name, preserve_order=True)
    if len(body_ids) != 1:
        raise RuntimeError(f"Expected exactly one body match for {body_name!r}. Got: {body_names}")
    body_id = int(body_ids[0])

    coms_cpu = asset.root_physx_view.get_coms().clone()  # (num_envs, num_bodies, >=3) on CPU
    coms = coms_cpu.to(device=env.device, dtype=torch.float32)[..., :3]

    key = "_p73_default_coms_xyz"
    if key not in env.__dict__:
        # Cache the initial COM reference (per env, per body)
        env.__dict__[key] = coms.detach().clone()

    default_coms = env.__dict__[key]
    if not isinstance(default_coms, torch.Tensor) or default_coms.shape != coms.shape:
        # Re-initialize cache if something is inconsistent (robustness)
        default_coms = coms.detach().clone()
        env.__dict__[key] = default_coms

    return coms[:, body_id, :] - default_coms[:, body_id, :]


def motor_joint_armature_stats(env: ManagerBasedEnv, *, joint_name_keys: str = ".*_motor") -> torch.Tensor:
    """Return mean/std of (armature / default_armature) across motor joints as 2D.

    This summarizes `randomize_joint_parameters(armature_distribution_params=...)` for prediction.
    """
    asset: Articulation = env.scene["robot"]
    joint_ids, _ = asset.find_joints(joint_name_keys, preserve_order=True)
    if len(joint_ids) == 0:
        return torch.zeros(env.num_envs, 2, device=env.device, dtype=torch.float32)
    jids = torch.tensor(joint_ids, device=env.device, dtype=torch.long)

    arm = asset.data.joint_armature[:, jids].to(dtype=torch.float32)
    arm0 = asset.data.default_joint_armature[:, jids].to(dtype=torch.float32).clamp_min(1.0e-8)
    scale = arm / arm0
    mean = scale.mean(dim=1)
    std = scale.std(dim=1, unbiased=False)
    return torch.stack((mean, std), dim=-1)


def persistent_push_force(env: ManagerBasedEnv) -> torch.Tensor:
    """Return the currently applied persistent push force as 3D (Fx, Fy, Fz) in world frame.

    Used as a TargetCfg observation so the DWM encoder learns to estimate external disturbances.
    Returns zeros if no push is active.
    """
    buf = env.__dict__.get("_persistent_push_force", None)
    if isinstance(buf, torch.Tensor) and buf.shape == (env.num_envs, 3):
        return buf
    return torch.zeros(env.num_envs, 3, device=env.device, dtype=torch.float32)


def actuator_command_delay(
    env: ManagerBasedEnv,
    *,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    actuator_name: str = "walker_motors",
    max_delay: int = 4,
) -> torch.Tensor:
    """Return the per-env command delay of the actuator, normalized to [0, 1]. Shape: (num_envs, 1).

    Reads the time_lags from the actuator's delay buffer and divides by max_delay.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    actuator = asset.actuators[actuator_name]
    time_lags = actuator.positions_delay_buffer.time_lags.float()  # (num_envs,)
    normalized = time_lags / max(max_delay, 1)
    return normalized.unsqueeze(-1)


def actuator_motor_strength(
    env: ManagerBasedEnv,
    *,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    actuator_name: str = "walker_motors",
) -> torch.Tensor:
    """Return the per-env, per-joint motor strength scale. Shape: (num_envs, num_joints).

    Reads the last-used motor_strength_scale from the custom actuator.
    Returns ones if the actuator does not have the attribute (no randomization).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    actuator = asset.actuators[actuator_name]
    if hasattr(actuator, "motor_strength_scale"):
        return actuator.motor_strength_scale.clone()
    return torch.ones(env.num_envs, actuator.num_joints, device=env.device)


def motor_joint_damping_stats(env: ManagerBasedEnv, *, joint_name_keys: str = ".*_motor") -> torch.Tensor:
    """Return mean/std of (damping - default_damping) across motor joints as 2D.

    This summarizes `randomize_actuator_gains(damping_distribution_params=..., operation=\"add\")` for prediction.
    """
    asset: Articulation = env.scene["robot"]
    joint_ids, _ = asset.find_joints(joint_name_keys, preserve_order=True)
    if len(joint_ids) == 0:
        return torch.zeros(env.num_envs, 2, device=env.device, dtype=torch.float32)
    jids = torch.tensor(joint_ids, device=env.device, dtype=torch.long)

    d = asset.data.joint_damping[:, jids].to(dtype=torch.float32)
    d0 = asset.data.default_joint_damping[:, jids].to(dtype=torch.float32)
    delta = d - d0
    mean = delta.mean(dim=1)
    std = delta.std(dim=1, unbiased=False)
    return torch.stack((mean, std), dim=-1)