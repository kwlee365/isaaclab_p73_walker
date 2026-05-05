from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_WALKER_LOWER_JOINT_NAMES_ORDERED: list[str] = [
    "L_HipRoll_Joint",
    "L_HipPitch_Joint",
    "L_HipYaw_Joint",
    "L_Knee_Joint",
    "L_AnklePitch_Joint",
    "L_AnkleRoll_Joint",
    "R_HipRoll_Joint",
    "R_HipPitch_Joint",
    "R_HipYaw_Joint",
    "R_Knee_Joint",
    "R_AnklePitch_Joint",
    "R_AnkleRoll_Joint",
]


def _assert_p73_joint_order(env: "ManagerBasedRLEnv") -> None:
    """Verify that symmetry-critical action/observation orders match expected lower-body order."""
    env_u = getattr(env, "unwrapped", env)
    if env_u.__dict__.get("_p73_symmetry_joint_order_ok", False):
        return

    try:
        action_term = env_u.action_manager.get_term("joint_pos")
        action_joint_ids = list(getattr(action_term, "_joint_ids"))
        action_asset = getattr(action_term, "_asset")
        action_joint_names = [action_asset.data.joint_names[i] for i in action_joint_ids]
    except Exception as e:
        raise RuntimeError(
            "Failed to access runtime action order from action term 'joint_pos' via '_joint_ids'."
        ) from e

    if action_joint_names != _WALKER_LOWER_JOINT_NAMES_ORDERED:
        mismatch = next(
            (
                (i, exp, got)
                for i, (exp, got) in enumerate(zip(_WALKER_LOWER_JOINT_NAMES_ORDERED, action_joint_names))
                if exp != got
            ),
            None,
        )
        raise RuntimeError(
            "Walker symmetry assumes a fixed 12-DOF lower-body order, but runtime action order differs.\n"
            f"First mismatch: idx={mismatch[0] if mismatch else 'N/A'} "
            f"expected={mismatch[1] if mismatch else 'N/A'} got={mismatch[2] if mismatch else 'N/A'}\n"
            "Fix: align ActionsCfg.joint_pos.lower_joint_names / _joint_ids with symmetry order."
        )

    cfg = cast(Any, getattr(env_u, "cfg", env.cfg))
    for term_name in ("motor_joint_pos", "motor_joint_vel"):
        term_cfg = cfg.observations.policy.__dict__[term_name]
        asset_cfg = term_cfg.params.get("asset_cfg", None)
        if asset_cfg is None:
            continue
        robot = env_u.scene[asset_cfg.name]
        joint_ids = asset_cfg.joint_ids
        func_name = getattr(getattr(term_cfg, "func", None), "__name__", "")
        if func_name in ("joint_pos_ordered_rel", "joint_vel_ordered"):
            obs_joint_names = list(getattr(asset_cfg, "joint_names", []) or [])
        else:
            if joint_ids == slice(None):
                obs_joint_names = list(getattr(robot, "joint_names"))
            else:
                obs_joint_names = [robot.joint_names[i] for i in joint_ids]
        if obs_joint_names != _WALKER_LOWER_JOINT_NAMES_ORDERED:
            mismatch = next(
                (
                    (i, exp, got)
                    for i, (exp, got) in enumerate(zip(_WALKER_LOWER_JOINT_NAMES_ORDERED, obs_joint_names))
                    if exp != got
                ),
                None,
            )
            raise RuntimeError(
                "Walker symmetry assumes a fixed 12-DOF lower-body order, but runtime observation order differs.\n"
                f"Term: policy.{term_name}\n"
                f"First mismatch: idx={mismatch[0] if mismatch else 'N/A'} "
                f"expected={mismatch[1] if mismatch else 'N/A'} got={mismatch[2] if mismatch else 'N/A'}\n"
                "Fix: keep policy motor_joint_pos/motor_joint_vel joint_names aligned with lower_joint_names."
            )

    env_u.__dict__["_p73_symmetry_joint_order_ok"] = True


def _flip_lowerbody_12_axis_aware(joint_tensor: torch.Tensor) -> torch.Tensor:
    """Mirror Walker lower-body 12D vector with axis-aware sign rules.

    Order:
      [0] L_HipRoll_Joint
      [1] L_HipPitch_Joint
      [2] L_HipYaw_Joint
      [3] L_Knee_Joint
      [4] L_AnklePitch_Joint
      [5] L_AnkleRoll_Joint
      [6] R_HipRoll_Joint
      [7] R_HipPitch_Joint
      [8] R_HipYaw_Joint
      [9] R_Knee_Joint
      [10] R_AnklePitch_Joint
      [11] R_AnkleRoll_Joint

    Rule (derived from URDF axial-vector sagittal mirror analysis):
      Under sagittal (xz-plane) reflection, a rotation axis (axial vector)
      transforms as (ax, ay, az) -> (-ax, ay, -az).  For each joint pair,
      if axial_mirror(L_axis) == R_axis → swap only (angles stay the same).
      Otherwise → swap + negate.

      Joint         L axis  R axis  axial_mirror(L)   == R?  Rule
      HipRoll       +x      -x      -x                yes    swap only
      HipPitch      -y      +y      -y                no     swap + negate
      HipYaw        -z      -z      +z                no     swap + negate
      Knee          +y      -y      +y                no     swap + negate
      AnklePitch    +y      -y      +y                no     swap + negate
      AnkleRoll     +x      +x      -x                no     swap + negate

    Verified against URDF forward kinematics: under this rule, FK(mirror(q))
    matches sagittal_mirror(FK(q)) at machine precision for arbitrary q.
    """
    if joint_tensor.shape[1] != 12:
        raise ValueError(f"Expected 12D lower-body tensor. Got: {joint_tensor.shape}.")
    out = torch.zeros_like(joint_tensor)

    # Left -> Right
    out[:, 6] = joint_tensor[:, 0]     # HipRoll:    swap only     (axial-mirror(+x) = -x = R_axis)
    out[:, 7] = -joint_tensor[:, 1]    # HipPitch:   swap + negate
    out[:, 8] = -joint_tensor[:, 2]    # HipYaw:     swap + negate (same L/R axes → +θ gives opposite physical rotation)
    out[:, 9] = -joint_tensor[:, 3]    # Knee:       swap + negate
    out[:, 10] = -joint_tensor[:, 4]   # AnklePitch: swap + negate
    out[:, 11] = -joint_tensor[:, 5]   # AnkleRoll:  swap + negate (same L/R axes)

    # Right -> Left
    out[:, 0] = joint_tensor[:, 6]     # HipRoll:    swap only
    out[:, 1] = -joint_tensor[:, 7]    # HipPitch:   swap + negate
    out[:, 2] = -joint_tensor[:, 8]    # HipYaw:     swap + negate
    out[:, 3] = -joint_tensor[:, 9]    # Knee:       swap + negate
    out[:, 4] = -joint_tensor[:, 10]   # AnklePitch: swap + negate
    out[:, 5] = -joint_tensor[:, 11]   # AnkleRoll:  swap + negate
    return out


def p73_data_augmentation_lowerbody_mirror(
    obs: torch.Tensor | None,
    actions: torch.Tensor | None,
    env: "ManagerBasedRLEnv",
    obs_type: str = "policy",
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Walker lower-body symmetry augmentation helper for mirror loss path.

    Policy single-frame layout is fixed to 47D:
      [0:3]   base_ang_vel        mirror: [-1, +1, -1]   (axial: roll/yaw flip)
      [3:6]   projected_gravity   mirror: [+1, -1, +1]   (polar: y flip)
      [6:9]   velocity_commands   mirror: [+1, -1, -1]   (lin_y flip, yaw rate flip)
      [9:11]  gait_phase_sin/cos  mirror: both × -1      (mirror = phase + 0.5 cycle)
      [11:23] motor_joint_pos (12D, axis-aware flip)
      [23:35] motor_joint_vel (12D, axis-aware flip)
      [35:47] actions(last_processed_action, 12D, axis-aware flip)
    """
    if (obs is not None and obs_type == "policy") or actions is not None:
        _assert_p73_joint_order(env)

    if obs is not None and obs.ndim == 1:
        obs = obs.unsqueeze(0)

    if actions is not None and actions.ndim == 1:
        actions = actions.unsqueeze(0)

    if obs is not None:
        if obs_type != "policy":
            raise ValueError(f"Walker lower-body symmetry currently supports obs_type='policy' only. Got: {obs_type}")
        single_frame_dim = 47
        if obs.shape[1] % single_frame_dim != 0:
            raise ValueError(f"Policy obs dim must be a multiple of 47. Got obs_dim={obs.shape[1]}.")
        history_length = obs.shape[1] // single_frame_dim
        obs_flipped = obs.clone()
        for i in range(history_length):
            start = i * single_frame_dim
            base_ang_vel = obs[:, start + 0 : start + 3]
            projected_gravity = obs[:, start + 3 : start + 6]
            vel_cmd = obs[:, start + 6 : start + 9]
            gait_phase = obs[:, start + 9 : start + 11]
            motor_joint_pos = obs[:, start + 11 : start + 23]
            motor_joint_vel = obs[:, start + 23 : start + 35]
            last_actions = obs[:, start + 35 : start + 47]

            base_ang_vel_flipped = base_ang_vel * torch.tensor([-1.0, 1.0, -1.0], device=obs.device)
            projected_gravity_flipped = projected_gravity * torch.tensor([1.0, -1.0, 1.0], device=obs.device)
            vel_cmd_flipped = vel_cmd * torch.tensor([1.0, -1.0, -1.0], device=obs.device)
            gait_phase_flipped = -gait_phase
            motor_joint_pos_flipped = _flip_lowerbody_12_axis_aware(motor_joint_pos)
            motor_joint_vel_flipped = _flip_lowerbody_12_axis_aware(motor_joint_vel)
            last_actions_flipped = _flip_lowerbody_12_axis_aware(last_actions)

            obs_flipped[:, start : start + single_frame_dim] = torch.cat(
                [
                    base_ang_vel_flipped,
                    projected_gravity_flipped,
                    vel_cmd_flipped,
                    gait_phase_flipped,
                    motor_joint_pos_flipped,
                    motor_joint_vel_flipped,
                    last_actions_flipped,
                ],
                dim=1,
            )
        obs_augmented = torch.cat([obs, obs_flipped], dim=0)
    else:
        obs_augmented = None

    if actions is not None:
        actions_flipped = _flip_lowerbody_12_axis_aware(actions)
        actions_augmented = torch.cat([actions, actions_flipped], dim=0)
    else:
        actions_augmented = None

    return obs_augmented, actions_augmented
