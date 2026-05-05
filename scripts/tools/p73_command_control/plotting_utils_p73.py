"""
Plotting utilities for P73 teleoperation playback (IsaacLab 5.1.0 repo local copy).

This module emits a JSON payload compatible with:
  tools/plot_contact_force_loco.py  (same schema as TOCABI pipeline)

Keys:
  - root_base_vel: cmd/true in yaw-only base frame
  - head_vel:      cmd/true/est for plot compatibility (P73 est=0)
  - foot_pos_base: left/right foot positions in yaw-only base frame (relative to root)
  - foot_force:    left/right contact net forces in yaw-only base frame
  - motor_torque:  12D lower-body motor torques (applied/target) + torque_limits
  - motor_vel:     12D lower-body motor joint velocities (measured) + velocity limits

Foot link names:
  - L_Foot_Link
  - R_Foot_Link
"""

from __future__ import annotations

import json
import sys
from typing import Any

import torch


def _is_piped_stdout() -> bool:
    return not sys.stdout.isatty()


def extract_plotting_data_p73(env, timestep: int) -> dict[str, Any] | None:
    try:
        actual_env = env.unwrapped
        robot = actual_env.scene["robot"]

        from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat

        yaw_q = yaw_quat(robot.data.root_quat_w)  # (N,4) yaw-only

        # ---- command: base_velocity (vx, vy, wz) ----
        cmd_base = [0.0, 0.0, 0.0]
        try:
            cmd = actual_env.command_manager.get_command("base_velocity")
            if cmd is not None:
                c0 = cmd[0].detach().to(device="cpu").to(dtype=torch.float32).numpy()
                cmd_base = [
                    float(c0[0]) if c0.shape[0] > 0 else 0.0,
                    float(c0[1]) if c0.shape[0] > 1 else 0.0,
                    float(c0[2]) if c0.shape[0] > 2 else 0.0,
                ]
        except Exception:
            pass

        # ---- true root vel in yaw-only base frame ----
        true_root_base = [0.0, 0.0, 0.0]
        try:
            root_lin_vel_w = robot.data.root_lin_vel_w[0]
            root_ang_vel_w = robot.data.root_ang_vel_w[0]
            lin_yaw = quat_apply_inverse(yaw_q[0].unsqueeze(0), root_lin_vel_w.unsqueeze(0))[0]
            true_root_base = [
                float(lin_yaw[0].detach().cpu()),
                float(lin_yaw[1].detach().cpu()),
                float(root_ang_vel_w[2].detach().cpu()),
            ]
        except Exception:
            pass

        # ---- head_vel (world XY + yawrate) for plot compatibility ----
        cmd_head_vel = [cmd_base[0], cmd_base[1], cmd_base[2]]
        try:
            cmd_xy_base = torch.tensor([[cmd_base[0], cmd_base[1], 0.0]], device=yaw_q.device, dtype=yaw_q.dtype)
            cmd_xy_w = quat_apply(yaw_q[0].unsqueeze(0), cmd_xy_base)[0]
            cmd_head_vel = [float(cmd_xy_w[0].detach().cpu()), float(cmd_xy_w[1].detach().cpu()), float(cmd_base[2])]
        except Exception:
            pass

        true_head_vel = [0.0, 0.0, 0.0]
        try:
            root_lin_vel_w = robot.data.root_lin_vel_w[0]
            root_ang_vel_w = robot.data.root_ang_vel_w[0]
            true_head_vel = [
                float(root_lin_vel_w[0].detach().cpu()),
                float(root_lin_vel_w[1].detach().cpu()),
                float(root_ang_vel_w[2].detach().cpu()),
            ]
        except Exception:
            pass

        est_head_vel = [0.0, 0.0, 0.0]

        # ---- resolve lower-body 12D joint selection (shared for torque/velocity plots) ----
        lower_joint_ids: list[int] = []
        lower_joint_names: list[str] = []
        torque_limits_cfg_12: list[float] = []
        try:
            # Resolve lower motor names and cfg torque limits from the action term.
            # P73 action term name is expected to be "joint_pos".
            action_term = actual_env.action_manager.get_term("joint_pos")
            action_cfg = getattr(action_term, "cfg", None)
            lower_names = list(getattr(action_cfg, "lower_joint_names", []))
            torque_limits_cfg = list(getattr(action_cfg, "torque_limits", []))
            torque_limits_cfg_12 = [float(x) for x in torque_limits_cfg[:12]]

            # Resolve joint ids in articulation in the explicit name order.
            try:
                joint_ids, joint_names_resolved = robot.find_joints(lower_names, preserve_order=True)
                lower_joint_ids = [int(j) for j in joint_ids]
                lower_joint_names = list(joint_names_resolved)
            except Exception:
                lower_joint_ids = [int(robot.data.joint_names.index(n)) for n in lower_names]
                lower_joint_names = [str(n) for n in lower_names]
        except Exception:
            lower_joint_ids = []
            lower_joint_names = []
            torque_limits_cfg_12 = []

        # ---- 12D lower-body motor torque (applied + target) and limits ----
        motor_torque = None
        try:
            if lower_joint_ids:
                applied = (
                    robot.data.applied_torque[0, lower_joint_ids]
                    .detach()
                    .to(device="cpu")
                    .to(dtype=torch.float32)
                    .numpy()
                    .tolist()
                )
                target = (
                    robot.data.joint_effort_target[0, lower_joint_ids]
                    .detach()
                    .to(device="cpu")
                    .to(dtype=torch.float32)
                    .numpy()
                    .tolist()
                )

                # Prefer action-cfg torque limits; fallback to sim effort limits.
                limit_list = [float(x) for x in torque_limits_cfg_12]
                if len(limit_list) != len(lower_joint_ids):
                    try:
                        sim_limits = (
                            robot.data.joint_effort_limits[0, lower_joint_ids]
                            .detach()
                            .to(device="cpu")
                            .to(dtype=torch.float32)
                            .numpy()
                            .tolist()
                        )
                        limit_list = [float(x) for x in sim_limits]
                    except Exception:
                        limit_list = [0.0 for _ in range(len(lower_joint_ids))]

                applied_list = [float(x) for x in applied]
                target_list = [float(x) for x in target]

                eps = 1e-6
                sat_applied = [
                    bool(abs(a) >= (l - eps)) if l > eps else False for a, l in zip(applied_list, limit_list)
                ]
                sat_target = [bool(abs(t) >= (l - eps)) if l > eps else False for t, l in zip(target_list, limit_list)]

                motor_torque = {
                    "names": lower_joint_names,
                    "applied": applied_list,
                    "target": target_list,
                    "limit": limit_list,
                    "sat_applied": sat_applied,
                    "sat_target": sat_target,
                }
        except Exception:
            motor_torque = None

        # ---- 12D lower-body joint velocity (measured) and velocity limits ----
        motor_vel = None
        try:
            if lower_joint_ids:
                measured = (
                    robot.data.joint_vel[0, lower_joint_ids]
                    .detach()
                    .to(device="cpu")
                    .to(dtype=torch.float32)
                    .numpy()
                    .tolist()
                )
                measured_list = [float(x) for x in measured]

                # Velocity limits are provided by simulation (rad/s). Use absolute magnitude.
                try:
                    vel_limits = (
                        robot.data.joint_vel_limits[0, lower_joint_ids]
                        .detach()
                        .to(device="cpu")
                        .to(dtype=torch.float32)
                        .numpy()
                        .tolist()
                    )
                    limit_list = [abs(float(x)) for x in vel_limits]
                except Exception:
                    limit_list = [0.0 for _ in range(len(lower_joint_ids))]

                eps = 1e-6
                sat_measured = [
                    bool(abs(v) >= (l - eps)) if l > eps else False for v, l in zip(measured_list, limit_list)
                ]

                motor_vel = {
                    "names": lower_joint_names,
                    "measured": measured_list,
                    "limit": limit_list,
                    "sat_measured": sat_measured,
                }
        except Exception:
            motor_vel = None

        # ---- contact forces: ContactSensor net forces ----
        foot_force_left = [0.0, 0.0, 0.0]
        foot_force_right = [0.0, 0.0, 0.0]
        try:
            contact_sensor = actual_env.scene.sensors["contact_forces"]
            body_ids, body_names = contact_sensor.find_bodies(["L_Foot_Link", "R_Foot_Link"], preserve_order=True)
            if len(body_ids) != 2:
                raise ValueError(f"Expected 2 foot bodies in ContactSensor, got {len(body_ids)}: {body_names}")

            net_forces_w = contact_sensor.data.net_forces_w
            left_force_w = net_forces_w[0, body_ids[0], :]
            right_force_w = net_forces_w[0, body_ids[1], :]

            left_force_base = quat_apply_inverse(yaw_q[0].unsqueeze(0), left_force_w.unsqueeze(0))[0]
            right_force_base = quat_apply_inverse(yaw_q[0].unsqueeze(0), right_force_w.unsqueeze(0))[0]

            foot_force_left = [
                float(left_force_base[0].detach().cpu()),
                float(left_force_base[1].detach().cpu()),
                float(left_force_base[2].detach().cpu()),
            ]
            foot_force_right = [
                float(right_force_base[0].detach().cpu()),
                float(right_force_base[1].detach().cpu()),
                float(right_force_base[2].detach().cpu()),
            ]
        except Exception:
            pass

        # ---- foot positions in yaw-only base frame (relative to root) ----
        foot_pos_left_base = [0.0, 0.0, 0.0]
        foot_pos_right_base = [0.0, 0.0, 0.0]
        try:
            foot_body_ids, foot_body_names = robot.find_bodies(["L_Foot_Link", "R_Foot_Link"], preserve_order=True)
            if len(foot_body_ids) != 2:
                raise ValueError(f"Expected 2 foot bodies in robot, got {len(foot_body_ids)}: {foot_body_names}")

            root_pos_w = robot.data.root_pos_w[0]
            left_pos_w = robot.data.body_pos_w[0, foot_body_ids[0], :]
            right_pos_w = robot.data.body_pos_w[0, foot_body_ids[1], :]

            left_rel_w = left_pos_w - root_pos_w
            right_rel_w = right_pos_w - root_pos_w

            left_rel_base = quat_apply_inverse(yaw_q[0].unsqueeze(0), left_rel_w.unsqueeze(0))[0]
            right_rel_base = quat_apply_inverse(yaw_q[0].unsqueeze(0), right_rel_w.unsqueeze(0))[0]

            foot_pos_left_base = [
                float(left_rel_base[0].detach().cpu()),
                float(left_rel_base[1].detach().cpu()),
                float(left_rel_base[2].detach().cpu()),
            ]
            foot_pos_right_base = [
                float(right_rel_base[0].detach().cpu()),
                float(right_rel_base[1].detach().cpu()),
                float(right_rel_base[2].detach().cpu()),
            ]
        except Exception:
            pass

        return {
            "root_base_vel": {"cmd": cmd_base, "true": true_root_base},
            "head_vel": {"cmd": cmd_head_vel, "true": true_head_vel, "est": est_head_vel},
            "foot_pos_base": {"left": foot_pos_left_base, "right": foot_pos_right_base},
            "foot_force": {"left": {"true": foot_force_left}, "right": {"true": foot_force_right}},
            "motor_torque": motor_torque,
            "motor_vel": motor_vel,
        }
    except Exception:
        return None


def print_plotting_data_p73(plot_data: dict[str, Any] | None, force_output: bool = False):
    if plot_data is None:
        return
    if _is_piped_stdout() or force_output:
        print(f"METRIC_DATA: {json.dumps(plot_data)}", flush=True)


def extract_and_print_plotting_data_p73(env, timestep: int, enable_plotting: bool = True, force_output: bool = False):
    if not enable_plotting:
        return
    plot_data = extract_plotting_data_p73(env=env, timestep=timestep)
    print_plotting_data_p73(plot_data, force_output=force_output)

