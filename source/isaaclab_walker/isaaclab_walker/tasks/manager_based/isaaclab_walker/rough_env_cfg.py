# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.envs import ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
)

from . import mdp

from isaaclab_walker import P73_CFG  # isort: skip

# =====================================================================
# Walker joint name constants
# =====================================================================
# Lower body: 12 joints (6 per leg), all RL-controlled
_LOWER_JOINT_NAMES = [
    "L_HipRoll_Joint", "L_HipPitch_Joint", "L_HipYaw_Joint",
    "L_Knee_Joint", "L_AnklePitch_Joint", "L_AnkleRoll_Joint",
    "R_HipRoll_Joint", "R_HipPitch_Joint", "R_HipYaw_Joint",
    "R_Knee_Joint", "R_AnklePitch_Joint", "R_AnkleRoll_Joint",
]
# Upper body: 1 joint (PD-held at default)
_UPPER_JOINT_NAMES = ["WaistYaw_Joint"]


@configclass
class KangarooRewards(RewardsCfg):
    """Reward terms for the MDP."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=3.5,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    # Anti-crouch height shaping (two terms):
    #   A. One-sided L2 below target — tall posture unpenalized, avoids
    #      the "fall vs crouch" trade-off of symmetric L2.
    #   C. Bell-curve reward at target (std=0.03) — dense signal near target,
    #      vanishing gradient far away (no incentive to fall further).
    body_height_below_target_l2 = RewTerm(
        func=mdp.base_height_below_target_l2,
        weight=-30.0,
        params={"target_height": 0.89},
    )

    body_height_target_exp = RewTerm(
        func=mdp.base_height_target_exp,
        weight=3.0,
        params={"target_height": 0.89, "std": 0.03},
    )

    # === 2. Stability & Safety Rewards ==============================================

    flat_base_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-30.0,
    )

    lin_vel_z_l2 = RewTerm(
        func=mdp.lin_vel_z_l2,
        weight=-0.3,
    )

    ang_vel_xy_l2 = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-0.3,
    )

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-5.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[
                "base_link",
                ".*_Thigh_Link",
                ".*_Knee_Link",
            ]),
            "threshold": 1.0,
        },
    )

    # === 3. Joint Control & Limits ==============================================

    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_AnklePitch_Joint", ".*_AnkleRoll_Joint"])},
    )

    all_joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    joint_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.001,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*")
        },
    )

    # === 4. Joint Deviation & Limits ==============================================

    joint_deviation_hipyaw = RewTerm(
        func=mdp.bio_mimetic_soft_hard_constraint_conditional,
        weight=-1.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["L_HipYaw_Joint", "R_HipYaw_Joint"]),
            "command_name": "base_velocity",
            "command_index": 2,                 # ang_z command
            "command_threshold": 0.01,
            "base_vel_index": 2,                # actual ang_vel_z
            "base_vel_threshold": 0.07,
            "target_offset": 0.0,
            "deadband_pos_active": 0.0,        # allowed during rotation
            "deadband_neg_active": 0.0,
            "deadband_pos_inactive": 0.0,       # tight otherwise
            "deadband_neg_inactive": 0.0,
            "stiffness_pos": 1.5,
            "stiffness_neg": 1.5,
        },
    )

    # HipRoll: L axis +x, R axis -x (mirrored URDF axes).
    # Both: pos = abduction, neg = adduction in own frame.
    # → L/R can share one term (same sign = same physical direction).
    joint_deviation_hiproll = RewTerm(
        func=mdp.bio_mimetic_soft_hard_constraint_conditional,
        weight=-1.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["L_HipRoll_Joint", "R_HipRoll_Joint"]),
            "command_name": "base_velocity",
            "command_index": 1,                 # vy command (lateral)
            "command_threshold": 0.01,
            "base_vel_index": 1,                # actual vy
            "base_vel_threshold": 0.07,
            "target_offset": 0.0,
            "deadband_pos_active": 0.0,         # abduction allowed under lateral motion / pushes
            "deadband_neg_active": 0.0,         # adduction limited under lateral motion / pushes
            "deadband_pos_inactive": 0.0,       # tight otherwise
            "deadband_neg_inactive": 0.0,
            "stiffness_pos": 1.5,
            "stiffness_neg": 2.5,
        },
    )

    # Knee: L axis +y [0, 2.56] default 0.35, R axis -y [-2.56, 0] default -0.35.
    # Axes are mirrored, so L/R need separate terms (sign of e flips per side).
    # Allowed range: q ∈ [0.15, 1.35] (L), q ∈ [-1.35, -0.15] (R).
    # Max flexion ~1.35 rad (~77°, slightly above human peak-gait flexion of
    # 60–70°) prevents the policy from dropping base height via a deep knee bend.
    joint_deviation_left_knee = RewTerm(
        func=mdp.bio_mimetic_soft_hard_constraint,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["L_Knee_Joint"]),
            "target_offset": 0.0,
            "deadband_pos": 1.0,     # flexion to 1.35 (1.35 - 0.35)
            "deadband_neg": 0.20,    # extension to 0.15 (0.35 - 0.15)
            "stiffness_pos": 0.2,    # soft on flexion
            "stiffness_neg": 5.0,    # strong against hyperextension
        },
    )

    joint_deviation_right_knee = RewTerm(
        func=mdp.bio_mimetic_soft_hard_constraint,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["R_Knee_Joint"]),
            "target_offset": 0.0,
            "deadband_pos": 0.20,    # extension to -0.15 (0.35 - 0.15)
            "deadband_neg": 1.0,     # flexion to -1.35 (1.35 - 0.35)
            "stiffness_pos": 5.0,    # strong against hyperextension
            "stiffness_neg": 0.2,    # soft on flexion
        },
    )

    # AnklePitch: L axis +y default -0.17, R axis -y default 0.17 (mirrored).
    # L: [-1.05, 0.7], R: [-0.7, 1.05]. Allowed q ∈ [-0.41, 0.41] (same intent both sides).
    joint_deviation_left_anklepitch = RewTerm(
        func=mdp.bio_mimetic_soft_hard_constraint,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["L_AnklePitch_Joint"]),
            "target_offset": 0.0,
            "deadband_pos": 0.58,    # up to 0.41 (0.41 - (-0.17))
            "deadband_neg": 0.24,    # down to -0.41 (-0.17 - (-0.41))
            "stiffness_pos": 3.0,
            "stiffness_neg": 3.0,
        },
    )

    joint_deviation_right_anklepitch = RewTerm(
        func=mdp.bio_mimetic_soft_hard_constraint,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["R_AnklePitch_Joint"]),
            "target_offset": 0.0,
            "deadband_pos": 0.24,    # up to 0.41 (0.41 - 0.17)
            "deadband_neg": 0.58,    # down to -0.41 (0.17 - (-0.41))
            "stiffness_pos": 3.0,
            "stiffness_neg": 3.0,
        },
    )

    # AnkleRoll: L/R both axis +x, default 0.0, range [-0.42, 0.42].
    # Same axis with symmetric range → L/R unified.
    joint_deviation_ankleroll = RewTerm(
        func=mdp.bio_mimetic_soft_hard_constraint,
        weight=-5.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["L_AnkleRoll_Joint", "R_AnkleRoll_Joint"]),
            "target_offset": 0.0,
            "deadband": 0.2,      # free within ±0.2 rad (~11°)
            "stiffness": 3.0,
        },
    )

    # === 5. Action & Energy Efficiency ==============================================

    # --- Action-rate L2, tiered by real-robot |Δa| measurement ---
    # Action index order follows _LOWER_JOINT_NAMES:
    #   0:L_HipRoll  1:L_HipPitch  2:L_HipYaw  3:L_Knee  4:L_AnklePitch  5:L_AnkleRoll
    #   6:R_HipRoll  7:R_HipPitch  8:R_HipYaw  9:R_Knee 10:R_AnklePitch 11:R_AnkleRoll
    #
    # Measured |Δa| p99 (real robot, 260420 — 6 CSVs, 316k rows, moving):
    #   Knee        0.248 / 0.195 (L/R)   max 0.56 / 0.39   (jerkiest)
    #   AnklePitch  0.202 / 0.211         max 0.30 / 0.42   (was 0.087 → 2.3× worse)
    #   HipPitch    0.115 / 0.117         max 0.23 / 0.27
    #   AnkleRoll   0.067 / 0.074         max 0.13 / 0.15
    #   HipYaw      0.065 / 0.056         max 0.13 / 0.12   (1.5× HipRoll)
    #   HipRoll     0.041 / 0.040         max 0.12 / 0.11   (quietest)
    #
    # Weight strategy: penalize jerkier joints harder. AnklePitch raised in line
    # with its 2.3× regression. Tier A split into HipRoll/HipYaw (1.5× ratio).

    # Tier A1 — HipRoll (very quiet): unchanged
    action_rate_hiproll = RewTerm(
        func=mdp.action_rate_l2_selective,
        weight=-0.02,
        params={"action_indices": [0, 6]},  # L/R HipRoll
    )

    # Tier A2 — HipYaw (real p99 1.5× HipRoll): slightly stronger
    action_rate_hipyaw = RewTerm(
        func=mdp.action_rate_l2_selective,
        weight=-0.035,
        params={"action_indices": [2, 8]},  # L/R HipYaw
    )

    # Tier B — HipPitch + AnkleRoll (moderate): unchanged
    action_rate_mid = RewTerm(
        func=mdp.action_rate_l2_selective,
        weight=-0.06,
        params={"action_indices": [1, 5, 7, 11]},  # L/R HipPitch + L/R AnkleRoll
    )

    # Tier C — AnklePitch: real p99 0.087 → 0.21 (2.3× worse) → raised proportionally
    action_rate_anklepitch = RewTerm(
        func=mdp.action_rate_l2_selective,
        weight=-0.15,
        params={"action_indices": [4, 10]},  # L/R AnklePitch
    )

    # Tier D — Knee (most jerk, p99 0.25): slight ↑
    action_rate_knee = RewTerm(
        func=mdp.action_rate_l2_selective,
        weight=-0.30,
        params={"action_indices": [3, 9]},  # L/R Knee
    )

    action_accel_l2 = RewTerm(
        func=mdp.action_accel_l2,
        weight=-0.004,
    )

    # --- Acceleration penalties (smoothness) ---
    # Baseline on all joints; distal joints (Knee/Ankle) get an extra term since
    # real-robot data shows their qdd p99 is ~2–4× higher than hip joints.
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-5.0e-7,
    )

    dof_acc_distal_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.0e-6,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[
                ".*_Knee_Joint", ".*_AnklePitch_Joint", ".*_AnkleRoll_Joint",
            ])
        },
    )

    # --- Torque L2 penalties, split by τ_max group so each joint sees a
    # comparable saturation-normalized penalty (w ∝ 1/τ_max²).
    # Reference: Knee/HipPitch (τ_max=220) at w=5.5e-6.
    # WaistYaw is excluded (real-robot data: max 17 Nm << 152 Nm limit).

    # τ_max = 352 (HipRoll) — rarely saturates → lightest weight
    dof_torques_hiproll_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-0.5e-6,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[
                "L_HipRoll_Joint", "R_HipRoll_Joint",
            ])
        },
    )

    # τ_max = 220 (HipPitch, Knee) — split to bias gait toward hip-driven walking.
    # HipPitch: 0.7× baseline (cheaper to use)  → encourages larger hip swing
    # Knee:     2.0× baseline (more expensive)  → discourages deep knee flexion
    dof_torques_hippitch_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-3.80e-6,  # 5.5e-6 × 0.7
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[
                ".*_HipPitch_Joint",
            ])
        },
    )

    dof_torques_knee_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.1e-5,  # 5.5e-6 × 2.0
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[
                ".*_Knee_Joint",
            ])
        },
    )

    # τ_max = 95 (HipYaw, Ankle*) — most saturation in real data → heaviest weight
    # Scaled by (220/95)² ≈ 5.4 so saturation-normalized penalty matches Knee group.
    dof_torques_small_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-3.0e-5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[
                ".*_HipYaw_Joint", ".*_AnklePitch_Joint", ".*_AnkleRoll_Joint",
            ])
        },
    )

    # Soft torque-limit penalty: activates only at |τ|/τ_max > 0.8.
    # Directly targets saturation, leaves normal torque usage unpenalized.
    # AnklePitch is excluded here — it gets its own stricter term below.
    dof_torques_limit_soft = RewTerm(
        func=mdp.joint_torques_limit_soft_l2,
        weight=-1.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[
                ".*_HipPitch_Joint", ".*_HipYaw_Joint",
                ".*_Knee_Joint", ".*_AnkleRoll_Joint",
            ]),
            "soft_ratio": 0.8,
        },
    )

    # AnklePitch-only soft-limit: earlier activation (soft_ratio=0.7) and stronger
    # weight to target the 40% real-robot saturation observed at this joint.
    dof_torques_anklepitch_limit_soft = RewTerm(
        func=mdp.joint_torques_limit_soft_l2,
        weight=-2.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_AnklePitch_Joint"]),
            "soft_ratio": 0.7,
        },
    )


    # === 6. Feet Contact & Stability ==============================================

    feet_air_time = RewTerm(
        func=mdp.feet_air_time_biped,
        weight=4.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["L_Foot_Link", "R_Foot_Link"],
                preserve_order=True,
            ),
            "asset_cfg": SceneEntityCfg("robot"),
            "threshold": 0.45,
            "velocity_threshold": 0.01,
            "stop_cmd_vel_max": 0.07,
            "yaw_threshold_deg": 5.0,
            "pos_threshold_m": 0.02,
            "stance_width_m": 0.205,
            # Lock parallel x stance at stop: |x_L - x_R| ~ x_sep_target_m (feet on same x-line).
            "x_sep_target_m": 0.0,
            "contact_threshold": 5.0,
            # Schedule gate: tocabi_should_walk_stop_align already keeps the
            # walking branch on while feet are misaligned at stop, so a fixed
            # grace window is redundant. Push-activate trigger remains.
            "cmd_zero_max": 0.0,
            "grace_steps": 0,
            "push_suppress_threshold": 1.0,
            "push_activates": True,
        },
    )

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-5.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Foot_Link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_Foot_Link"),
        },
    )

    feet_ground_parallel = RewTerm(
        func=mdp.feet_ground_parallel,
        weight=-15.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Foot_Link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_Foot_Link"),
            "threshold": 1.0,
        },
    )

    # Swing-phase counterpart: keeps the foot flat while airborne so the gait
    # doesn't show toe-down dangling. Lower weight than the contact term to
    # avoid overdriving ankle torque near touchdown / takeoff.
    feet_ground_parallel_swing = RewTerm(
        func=mdp.feet_ground_parallel_swing,
        weight=-10.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Foot_Link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_Foot_Link"),
            "threshold": 1.0,
        },
    )

    feet_parallel = RewTerm(
        func=mdp.feet_parallel,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Foot_Link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_Foot_Link"),
            "threshold": 1.0,
        },
    )

    feet_yaw_symmetry_about_base = RewTerm(
        func=mdp.feet_yaw_symmetry_about_base_l2,
        weight=-3.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["L_Foot_Link", "R_Foot_Link"],
                preserve_order=True,
            ),
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["L_Foot_Link", "R_Foot_Link"],
                preserve_order=True,
            ),
            "reference_body_name": "base_link",
            "threshold": 1.0,
        },
    )

    feet_yaw_align_cmd_gated_l2 = RewTerm(
        func=mdp.feet_yaw_align_cmd_gated_l2,
        weight=-3.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_Foot_Link"),
            "wz_threshold": 0.01,
        },
    )

    feet_clearance_penalty = RewTerm(
        func=mdp.feet_clearance_height,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_Foot_Link"),
            "target_height": 0.07,
        },
    )

    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Foot_Link"),
            "threshold_ratio": 3.0,
        },
    )

    contact_momentum = RewTerm(
        func=mdp.contact_momentum,
        weight=-4.5e-4,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Foot_Link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_Foot_Link"),
        },
    )

    contact_force_limit = RewTerm(
        func=mdp.contact_force_limit_soft_penalty,
        weight=-15.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Foot_Link"),
            "robot_mass": 60.0,
            "safety_factor": 1.5,
            "std": 200.0,
        },
    )

    contact_schedule_biped_ds = RewTerm(
        func=mdp.contact_schedule_reward_biped_ds,
        weight=5.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["L_Foot_Link", "R_Foot_Link"],
                preserve_order=True,
            ),
            "asset_cfg": SceneEntityCfg("robot"),
            "period_steps": 40,
            "ds_ratio": 0.20,
            "contact_threshold": 5.0,
            "cmd_zero_max": 0.0,
            # misalign_activates keeps the schedule on until feet align — that's
            # a more precise replacement for the fixed grace window.
            "grace_steps": 0,
            "push_suppress_threshold": 1.0,
            "push_activates": True,
            # Keep stepping while feet aren't aligned at stop intent.
            # Thresholds match feet_air_time_biped's stop-alignment switch.
            "misalign_activates": True,
            "stop_cmd_vel_max": 0.07,
            "yaw_threshold_deg": 5.0,
            "pos_threshold_m": 0.02,
            "stance_width_m": 0.205,
            "x_sep_target_m": 0.0,
        },
    )

    swing_clearance_min_profile_penalty = RewTerm(
        func=mdp.swing_clearance_min_profile_penalty,
        weight=-60.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["L_Foot_Link", "R_Foot_Link"],
                preserve_order=True,
            ),
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["L_Foot_Link", "R_Foot_Link"],
                preserve_order=True,
            ),
            "period_steps": 40,
            "ds_ratio": 0.20,
            "clearance_height": 0.13,
            "profile_offset": 0.0075,
            "contact_threshold": 5.0,
            "cmd_zero_max": 0.0,
            # misalign_activates supersedes the fixed grace window.
            "grace_steps": 0,
            "push_suppress_threshold": 1.0,
            # Keep swing-clearance penalty active while feet aren't aligned at
            # stop intent — so the robot keeps lifting the foot during the
            # alignment-correction stepping.
            "misalign_activates": True,
            "stop_cmd_vel_max": 0.07,
            "yaw_threshold_deg": 5.0,
            "pos_threshold_m": 0.02,
            "stance_width_m": 0.205,
            "x_sep_target_m": 0.0,
        },
    )

    spring_compliance_pos_match = RewTerm(
        func=mdp.spring_compliance_pos_match_reward,
        weight=-0.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["L_Foot_Link", "R_Foot_Link"],
                preserve_order=True,
            ),
            "asset_cfg": SceneEntityCfg("robot"),
            "head_body_name": "base_link",
            "k_spring": 9000.0,
            "contact_threshold": 5.0,
            "robot_mass": 60.3,
            "baseline_margin": 0.5,
            "gravity": 9.81,
            "command_name": "base_velocity",
            "velocity_threshold": 0.0,
            "cmd_vel_eps": 0.0,
        },
    )

    feet_height_symmetry_penalty = RewTerm(
        func=mdp.feet_height_symmetry_penalty,
        weight=-15.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["L_Foot_Link", "R_Foot_Link"]),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["L_Foot_Link", "R_Foot_Link"]),
            "command_name": "base_velocity",
            "velocity_threshold": 0.0,
            "contact_threshold": 5.0,
        },
    )

    air_time_variance = RewTerm(
        func=mdp.air_time_variance_penalty_gated,
        weight=-20.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["L_Foot_Link", "R_Foot_Link"]),
            "command_name": "base_velocity",
            "velocity_threshold": 0.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class ActionsCfg:
    # PD gains, torque limits, LUT, delay, and motor randomization are now
    # configured in the actuator (DelayedPDActuatorLUTCfg) inside p73_walker.py.
    # The action term only handles scaling, clipping, and position target computation.
    joint_pos = mdp.LowerBodyActionsCfg(
        asset_name="robot",
        clip={".*": (-1.0, 1.0)},
        scale=0.5,
        lower_joint_names=_LOWER_JOINT_NAMES,
        upper_joint_names=_UPPER_JOINT_NAMES,

        # Joint position limits from URDF (L leg then R leg)
        # New URDF: L/R axes are mirrored, limits reflect the new axis directions
        joint_pos_limits=[(-0.58, 0.3), (-1.57, 2.09), (-0.78, 0.78), (0.0, 2.56), (-1.05, 0.7), (-0.42, 0.42),
                          (-0.58, 0.3), (-2.09, 1.57), (-0.78, 0.78), (-2.56, 0.0), (-0.7, 1.05), (-0.42, 0.42)],
    )


@configclass
class ObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group.

        Walker has NO passive joints — all 12 lower-body joints are motor joints.
        Policy obs per frame: 3+3+3+2+12+12+12 = 47D
        """

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.4, n_max=0.4)) # 0.2 -> 0.4
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)) # 0.05 -> 0.1 -> 0.2 -> 0.05
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        gait_phase_sin = ObsTerm(
            func=mdp.gait_phase_sin,
            params={"period_steps": 40, "command_name": "base_velocity", "cmd_zero_max": 0.01},
        )
        gait_phase_cos = ObsTerm(
            func=mdp.gait_phase_cos,
            params={"period_steps": 40, "command_name": "base_velocity", "cmd_zero_max": 0.01},
        )
        motor_joint_pos = ObsTerm(
            func=mdp.joint_pos_ordered_rel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=_LOWER_JOINT_NAMES)
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        motor_joint_vel = ObsTerm(
            func=mdp.joint_vel_ordered,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=_LOWER_JOINT_NAMES)
            },
            noise=Unoise(n_min=-1.5, n_max=1.5), # (-1.0, 1.0)
            clip=(-30.0, 30.0),
            scale=1.0 / 30.0,
        )
        actions = ObsTerm(func=mdp.last_processed_action, params={"action_name": "joint_pos"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 10

    @configclass
    class TargetCfg(ObsGroup):
        """Target observation group for PPOFuture."""

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        gait_phase_sin = ObsTerm(
            func=mdp.gait_phase_sin,
            params={"period_steps": 40, "command_name": "base_velocity", "cmd_zero_max": 0.01},
        )
        gait_phase_cos = ObsTerm(
            func=mdp.gait_phase_cos,
            params={"period_steps": 40, "command_name": "base_velocity", "cmd_zero_max": 0.01},
        )
        motor_joint_pos = ObsTerm(
            func=mdp.joint_pos_ordered_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=_LOWER_JOINT_NAMES)},
        )
        motor_joint_vel = ObsTerm(
            func=mdp.joint_vel_ordered,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=_LOWER_JOINT_NAMES)},
            clip=(-30.0, 30.0),
            scale=1.0 / 30.0,
        )
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )
        actions = ObsTerm(func=mdp.last_processed_action, params={"action_name": "joint_pos"})

        physics_material = ObsTerm(func=mdp.physics_material_sd)
        base_mass_delta = ObsTerm(func=mdp.base_link_mass_delta)
        base_com_offset = ObsTerm(func=mdp.base_link_com_offset)
        motor_armature_stats = ObsTerm(func=mdp.motor_joint_armature_stats, params={"joint_name_keys": ".*_Joint"})
        motor_damping_stats = ObsTerm(func=mdp.motor_joint_damping_stats, params={"joint_name_keys": ".*_Joint"})
        push_force = ObsTerm(func=mdp.persistent_push_force, scale=1.0 / 100.0)
        # Actuator privileged info
        command_delay = ObsTerm(
            func=mdp.actuator_command_delay,
            params={"actuator_name": "walker_motors", "max_delay": 2},
        )
        motor_strength = ObsTerm(func=mdp.actuator_motor_strength, params={"actuator_name": "walker_motors"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 1
            self.flatten_history_dim = True

    @configclass
    class CriticCfg(ObsGroup):
        """Critic observation group for PPOFuture."""

        gt_vel3 = ObsTerm(func=mdp.base_vel_xy_yawrate)
        gt_foot_force6 = ObsTerm(
            func=mdp.feet_contact_forces_l_r,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=["L_Foot_Link", "R_Foot_Link"],
                    preserve_order=True,
                )
            },
            scale=(
                1.0 / 300.0, 1.0 / 300.0, 1.0 / 600.0,
                1.0 / 300.0, 1.0 / 300.0, 1.0 / 600.0,
            ),
        )

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        gait_phase_sin = ObsTerm(
            func=mdp.gait_phase_sin,
            params={"period_steps": 40, "command_name": "base_velocity", "cmd_zero_max": 0.01},
        )
        gait_phase_cos = ObsTerm(
            func=mdp.gait_phase_cos,
            params={"period_steps": 40, "command_name": "base_velocity", "cmd_zero_max": 0.01},
        )
        motor_joint_pos = ObsTerm(
            func=mdp.joint_pos_ordered_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=_LOWER_JOINT_NAMES)},
        )
        motor_joint_vel = ObsTerm(
            func=mdp.joint_vel_ordered,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=_LOWER_JOINT_NAMES)},
            clip=(-30.0, 30.0),
            scale=1.0 / 30.0,
        )
        actions = ObsTerm(func=mdp.last_processed_action, params={"action_name": "joint_pos"})

        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )
        physics_material = ObsTerm(func=mdp.physics_material_sd)
        base_mass_delta = ObsTerm(func=mdp.base_link_mass_delta)
        base_com_offset = ObsTerm(func=mdp.base_link_com_offset)
        motor_armature_stats = ObsTerm(func=mdp.motor_joint_armature_stats, params={"joint_name_keys": ".*_Joint"})
        motor_damping_stats = ObsTerm(func=mdp.motor_joint_damping_stats, params={"joint_name_keys": ".*_Joint"})
        # Actuator privileged info
        command_delay = ObsTerm(
            func=mdp.actuator_command_delay,
            params={"actuator_name": "walker_motors", "max_delay": 2},
        )
        motor_strength = ObsTerm(func=mdp.actuator_motor_strength, params={"actuator_name": "walker_motors"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 1
            self.flatten_history_dim = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    target: TargetCfg = TargetCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material_and_cache,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (0.2, 1.4),
            "dynamic_friction_range": (0.2, 1.4),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    randomize_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-5.0, 10.0), # -10.0, 20.0
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "com_range": {
                "x": (-0.10, 0.10),
                "y": (-0.10, 0.10),
                "z": (-0.10, 0.10),
            },
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5), "pitch": (-0.5, 0.5), "yaw": (-0.5, 0.5),
            },
        },
    )
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset, mode="reset",
        params={"position_range": (-0.1, 0.1), "velocity_range": (-0.1, 0.1)},
    )
    # Armature DR DISABLED for LSTM actuator-net configuration.
    # Reason: armature is a motor-side parameter; varying it (0.6×–1.4×) makes the
    # joint's qdd-response to a given torque differ by up to ±40%. The LSTM was
    # trained on real-robot data with one fixed armature, so randomized values put
    # the LSTM input (q̇) out-of-distribution and produce unreliable torques.
    # Plant-side DR (link mass, base mass/COM, friction, push) remains active to
    # preserve sim2real robustness without contaminating the actuator-net domain.
    randomize_armature = EventTerm(
        func=mdp.randomize_joint_parameters, mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_Joint"),
            "friction_distribution_params": (0.0, 0.0),
            "armature_distribution_params": (1.0, 1.0), # 0.6, 1.4
            "operation": "scale",
        },
    )
    randomize_damping = EventTerm(
        func=mdp.randomize_actuator_gains, mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_Joint"),
            "stiffness_distribution_params": (0.0, 0.0),
            "damping_distribution_params": (0.0, 0.0),
            "operation": "add",
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_persistent_force,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={
            "force_range": {
                "x": (-60.0, 60.0), # (-60.0, 60.0)
                "y": (-60.0, 60.0), # (-60.0, 60.0)
                "z": (-30.0, 30.0), # (-30.0, 30.0)
            },
            "duration_s": (0.5, 2.0),
        },
    )



@configclass
class WalkerViewerCfg(ViewerCfg):
    resolution: tuple[int, int] = (1920, 1080)
    eye: tuple[float, float, float] = (10, 10, 10)


@configclass
class P73RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: KangarooRewards = KangarooRewards()
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    viewer: WalkerViewerCfg = WalkerViewerCfg()

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005

        # Stop/restart transition coverage: boost zero-cmd sampling so the policy
        # sees more walk→stop→walk cycles during training (was 0.02 from base cfg).
        self.commands.base_velocity.rel_standing_envs = 0.15

        # Scene
        self.scene.num_envs = 4096
        self.scene.robot = P73_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base_link"
        self.scene.height_scanner.offset.pos = (0.0, 0.0, 0.0)
        self.scene.height_scanner.pattern_cfg.size = [1.2, 1.2]
        self.scene.height_scanner.pattern_cfg.resolution = 0.15
        self.scene.height_scanner.debug_vis = True

        # Terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base_link"

        # Phase-gait hyperparameters
        gait_period_steps = 70
        gait_ds_ratio = 0.25

        self.observations.policy.gait_phase_sin.params["period_steps"] = gait_period_steps
        self.observations.policy.gait_phase_cos.params["period_steps"] = gait_period_steps
        self.observations.critic.gait_phase_sin.params["period_steps"] = gait_period_steps
        self.observations.critic.gait_phase_cos.params["period_steps"] = gait_period_steps
        self.observations.target.gait_phase_sin.params["period_steps"] = gait_period_steps
        self.observations.target.gait_phase_cos.params["period_steps"] = gait_period_steps

        self.rewards.contact_schedule_biped_ds.params["period_steps"] = gait_period_steps
        self.rewards.contact_schedule_biped_ds.params["ds_ratio"] = gait_ds_ratio
        self.rewards.swing_clearance_min_profile_penalty.params["period_steps"] = gait_period_steps
        self.rewards.swing_clearance_min_profile_penalty.params["ds_ratio"] = gait_ds_ratio
