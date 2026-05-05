import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab_walker.actuators import ActuatorNetLSTMWalkerCfg
from isaaclab_walker.assets import P73_ASSETS_DATA_DIR

import os

_TORQUE_LUT_DIR = os.path.join(P73_ASSETS_DATA_DIR, "p73_walker", "lut")
_ACTUATOR_NET_DIR = os.path.join(P73_ASSETS_DATA_DIR, "p73_walker", "actuator_nets")

# Per-joint LSTM network files (trained on real-robot data with disturbance).
# Keys must match the exact joint names below.
_LSTM_NETWORK_FILES = {
    "L_HipRoll_Joint":     "p73_lstm_left_hip_roll.pt",
    "L_HipPitch_Joint":    "p73_lstm_left_hip_pitch.pt",
    "L_HipYaw_Joint":      "p73_lstm_left_hip_yaw.pt",
    "L_Knee_Joint":        "p73_lstm_left_knee_pitch.pt",
    "L_AnklePitch_Joint":  "p73_lstm_left_ankle_pitch.pt",
    "L_AnkleRoll_Joint":   "p73_lstm_left_ankle_roll.pt",
    "R_HipRoll_Joint":     "p73_lstm_right_hip_roll.pt",
    "R_HipPitch_Joint":    "p73_lstm_right_hip_pitch.pt",
    "R_HipYaw_Joint":      "p73_lstm_right_hip_yaw.pt",
    "R_Knee_Joint":        "p73_lstm_right_knee_pitch.pt",
    "R_AnklePitch_Joint":  "p73_lstm_right_ankle_pitch.pt",
    "R_AnkleRoll_Joint":   "p73_lstm_right_ankle_roll.pt",
}

_LEG_JOINT_NAMES = list(_LSTM_NETWORK_FILES.keys())

# Walker initial joint positions (standing pose)
# All joints are directly actuated — no passive/4-bar joints.
# Order: L leg (6) + R leg (6) + WaistYaw (1) = 13 joints
# New URDF: L/R axes are mirrored — opposite sign = same physical motion
init = {
    # Left leg  (HipRoll +X, HipPitch -Y, Knee +Y, AnklePitch +Y)
    "L_HipRoll_Joint": 0.0,
    "L_HipPitch_Joint": 0.18,
    "L_HipYaw_Joint": 0.0,
    "L_Knee_Joint": 0.35,
    "L_AnklePitch_Joint": -0.17,
    "L_AnkleRoll_Joint": 0.0,
    # Right leg (HipRoll -X, HipPitch +Y, Knee -Y, AnklePitch -Y)
    "R_HipRoll_Joint": 0.0,
    "R_HipPitch_Joint": -0.18,
    "R_HipYaw_Joint": 0.0,
    "R_Knee_Joint": -0.35,
    "R_AnklePitch_Joint": 0.17,
    "R_AnkleRoll_Joint": 0.0,
    # Waist
    "WaistYaw_Joint": 0.0,
}


P73_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{P73_ASSETS_DATA_DIR}/p73_walker/p73_walker.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.895),
        joint_pos=init,
    ),
    soft_joint_pos_limit_factor=1.0,
    actuators={
        # ---- 12 leg joints: per-joint LSTM actuator networks ----
        # Networks replace PD control; LUT, motor-strength randomization, and
        # command delay are preserved from the previous setup.
        "walker_motors": ActuatorNetLSTMWalkerCfg(
            joint_names_expr=_LEG_JOINT_NAMES,
            # Per-joint trained LSTM weights
            network_files=_LSTM_NETWORK_FILES,
            network_dir=_ACTUATOR_NET_DIR,
            # Networks were trained with torque_scaling=0.01 → recover Nm with ×100
            torque_scale=100.0,
            # Continuous (rated/thermal) torque — sustained operating limit.
            # Acts as the upper envelope cap in the DC-motor BEMF clipping:
            #   τ_max(q̇) = min( saturation_effort × (1 − q̇/q̇_max), effort_limit ).
            # At low speeds the continuous cap binds; at higher speeds the BEMF
            # curve from saturation_effort drops below this and binds instead.
            effort_limit={
                ".*_HipRoll_Joint": 264.0,
                ".*_HipPitch_Joint": 165.0,
                ".*_HipYaw_Joint": 69.0,
                ".*_Knee_Joint": 165.0,
                ".*_AnklePitch_Joint": 69.0,
                ".*_AnkleRoll_Joint": 69.0,
            },
            effort_limit_sim={
                ".*_HipRoll_Joint": 264.0,
                ".*_HipPitch_Joint": 165.0,
                ".*_HipYaw_Joint": 69.0,
                ".*_Knee_Joint": 165.0,
                ".*_AnklePitch_Joint": 69.0,
                ".*_AnkleRoll_Joint": 69.0,
            },
            # Peak/stall torque (motor's maximum at zero speed). Drives the BEMF
            # torque-speed curve; physically the absolute electromagnetic limit.
            # Ratio peak/continuous ≈ 1.33 here (from P73 motor spec).
            saturation_effort={
                ".*_HipRoll_Joint": 352.0,
                ".*_HipPitch_Joint": 220.0,
                ".*_HipYaw_Joint": 95.0,
                ".*_Knee_Joint": 220.0,
                ".*_AnklePitch_Joint": 95.0,
                ".*_AnkleRoll_Joint": 95.0,
            },
            # No-load speed used by both PhysX and the BEMF clipping curve.
            # Setting `velocity_limit` (not just _sim) ensures DCMotor's
            # torque-speed model uses the same value the simulator solver does.
            velocity_limit={
                ".*_HipRoll_Joint": 4.86,
                ".*_HipPitch_Joint": 7.78,
                ".*_HipYaw_Joint": 11.81,
                ".*_Knee_Joint": 7.78,
                ".*_AnklePitch_Joint": 11.81,
                ".*_AnkleRoll_Joint": 11.81,
            },
            velocity_limit_sim={
                ".*_HipRoll_Joint": 4.86,
                ".*_HipPitch_Joint": 7.78,
                ".*_HipYaw_Joint": 11.81,
                ".*_Knee_Joint": 7.78,
                ".*_AnklePitch_Joint": 11.81,
                ".*_AnkleRoll_Joint": 11.81,
            },
            # Friction parameters from system identification
            viscous_friction={
                "L_HipRoll_Joint": 15.0000,
                "L_HipPitch_Joint": 15.0000,
                "L_HipYaw_Joint": 0.0031346,
                "L_Knee_Joint": 10.7850,
                "L_AnklePitch_Joint": 5.1401,
                "L_AnkleRoll_Joint": 11.7930,
                "R_HipRoll_Joint": 15.0000,
                "R_HipPitch_Joint": 15.0000,
                "R_HipYaw_Joint": 3.3811,
                "R_Knee_Joint": 11.8920,
                "R_AnklePitch_Joint": 6.6517,
                "R_AnkleRoll_Joint": 12.6730,
            },
            friction={
                "L_HipRoll_Joint": 15.0000,
                "L_HipPitch_Joint": 14.2060,
                "L_HipYaw_Joint": 3.1658,
                "L_Knee_Joint": 0.056750,
                "L_AnklePitch_Joint": 2.4292,
                "L_AnkleRoll_Joint": 0.62729,
                "R_HipRoll_Joint": 15.0000,
                "R_HipPitch_Joint": 10.4270,
                "R_HipYaw_Joint": 0.00041395,
                "R_Knee_Joint": 1.3967,
                "R_AnklePitch_Joint": 2.6899,
                "R_AnkleRoll_Joint": 0.000034422,
            },
            dynamic_friction={
                "L_HipRoll_Joint": 15.0000,
                "L_HipPitch_Joint": 14.2060,
                "L_HipYaw_Joint": 3.1658,
                "L_Knee_Joint": 0.056750,
                "L_AnklePitch_Joint": 2.4292,
                "L_AnkleRoll_Joint": 0.62729,
                "R_HipRoll_Joint": 15.0000,
                "R_HipPitch_Joint": 10.4270,
                "R_HipYaw_Joint": 0.00041395,
                "R_Knee_Joint": 1.3967,
                "R_AnklePitch_Joint": 2.6899,
                "R_AnkleRoll_Joint": 0.000034422,
            },
            armature={
                "L_HipRoll_Joint": 0.96,
                "L_HipPitch_Joint": 0.375,
                "L_HipYaw_Joint": 0.0625,
                "L_Knee_Joint": 0.35630,
                "L_AnklePitch_Joint": 0.12886,
                "L_AnkleRoll_Joint": 0.12883,
                "R_HipRoll_Joint": 0.96,
                "R_HipPitch_Joint": 0.375,
                "R_HipYaw_Joint": 0.0625,
                "R_Knee_Joint": 0.35630,
                "R_AnklePitch_Joint": 0.12886,
                "R_AnkleRoll_Joint": 0.12883,
            },
            # Command delay: fixed = 1 physics step because the actuator net was
            # trained with delay=1. Even though the raw system-ID delay value is
            # 0.3943 physics steps (which rounds to 0), training used 1, so sim
            # must match training — not the raw ID value — to stay in distribution.
            min_delay=1,
            max_delay=1,
            # Angle-dependent torque limits for 4-bar linkage joints
            torque_lut_dir=_TORQUE_LUT_DIR,
            # Per-step motor strength randomization — DISABLED.
            # Re-randomized every 5 ms (200 Hz) acts as multiplicative white noise on
            # torque, exciting closed-loop resonances of the LSTM actuator net and
            # appearing as 25 Hz alias oscillation in 50 Hz logs. The actuator net
            # already internalizes the real robot's motor variability, so this term
            # is redundant. If reintroduced later, randomize per-reset only, not per-step.
            rand_motor_scale_range=None,
        ),
        # ---- Waist: no trained network → keep simple delayed PD ----
        "waist": DelayedPDActuatorCfg(
            joint_names_expr=["WaistYaw_Joint"],
            stiffness={"WaistYaw_Joint": 576.0},
            damping={"WaistYaw_Joint": 19.2},
            effort_limit={"WaistYaw_Joint": 152.0},
            effort_limit_sim={"WaistYaw_Joint": 152.0},
            velocity_limit_sim={"WaistYaw_Joint": 4.03},
            viscous_friction={"WaistYaw_Joint": 7.8947},
            friction={"WaistYaw_Joint": 7.7742},
            dynamic_friction={"WaistYaw_Joint": 7.7742},
            armature={"WaistYaw_Joint": 0.16},
            min_delay=0,
            max_delay=2,
        ),
    },
)
