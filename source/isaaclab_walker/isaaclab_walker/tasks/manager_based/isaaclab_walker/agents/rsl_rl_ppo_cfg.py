# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# Copyright (c) 2025, Jaeyong Shin (jasonshin0537@snu.ac.kr).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)


# ============================================================================
# Per-term obs scale vectors (applied as constant buffers inside the model).
#
# Design intent:
# - Env config already normalizes a lot of terms via `scale=` / `clip=` on
#   ObservationTermCfg. For those terms we pass 1.0 here (no double-scaling).
# - Only dims whose raw magnitude is clearly off from O(1) get a non-unit scale.
# - Supervised latent targets (critic_obs[0:9] = gt_vel3 + gt_foot_force6) MUST
#   stay at 1.0 so L_vel / L_foot supervision is consistent.
# - These vectors are baked into the ONNX export graph via act_inference, so
#   deployment code (cc.cpp) can keep sending raw obs.
#
# Index order follows the class-body declaration order in
#   rough_env_cfg.py :: ObservationsCfg::{PolicyCfg, TargetCfg, CriticCfg}
# ============================================================================

# --- PolicyCfg single frame (47D, history_length=10) ---
# base_ang_vel(3) + projected_gravity(3) + velocity_commands(3)
# + gait_sin(1) + gait_cos(1) + motor_joint_pos(12)
# + motor_joint_vel(12, env *1/30 already) + actions(12)
_POLICY_OBS_SCALE_SINGLE: list[float] = (
    [1.0] * 3    # base_ang_vel
    + [1.0] * 3  # projected_gravity  (unit vector)
    + [1.0] * 3  # velocity_commands  (~±1 m/s)
    + [1.0]      # gait_phase_sin
    + [1.0]      # gait_phase_cos
    + [1.0] * 12 # motor_joint_pos (rel, ±0.5 rad)
    + [1.0] * 12 # motor_joint_vel (env scale=1/30 already applied)
    + [1.0] * 12 # actions (clip ±1)
)
assert len(_POLICY_OBS_SCALE_SINGLE) == 47, f"policy scale single must be 47D, got {len(_POLICY_OBS_SCALE_SINGLE)}"

# --- TargetCfg (154D, history_length=1) ---
# base_ang_vel(3) + gravity(3) + cmd(3) + gait(2) + joint_pos(12)
# + joint_vel(12, env/30) + height_scan(81, clipped ±1) + actions(12)
# + physics_material(2) + base_mass_delta(1, *0.1) + base_com_offset(3, *5)
# + armature(2) + damping(2) + push_force(3, env/100)
# + command_delay(1, normalized) + motor_strength(12, [0.8,1.2])
# Note: motor_strength is 12 (not 13) because the "walker_motors" actuator
# group now covers only the 12 leg joints (WaistYaw is in a separate PD group).
_TARGET_OBS_SCALE: list[float] = (
    [1.0] * 3    # base_ang_vel
    + [1.0] * 3  # projected_gravity
    + [1.0] * 3  # velocity_commands
    + [1.0]      # gait_phase_sin
    + [1.0]      # gait_phase_cos
    + [1.0] * 12 # motor_joint_pos
    + [1.0] * 12 # motor_joint_vel (env /30)
    + [1.0] * 81 # height_scan (env clip ±1)
    + [1.0] * 12 # actions
    + [1.0] * 2  # physics_material (sd, O(1))
    + [0.1]      # base_mass_delta (DR range [-5, 10] kg -> ±1.0)
    + [5.0] * 3  # base_com_offset (±0.1 m -> ±0.5)
    + [1.0] * 2  # motor_armature_stats
    + [1.0] * 2  # motor_damping_stats
    + [1.0] * 3  # push_force (env /100)
    + [1.0]      # command_delay (normalized [0,1])
    + [1.0] * 12 # motor_strength (per-joint over walker_motors group, [0.8, 1.2])
)
assert len(_TARGET_OBS_SCALE) == 154, f"target scale must be 154D, got {len(_TARGET_OBS_SCALE)}"

# --- CriticCfg (160D, history_length=1) ---
# gt_vel3(3, LOCKED) + gt_foot_force6(6, LOCKED) + base_ang_vel(3)
# + gravity(3) + cmd(3) + gait(2) + joint_pos(12) + joint_vel(12, env/30)
# + actions(12) + height_scan(81, clipped) + physics_material(2)
# + base_mass_delta(1, *0.1) + base_com_offset(3, *5) + armature(2) + damping(2)
# + command_delay(1, normalized) + motor_strength(12, [0.8,1.2])
_CRITIC_OBS_SCALE: list[float] = (
    [1.0] * 3    # gt_vel3 — L_vel supervision target (LOCKED at 1.0)
    + [1.0] * 6  # gt_foot_force6 — L_foot supervision target (LOCKED at 1.0)
    + [1.0] * 3  # base_ang_vel
    + [1.0] * 3  # projected_gravity
    + [1.0] * 3  # velocity_commands
    + [1.0]      # gait_phase_sin
    + [1.0]      # gait_phase_cos
    + [1.0] * 12 # motor_joint_pos
    + [1.0] * 12 # motor_joint_vel (env /30)
    + [1.0] * 12 # actions
    + [1.0] * 81 # height_scan (env clip ±1)
    + [1.0] * 2  # physics_material
    + [0.1]      # base_mass_delta
    + [5.0] * 3  # base_com_offset
    + [1.0] * 2  # motor_armature_stats
    + [1.0] * 2  # motor_damping_stats
    + [1.0]      # command_delay (normalized [0,1])
    + [1.0] * 12 # motor_strength (per-joint over walker_motors group, [0.8, 1.2])
)
assert len(_CRITIC_OBS_SCALE) == 160, f"critic scale must be 160D, got {len(_CRITIC_OBS_SCALE)}"


@configclass
class P73RslRlPpoActorCriticFutureCfg(RslRlPpoActorCriticCfg):
    """Config schema for TOCABI-style ActorCriticAdaptationFuture in IsaacLab 5.1.

    Keep only type schema/class binding here and define concrete hyper-parameters
    in the runner config below to avoid duplicated sources of truth.
    """

    class_name: str = "ActorCriticAdaptationFuture"
    encoder_hidden_dims: list[int] = MISSING
    latent_dim: int = MISSING
    num_single_obs: int = MISSING
    target_obs_dim: int = MISSING
    target_encoder_hidden_dims: list[int] = MISSING
    future_dim: int = MISSING
    policy_obs_scale_single: list[float] = MISSING
    critic_obs_scale: list[float] = MISSING
    target_obs_scale: list[float] = MISSING


@configclass
class P73RslRlPpoFutureAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Config schema for PPOFuture auxiliary losses."""

    class_name: str = "PPOFuture"
    w_vel: float = MISSING
    w_foot: float = MISSING
    w_future: float = MISSING
    target_obs_dim: int = MISSING


@configclass
class P73RoughPPORunnerFutureCfg(RslRlOnPolicyRunnerCfg):
    """PPOFuture runner config (TOCABI-style latent supervision) for Walker.

    Structure parity with TOCABI framework:
    - history encoder -> latent
    - latent prefix supervision (vel/foot/future)
    - O(t+1) target encoder matching

    Walker differences vs p73:
    - No 4-bar linkage, all 12 lower-body joints directly actuated
    - No measured (passive) joints → policy obs 47D instead of 59D
    - No upper-body arms → only WaistYaw as PD-held upper body
    """

    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 200
    experiment_name = "walker_flat"
    empirical_normalization = False
    logger = "wandb"
    wandb_project = "walker_flat"

    # Actor-Critic with encoder + target-encoder
    # Single source of truth for architecture hyper-parameters.
    #
    # Walker single-frame policy dim:
    #   base_ang_vel(3) + projected_gravity(3) + cmd(3) + gait_phase(2)
    #   + motor_pos(12) + motor_vel(12) + actions(12) = 47D
    #
    # Walker target obs dim:
    #   base_ang_vel(3) + projected_gravity(3) + cmd(3) + gait_phase(2)
    #   + motor_pos(12) + motor_vel(12) + height_scan(81) + actions(12)
    #   + physics_material(2) + base_mass_delta(1) + base_com_offset(3)
    #   + motor_armature_stats(2) + motor_damping_stats(2)
    #   + push_force(3) + command_delay(1) + motor_strength(12) = 154D
    policy = P73RslRlPpoActorCriticFutureCfg(
        init_noise_std=1.0,
        noise_std_type="log",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        # Encoder / latent
        encoder_hidden_dims=[512, 512, 256],
        latent_dim=64,
        num_single_obs=47,
        target_obs_dim=154,
        target_encoder_hidden_dims=[128],
        future_dim=30,
        # Per-term scale buffers (baked into ONNX graph via act_inference)
        policy_obs_scale_single=_POLICY_OBS_SCALE_SINGLE,
        critic_obs_scale=_CRITIC_OBS_SCALE,
        target_obs_scale=_TARGET_OBS_SCALE,
    )

    # PPOFuture algorithm config (aux losses + PPO core)
    algorithm = P73RslRlPpoFutureAlgorithmCfg(
        value_loss_coef=5.0,
        use_clipped_value_loss=True,
        clip_param=0.16,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.97,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        # Auxiliary latent supervision weights
        w_vel=1.0,
        w_foot=1.0,
        w_future=1.0,
        target_obs_dim=154,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=False,
            use_mirror_loss=True,
            mirror_loss_coeff=2.0,
            data_augmentation_func=(
                "isaaclab_walker.tasks.manager_based.isaaclab_walker.mdp.symmetry:"
                "p73_data_augmentation_lowerbody_mirror"
            ),
        ),
    )
