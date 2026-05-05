"""Smoke test for ActuatorNetLSTMWalker integration.

Boots Isaac Sim headless, instantiates the P73 articulation with a couple of
envs, resets, runs a handful of physics steps, and prints diagnostics. Passes
if no exception is raised, all tensors stay finite, and the LSTM/LUT/motor-
scale buffers look right.

Usage:
    ./isaaclab.sh -p scripts/smoke_actuator_net.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Smoke test: ActuatorNetLSTMWalker.")
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--steps", type=int, default=50)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest after SimulationApp boot."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from isaaclab_walker import P73_CFG
from isaaclab_walker.actuators import ActuatorNetLSTMWalker


def _status(msg: str, ok: bool = True):
    tag = "[PASS]" if ok else "[FAIL]"
    print(f"{tag} {msg}")


def main():
    # -- World setup --
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Replicate across num_envs
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.utils import configclass

    @configclass
    class _SceneCfg(InteractiveSceneCfg):
        robot = P73_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    scene_cfg = _SceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO] Scene constructed.")

    robot: Articulation = scene["robot"]

    # -- Assertions on actuator structure --
    actuators = robot.actuators
    print(f"[INFO] Actuator groups: {list(actuators.keys())}")
    _status("walker_motors present", "walker_motors" in actuators)
    _status("waist present", "waist" in actuators)

    wm = actuators["walker_motors"]
    assert isinstance(wm, ActuatorNetLSTMWalker), f"Got {type(wm).__name__}"
    _status(f"walker_motors class = {type(wm).__name__}")
    _status(f"walker_motors num_joints = {wm.num_joints} (expect 12)", wm.num_joints == 12)
    _status(f"num networks loaded = {len(wm.networks)}", len(wm.networks) == wm.num_joints)

    # Buffer shapes
    B = args_cli.num_envs
    print(f"[INFO] sea_hidden_state.shape = {tuple(wm.sea_hidden_state.shape)}"
          f"  (expect (12, num_layers, {B}, hidden))")
    print(f"[INFO] motor_strength_scale.shape = {tuple(wm.motor_strength_scale.shape)}"
          f"  (expect ({B}, 12))")
    print(f"[INFO] positions_delay_buffer.time_lags.shape = {tuple(wm.positions_delay_buffer.time_lags.shape)}")
    print(f"[INFO] torque_scale = {wm._torque_scale}")
    print(f"[INFO] use LUT = {wm._use_lut}")
    print(f"[INFO] motor_scale_range = {wm._motor_scale_range}")

    # -- Reset & run steps --
    joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.reset()

    any_nan = False
    torque_stats = []
    for i in range(args_cli.steps):
        # Hold default pose as position target
        robot.set_joint_position_target(joint_pos)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.get_physics_dt())

        if torch.isnan(wm.applied_effort).any() or torch.isinf(wm.applied_effort).any():
            any_nan = True
            print(f"[FAIL] NaN/Inf in applied_effort at step {i}")
            break

        if i in (0, 1, args_cli.steps // 2, args_cli.steps - 1):
            ae = wm.applied_effort
            torque_stats.append(
                (i, ae.abs().mean().item(), ae.abs().max().item(),
                 wm.motor_strength_scale.mean().item())
            )

    _status("applied_effort finite over all steps", not any_nan)

    print("\n[INFO] applied_effort (Nm) stats at selected steps:")
    print(f"  {'step':>6}  {'|tau|.mean':>12}  {'|tau|.max':>12}  {'motor_scale.mean':>18}")
    for s, m, mx, ms in torque_stats:
        print(f"  {s:>6}  {m:>12.3f}  {mx:>12.3f}  {ms:>18.3f}")

    # -- Reset again and verify hidden state zeroing --
    before = wm.sea_hidden_state.abs().sum().item()
    robot.reset()
    after = wm.sea_hidden_state.abs().sum().item()
    _status(
        f"sea_hidden_state zeroed on reset (before={before:.3f} → after={after:.3f})",
        after == 0.0,
    )

    print("\n[INFO] Smoke test complete.")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
