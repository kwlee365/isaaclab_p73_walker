"""
Spawn walker robot and verify USD is correctly generated.

Usage (run from IsaacLab directory):
    ./isaaclab.sh -p /home/piene/isaaclab5.2/isaaclab_walker/scripts/spawn_walker.py
"""

import argparse
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Spawn walker robot to verify USD.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from isaaclab_walker import P73_CFG  # isort: skip


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[3.0, 0.0, 2.25], target=[0.0, 0.0, 0.9])

    # Ground + Light
    sim_utils.GroundPlaneCfg().func("/World/defaultGroundPlane", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)).func("/World/Light", sim_utils.DomeLightCfg(intensity=2000.0))

    # Spawn walker
    walker = Articulation(P73_CFG.replace(prim_path="/World/Walker"))

    sim.reset()

    # Print joint info for verification
    print("\n" + "=" * 60)
    print("Walker USD Verification")
    print("=" * 60)
    print(f"Number of joints: {walker.num_joints}")
    print(f"Number of bodies: {walker.num_bodies}")
    print(f"\nJoint names ({walker.num_joints}):")
    for i, name in enumerate(walker.data.joint_names):
        print(f"  [{i:2d}] {name}")
    print(f"\nBody names ({walker.num_bodies}):")
    for i, name in enumerate(walker.data.body_names):
        print(f"  [{i:2d}] {name}")
    print(f"\nDefault joint positions:")
    for i, (name, pos) in enumerate(zip(walker.data.joint_names, walker.data.default_joint_pos[0].tolist())):
        print(f"  [{i:2d}] {name}: {pos:.4f}")
    print("=" * 60 + "\n")

    # Run simulation loop
    sim_dt = sim.get_physics_dt()
    count = 0
    while simulation_app.is_running():
        if count % 500 == 0:
            joint_pos, joint_vel = walker.data.default_joint_pos, walker.data.default_joint_vel
            walker.write_joint_state_to_sim(joint_pos, joint_vel)
            root_state = walker.data.default_root_state.clone()
            walker.write_root_pose_to_sim(root_state[:, :7])
            walker.write_root_velocity_to_sim(root_state[:, 7:])
            walker.reset()
            print(">>>>>>>> Reset!")

        walker.write_data_to_sim()
        sim.step()
        count += 1
        walker.update(sim_dt)


if __name__ == "__main__":
    main()
    simulation_app.close()
