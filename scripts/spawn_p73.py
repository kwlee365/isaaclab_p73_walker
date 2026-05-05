# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# Copyright (c) 2025, Jaeyong Shin (jasonshin0537@snu.ac.kr).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to simulate kangaroo.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/custom/spawn_kangaroo.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

import torch
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates how to simulate bipedal robots.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

##
# Pre-defined configs
##
from isaaclab_walker import P73_CFG  # isort: skip


def design_scene(sim: sim_utils.SimulationContext) -> tuple[list, torch.Tensor]:
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Define origins
    origins = torch.tensor([
        [0.0, 0.0, 0.0],
    ]).to(device=sim.device)

    # Robots
    p73 = Articulation(P73_CFG.replace(prim_path="/World/p73"))
    robots = [p73]

    return robots, origins


def run_simulator(sim: sim_utils.SimulationContext, robots: list[Articulation], origins: torch.Tensor):
    """Runs the simulation loop."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    # Simulate physics
    while simulation_app.is_running():
        # reset
        if count % 400 == 0:
            # reset counters
            sim_time = 0.0
            count = 0
            for index, robot in enumerate(robots):
                # reset dof state
                joint_pos, joint_vel = (
                    robot.data.default_joint_pos,
                    robot.data.default_joint_vel,
                )
                robot.write_joint_state_to_sim(joint_pos, joint_vel)
                root_state = robot.data.default_root_state.clone()
                root_state[:, :3] += origins[index]
                robot.write_root_pose_to_sim(root_state[:, :7])
                robot.write_root_velocity_to_sim(root_state[:, 7:])
                robot.reset()
            # reset command
            print(">>>>>>>> Reset!")
        # apply action to the robot
        for robot in robots:
            init_pose = torch.tensor([0.0, 0.0, 0.0, 0.36, -0.36, 0.0, 0.0, 0.0, 0.0, 
            0.785, -0.785, 0.3, -0.3, -0.325, 0.325, 0.325,-0.325, 0.0, 0.0, 0.0, 0.0,
            -1.27, 1.27, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).to(device=sim.device)
            joint_ids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8,
            10, 12, 13, 14, 16, 17, 20, 21, 23, 24, 
            39, 40, 41, 42, 43, 44, 45, 46, 47, 48]).to(device=sim.device)

            robot.set_joint_position_target(init_pose, joint_ids=joint_ids)

            robot.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        for robot in robots:
            robot.update(sim_dt)


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[3.0, 0.0, 2.25], target=[0.0, 0.0, 1.0])

    # design scene
    robots, origins = design_scene(sim)

    # Play the simulator
    sim.reset()

    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Run the simulator
    run_simulator(sim, robots, origins)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
