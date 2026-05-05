# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# Copyright (c) 2025, Jaeyong Shin (jasonshin0537@snu.ac.kr).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from . import mdp
from .rough_env_cfg import P73RoughEnvCfg


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    falling = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.6, "asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class P73FlatEnvCfg(P73RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None
        # Height scan:
        # - We keep `height_scanner` enabled so `target`/`critic` can include height_scan (privileged obs).
        # - The policy observation group does NOT include height_scan by design (see ObservationsCfg.PolicyCfg).
        # On a plane terrain, the height scan will be near-constant; this still keeps obs dimensions consistent.
        self.scene.height_scanner.debug_vis = False

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (-0.6, 0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.4, 0.4)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.6, 0.6)

        # Terminations
        self.terminations = TerminationsCfg()

class P73FlatPlayEnvCfg(P73FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False

        # Viewport camera: track the robot's base_link and keep a diagonal 3/4 view.
        # This uses ViewerCfg from Isaac Lab (isaaclab.envs.common.ViewerCfg):
        #   cam_eye = origin(base_link) + eye, cam_target = origin(base_link) + lookat
        #self.viewer.origin_type = "asset_body"
        #self.viewer.asset_name = "robot"
        #self.viewer.body_name = "base_link"
        #self.viewer.env_index = 0
        # Diagonal (3/4) view offsets (in meters) in the chosen origin frame.
        #self.viewer.eye = (3.0, 6.0, 2.0)
        #self.viewer.lookat = (0.0, 0.0, 1.0)
        self.events.push_robot = None
                # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.0, 0.0)
