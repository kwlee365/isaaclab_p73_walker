from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.string as string_utils
import omni.log
import torch
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class LowerBodyActions(ActionTerm):
    """Action term that sets joint position targets for lower-body (RL-controlled) and upper-body (PD-held) joints.

    PD computation, torque clipping (including angle-dependent LUT), command delay,
    and motor strength randomization are all handled by the actuator model
    (DelayedPDActuatorLUT) configured on the robot asset.

    This term only:
      - Scales and clips the raw policy output (12D normalized delta)
      - Converts to desired positions: q_des = q_default + delta
      - Clamps to joint position limits
      - Sets position targets for the actuator to process
    """

    cfg: LowerBodyActionsCfg
    _asset: Articulation
    _scale: torch.Tensor | float
    _clip: torch.Tensor

    def __init__(self, cfg: LowerBodyActionsCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # Resolve lower body joints
        self._joint_ids = [self._asset.data.joint_names.index(name) for name in self.cfg.lower_joint_names]
        self._joint_names = list(self.cfg.lower_joint_names)

        # Resolve upper body joints
        self._upper_joint_ids = [self._asset.data.joint_names.index(name) for name in self.cfg.upper_joint_names]
        self._default_upper_joint_pos = self._asset.data.default_joint_pos[:, self._upper_joint_ids]

        # Joint position limits for lower body
        self._lower_joint_pos_limits = torch.tensor(self.cfg.joint_pos_limits, device=self.device)

        self._num_lower = len(self._joint_ids)
        self._num_upper = len(self._upper_joint_ids)
        self._num_joints = self._num_lower

        omni.log.info(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
            f" {self._joint_names} [{self._joint_ids}]"
        )

        # Create action tensors
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

        # Parse scale
        if isinstance(cfg.scale, (float, int)):
            self._scale = float(cfg.scale)
        elif isinstance(cfg.scale, dict):
            self._scale = torch.ones(self.num_envs, self.action_dim, device=self.device)
            index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.scale, self._joint_names)
            self._scale[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported scale type: {type(cfg.scale)}. Supported types are float and dict.")

        # Parse clip
        if self.cfg.clip is not None:
            if isinstance(cfg.clip, dict):
                self._clip = torch.tensor([[-float("inf"), float("inf")]], device=self.device).repeat(
                    self.num_envs, self.action_dim, 1
                )
                index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.clip, self._joint_names)
                self._clip[:, index_list] = torch.tensor(value_list, device=self.device)
            else:
                raise ValueError(f"Unsupported clip type: {type(cfg.clip)}. Supported types are dict.")

    @property
    def action_dim(self) -> int:
        return self._num_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        processed = actions * self._scale
        if self.cfg.clip is not None:
            processed = torch.clamp(processed, min=self._clip[:, :, 0], max=self._clip[:, :, 1])
        processed = processed.clamp(-1.0, 1.0)
        self._raw_actions[:] = processed
        self._processed_actions[:] = processed

    def apply_actions(self):
        # Lower body: desired position = default + delta
        lower_lim = self._lower_joint_pos_limits[:, 0].view(1, -1)
        upper_lim = self._lower_joint_pos_limits[:, 1].view(1, -1)

        q_default_lower = self._asset.data.default_joint_pos[:, self._joint_ids]
        q_des_lower = q_default_lower + self.processed_actions
        q_des_lower = torch.clamp(q_des_lower, min=lower_lim, max=upper_lim)

        # Upper body: hold at default pose
        q_des_upper = self._default_upper_joint_pos

        # Set position targets — the actuator (DelayedPDActuatorLUT) handles
        # PD computation, delay, LUT clipping, and motor strength randomization
        joint_ids_ordered = self._joint_ids + self._upper_joint_ids
        target_pos = torch.cat([q_des_lower, q_des_upper], dim=1)
        self._asset.set_joint_position_target(target_pos, joint_ids=joint_ids_ordered)


from dataclasses import MISSING
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass


@configclass
class LowerBodyActionsCfg(ActionTermCfg):
    class_type: type[ActionTerm] = LowerBodyActions
    lower_joint_names: list[str] = MISSING
    upper_joint_names: list[str] = MISSING
    scale: float | dict[str, float] = 1.0
    joint_pos_limits: list[tuple[float, float]] = MISSING
