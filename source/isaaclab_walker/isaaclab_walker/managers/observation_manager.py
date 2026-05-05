# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# Copyright (c) 2025, Jaeyong Shin (jasonshin0537@snu.ac.kr).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import ObservationManager
import torch
from collections.abc import Sequence
from isaaclab.utils import class_to_dict, modifiers, noise
from isaaclab.utils.buffers import CircularBuffer
import numpy as np


class P73ObservationManager(ObservationManager):
    """Custom observation manager for P73 robot.
    
    This class inherits from the standard ObservationManager and can be customized
    to override default behaviors.
    """
    
    def __init__(self, cfg, env):
        # You can add custom initialization logic here
        super().__init__(cfg, env)
        # Example: print a message to verify usage
        print("[INFO] Using custom P73ObservationManager")


    def compute_group(self, group_name: str, update_history: bool = False) -> torch.Tensor | dict[str, torch.Tensor]:
        # check ig group name is valid
        if group_name not in self._group_obs_term_names:
            raise ValueError(
                f"Unable to find the group '{group_name}' in the observation manager."
                f" Available groups are: {list(self._group_obs_term_names.keys())}"
            )
        # iterate over all the terms in each group
        group_term_names = self._group_obs_term_names[group_name]
        # buffer to store obs per group
        group_obs = dict.fromkeys(group_term_names, None)
        # read attributes for each term
        obs_terms = zip(group_term_names, self._group_obs_term_cfgs[group_name])

        # evaluate terms: compute, add noise, clip, scale, custom modifiers
        for term_name, term_cfg in obs_terms:
            # compute term's value
            obs: torch.Tensor = term_cfg.func(self._env, **term_cfg.params).clone()
            # apply post-processing
            if term_cfg.modifiers is not None:
                for modifier in term_cfg.modifiers:
                    obs = modifier.func(obs, **modifier.params)
            if isinstance(term_cfg.noise, noise.NoiseCfg):
                obs = term_cfg.noise.func(obs, term_cfg.noise)
            elif isinstance(term_cfg.noise, noise.NoiseModelCfg) and term_cfg.noise.func is not None:
                obs = term_cfg.noise.func(obs)
            if term_cfg.clip:
                obs = obs.clip_(min=term_cfg.clip[0], max=term_cfg.clip[1])
            if term_cfg.scale is not None:
                obs = obs.mul_(term_cfg.scale)
            # Update the history buffer if observation term has history enabled
            if term_cfg.history_length > 0:
                circular_buffer = self._group_obs_term_history_buffer[group_name][term_name]
                if update_history:
                    circular_buffer.append(obs)
                elif circular_buffer._buffer is None:
                    # because circular buffer only exits after the simulation steps,
                    # this guards history buffer from corruption by external calls before simulation start
                    circular_buffer = CircularBuffer(
                        max_len=circular_buffer.max_length,
                        batch_size=circular_buffer.batch_size,
                        device=circular_buffer.device,
                    )
                    circular_buffer.append(obs)

                if term_cfg.flatten_history_dim:
                    # group_obs[term_name] = circular_buffer.buffer.reshape(self._env.num_envs, -1)
                    #### yongarry edit ####
                    ###### purpose: to make the history observation order with another way ######
                    group_obs[term_name] = circular_buffer.buffer
                else:
                    group_obs[term_name] = circular_buffer.buffer
            else:
                group_obs[term_name] = obs

        # concatenate all observations in the group together
        if self._group_obs_concatenate[group_name]:
            # set the concatenate dimension, account for the batch dimension if positive dimension is given
            # return torch.cat(list(group_obs.values()), dim=self._group_obs_concatenate_dim[group_name])
            #### yongarry edit ####
            ###### purpose: to make the history observation order with another way ######
            return torch.cat(list(group_obs.values()), dim=self._group_obs_concatenate_dim[group_name]).reshape(self._env.num_envs, -1)
        else:
            return group_obs

    def get_active_iterable_terms(self, env_idx: int) -> Sequence[tuple[str, Sequence[float]]]:
        """Returns the active terms as iterable sequence of tuples.

        The first element of the tuple is the name of the term and the second element is the raw value(s) of the term.

        Args:
            env_idx: The specific environment to pull the active terms from.

        Returns:
            The active terms.
        """
        terms = []

        if self._obs_buffer is None:
            self.compute()
        obs_buffer: dict[str, torch.Tensor | dict[str, torch.Tensor]] = self._obs_buffer

        for group_name, _ in self._group_obs_dim.items():
            if not self.group_obs_concatenate[group_name]:
                for name, term in obs_buffer[group_name].items():
                    terms.append((group_name + "-" + name, term[env_idx].cpu().tolist()))
                continue

            idx = 0
            # add info for each term
            data = obs_buffer[group_name]
            for name, shape in zip(
                self._group_obs_term_names[group_name],
                self._group_obs_term_dim[group_name],
            ):
                data_length = np.prod(shape)
                # term = data[env_idx, idx : idx + data_length]
                # yongarry edit: only vizualizing the current observation, not the history
                term = data[env_idx, idx : idx + (data_length // self._group_obs_term_history_buffer[group_name][name].max_length)]                
                terms.append((group_name + "-" + name, term.cpu().tolist()))
                idx += data_length // self._group_obs_term_history_buffer[group_name][name].max_length

        return terms