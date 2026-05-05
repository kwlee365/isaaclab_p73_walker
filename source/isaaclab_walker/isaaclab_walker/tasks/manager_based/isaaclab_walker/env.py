# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# Copyright (c) 2025, Jaeyong Shin (jasonshin0537@snu.ac.kr).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_walker.managers import P73ObservationManager

class P73ManagerBasedRLEnv(ManagerBasedRLEnv):
    """Custom RL environment for P73 that uses a custom ObservationManager."""

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        # Initialize the parent class
        super().__init__(cfg, render_mode, **kwargs)

    def load_managers(self):
        """Load managers and ensure EventManager terms are resolved before applying startup events.

        In IsaacLab standalone mode, the EventManager can be constructed before the simulation starts playing.
        Its class-based terms (e.g., `randomize_rigid_body_mass`) are only instantiated when the timeline enters PLAY.
        If the PLAY callback is missed (or startup events are applied before term resolution), calling the terms
        will attempt to call the class constructor with term params, causing errors like:
        `TypeError: randomize_rigid_body_mass.__init__() got an unexpected keyword argument 'asset_cfg'`.
        """
        # Force-resolve deferred term configs once the simulation is playing.
        # This makes sure class-based event terms are instantiated before `apply(mode="startup")`.
        if not getattr(self.event_manager, "_is_scene_entities_resolved", True) and self.sim.is_playing():
            self.event_manager._resolve_terms_callback(None)  # noqa: SLF001

        # Let the base class create all managers and apply startup events.
        super().load_managers()

        # Replace the standard ObservationManager with our custom one.
        if self.cfg.observations is not None:
            self.observation_manager = P73ObservationManager(self.cfg.observations, self)
            print("[INFO] P73ManagerBasedRLEnv: Replaced ObservationManager with P73ObservationManager")

