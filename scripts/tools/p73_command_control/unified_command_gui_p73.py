"""
P73 Unified Teleop GUI (IsaacLab 5.1)

- Base velocity command sliders (vx, vy, wz) via env.command_manager
- Disturbance/Push buttons (random/custom) via push_by_setting_velocity (env0 only)
- Foot external force impulse (random direction, magnitude + duration) via robot.set_external_force_and_torque (env0 only)

Notes
-----
This module is intentionally kept inside `isaaclab_walker/` (P73 repo) and should not modify the vanilla `IsaacLab/`.
We exclude EE pose control by design (per user request).
"""

from __future__ import annotations

from typing import Any, cast

import torch
import omni.ui as ui

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp.events import push_by_setting_velocity


class P73UnifiedCommandGUI:
    """Unified Omniverse UI to teleoperate P73 commands + disturbances.

    Features (env0 only for non-command effects):
    - Base velocity: writes directly to the command term buffer and resets time_left to prevent auto-resampling.
    - Push: calls `push_by_setting_velocity(env, env_ids=[0], velocity_range=...)`.
    - Foot external force: calls `robot.set_external_force_and_torque(...)` for foot links, auto-cleared after duration.
    """

    def __init__(
        self,
        env: ManagerBasedRLEnv,
        *,
        base_command_name: str = "base_velocity",
        window_name: str = "P73 Unified Command Control",
        range_scale: float = 1.0,
        use_env_cfg_ranges: bool = True,
        # Base Velocity defaults (fallback)
        lin_vel_x_range: tuple[float, float] = (-1.0, 1.0),
        lin_vel_y_range: tuple[float, float] = (-0.5, 0.5),
        ang_vel_z_range: tuple[float, float] = (-1.0, 1.0),
    ):
        self._env = env
        self._base_command_name = str(base_command_name)
        self._use_env_cfg_ranges = bool(use_env_cfg_ranges)

        # Range scaling (OOD testing). Expect positive scale.
        try:
            self._range_scale = float(range_scale)
        except Exception:
            self._range_scale = 1.0
        if self._range_scale <= 0.0:
            self._range_scale = abs(self._range_scale) if self._range_scale != 0.0 else 1.0

        self._lin_vel_x_range = lin_vel_x_range
        self._lin_vel_y_range = lin_vel_y_range
        self._ang_vel_z_range = ang_vel_z_range

        # Base velocity values
        self._lin_vel_x = 0.0
        self._lin_vel_y = 0.0
        self._ang_vel_z = 0.0

        # Push values (custom direct delta)
        self._push_x = 0.0
        self._push_y = 0.0
        self._push_yaw = 0.0

        # Foot external force (env0 impulse)
        self._foot_force_mag = 0.0  # N
        self._foot_force_duration_s = 0.2  # seconds
        self._pending_foot_force_clear_steps = 0
        self._cached_body_id_left_foot: int | None = None
        self._cached_body_id_right_foot: int | None = None

        if self._use_env_cfg_ranges:
            self._maybe_override_ranges_from_env()

        self._build_ui(window_name)

    # ---------------------------------------------------------------------
    # Range helpers
    # ---------------------------------------------------------------------

    def _scaled_range_zero_centered(self, r: tuple[float, float]) -> tuple[float, float]:
        a, b = float(r[0]), float(r[1])
        a_s, b_s = a * self._range_scale, b * self._range_scale
        return (min(a_s, b_s), max(a_s, b_s))

    def _normalize_range(self, r: tuple[float, float]) -> tuple[float, float]:
        a, b = float(r[0]), float(r[1])
        return (a, b) if a <= b else (b, a)

    def _term_cfg_ranges(self, term_name: str):
        """Return cfg.ranges object for a command term, if available."""
        try:
            command_manager = self._env.command_manager
            if term_name not in command_manager._terms:
                return None
            term = command_manager._terms[term_name]
            cfg = getattr(term, "cfg", None)
            if cfg is None:
                return None
            return getattr(cfg, "ranges", None)
        except Exception:
            return None

    def _maybe_override_ranges_from_env(self):
        base_ranges = self._term_cfg_ranges(self._base_command_name)
        if base_ranges is None:
            return
        if not all(hasattr(base_ranges, k) for k in ("lin_vel_x", "lin_vel_y", "ang_vel_z")):
            return

        rx = self._normalize_range(getattr(base_ranges, "lin_vel_x"))
        ry = self._normalize_range(getattr(base_ranges, "lin_vel_y"))
        rz = self._normalize_range(getattr(base_ranges, "ang_vel_z"))

        def expand_if_degenerate(r: tuple[float, float], fallback: tuple[float, float]):
            if abs(r[1] - r[0]) > 1e-6:
                return r
            v = float(r[0])
            if abs(v) > 1e-6:
                return (-abs(v), abs(v))
            return fallback

        rx = expand_if_degenerate(rx, self._lin_vel_x_range)
        ry = expand_if_degenerate(ry, self._lin_vel_y_range)
        rz = expand_if_degenerate(rz, self._ang_vel_z_range)

        self._lin_vel_x_range = self._scaled_range_zero_centered(rx)
        self._lin_vel_y_range = self._scaled_range_zero_centered(ry)
        self._ang_vel_z_range = self._scaled_range_zero_centered(rz)

    # ---------------------------------------------------------------------
    # Push helpers
    # ---------------------------------------------------------------------

    def _get_push_velocity_range_from_runtime_cfg(self) -> dict[str, tuple[float, float]] | None:
        """Best-effort: read velocity_range from env.event_manager.cfg.push_robot.params."""
        try:
            if not hasattr(self._env, "event_manager"):
                return None
            event_manager_any = cast(Any, getattr(self._env, "event_manager"))
            cfg = getattr(event_manager_any, "cfg", None)
            if cfg is None:
                return None

            term_cfg = None
            if isinstance(cfg, dict):
                term_cfg = cfg.get("push_robot", None)
            else:
                term_cfg = getattr(cfg, "push_robot", None)
            if term_cfg is None:
                return None

            params = getattr(term_cfg, "params", None)
            if not isinstance(params, dict):
                return None

            velocity_range = params.get("velocity_range", None)
            if not isinstance(velocity_range, dict):
                return None

            out: dict[str, tuple[float, float]] = {}
            for k, v in velocity_range.items():
                if not isinstance(k, str):
                    continue
                if isinstance(v, tuple) and len(v) == 2:
                    out[k] = (float(v[0]), float(v[1]))
            return out if len(out) > 0 else None
        except Exception:
            return None

    def _push_env0_once(self, velocity_range: dict[str, tuple[float, float]]):
        env_ids = torch.tensor([0], device=self._env.device, dtype=torch.long)
        push_by_setting_velocity(self._env, env_ids, velocity_range=velocity_range)

    # ---------------------------------------------------------------------
    # Foot external force helpers
    # ---------------------------------------------------------------------

    def _resolve_foot_body_id(self, side: str) -> int | None:
        """Resolve body id for a foot link from the robot articulation."""
        try:
            robot = self._env.scene["robot"]
            link_name = "L_Foot_Link" if side.lower().startswith("l") else "R_Foot_Link"
            body_ids, _ = robot.find_bodies([link_name], preserve_order=True)
            if len(body_ids) > 0:
                return int(body_ids[0])
            return None
        except Exception:
            return None

    def _apply_external_force_env0_one_step(self, body_id: int, force_vec_local: torch.Tensor):
        """Apply external force (local frame) to one body for env0, then auto-clear after duration."""
        robot = self._env.scene["robot"]
        env_ids = [0]
        body_ids = [body_id]
        forces = force_vec_local.view(1, 1, 3).to(device=self._env.device, dtype=torch.float32)
        torques = torch.zeros((1, 1, 3), device=self._env.device, dtype=torch.float32)
        robot.set_external_force_and_torque(forces=forces, torques=torques, body_ids=body_ids, env_ids=env_ids)

        # clear after duration to create an impulse-like effect
        dt = getattr(self._env, "step_dt", None)
        try:
            dt = float(dt) if dt is not None else 0.005
        except Exception:
            dt = 0.005
        dur_s = max(0.0, float(self._foot_force_duration_s))
        steps = int(round(dur_s / max(dt, 1e-6)))
        self._pending_foot_force_clear_steps = max(1, steps)

    def _clear_external_force_env0(self):
        try:
            robot = self._env.scene["robot"]
            env_ids = [0]
            for body_id in [self._cached_body_id_left_foot, self._cached_body_id_right_foot]:
                if body_id is None:
                    continue
                forces = torch.zeros((1, 1, 3), device=self._env.device, dtype=torch.float32)
                torques = torch.zeros((1, 1, 3), device=self._env.device, dtype=torch.float32)
                robot.set_external_force_and_torque(forces=forces, torques=torques, body_ids=[body_id], env_ids=env_ids)
        except Exception:
            pass

    def _sample_random_unit_vector_3d(self) -> torch.Tensor:
        v = torch.randn(3, device=self._env.device, dtype=torch.float32)
        n = torch.linalg.norm(v)
        if float(n) < 1e-6:
            return torch.tensor([1.0, 0.0, 0.0], device=self._env.device, dtype=torch.float32)
        return v / n

    # ---------------------------------------------------------------------
    # Public hook
    # ---------------------------------------------------------------------

    def post_step(self):
        """Call once per simulation step to implement timed foot-force clearing."""
        if self._pending_foot_force_clear_steps > 0:
            self._pending_foot_force_clear_steps -= 1
            if self._pending_foot_force_clear_steps == 0:
                self._clear_external_force_env0()

    # ---------------------------------------------------------------------
    # UI
    # ---------------------------------------------------------------------

    def _build_ui(self, window_name: str):
        # Destroy previous window instance to avoid stale UI state across restarts.
        try:
            if hasattr(self, "_window") and self._window:
                self._window.destroy()
        except Exception:
            pass

        self._window = ui.Window(
            window_name,
            width=450,
            height=660,
            visible=True,
            dockPreference=ui.DockPreference.RIGHT_BOTTOM,
        )

        with self._window.frame:
            with ui.VStack(spacing=6, height=0):
                # =============================================================
                # SECTION 1: BASE VELOCITY CONTROL
                # =============================================================
                ui.Label(
                    "Base Velocity Control",
                    alignment=ui.Alignment.CENTER,
                    style={"font_size": 18, "color": 0xFF0088FF},
                )
                ui.Line(style={"color": 0xFF444444}, height=2)
                ui.Spacer(height=4)

                # Lin Vel X
                with ui.HStack(spacing=5):
                    ui.Label("Lin Vel X (m/s):", width=140)
                    self._lin_vel_x_label = ui.Label(f"{self._lin_vel_x:.3f}", width=70, alignment=ui.Alignment.RIGHT)
                with ui.HStack(spacing=5):
                    ui.Label("←Back", width=55)
                    self._lin_vel_x_slider = ui.FloatSlider(
                        min=self._lin_vel_x_range[0], max=self._lin_vel_x_range[1], step=0.01, height=18
                    )
                    ui.Label("Fwd→", width=55, alignment=ui.Alignment.RIGHT)
                self._lin_vel_x_slider.model.set_value(self._lin_vel_x)
                self._lin_vel_x_slider.model.add_value_changed_fn(
                    lambda m: self._on_lin_vel_x_changed(m.get_value_as_float())
                )

                # Lin Vel Y
                with ui.HStack(spacing=5):
                    ui.Label("Lin Vel Y (m/s):", width=140)
                    self._lin_vel_y_label = ui.Label(f"{self._lin_vel_y:.3f}", width=70, alignment=ui.Alignment.RIGHT)
                with ui.HStack(spacing=5):
                    ui.Label("←Left", width=55)
                    self._lin_vel_y_slider = ui.FloatSlider(
                        min=self._lin_vel_y_range[0], max=self._lin_vel_y_range[1], step=0.01, height=18
                    )
                    ui.Label("Right→", width=55, alignment=ui.Alignment.RIGHT)
                self._lin_vel_y_slider.model.set_value(self._lin_vel_y)
                self._lin_vel_y_slider.model.add_value_changed_fn(
                    lambda m: self._on_lin_vel_y_changed(m.get_value_as_float())
                )

                # Ang Vel Z
                with ui.HStack(spacing=5):
                    ui.Label("Ang Vel Z (rad/s):", width=140)
                    self._ang_vel_z_label = ui.Label(f"{self._ang_vel_z:.3f}", width=70, alignment=ui.Alignment.RIGHT)
                with ui.HStack(spacing=5):
                    ui.Label("↶CCW", width=55)
                    self._ang_vel_z_slider = ui.FloatSlider(
                        min=self._ang_vel_z_range[0], max=self._ang_vel_z_range[1], step=0.01, height=18
                    )
                    ui.Label("CW↷", width=55, alignment=ui.Alignment.RIGHT)
                self._ang_vel_z_slider.model.set_value(self._ang_vel_z)
                self._ang_vel_z_slider.model.add_value_changed_fn(
                    lambda m: self._on_ang_vel_z_changed(m.get_value_as_float())
                )

                with ui.HStack(spacing=10):
                    ui.Button("Reset", clicked_fn=self._on_reset_base, height=28)
                    ui.Button("STOP", clicked_fn=self._on_emergency_stop, height=28)

                self._base_status_label = ui.Label(
                    f"X={self._lin_vel_x:.2f}, Y={self._lin_vel_y:.2f}, Z={self._ang_vel_z:.2f}",
                    alignment=ui.Alignment.CENTER,
                )

                ui.Spacer(height=6)
                ui.Line(style={"color": 0xFF444444}, height=2)
                ui.Spacer(height=4)

                # =============================================================
                # SECTION 2: DISTURBANCE / PUSH (env0 only)
                # =============================================================
                ui.Label(
                    "Disturbance / Push (env0)",
                    alignment=ui.Alignment.CENTER,
                    style={"font_size": 16, "color": 0xFFFF7777},
                )
                ui.Spacer(height=2)

                self._push_status_label = ui.Label("Last push: N/A", alignment=ui.Alignment.CENTER)

                with ui.HStack(spacing=10):
                    ui.Button(
                        "Random Push",
                        clicked_fn=self._on_random_push,
                        height=28,
                        style={"Button": {"background_color": 0xFFAA4444}},
                    )

                ui.Spacer(height=6)
                ui.Label(
                    "Custom Push (direct delta on root vel)",
                    alignment=ui.Alignment.CENTER,
                    style={"font_size": 12, "color": 0xFFCCCCCC},
                )

                # Push X
                with ui.HStack(spacing=5):
                    ui.Label("Push X (m/s):", width=140)
                    self._push_x_label = ui.Label(f"{self._push_x:.3f}", width=70, alignment=ui.Alignment.RIGHT)
                with ui.HStack(spacing=5):
                    self._push_x_slider = ui.FloatSlider(min=-0.8, max=0.8, step=0.01, height=18)
                self._push_x_slider.model.set_value(self._push_x)
                self._push_x_slider.model.add_value_changed_fn(lambda m: self._on_push_x_changed(m.get_value_as_float()))

                # Push Y
                with ui.HStack(spacing=5):
                    ui.Label("Push Y (m/s):", width=140)
                    self._push_y_label = ui.Label(f"{self._push_y:.3f}", width=70, alignment=ui.Alignment.RIGHT)
                with ui.HStack(spacing=5):
                    self._push_y_slider = ui.FloatSlider(min=-0.8, max=0.8, step=0.01, height=18)
                self._push_y_slider.model.set_value(self._push_y)
                self._push_y_slider.model.add_value_changed_fn(lambda m: self._on_push_y_changed(m.get_value_as_float()))

                # Push Yaw
                with ui.HStack(spacing=5):
                    ui.Label("Push Yaw (rad/s):", width=140)
                    self._push_yaw_label = ui.Label(f"{self._push_yaw:.3f}", width=70, alignment=ui.Alignment.RIGHT)
                with ui.HStack(spacing=5):
                    self._push_yaw_slider = ui.FloatSlider(min=-0.2, max=0.2, step=0.01, height=18)
                self._push_yaw_slider.model.set_value(self._push_yaw)
                self._push_yaw_slider.model.add_value_changed_fn(
                    lambda m: self._on_push_yaw_changed(m.get_value_as_float())
                )

                with ui.HStack(spacing=10):
                    ui.Button(
                        "Apply Custom Push",
                        clicked_fn=self._on_custom_push,
                        height=28,
                        style={"Button": {"background_color": 0xFFCC5555}},
                    )

                ui.Spacer(height=6)
                ui.Line(style={"color": 0xFF444444}, height=2)
                ui.Spacer(height=4)

                # =============================================================
                # SECTION 3: FOOT EXTERNAL FORCE (env0 only)
                # =============================================================
                ui.Label(
                    "Foot External Force (env0, impulse)",
                    alignment=ui.Alignment.CENTER,
                    style={"font_size": 16, "color": 0xFF66CCFF},
                )
                ui.Spacer(height=2)

                self._foot_force_status_label = ui.Label("Last foot force: N/A", alignment=ui.Alignment.CENTER)

                # Force magnitude
                with ui.HStack(spacing=5):
                    ui.Label("Force Mag (N):", width=140)
                    self._foot_force_mag_label = ui.Label(
                        f"{self._foot_force_mag:.1f}", width=70, alignment=ui.Alignment.RIGHT
                    )
                with ui.HStack(spacing=5):
                    ui.Label("0", width=30)
                    self._foot_force_mag_slider = ui.FloatSlider(min=0.0, max=800.0, step=1.0, height=18)
                    ui.Label("800", width=40, alignment=ui.Alignment.RIGHT)
                self._foot_force_mag_slider.model.set_value(self._foot_force_mag)
                self._foot_force_mag_slider.model.add_value_changed_fn(
                    lambda m: self._on_foot_force_mag_changed(m.get_value_as_float())
                )

                # Force duration
                with ui.HStack(spacing=5):
                    ui.Label("Duration (s):", width=140)
                    self._foot_force_dur_label = ui.Label(
                        f"{self._foot_force_duration_s:.2f}", width=70, alignment=ui.Alignment.RIGHT
                    )
                with ui.HStack(spacing=5):
                    ui.Label("0.01", width=40)
                    self._foot_force_dur_slider = ui.FloatSlider(min=0.01, max=1.00, step=0.01, height=18)
                    ui.Label("1.00", width=40, alignment=ui.Alignment.RIGHT)
                self._foot_force_dur_slider.model.set_value(self._foot_force_duration_s)
                self._foot_force_dur_slider.model.add_value_changed_fn(
                    lambda m: self._on_foot_force_duration_changed(m.get_value_as_float())
                )

                with ui.HStack(spacing=10):
                    ui.Button(
                        "Left Foot Force",
                        clicked_fn=self._on_left_foot_force,
                        height=28,
                        style={"Button": {"background_color": 0xFF336699}},
                    )
                    ui.Button(
                        "Right Foot Force",
                        clicked_fn=self._on_right_foot_force,
                        height=28,
                        style={"Button": {"background_color": 0xFF336699}},
                    )

                ui.Spacer(height=6)
                ui.Line(style={"color": 0xFF444444}, height=2)
                ui.Spacer(height=4)

                # =============================================================
                # SECTION 4: POSE CAPTURE
                # =============================================================
                ui.Label(
                    "Pose Capture (env0)",
                    alignment=ui.Alignment.CENTER,
                    style={"font_size": 16, "color": 0xFF88FF88},
                )
                ui.Spacer(height=2)
                ui.Button(
                    "Capture Pose",
                    clicked_fn=self._on_capture_pose,
                    height=32,
                    style={"Button": {"background_color": 0xFF228822}},
                )

    # ---------------------------------------------------------------------
    # Base velocity callbacks / updates
    # ---------------------------------------------------------------------

    def _on_lin_vel_x_changed(self, value: float):
        self._lin_vel_x = float(value)
        self._lin_vel_x_label.text = f"{self._lin_vel_x:.3f}"
        self._update_base_command()

    def _on_lin_vel_y_changed(self, value: float):
        self._lin_vel_y = float(value)
        self._lin_vel_y_label.text = f"{self._lin_vel_y:.3f}"
        self._update_base_command()

    def _on_ang_vel_z_changed(self, value: float):
        self._ang_vel_z = float(value)
        self._ang_vel_z_label.text = f"{self._ang_vel_z:.3f}"
        self._update_base_command()

    def _on_reset_base(self):
        self._lin_vel_x = 0.0
        self._lin_vel_y = 0.0
        self._ang_vel_z = 0.0
        self._lin_vel_x_slider.model.set_value(0.0)
        self._lin_vel_y_slider.model.set_value(0.0)
        self._ang_vel_z_slider.model.set_value(0.0)
        self._lin_vel_x_label.text = "0.000"
        self._lin_vel_y_label.text = "0.000"
        self._ang_vel_z_label.text = "0.000"
        self._update_base_command()

    def _on_emergency_stop(self):
        self._on_reset_base()
        self._push_status_label.text = "Last push: STOP"
        self._foot_force_status_label.text = "Last foot force: STOP"
        self._clear_external_force_env0()
        self._pending_foot_force_clear_steps = 0

    def _update_base_command(self):
        try:
            command_manager = self._env.command_manager
            if self._base_command_name not in command_manager._terms:
                self._base_status_label.text = f"Base Error: term '{self._base_command_name}' not found"
                return
            command_term_any = cast(Any, command_manager._terms[self._base_command_name])

            if hasattr(command_term_any, "command"):
                command_term_any.command[:, 0] = self._lin_vel_x
                command_term_any.command[:, 1] = self._lin_vel_y
                command_term_any.command[:, 2] = self._ang_vel_z

            # Prevent auto-resampling from overwriting user inputs
            if hasattr(command_term_any, "time_left"):
                cfg_any = getattr(command_term_any, "cfg", None)
                resampling_time_range = getattr(cfg_any, "resampling_time_range", (0.0, 1e6))
                try:
                    max_time = float(resampling_time_range[1])
                except Exception:
                    max_time = 1e6
                command_term_any.time_left[:] = max_time

            self._base_status_label.text = f"X={self._lin_vel_x:.2f}, Y={self._lin_vel_y:.2f}, Z={self._ang_vel_z:.2f}"
        except Exception as e:
            self._base_status_label.text = f"Base Error: {str(e)}"

    # ---------------------------------------------------------------------
    # Push callbacks
    # ---------------------------------------------------------------------

    def _on_push_x_changed(self, value: float):
        self._push_x = float(value)
        self._push_x_label.text = f"{self._push_x:.3f}"

    def _on_push_y_changed(self, value: float):
        self._push_y = float(value)
        self._push_y_label.text = f"{self._push_y:.3f}"

    def _on_push_yaw_changed(self, value: float):
        self._push_yaw = float(value)
        self._push_yaw_label.text = f"{self._push_yaw:.3f}"

    def _on_random_push(self):
        try:
            velocity_range = self._get_push_velocity_range_from_runtime_cfg()
            if velocity_range is None:
                velocity_range = {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-0.2, 0.2)}
            self._push_env0_once(velocity_range)
            self._push_status_label.text = (
                f"Last push: Random (x={velocity_range.get('x')}, y={velocity_range.get('y')}, yaw={velocity_range.get('yaw')})"
            )
            print(f"[P73UnifiedCommandGUI] Random push applied to env0: {velocity_range}", flush=True)
        except Exception as e:
            self._push_status_label.text = f"Push error: {str(e)}"

    def _on_custom_push(self):
        try:
            velocity_range = {"x": (self._push_x, self._push_x), "y": (self._push_y, self._push_y), "yaw": (self._push_yaw, self._push_yaw)}
            self._push_env0_once(velocity_range)
            self._push_status_label.text = f"Last push: Custom (x={self._push_x:+.2f}, y={self._push_y:+.2f}, yaw={self._push_yaw:+.2f})"
            print(f"[P73UnifiedCommandGUI] Custom push applied to env0: {velocity_range}", flush=True)
        except Exception as e:
            self._push_status_label.text = f"Push error: {str(e)}"

    # ---------------------------------------------------------------------
    # Foot force callbacks
    # ---------------------------------------------------------------------

    def _on_foot_force_mag_changed(self, value: float):
        self._foot_force_mag = float(value)
        self._foot_force_mag_label.text = f"{self._foot_force_mag:.1f}"

    def _on_foot_force_duration_changed(self, value: float):
        self._foot_force_duration_s = float(value)
        self._foot_force_dur_label.text = f"{self._foot_force_duration_s:.2f}"

    def _on_left_foot_force(self):
        try:
            if self._cached_body_id_left_foot is None:
                self._cached_body_id_left_foot = self._resolve_foot_body_id("left")
            if self._cached_body_id_left_foot is None:
                self._foot_force_status_label.text = "Left foot body not found (expected: L_Foot_Link)"
                return
            force_vec = self._foot_force_mag * self._sample_random_unit_vector_3d()
            self._apply_external_force_env0_one_step(self._cached_body_id_left_foot, force_vec)
            self._foot_force_status_label.text = (
                f"Last foot force: LEFT |mag|={self._foot_force_mag:.1f} N, dur={self._foot_force_duration_s:.2f}s"
            )
            print(f"[P73UnifiedCommandGUI] Left foot external force (env0): {force_vec.detach().cpu().numpy()}", flush=True)
        except Exception as e:
            self._foot_force_status_label.text = f"Foot force error: {str(e)}"

    def _on_right_foot_force(self):
        try:
            if self._cached_body_id_right_foot is None:
                self._cached_body_id_right_foot = self._resolve_foot_body_id("right")
            if self._cached_body_id_right_foot is None:
                self._foot_force_status_label.text = "Right foot body not found (expected: R_Foot_Link)"
                return
            force_vec = self._foot_force_mag * self._sample_random_unit_vector_3d()
            self._apply_external_force_env0_one_step(self._cached_body_id_right_foot, force_vec)
            self._foot_force_status_label.text = (
                f"Last foot force: RIGHT |mag|={self._foot_force_mag:.1f} N, dur={self._foot_force_duration_s:.2f}s"
            )
            print(f"[P73UnifiedCommandGUI] Right foot external force (env0): {force_vec.detach().cpu().numpy()}", flush=True)
        except Exception as e:
            self._foot_force_status_label.text = f"Foot force error: {str(e)}"

    # ---------------------------------------------------------------------
    # Pose capture callback
    # ---------------------------------------------------------------------

    _CAPTURE_JOINT_NAMES = [
        "L_HipRoll_Joint", "L_HipPitch_Joint", "L_HipYaw_Joint",
        "L_Knee_Joint", "L_AnklePitch_Joint", "L_AnkleRoll_Joint",
        "R_HipRoll_Joint", "R_HipPitch_Joint", "R_HipYaw_Joint",
        "R_Knee_Joint", "R_AnklePitch_Joint", "R_AnkleRoll_Joint",
        "WaistYaw_Joint",
    ]

    def _on_capture_pose(self):
        try:
            robot = self._env.scene["robot"]
            eid = 0
            h = float(robot.data.root_pos_w[eid, 2].cpu())
            jp = robot.data.joint_pos[eid].cpu()
            dp = robot.data.default_joint_pos[eid].cpu()

            cmd = None
            try:
                c = self._env.command_manager.get_command(self._base_command_name)
                cmd = c[eid].cpu()
            except Exception:
                pass

            if not hasattr(self, "_capture_count"):
                self._capture_count = 0
            self._capture_count += 1

            print(f"\n{'='*60}", flush=True)
            print(f"  POSE CAPTURE #{self._capture_count}", flush=True)
            print(f"{'='*60}", flush=True)
            print(f"  base_height: {h:.4f} m", flush=True)
            if cmd is not None:
                print(f"  cmd_vel: vx={cmd[0]:.3f}  vy={cmd[1]:.3f}  wz={cmd[2]:.3f}", flush=True)

            print(f"\n  {'Joint':<25s} {'Abs':>8s}  {'Rel':>8s}", flush=True)
            print(f"  {'-'*25} {'-'*8}  {'-'*8}", flush=True)

            jmap = {}
            for n in self._CAPTURE_JOINT_NAMES:
                try:
                    jmap[n] = robot.data.joint_names.index(n)
                except ValueError:
                    pass

            for name, idx in jmap.items():
                a = float(jp[idx])
                r = float(jp[idx] - dp[idx])
                print(f"  {name:<25s} {a:>8.4f}  {r:>8.4f}", flush=True)

            print(f"\n  # Copy-paste for standing_pose:", flush=True)
            print(f'  "standing_pose": {{', flush=True)
            items = list(jmap.items())
            for i, (name, idx) in enumerate(items):
                a = round(float(jp[idx]), 4)
                comma = "," if i < len(items) - 1 else ""
                print(f'      "{name}": {a}{comma}', flush=True)
            print(f"  }}", flush=True)
            print(f'  # target_height: {h:.4f}', flush=True)
            print(f"{'='*60}\n", flush=True)
        except Exception as e:
            print(f"[PoseCapture] Error: {e}", flush=True)

