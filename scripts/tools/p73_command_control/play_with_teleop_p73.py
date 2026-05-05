"""
P73 policy playback with interactive base-velocity GUI + METRIC_DATA streaming.

This file lives inside `isaaclab_walker` to keep all P73-specific tools out of the vanilla `IsaacLab/` repo.

Usage (example):
  # In conda env: p73
  cd /home/piene/p73/isaaclab_walker && \\
  TERM=xterm-256color OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 \\
  python scripts/tools/p73_command_control/play_with_teleop_p73.py \\
    --task P73-Flat-Play \\
    --checkpoint logs/rsl_rl/p73_flat/2026-02-11_10-29-56/model_9000.pt \\
    --num_envs 1 --control_mode gui \\
    2>&1 | python tools/plot_contact_force_loco.py
"""

from __future__ import annotations

# Launch Isaac Sim Simulator first.

import argparse
import os
import sys
import time
from typing import Any, cast

from isaaclab.app import AppLauncher


# -----------------------------------------------------------------------------
# CLI args
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Play P73 RL agent with base-velocity GUI + plotting.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of recorded video (in steps).")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Gym task id (e.g., P73-Flat-Play).")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time.")

# Interactive control args (GUI only; keep minimal)
parser.add_argument(
    "--control_mode",
    type=str,
    default="none",
    choices=["none", "gui"],
    help="Interactive control mode: 'none' or 'gui'",
)
parser.add_argument(
    "--teleop_range_scale",
    type=float,
    default=1.0,
    help="Scale factor applied to base_velocity cfg ranges for teleop GUI sliders.",
)
parser.add_argument(
    "--disable_auto_push",
    action="store_true",
    default=False,
    help=(
        "Disable interval-based push_robot event in the env cfg (recommended for teleop). "
        "Manual push via GUI still works."
    ),
)

# Import IsaacLab play cli args (same module used by scripts/reinforcement_learning/rsl_rl/play.py)
_this_dir = os.path.dirname(__file__)
_rslrl_cli_dir = os.path.abspath(os.path.join(_this_dir, "..", "..", "rsl_rl"))
if _rslrl_cli_dir not in sys.path:
    sys.path.append(_rslrl_cli_dir)
import cli_args  # isort: skip  # noqa: E402

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

# Clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# Launch app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# Rest everything follows.

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import isaaclab_walker.tasks  # noqa: F401, E402

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg  # noqa: E402
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint  # noqa: E402

from rsl_rl.runners import OnPolicyRunner  # noqa: E402

try:
    from isaaclab_walker.algorithms.rsl_rl import P73OnPolicyRunner as OnPolicyRunner  # type: ignore

    print("[INFO] Using custom P73OnPolicyRunner")
except Exception:
    pass

from unified_command_gui_p73 import P73UnifiedCommandGUI  # noqa: E402


class BaseVelocityGUI:
    """Minimal Omniverse UI to teleoperate base velocity command (vx, vy, wz)."""

    def __init__(
        self,
        env,
        base_command_name: str = "base_velocity",
        window_name: str = "P73 Base Velocity Control",
        range_scale: float = 1.0,
        use_env_cfg_ranges: bool = True,
        lin_vel_x_range: tuple[float, float] = (-1.0, 1.0),
        lin_vel_y_range: tuple[float, float] = (-0.5, 0.5),
        ang_vel_z_range: tuple[float, float] = (-1.0, 1.0),
    ):
        import omni.ui as ui  # local import (requires Isaac Sim)

        self._ui = ui
        self._env = env
        self._base_command_name = base_command_name
        self._use_env_cfg_ranges = bool(use_env_cfg_ranges)

        try:
            self._range_scale = float(range_scale)
        except Exception:
            self._range_scale = 1.0
        if self._range_scale <= 0.0:
            self._range_scale = abs(self._range_scale) if self._range_scale != 0.0 else 1.0

        self._lin_vel_x_range = lin_vel_x_range
        self._lin_vel_y_range = lin_vel_y_range
        self._ang_vel_z_range = ang_vel_z_range

        self._lin_vel_x = 0.0
        self._lin_vel_y = 0.0
        self._ang_vel_z = 0.0

        if self._use_env_cfg_ranges:
            self._maybe_override_ranges_from_env()

        self._build_ui(window_name)

    def _scaled_range_zero_centered(self, r: tuple[float, float]) -> tuple[float, float]:
        a, b = float(r[0]), float(r[1])
        a_s, b_s = a * self._range_scale, b * self._range_scale
        return (min(a_s, b_s), max(a_s, b_s))

    def _normalize_range(self, r: tuple[float, float]) -> tuple[float, float]:
        a, b = float(r[0]), float(r[1])
        return (a, b) if a <= b else (b, a)

    def _term_cfg_ranges(self, term_name: str):
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

    def _build_ui(self, window_name: str):
        ui = self._ui
        self._window = ui.Window(
            window_name,
            width=420,
            height=360,
            visible=True,
            dockPreference=ui.DockPreference.RIGHT_BOTTOM,
        )

        with self._window.frame:
            with ui.VStack(spacing=6, height=0):
                ui.Label("Base Velocity Control", alignment=ui.Alignment.CENTER, style={"font_size": 18})

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
                    ui.Button("Reset", clicked_fn=self._on_reset, height=28)
                    ui.Button("STOP", clicked_fn=self._on_stop, height=28)

                self._status_label = ui.Label(
                    f"X={self._lin_vel_x:.2f}, Y={self._lin_vel_y:.2f}, Z={self._ang_vel_z:.2f}",
                    alignment=ui.Alignment.CENTER,
                )

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

    def _on_reset(self):
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

    def _on_stop(self):
        self._on_reset()
        print("[BaseVelocityGUI] 🛑 STOP")

    def _update_base_command(self):
        try:
            command_manager = self._env.command_manager
            if self._base_command_name not in command_manager._terms:
                return
            command_term_any = cast(Any, command_manager._terms[self._base_command_name])
            if hasattr(command_term_any, "command"):
                command_term_any.command[:, 0] = self._lin_vel_x
                command_term_any.command[:, 1] = self._lin_vel_y
                command_term_any.command[:, 2] = self._ang_vel_z
            if hasattr(command_term_any, "time_left"):
                cfg_any = getattr(command_term_any, "cfg", None)
                resampling_time_range = getattr(cfg_any, "resampling_time_range", (0.0, 1e6))
                try:
                    max_time = float(resampling_time_range[1])
                except Exception:
                    max_time = 1e6
                command_term_any.time_left[:] = max_time
            self._status_label.text = f"X={self._lin_vel_x:.2f}, Y={self._lin_vel_y:.2f}, Z={self._ang_vel_z:.2f}"
        except Exception as e:
            self._status_label.text = f"Error: {str(e)}"


class _FullPipelineExporter(torch.nn.Module):
    """Wraps the full ActorCriticAdaptationFuture inference pipeline for export.

    Input:  history_obs (1, num_actor_obs)  — e.g. (1, 470) = 47D × 10 frames
    Output: actions     (1, num_actions)    — e.g. (1, 12)

    Internally runs: normalizer → encoder → extract current_obs → concat → actor
    This matches ActorCriticAdaptationFuture.act_inference() exactly.
    """

    def __init__(self, policy_nn):
        super().__init__()
        import copy

        self.normalizer = copy.deepcopy(policy_nn.actor_obs_normalizer)
        self.encoder = copy.deepcopy(policy_nn.encoder)
        self.actor = copy.deepcopy(policy_nn.actor)
        self.num_single_obs = policy_nn.num_single_obs

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = self.normalizer(obs)
        latent = self.encoder(obs)
        current_obs = obs[:, -self.num_single_obs :]
        actor_input = torch.cat((current_obs, latent), dim=-1)
        return self.actor(actor_input)


def _export_full_pipeline_onnx(policy_nn, export_dir: str):
    """Export ActorCriticAdaptationFuture as ONNX with full pipeline (history_obs → actions)."""
    os.makedirs(export_dir, exist_ok=True)
    exporter = _FullPipelineExporter(policy_nn)
    exporter.to("cpu")
    exporter.eval()

    dummy_obs = torch.zeros(1, policy_nn.num_actor_obs)
    onnx_path = os.path.join(export_dir, "policy.onnx")
    torch.onnx.export(
        exporter,
        dummy_obs,
        onnx_path,
        export_params=True,
        opset_version=18,
        verbose=False,
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={},
    )
    print(f"[INFO] Exported full-pipeline ONNX: input=({1}, {policy_nn.num_actor_obs}) → output=({1}, {policy_nn.num_actions})")


def _export_full_pipeline_jit(policy_nn, export_dir: str):
    """Export ActorCriticAdaptationFuture as TorchScript JIT with full pipeline."""
    os.makedirs(export_dir, exist_ok=True)
    exporter = _FullPipelineExporter(policy_nn)
    exporter.to("cpu")
    exporter.eval()

    jit_path = os.path.join(export_dir, "policy.pt")
    scripted = torch.jit.script(exporter)
    scripted.save(jit_path)
    print(f"[INFO] Exported full-pipeline JIT: input=({1}, {policy_nn.num_actor_obs}) → output=({1}, {policy_nn.num_actions})")


def main():
    task_name = args_cli.task.split(":")[-1]

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    agent_cfg = cli_args.parse_rsl_rl_cfg(task_name, args_cli)

    # interactive play: disable timeout + freeze auto-resampling
    try:
        if hasattr(env_cfg, "terminations") and hasattr(env_cfg.terminations, "time_out"):
            env_cfg.terminations.time_out = None
    except Exception:
        pass
    try:
        if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "base_velocity"):
            env_cfg.commands.base_velocity.resampling_time_range = (1000.0, 1000.0)
            # IMPORTANT: IsaacLab default locomotion cfg uses heading_command=True, which overwrites command[:, 2]
            # (wz) each step from heading_target. For teleop yaw-rate control, force heading mode off.
            env_cfg.commands.base_velocity.heading_command = False
            env_cfg.commands.base_velocity.rel_heading_envs = 0.0
            env_cfg.commands.base_velocity.rel_standing_envs = 0.0
            print("[PLAY] base_velocity.heading_command forced to False (teleop wz enabled).", flush=True)
    except Exception:
        pass
    # optional: disable interval push (teleop-friendly)
    if bool(getattr(args_cli, "disable_auto_push", False)):
        try:
            if hasattr(env_cfg, "events") and hasattr(env_cfg.events, "push_robot") and env_cfg.events.push_robot is not None:
                # make it effectively never trigger
                env_cfg.events.push_robot.interval_range_s = (1.0e9, 1.0e9)
                print("[PLAY] Auto push_robot event disabled (interval_range_s -> 1e9s).", flush=True)
        except Exception as e:
            print(f"[PLAY] Warning: failed to disable auto push_robot: {e}", flush=True)

    # checkpoint path
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # env
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    actual_env = env.unwrapped
    print("[PLAY] Resetting env once to initialize managers...", flush=True)
    actual_env.reset()

    gui = None
    if args_cli.control_mode == "gui":
        gui = P73UnifiedCommandGUI(
            env=actual_env,
            base_command_name="base_velocity",
            window_name=f"P73 Command Control - {task_name}",
            range_scale=float(args_cli.teleop_range_scale),
            use_env_cfg_ranges=True,
        )

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during playback.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=actual_env.device)

    # --- Export policy as ONNX / JIT ---
    try:
        policy_nn = ppo_runner.alg.policy
    except AttributeError:
        policy_nn = ppo_runner.alg.actor_critic

    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

    # Check if this is an ActorCriticAdaptationFuture model (encoder-based).
    # The standard exporter only exports policy.actor, which expects
    # (num_single_obs + latent_dim) input. For deployment, we need the full
    # pipeline: history_obs(470) -> normalizer -> encoder -> latent -> actor -> actions.
    from isaaclab_walker.algorithms.rsl_rl import ActorCriticAdaptationFuture

    if isinstance(policy_nn, ActorCriticAdaptationFuture):
        _export_full_pipeline_onnx(policy_nn, export_model_dir)
        _export_full_pipeline_jit(policy_nn, export_model_dir)
    else:
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")
    print(f"[INFO] Exported policy to: {export_model_dir}")

    enable_plotting = False
    try:
        enable_plotting = int(env.unwrapped.num_envs) == 1
    except Exception:
        enable_plotting = False

    from plotting_utils_p73 import extract_and_print_plotting_data_p73  # noqa: WPS433

    dt = env.unwrapped.step_dt
    obs, _ = env.reset()
    timestep = 0

    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, rewards, terminated, info = env.step(actions)
            if enable_plotting:
                extract_and_print_plotting_data_p73(env=env, timestep=timestep, enable_plotting=True)
            if gui is not None:
                try:
                    gui.post_step()
                except Exception:
                    pass
        timestep += 1
        if args_cli.video and timestep >= int(args_cli.video_length):
            break
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

