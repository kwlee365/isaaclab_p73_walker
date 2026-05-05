# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to plot real-time loco-manipulation-style data from standard input.

This is copied into the IsaacLab 5.1.0 repo (P73) to keep the toolchain independent
from other IsaacLab installs.

The script expects a stream of JSON objects from stdin with the following structure:
{
    "root_base_vel": {   # RECOMMENDED (explicit root/base velocity in body frame)
        "cmd":  [x, y, z],
        "true": [x, y, z]
    },
    "head_vel": {        # RECOMMENDED (head-centric vel3 in world; z may be yaw_dot)
        "true": [x, y, z],
        "est":  [x, y, z]
    },
    "base_vel": {
        "cmd": [x, y, z],
        "true": [x, y, z],
        "est": [x, y, z]
    },
    "foot_pos_base": {   # OPTIONAL: foot positions expressed in base gravity-aligned frame
        "left":  [x, y, z],
        "right": [x, y, z]
    },
    "foot_force": {
        "left": {
            "true": [x, y, z]
        },
        "right": {
            "true": [x, y, z]
        }
    }
}

Usage:
    ./isaaclab.sh -p <play_script.py> ... 2>&1 | python tools/plot_contact_force_loco.py
"""

import argparse
import json
import sys
import threading
import queue
import re
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque


class RealTimePlotter:
    def __init__(self, buffer_size=200, ylim_pos=None, ylim_vel=None, ylim_force=None, ylim_contact=None):
        self.buffer_size = buffer_size
        self.ylim_pos = ylim_pos
        self.ylim_vel = ylim_vel
        self.ylim_force = ylim_force
        self.ylim_contact = ylim_contact

        # Data buffers
        self.steps = deque(maxlen=buffer_size)
        self.data_store = {
            "head_vel": {
                "x": {"cmd": deque(maxlen=buffer_size), "true": deque(maxlen=buffer_size), "est": deque(maxlen=buffer_size)},
                "y": {"cmd": deque(maxlen=buffer_size), "true": deque(maxlen=buffer_size), "est": deque(maxlen=buffer_size)},
                "z": {"cmd": deque(maxlen=buffer_size), "true": deque(maxlen=buffer_size), "est": deque(maxlen=buffer_size)},
            },
            "root_base_vel": {
                "x": {"cmd": deque(maxlen=buffer_size), "true": deque(maxlen=buffer_size)},
                "y": {"cmd": deque(maxlen=buffer_size), "true": deque(maxlen=buffer_size)},
                "z": {"cmd": deque(maxlen=buffer_size), "true": deque(maxlen=buffer_size)},
            },
            "base_vel": {
                "x": {"cmd": deque(maxlen=buffer_size), "true": deque(maxlen=buffer_size), "est": deque(maxlen=buffer_size)},
                "y": {"cmd": deque(maxlen=buffer_size), "true": deque(maxlen=buffer_size), "est": deque(maxlen=buffer_size)},
                "z": {"cmd": deque(maxlen=buffer_size), "true": deque(maxlen=buffer_size), "est": deque(maxlen=buffer_size)},
            },
            "foot_pos_left": {"x": deque(maxlen=buffer_size), "y": deque(maxlen=buffer_size), "z": deque(maxlen=buffer_size)},
            "foot_pos_right": {"x": deque(maxlen=buffer_size), "y": deque(maxlen=buffer_size), "z": deque(maxlen=buffer_size)},
            "foot_left": {"x": {"true": deque(maxlen=buffer_size)}, "y": {"true": deque(maxlen=buffer_size)}, "z": {"true": deque(maxlen=buffer_size)}},
            "foot_right": {"x": {"true": deque(maxlen=buffer_size)}, "y": {"true": deque(maxlen=buffer_size)}, "z": {"true": deque(maxlen=buffer_size)}},
        }

        self.data_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.step_count = 0

        # Setup Plot - 5x3 grid
        self.fig, self.axs = plt.subplots(5, 3, figsize=(15, 16))
        self.fig.suptitle("Real-time Monitor (Vel + Foot Pos + Foot Forces)", fontsize=16)

        self.configs = [
            {"key": "head_vel", "axis": "x", "ax": self.axs[0, 0], "title": "Head Vel X (m/s)", "ylabel": "Velocity", "ylim": "vel", "ylim_custom": (-1.5, 1.5)},
            {"key": "head_vel", "axis": "y", "ax": self.axs[0, 1], "title": "Head Vel Y (m/s)", "ylabel": "Velocity", "ylim": "vel", "ylim_custom": (-1.5, 1.5)},
            {"key": "head_vel", "axis": "z", "ax": self.axs[0, 2], "title": "Head Yawrate Z (rad/s)", "ylabel": "Yawrate", "ylim": "vel", "ylim_custom": (-1.5, 1.5)},
            {"key": "base_vel", "axis": "x", "ax": self.axs[1, 0], "title": "Base Vel X (m/s)", "ylabel": "Velocity", "ylim": "vel", "ylim_custom": (-1.5, 1.5)},
            {"key": "base_vel", "axis": "y", "ax": self.axs[1, 1], "title": "Base Vel Y (m/s)", "ylabel": "Velocity", "ylim": "vel", "ylim_custom": (-1.5, 1.5)},
            {"key": "base_vel", "axis": "z", "ax": self.axs[1, 2], "title": "Base Yawrate Z (rad/s)", "ylabel": "Yawrate", "ylim": "vel", "ylim_custom": (-1.5, 1.5)},
            {"key": "foot_pos", "axis": "x", "ax": self.axs[2, 0], "title": "Foot Pos X in Base (m)", "ylabel": "Position (m)", "ylim": "pos", "ylim_custom": None},
            {"key": "foot_pos", "axis": "y", "ax": self.axs[2, 1], "title": "Foot Pos Y in Base (m)", "ylabel": "Position (m)", "ylim": "pos", "ylim_custom": None},
            {"key": "foot_pos", "axis": "z", "ax": self.axs[2, 2], "title": "Foot Pos Z in Base (m)", "ylabel": "Position (m)", "ylim": "pos", "ylim_custom": None},
            {"key": "foot_left", "axis": "x", "ax": self.axs[3, 0], "title": "Left Foot Force X (N)", "ylabel": "Force (N)", "ylim": "contact", "ylim_custom": (-500, 500)},
            {"key": "foot_left", "axis": "y", "ax": self.axs[3, 1], "title": "Left Foot Force Y (N)", "ylabel": "Force (N)", "ylim": "contact", "ylim_custom": (-500, 500)},
            {"key": "foot_left", "axis": "z", "ax": self.axs[3, 2], "title": "Left Foot Force Z (N)", "ylabel": "Force (N)", "ylim": "contact", "ylim_custom": (0, 5000)},
            {"key": "foot_right", "axis": "x", "ax": self.axs[4, 0], "title": "Right Foot Force X (N)", "ylabel": "Force (N)", "ylim": "contact", "ylim_custom": (-500, 500)},
            {"key": "foot_right", "axis": "y", "ax": self.axs[4, 1], "title": "Right Foot Force Y (N)", "ylabel": "Force (N)", "ylim": "contact", "ylim_custom": (-500, 500)},
            {"key": "foot_right", "axis": "z", "ax": self.axs[4, 2], "title": "Right Foot Force Z (N)", "ylabel": "Force (N)", "ylim": "contact", "ylim_custom": (0, 5000)},
        ]

        self.lines = {}
        self.styles = {
            "cmd": {"color": "red", "linestyle": "--", "label": "cmd", "linewidth": 1.5},
            "true": {"color": "black", "linestyle": "-", "label": "true", "alpha": 0.7, "linewidth": 1.5},
            "est": {"color": "blue", "linestyle": "-", "label": "est", "linewidth": 2.0},
            "foot_left_pos": {"color": "red", "linestyle": "-", "label": "Left Foot", "linewidth": 2.0, "alpha": 0.9},
            "foot_right_pos": {"color": "blue", "linestyle": "-", "label": "Right Foot", "linewidth": 2.0, "alpha": 0.9},
        }

        for cfg in self.configs:
            ax = cfg["ax"]
            ax.set_title(cfg["title"])
            ax.set_xlabel("Step")
            ax.set_ylabel(cfg["ylabel"])
            ax.grid(True, alpha=0.3)

            key = cfg["key"]
            self.lines[id(ax)] = {}

            if key in ["foot_left", "foot_right"]:
                for type_name in ["true"]:
                    style = self.styles[type_name]
                    line, = ax.plot([], [], **style)
                    self.lines[id(ax)][type_name] = line
            elif key == "foot_pos":
                for type_name in ["foot_left_pos", "foot_right_pos"]:
                    style = self.styles[type_name]
                    line, = ax.plot([], [], **style)
                    self.lines[id(ax)][type_name] = line
            else:
                for type_name in ["cmd", "true", "est"]:
                    style = self.styles[type_name]
                    line, = ax.plot([], [], **style)
                    self.lines[id(ax)][type_name] = line

            ax.legend(loc="upper right", fontsize="small")

        plt.tight_layout()

    def _get_vec3(self, dct, keys, default=None):
        if default is None:
            default = [0.0, 0.0, 0.0]
        cur = dct
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        if isinstance(cur, (list, tuple)) and len(cur) >= 3:
            try:
                return [float(cur[0]), float(cur[1]), float(cur[2])]
            except Exception:
                return default
        return default

    def read_stdin(self):
        json_pattern = re.compile(r"METRIC_DATA: ({.*})")
        while not self.stop_event.is_set():
            line = sys.stdin.readline()
            if not line:
                break
            match = json_pattern.search(line)
            if match:
                try:
                    data = json.loads(match.group(1))
                    self.data_queue.put(data)
                except json.JSONDecodeError:
                    pass

    def update(self, frame):
        while not self.data_queue.empty():
            try:
                data = self.data_queue.get_nowait()
                self.step_count += 1
                self.steps.append(self.step_count)

                head_cmd = self._get_vec3(data, ["root_base_vel", "cmd"], default=self._get_vec3(data, ["base_vel", "cmd"]))
                head_true = self._get_vec3(data, ["head_vel", "true"], default=self._get_vec3(data, ["base_vel", "true"]))
                head_est = self._get_vec3(data, ["head_vel", "est"], default=self._get_vec3(data, ["base_vel", "est"]))
                for i, axis in enumerate(["x", "y", "z"]):
                    self.data_store["head_vel"][axis]["cmd"].append(head_cmd[i])
                    self.data_store["head_vel"][axis]["true"].append(head_true[i])
                    self.data_store["head_vel"][axis]["est"].append(head_est[i])

                base_true = self._get_vec3(data, ["root_base_vel", "true"], default=self._get_vec3(data, ["base_vel", "true"]))
                for i, axis in enumerate(["x", "y", "z"]):
                    self.data_store["base_vel"][axis]["cmd"].append(head_cmd[i])
                    self.data_store["base_vel"][axis]["true"].append(base_true[i])
                    self.data_store["base_vel"][axis]["est"].append(head_est[i])

                if "foot_pos_base" in data:
                    try:
                        for i, axis in enumerate(["x", "y", "z"]):
                            self.data_store["foot_pos_left"][axis].append(data["foot_pos_base"]["left"][i])
                            self.data_store["foot_pos_right"][axis].append(data["foot_pos_base"]["right"][i])
                    except Exception:
                        for axis in ["x", "y", "z"]:
                            self.data_store["foot_pos_left"][axis].append(0.0)
                            self.data_store["foot_pos_right"][axis].append(0.0)
                else:
                    for axis in ["x", "y", "z"]:
                        self.data_store["foot_pos_left"][axis].append(0.0)
                        self.data_store["foot_pos_right"][axis].append(0.0)

                if "foot_force" in data:
                    for i, axis in enumerate(["x", "y", "z"]):
                        try:
                            self.data_store["foot_left"][axis]["true"].append(data["foot_force"]["left"]["true"][i])
                        except Exception:
                            self.data_store["foot_left"][axis]["true"].append(0.0)
                        try:
                            self.data_store["foot_right"][axis]["true"].append(data["foot_force"]["right"]["true"][i])
                        except Exception:
                            self.data_store["foot_right"][axis]["true"].append(0.0)
            except queue.Empty:
                break

        if not self.steps:
            return [line for ax_lines in self.lines.values() for line in ax_lines.values()]

        for cfg in self.configs:
            ax = cfg["ax"]
            key = cfg["key"]
            axis = cfg["axis"]

            if key in ["foot_left", "foot_right"]:
                self.lines[id(ax)]["true"].set_data(self.steps, self.data_store[key][axis]["true"])
            elif key == "foot_pos":
                self.lines[id(ax)]["foot_left_pos"].set_data(self.steps, self.data_store["foot_pos_left"][axis])
                self.lines[id(ax)]["foot_right_pos"].set_data(self.steps, self.data_store["foot_pos_right"][axis])
            else:
                for type_name in ["cmd", "true", "est"]:
                    self.lines[id(ax)][type_name].set_data(self.steps, self.data_store[key][axis][type_name])

            ax.relim()
            ax.autoscale_view()

            if "ylim_custom" in cfg and cfg["ylim_custom"] is not None:
                ax.set_ylim(cfg["ylim_custom"])
            else:
                ylim_type = cfg.get("ylim", None)
                if ylim_type == "pos" and self.ylim_pos:
                    ax.set_ylim(self.ylim_pos)
                elif ylim_type == "vel" and self.ylim_vel:
                    ax.set_ylim(self.ylim_vel)
                elif ylim_type == "force" and self.ylim_force:
                    ax.set_ylim(self.ylim_force)
                elif ylim_type == "contact" and self.ylim_contact:
                    ax.set_ylim(self.ylim_contact)

        return [line for ax_lines in self.lines.values() for line in ax_lines.values()]

    def start(self):
        print("[PLOT_DEBUG] Starting real-time plotter...")
        thread = threading.Thread(target=self.read_stdin, daemon=True)
        thread.start()
        self._ani = animation.FuncAnimation(self.fig, self.update, interval=50, blit=False, cache_frame_data=False)
        plt.show()
        self.stop_event.set()
        thread.join(timeout=1.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot real-time data.")
    parser.add_argument("--ylim_pos", type=float, nargs=2, default=None)
    parser.add_argument("--ylim_vel", type=float, nargs=2, default=None)
    parser.add_argument("--ylim_force", type=float, nargs=2, default=None)
    parser.add_argument("--ylim_contact", type=float, nargs=2, default=None)
    args = parser.parse_args()

    plotter = RealTimePlotter(
        ylim_pos=args.ylim_pos,
        ylim_vel=args.ylim_vel,
        ylim_force=args.ylim_force,
        ylim_contact=args.ylim_contact,
    )
    plotter.start()

