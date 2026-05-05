# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Real-time plotter for P73 lower-body 12D motor torques (applied/target) with torque limits.

Input:
  Reads lines from stdin that contain:
    METRIC_DATA: { ... json ... }

Expected JSON fragment (produced by scripts/tools/p73_command_control/plotting_utils_p73.py):
  "motor_torque": {
      "names": [... 12 joint names ...],
      "applied": [... 12 floats ...],   # applied torque (after clipping)
      "target":  [... 12 floats ...],   # effort target
      "limit":   [... 12 floats ...],   # torque_limits (constant)
      "sat_applied": [... 12 bool ...], # optional
      "sat_target":  [... 12 bool ...], # optional
  }

Usage:
  ./isaaclab.sh -p <play_script.py> ... 2>&1 | python tools/plot_motor_torque_12d.py
"""

from __future__ import annotations

import argparse
import json
import queue
import re
import sys
import threading
from collections import deque

import matplotlib.animation as animation
import matplotlib.pyplot as plt


class MotorTorquePlotter12D:
    def __init__(self, buffer_size: int = 300):
        self.buffer_size = int(buffer_size)

        self.steps = deque(maxlen=self.buffer_size)
        self.step_count = 0

        # filled on first packet
        self.names: list[str] | None = None
        self.limits: list[float] | None = None

        # per-joint buffers
        self.applied_buf: list[deque] | None = None
        self.target_buf: list[deque] | None = None

        self.data_queue: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()

        self.fig, self.axs = plt.subplots(4, 3, figsize=(18, 10), sharex=True)
        self.fig.suptitle("P73 Lower-body Motor Torque (12D): applied vs target with ±limit", fontsize=14)

        self.lines_applied = []
        self.lines_target = []
        self.lines_limit_pos = []
        self.lines_limit_neg = []

        # pre-create empty lines
        for i in range(12):
            ax = self.axs[i // 3, i % 3]
            ax.grid(True, alpha=0.3)
            ax.set_ylabel("Torque (N·m)")
            (l_applied,) = ax.plot([], [], color="black", linewidth=1.5, label="applied")
            (l_target,) = ax.plot([], [], color="blue", linewidth=1.5, label="target")
            (l_lim_pos,) = ax.plot([], [], color="red", linestyle="--", linewidth=1.0, label="+limit")
            (l_lim_neg,) = ax.plot([], [], color="red", linestyle="--", linewidth=1.0, label="-limit")
            self.lines_applied.append(l_applied)
            self.lines_target.append(l_target)
            self.lines_limit_pos.append(l_lim_pos)
            self.lines_limit_neg.append(l_lim_neg)

        # legend once
        self.axs[0, 0].legend(loc="upper right", fontsize="small")
        plt.tight_layout()

    def read_stdin(self):
        json_pattern = re.compile(r"METRIC_DATA: ({.*})")
        while not self.stop_event.is_set():
            line = sys.stdin.readline()
            if not line:
                break
            match = json_pattern.search(line)
            if not match:
                continue
            try:
                data = json.loads(match.group(1))
                self.data_queue.put(data)
            except Exception:
                continue

    def _init_buffers_if_needed(self, motor_torque: dict):
        if self.names is not None and self.limits is not None and self.applied_buf is not None and self.target_buf is not None:
            return
        names = motor_torque.get("names", None)
        limits = motor_torque.get("limit", None)
        if not isinstance(names, list) or not isinstance(limits, list) or len(names) < 12 or len(limits) < 12:
            return
        self.names = [str(n) for n in names[:12]]
        self.limits = [float(x) for x in limits[:12]]
        self.applied_buf = [deque(maxlen=self.buffer_size) for _ in range(12)]
        self.target_buf = [deque(maxlen=self.buffer_size) for _ in range(12)]

        # set titles + initial y-limits
        for i in range(12):
            ax = self.axs[i // 3, i % 3]
            ax.set_title(self.names[i])
            lim = abs(self.limits[i])
            if lim > 1e-6:
                ax.set_ylim((-1.25 * lim, 1.25 * lim))

    def update(self, frame):
        # drain queue
        while not self.data_queue.empty():
            try:
                data = self.data_queue.get_nowait()
            except queue.Empty:
                break

            motor_torque = data.get("motor_torque", None)
            if not isinstance(motor_torque, dict):
                continue

            self._init_buffers_if_needed(motor_torque)
            if self.applied_buf is None or self.target_buf is None or self.limits is None:
                continue

            applied = motor_torque.get("applied", None)
            target = motor_torque.get("target", None)
            limit = motor_torque.get("limit", None)
            if not (isinstance(applied, list) and isinstance(target, list) and isinstance(limit, list)):
                continue
            if len(applied) < 12 or len(target) < 12 or len(limit) < 12:
                continue

            # update step
            self.step_count += 1
            self.steps.append(self.step_count)

            for i in range(12):
                self.applied_buf[i].append(float(applied[i]))
                self.target_buf[i].append(float(target[i]))
                # limits can be static but allow runtime update if it changes
                self.limits[i] = float(limit[i])

        # draw
        if not self.steps or self.applied_buf is None or self.target_buf is None or self.limits is None:
            return self.lines_applied + self.lines_target + self.lines_limit_pos + self.lines_limit_neg

        xs = list(self.steps)
        for i in range(12):
            ax = self.axs[i // 3, i % 3]
            self.lines_applied[i].set_data(xs, list(self.applied_buf[i]))
            self.lines_target[i].set_data(xs, list(self.target_buf[i]))

            lim = float(self.limits[i])
            self.lines_limit_pos[i].set_data(xs, [lim for _ in xs])
            self.lines_limit_neg[i].set_data(xs, [-lim for _ in xs])

            ax.relim()
            ax.autoscale_view(scalex=True, scaley=False)

        return self.lines_applied + self.lines_target + self.lines_limit_pos + self.lines_limit_neg

    def start(self):
        print("[PLOT_TORQUE] Waiting for METRIC_DATA... (motor_torque)", flush=True)
        t = threading.Thread(target=self.read_stdin, daemon=True)
        t.start()
        self._ani = animation.FuncAnimation(self.fig, self.update, interval=50, blit=False, cache_frame_data=False)
        plt.show()
        self.stop_event.set()
        t.join(timeout=1.0)


def main():
    parser = argparse.ArgumentParser(description="Plot P73 12D motor torque (applied/target) with limits.")
    parser.add_argument("--buffer_size", type=int, default=300, help="Number of points to keep in the rolling plot.")
    args = parser.parse_args()

    plotter = MotorTorquePlotter12D(buffer_size=args.buffer_size)
    plotter.start()


if __name__ == "__main__":
    main()

