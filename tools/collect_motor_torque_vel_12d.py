"""
Offline collector for P73 lower-body 12D motor DOF torque/velocity from METRIC_DATA stream.

Why this exists:
- The real-time plotters (`tools/plot_motor_torque_12d.py`, `tools/plot_motor_vel_12d.py`) are for visualization.
- For motor/gearbox sizing, we want an offline dataset of (torque, velocity) points at a target gait condition
  (e.g., forward 1.0 m/s) and then sweep gearbox ratios to convert to motor-side requirements.

Input (stdin):
  Lines containing:
    METRIC_DATA: { ... json ... }

Expected JSON fragments (from `scripts/tools/p73_command_control/plotting_utils_p73.py`):
  "motor_torque": {"names":[...], "applied":[...12...], ...}
  "motor_vel":    {"names":[...], "measured":[...12...], ...}

Output:
  NPZ file containing:
    - names: (12,) joint/motor DOF names
    - torque_dof: (T,12) applied DOF torque [N·m]
    - vel_dof:    (T,12) measured DOF velocity [rad/s]
    - dt: float (assumed step dt used to compute warmup/collect steps)

Notes:
- This collector gates by *steps*, not wall-clock time, to work both in real-time and faster-than-real-time runs.
- Default dt is set for P73 Rough/Flat typical config: sim.dt=0.005 and decimation=4 => step_dt ≈ 0.02 s.
  If your config differs, pass `--dt` explicitly.

Usage example:
  python ...play_with_teleop_p73.py ... 2>&1 | \\
    python tools/collect_motor_torque_vel_12d.py \\
      --dt 0.02 --warmup_s 3 --collect_s 10 --out logs/motor_scatter_run1.npz

Then plot ratio sweep scatter:
  python tools/plot_motor_torque_speed_scatter_12d.py \\
    --npz logs/motor_scatter_run1.npz --ratios 50 80 100 --out logs/motor_scatter_run1_ratios.png
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any

import numpy as np


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect 12D motor DOF torque/velocity from METRIC_DATA to NPZ.")
    p.add_argument("--dt", type=float, default=0.02, help="Environment step dt [s] used to convert seconds->steps.")
    p.add_argument("--warmup_s", type=float, default=3.0, help="Warmup duration [s] to skip (steady-state gating).")
    p.add_argument("--collect_s", type=float, default=10.0, help="Collection duration [s] after warmup.")
    p.add_argument("--out", type=str, required=True, help="Output NPZ path.")
    p.add_argument(
        "--require_both",
        action="store_true",
        default=False,
        help="If set, only collect steps where both motor_torque and motor_vel are present in the same packet.",
    )
    return p.parse_args()


def _safe_float_list(x: Any, n: int) -> list[float] | None:
    if not isinstance(x, list) or len(x) < n:
        return None
    try:
        return [float(v) for v in x[:n]]
    except Exception:
        return None


def main() -> None:
    args = _parse_args()

    dt = float(args.dt)
    if dt <= 0.0:
        raise ValueError("--dt must be > 0")

    warmup_steps = int(round(float(args.warmup_s) / dt))
    collect_steps = int(round(float(args.collect_s) / dt))
    total_steps = warmup_steps + collect_steps

    json_pattern = re.compile(r"METRIC_DATA: ({.*})")

    names: list[str] | None = None
    torque_buf: list[list[float]] = []
    vel_buf: list[list[float]] = []

    step_idx = 0

    print(
        f"[COLLECT] dt={dt:.6f}s warmup_steps={warmup_steps} collect_steps={collect_steps} total_steps={total_steps}",
        flush=True,
    )
    print(f"[COLLECT] writing to: {args.out}", flush=True)

    for line in sys.stdin:
        match = json_pattern.search(line)
        if not match:
            continue

        try:
            data = json.loads(match.group(1))
        except Exception:
            continue

        mt = data.get("motor_torque", None)
        mv = data.get("motor_vel", None)

        torque_12: list[float] | None = None
        vel_12: list[float] | None = None

        if isinstance(mt, dict):
            n = mt.get("names", None)
            if names is None and isinstance(n, list) and len(n) >= 12:
                names = [str(x) for x in n[:12]]
            torque_12 = _safe_float_list(mt.get("applied", None), 12)

        if isinstance(mv, dict):
            n = mv.get("names", None)
            if names is None and isinstance(n, list) and len(n) >= 12:
                names = [str(x) for x in n[:12]]
            vel_12 = _safe_float_list(mv.get("measured", None), 12)

        if args.require_both and (torque_12 is None or vel_12 is None):
            continue

        # advance step when we have at least one signal
        if torque_12 is None and vel_12 is None:
            continue

        step_idx += 1

        # gate warmup
        if step_idx <= warmup_steps:
            continue

        # stop after enough samples
        if step_idx > total_steps:
            break

        # If one is missing, repeat last value (keeps arrays aligned).
        if torque_12 is None:
            if torque_buf:
                torque_12 = torque_buf[-1]
            else:
                continue
        if vel_12 is None:
            if vel_buf:
                vel_12 = vel_buf[-1]
            else:
                continue

        torque_buf.append(torque_12)
        vel_buf.append(vel_12)

    if names is None:
        raise RuntimeError("No motor names received. Ensure METRIC_DATA contains motor_torque/motor_vel.")
    if not torque_buf or not vel_buf:
        raise RuntimeError("No samples collected. Check warmup/collect settings and input stream.")

    torque_arr = np.asarray(torque_buf, dtype=np.float32)  # (T,12)
    vel_arr = np.asarray(vel_buf, dtype=np.float32)  # (T,12)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(
        args.out,
        names=np.asarray(names, dtype=object),
        torque_dof=torque_arr,
        vel_dof=vel_arr,
        dt=np.asarray(dt, dtype=np.float32),
        warmup_steps=np.asarray(warmup_steps, dtype=np.int32),
        collect_steps=np.asarray(collect_steps, dtype=np.int32),
    )

    print(f"[COLLECT] saved: T={torque_arr.shape[0]} joints={torque_arr.shape[1]}", flush=True)


if __name__ == "__main__":
    main()

