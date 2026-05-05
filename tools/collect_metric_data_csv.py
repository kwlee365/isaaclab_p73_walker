"""
Offline CSV collector for P73 METRIC_DATA stream (locomotion + motor).

Reads METRIC_DATA JSON lines from stdin (produced by
`scripts/tools/p73_command_control/plotting_utils_p73.py`), flattens the
fields, writes one CSV row per env step. Intended for reward re-tuning
workflows where we compare distributions (torque p50/p95/p99, saturation %,
qdd from vel diffs, foot force peaks) between PD-actuator sim, Actuator-Net
sim, and real-robot data.

Captured per step:
  - cmd (vx, vy, wz)
  - root base vel (body-frame true) / head vel (world-frame true)
  - foot position in yaw-only base frame (L / R)
  - foot contact force in yaw-only base frame (L / R)
  - per-joint motor torque: applied / target / limit / sat_applied
  - per-joint motor velocity: measured / limit / sat_vel

Usage — plain dump:
  python scripts/tools/p73_command_control/play_with_teleop_p73.py \\
      --task=Walker-Flat-Play --num_envs=1 \\
      --checkpoint=<ckpt> --control_mode gui --real-time 2>&1 \\
    | python tools/collect_metric_data_csv.py --out logs/metric/run1.csv

Usage — collect and chain a plotter at the same time (passthrough):
  ... 2>&1 \\
    | python tools/collect_metric_data_csv.py --out run1.csv --passthrough \\
    | python tools/plot_motor_torque_12d.py

Step-based warmup / duration gating (works regardless of real-time):
  --dt 0.02 --warmup_s 3 --collect_s 10
Set --collect_s 0 (default) to collect until stdin closes / Ctrl+C.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from typing import Any


_METRIC_RE = re.compile(r"METRIC_DATA: ({.*})")

_FIXED_COLS: list[str] = [
    "step", "sim_time_s",
    "cmd_vx", "cmd_vy", "cmd_wz",
    "true_vx_base", "true_vy_base", "true_wz_base",
    "true_vx_world", "true_vy_world", "true_wz_world",
    "foot_pos_l_x", "foot_pos_l_y", "foot_pos_l_z",
    "foot_pos_r_x", "foot_pos_r_y", "foot_pos_r_z",
    "foot_force_l_x", "foot_force_l_y", "foot_force_l_z",
    "foot_force_r_x", "foot_force_r_y", "foot_force_r_z",
]


def _vec3(d: Any, keys: list[str]) -> list[float]:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return [0.0, 0.0, 0.0]
        cur = cur[k]
    if isinstance(cur, (list, tuple)) and len(cur) >= 3:
        try:
            return [float(cur[0]), float(cur[1]), float(cur[2])]
        except Exception:
            return [0.0, 0.0, 0.0]
    return [0.0, 0.0, 0.0]


def _get_float(lst: Any, i: int) -> float:
    if isinstance(lst, list) and i < len(lst):
        try:
            return float(lst[i])
        except Exception:
            return 0.0
    return 0.0


def _get_bool01(lst: Any, i: int) -> int:
    if isinstance(lst, list) and i < len(lst):
        return int(bool(lst[i]))
    return 0


def _per_joint_cols(names: list[str]) -> list[str]:
    cols: list[str] = []
    for n in names:
        cols += [
            f"torque_applied__{n}",
            f"torque_target__{n}",
            f"torque_limit__{n}",
            f"sat_applied__{n}",
            f"vel_measured__{n}",
            f"vel_limit__{n}",
            f"sat_vel__{n}",
        ]
    return cols


def _log(msg: str) -> None:
    print(f"[CSV] {msg}", file=sys.stderr, flush=True)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Flatten METRIC_DATA JSON stream to wide CSV.")
    p.add_argument("--out", type=str, required=True, help="Output CSV path.")
    p.add_argument("--dt", type=float, default=0.02,
                   help="Env step dt [s] used to convert seconds to step counts.")
    p.add_argument("--warmup_s", type=float, default=0.0,
                   help="Seconds to skip at the start (no rows written).")
    p.add_argument("--collect_s", type=float, default=0.0,
                   help="Seconds to collect after warmup. 0 = until stdin closes / Ctrl+C.")
    p.add_argument("--passthrough", action="store_true",
                   help="Echo stdin to stdout so a downstream tool (plotter) can chain.")
    p.add_argument("--flush_every", type=int, default=50,
                   help="Flush CSV buffer every N rows.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    dt = float(args.dt)
    if dt <= 0.0:
        raise ValueError("--dt must be > 0")

    warmup_steps = int(round(float(args.warmup_s) / dt))
    collect_steps = int(round(float(args.collect_s) / dt)) if args.collect_s > 0 else 0

    out_path = os.path.abspath(args.out)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    _log(f"dt={dt:.4f}s  warmup_steps={warmup_steps}  collect_steps={collect_steps or 'inf'}")
    _log(f"out: {out_path}")
    if args.passthrough:
        _log("passthrough ON — forwarding stdin -> stdout")

    csv_file = None
    writer: Any = None
    header_cols: list[str] | None = None
    joint_names: list[str] | None = None

    step_idx = 0
    written = 0

    try:
        for line in sys.stdin:
            if args.passthrough:
                sys.stdout.write(line)
                sys.stdout.flush()

            m = _METRIC_RE.search(line)
            if not m:
                continue
            try:
                data = json.loads(m.group(1))
            except Exception:
                continue

            # Resolve joint names from the first packet that carries motor_torque or motor_vel.
            if joint_names is None:
                mt = data.get("motor_torque")
                mv = data.get("motor_vel")
                names_src: Any = None
                if isinstance(mt, dict) and isinstance(mt.get("names"), list):
                    names_src = mt["names"]
                elif isinstance(mv, dict) and isinstance(mv.get("names"), list):
                    names_src = mv["names"]
                if isinstance(names_src, list) and len(names_src) >= 12:
                    joint_names = [str(n) for n in names_src[:12]]
                    header_cols = _FIXED_COLS + _per_joint_cols(joint_names)
                    csv_file = open(out_path, "w", newline="")
                    writer = csv.writer(csv_file)
                    writer.writerow(header_cols)
                    _log(f"joints resolved ({len(joint_names)}): header written")
                else:
                    # Cannot write rows yet; skip until joint names arrive.
                    continue

            if writer is None or header_cols is None or joint_names is None:
                continue

            step_idx += 1
            if step_idx <= warmup_steps:
                continue
            if collect_steps > 0 and (step_idx - warmup_steps) > collect_steps:
                break

            cmd = _vec3(data, ["root_base_vel", "cmd"])
            if cmd == [0.0, 0.0, 0.0]:
                cmd = _vec3(data, ["head_vel", "cmd"])
            tr_base = _vec3(data, ["root_base_vel", "true"])
            tr_world = _vec3(data, ["head_vel", "true"])
            fpl = _vec3(data, ["foot_pos_base", "left"])
            fpr = _vec3(data, ["foot_pos_base", "right"])
            ffl = _vec3(data, ["foot_force", "left", "true"])
            ffr = _vec3(data, ["foot_force", "right", "true"])

            mt = data.get("motor_torque")
            mv = data.get("motor_vel")
            applied = mt.get("applied") if isinstance(mt, dict) else None
            target = mt.get("target") if isinstance(mt, dict) else None
            t_limit = mt.get("limit") if isinstance(mt, dict) else None
            sat_a = mt.get("sat_applied") if isinstance(mt, dict) else None
            measured = mv.get("measured") if isinstance(mv, dict) else None
            v_limit = mv.get("limit") if isinstance(mv, dict) else None
            sat_v = mv.get("sat_measured") if isinstance(mv, dict) else None

            row: list[Any] = [
                step_idx, step_idx * dt,
                cmd[0], cmd[1], cmd[2],
                tr_base[0], tr_base[1], tr_base[2],
                tr_world[0], tr_world[1], tr_world[2],
                fpl[0], fpl[1], fpl[2],
                fpr[0], fpr[1], fpr[2],
                ffl[0], ffl[1], ffl[2],
                ffr[0], ffr[1], ffr[2],
            ]
            for i in range(len(joint_names)):
                row.append(_get_float(applied, i))
                row.append(_get_float(target, i))
                row.append(_get_float(t_limit, i))
                row.append(_get_bool01(sat_a, i))
                row.append(_get_float(measured, i))
                row.append(_get_float(v_limit, i))
                row.append(_get_bool01(sat_v, i))

            writer.writerow(row)
            written += 1
            if args.flush_every > 0 and written % int(args.flush_every) == 0:
                csv_file.flush()
    except KeyboardInterrupt:
        _log("interrupted by keyboard")
    finally:
        if csv_file is not None:
            csv_file.flush()
            csv_file.close()
        _log(f"rows written: {written}")
        _log(f"saved: {out_path}")


if __name__ == "__main__":
    main()
