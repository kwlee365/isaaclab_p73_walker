# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Scatter plot: motor-side OR output-side torque vs speed (12D) with gearbox ratio sweep.

This tool is intended for motor/gearbox sizing using offline-collected METRIC_DATA logs.

Input:
  NPZ created by:
    tools/collect_motor_torque_vel_12d.py

Assumptions:
- The collected 12 DOFs are already the motor DOFs (P73 4bar linkage handled in USD DOF definition).
- Convert DOF quantities to motor-side quantities using gear ratio g (ideal, eta=1):
    omega_motor = omega_dof * g
    tau_motor   = tau_dof / g

Output-side mode:
- Keep scatter points in output-side (DOF) coordinates: (omega_out, tau_out) = (omega_dof, tau_dof)
- Overlay gearbox-ratio-dependent continuous limits projected to output-side using a simple box model:
    |tau_out| <= torque_margin * (tau_m_cont * g * eta)
    |omega_out| <= (omega_m_max / g)
  where eta defaults to 1.0 (lossless) and torque_margin defaults to 0.70 (30% margin).

Limit/spec sources:
- Default: built-in P73 motor spec mapping (QTR105/QTR78 @63V) (hardcoded).
- Optional: read actuator limits from `p73.py` (no file modification) and *derive motor spec* assuming those limits
  correspond to a known base ratio (e.g., 100:1):
    tau_m_cont = effort_limit_sim_out / base_ratio
    omega_m_max = velocity_limit_sim_out * base_ratio

Motor spec-up overlay:
- You can add the token `spec` into `--ratios` to also overlay the upgraded motor spec (keeping the same gear ratios).
  Example: `--ratios 50 80 spec` overlays:
    - base motors @ g=50, g=80
    - spec-up motors @ g=50, g=80

Usage:
  python tools/plot_motor_torque_speed_scatter_12d.py \\
    --npz logs/motor_scatter_run1.npz \\
    --ratios 50 80 100 \\
    --out logs/motor_scatter_run1_ratios.png

Tip:
  If points are too dense, reduce `--alpha` and/or set `--max_points` (downsampling).
"""

from __future__ import annotations

import ast
import argparse
import os
import re
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot motor-side torque-speed scatter (12D) for ratio sweep.")
    p.add_argument("--npz", type=str, required=True, help="Input NPZ path from collector.")
    p.add_argument(
        "--domain",
        type=str,
        default="motor",
        choices=["motor", "output"],
        help="Plot domain: 'motor' (default) converts DOF->motor via ratio; 'output' keeps DOF scatter fixed.",
    )
    p.add_argument(
        "--ratios",
        type=str,
        nargs="+",
        default=["50", "80", "100"],
        help=(
            "Gear ratios to overlay. Tokens are parsed as floats. "
            "You may also include the special token 'spec' to add motor spec-up overlays "
            "for the same gear ratios (output domain only)."
        ),
    )
    p.add_argument("--out", type=str, required=True, help="Output image path (png/pdf).")
    p.add_argument("--alpha", type=float, default=0.15, help="Scatter alpha (transparency).")
    p.add_argument("--s", type=float, default=3.0, help="Scatter marker size.")
    p.add_argument(
        "--max_points",
        type=int,
        default=20000,
        help="Max points per joint (downsample if larger). Set <=0 to disable downsampling.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for downsampling.",
    )
    p.add_argument(
        "--percentile_clip",
        type=float,
        default=99.5,
        help="Axis limit clip percentile (symmetric). Set <=0 to disable.",
    )

    # --- output-side limit overlay controls (used only when --domain output) ---
    p.add_argument("--eta", type=float, default=1.0, help="Gear efficiency eta (used in output limit projection).")
    p.add_argument(
        "--torque_margin",
        type=float,
        default=0.70,
        help="Fraction of continuous output torque allowed (0.70 => 30%% margin).",
    )
    p.add_argument(
        "--spec_source",
        type=str,
        default="p73_default",
        choices=["p73_default", "p73_py"],
        help=(
            "Spec source for output-side limit overlay. "
            "'p73_default' uses built-in QTR105/QTR78 @63V mapping. "
            "'p73_py' reads effort/velocity limits from p73.py and derives motor spec using --p73_base_ratio."
        ),
    )
    p.add_argument(
        "--p73_py_path",
        type=str,
        default="source/isaaclab_p73/isaaclab_p73/assets/p73.py",
        help="Path to p73.py (used when --spec_source p73_py).",
    )
    p.add_argument(
        "--p73_base_ratio",
        type=float,
        default=100.0,
        help="Base gearbox ratio assumed for limits in p73.py (used to derive motor spec).",
    )
    return p.parse_args()


def _downsample_indices(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    if k <= 0 or k >= n:
        return np.arange(n, dtype=np.int64)
    return rng.choice(n, size=k, replace=False)


def _as_str_list(arr: np.ndarray) -> list[str]:
    return [str(x) for x in arr.tolist()]


def _clip_symmetric(v: np.ndarray, p: float) -> float | None:
    if p <= 0.0 or v.size == 0:
        return None
    a = np.abs(v)
    lim = float(np.percentile(a, p))
    if not np.isfinite(lim) or lim <= 0.0:
        return None
    return lim


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)


def _rpm_to_rad_s(rpm: float) -> float:
    return float(rpm) * (2.0 * float(np.pi) / 60.0)


def _p73_default_motor_spec_for_name(name: str) -> tuple[float, float] | None:
    """Return (tau_m_cont_Nm, omega_m_max_rad_s) for a given motor DOF name.

    Based on user-provided spec table:
    - QTR105-25-Z: cont=3.3 N·m, input RPM@63V=3712.871287
    - QTR78-25:    cont=1.38 N·m, input RPM@63V=5639.804487
    and harmonic drive ratio assumed variable in plotting (g in --ratios).
    """
    n = str(name)
    # QTR105 group
    if ("HipRoll" in n) or ("HipPitch" in n) or ("KneeUpper" in n):
        return (3.3, _rpm_to_rad_s(3712.871287))
    # QTR78 group
    if ("HipYaw" in n) or ("AnkleM1" in n) or ("AnkleM2" in n):
        return (1.38, _rpm_to_rad_s(5639.804487))
    return None


def _p73_spec_up_motor_spec_for_name(name: str) -> tuple[float, float] | None:
    """Return (tau_m_cont_Nm, omega_m_max_rad_s) for spec-up motors (user-provided).

    Spec-up table:
    - QTR105-34-Z: cont=5.2 N·m, input RPM@63V=2520.0
    - QTR78-34:    cont=2.10 N·m, input RPM@63V=3705.88
    """
    n = str(name)
    # QTR105 group
    if ("HipRoll" in n) or ("HipPitch" in n) or ("KneeUpper" in n):
        return (5.2, _rpm_to_rad_s(2520.0))
    # QTR78 group
    if ("HipYaw" in n) or ("AnkleM1" in n) or ("AnkleM2" in n):
        return (2.10, _rpm_to_rad_s(3705.88))
    return None


def _extract_implicit_actuator_limit_dicts_from_p73_py(
    p73_py_path: str,
) -> tuple[dict[str, float], dict[str, float]] | None:
    """Extract (effort_limit_sim, velocity_limit_sim) dicts from p73.py via AST parsing.

    We intentionally DO NOT import/execute p73.py to avoid heavy Isaac Sim dependencies.
    Only literal dicts (string->number) are supported.
    """
    path = os.path.abspath(p73_py_path)
    if not os.path.isfile(path):
        return None
    try:
        text = open(path, "r", encoding="utf-8").read()
        tree = ast.parse(text, filename=path)
    except Exception:
        return None

    def _literal_dict_to_float_map(node: ast.AST) -> dict[str, float] | None:
        try:
            obj = ast.literal_eval(node)
        except Exception:
            return None
        if not isinstance(obj, dict):
            return None
        out: dict[str, float] = {}
        for k, v in obj.items():
            if not isinstance(k, str):
                continue
            try:
                out[str(k)] = float(v)
            except Exception:
                continue
        return out

    # Find a call to ImplicitActuatorCfg(..., effort_limit_sim={...}, velocity_limit_sim={...})
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        # function name may be ImplicitActuatorCfg or something.Attribute
        func_name = None
        if isinstance(n.func, ast.Name):
            func_name = n.func.id
        elif isinstance(n.func, ast.Attribute):
            func_name = n.func.attr
        if func_name != "ImplicitActuatorCfg":
            continue

        effort_node = None
        vel_node = None
        for kw in n.keywords:
            if kw.arg == "effort_limit_sim":
                effort_node = kw.value
            elif kw.arg == "velocity_limit_sim":
                vel_node = kw.value

        if effort_node is None or vel_node is None:
            continue

        effort_map = _literal_dict_to_float_map(effort_node)
        vel_map = _literal_dict_to_float_map(vel_node)
        if effort_map is None or vel_map is None:
            continue
        return (effort_map, vel_map)

    return None


def _regex_lookup(name: str, mapping: dict[str, float]) -> float | None:
    """Match name against mapping keys which may be regex patterns."""
    # Prefer exact key match if present.
    if name in mapping:
        return float(mapping[name])
    for pat, val in mapping.items():
        try:
            if re.match(pat + r"\Z", name) is not None:
                return float(val)
        except re.error:
            # If pat isn't a valid regex, fall back to substring equality
            if pat == name:
                return float(val)
    return None


def _motor_spec_from_p73_py_limits(
    *,
    joint_name: str,
    effort_limit_sim: dict[str, float],
    velocity_limit_sim: dict[str, float],
    base_ratio: float,
) -> tuple[float, float] | None:
    """Derive (tau_m_cont, omega_m_max) from output-side limits in p73.py for a given base_ratio."""
    br = float(base_ratio)
    if br <= 0.0:
        return None
    tau_out = _regex_lookup(joint_name, effort_limit_sim)
    omega_out = _regex_lookup(joint_name, velocity_limit_sim)
    if tau_out is None or omega_out is None:
        return None
    # Interpret p73.py sim limits as output-side continuous torque and max speed at base_ratio.
    tau_m_cont = float(tau_out) / br
    omega_m_max = float(omega_out) * br
    return (tau_m_cont, omega_m_max)


def _parse_ratio_tokens(tokens: Sequence[str]) -> tuple[list[float], bool]:
    """Parse --ratios tokens into numeric ratios and a flag for spec-up overlay."""
    ratios: list[float] = []
    spec_up = False
    for t in tokens:
        s = str(t).strip().lower()
        if s in ("spec", "spec_up", "motor_up", "specup"):
            spec_up = True
            continue
        try:
            r = float(s)
        except Exception:
            raise ValueError(f"Invalid ratio token: {t!r}. Use numbers (e.g., 50 80) or 'spec'.") from None
        if r <= 0.0:
            raise ValueError(f"Invalid ratio: {r} (must be > 0)")
        ratios.append(r)
    if not ratios:
        raise ValueError("No numeric ratios provided. Example: --ratios 50 80 spec")
    return ratios, spec_up


def _overlay_output_limit_box(
    ax: plt.Axes,
    *,
    ratio: float,
    tau_m_cont: float,
    omega_m_max: float,
    eta: float,
    torque_margin: float,
    color: str,
):
    """Overlay output-side limit box and torque-margin box.

    Box model:
      omega_out_max = omega_m_max / ratio
      tau_out_cont  = tau_m_cont * ratio * eta
      tau_out_allow = torque_margin * tau_out_cont
    """
    r = float(ratio)
    if r <= 0.0:
        return
    omega_out_max = float(omega_m_max) / r
    tau_out_cont = float(tau_m_cont) * r * float(eta)
    tau_out_allow = float(torque_margin) * tau_out_cont

    # Speed limits (dashed vertical)
    ax.axvline(+omega_out_max, color=color, linestyle="--", linewidth=1.0, alpha=0.9)
    ax.axvline(-omega_out_max, color=color, linestyle="--", linewidth=1.0, alpha=0.9)

    # Continuous torque (lighter dotted)
    ax.axhline(+tau_out_cont, color=color, linestyle=":", linewidth=1.0, alpha=0.5)
    ax.axhline(-tau_out_cont, color=color, linestyle=":", linewidth=1.0, alpha=0.5)

    # Allowed torque with margin (solid)
    ax.axhline(+tau_out_allow, color=color, linestyle="-", linewidth=1.0, alpha=0.9)
    ax.axhline(-tau_out_allow, color=color, linestyle="-", linewidth=1.0, alpha=0.9)


def main() -> None:
    args = _parse_args()

    data = np.load(args.npz, allow_pickle=True)
    names = _as_str_list(data["names"])
    torque_dof = np.asarray(data["torque_dof"], dtype=np.float32)  # (T,12)
    vel_dof = np.asarray(data["vel_dof"], dtype=np.float32)  # (T,12)

    if torque_dof.ndim != 2 or vel_dof.ndim != 2 or torque_dof.shape != vel_dof.shape:
        raise ValueError(f"Invalid shapes: torque_dof={torque_dof.shape}, vel_dof={vel_dof.shape}")
    if torque_dof.shape[1] < 12:
        raise ValueError(f"Expected at least 12 columns, got {torque_dof.shape[1]}")

    # Use first 12
    torque_dof = torque_dof[:, :12]
    vel_dof = vel_dof[:, :12]
    names = names[:12] if len(names) >= 12 else [f"joint_{i}" for i in range(12)]

    ratios, want_spec_up = _parse_ratio_tokens(args.ratios)
    ratios = list(ratios)

    rng = np.random.default_rng(int(args.seed))

    fig, axs = plt.subplots(4, 3, figsize=(18, 10), sharex=False, sharey=False)
    if str(args.domain) == "output":
        fig.suptitle("P73 Output-side Torque–Speed Scatter (12D) + Ratio-projected Limits", fontsize=14)
    else:
        fig.suptitle("P73 Motor-side Torque–Speed Scatter (12D) with Gear Ratio Sweep", fontsize=14)

    # Colors: stable mapping for up to a few ratios
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3", "C4"])

    T = torque_dof.shape[0]
    idx = _downsample_indices(T, int(args.max_points), rng) if int(args.max_points) > 0 else np.arange(T)

    # Precompute clipped axis limits per joint over all ratios (optional)
    clip_p = float(args.percentile_clip)
    x_lim_per_joint: list[float | None] = []
    y_lim_per_joint: list[float | None] = []

    # Optional: read p73.py actuator limits once (used to derive per-joint motor specs)
    p73_limits = None
    if str(args.domain) == "output" and str(args.spec_source) == "p73_py":
        p73_limits = _extract_implicit_actuator_limit_dicts_from_p73_py(str(args.p73_py_path))

    variants: list[str] = ["base"]
    if bool(want_spec_up):
        variants.append("spec_up")

    for j in range(12):
        if str(args.domain) == "output":
            xs = vel_dof[idx, j]
            ys = torque_dof[idx, j]
        else:
            xs_all = []
            ys_all = []
            for r in ratios:
                xs_all.append(vel_dof[idx, j] * r)
                ys_all.append(torque_dof[idx, j] / r)
            xs = np.concatenate(xs_all, axis=0) if xs_all else vel_dof[idx, j]
            ys = np.concatenate(ys_all, axis=0) if ys_all else torque_dof[idx, j]

        x_lim = _clip_symmetric(xs, clip_p)
        y_lim = _clip_symmetric(ys, clip_p)

        # In output-side domain, also include ratio-projected limit lines in axis limits
        # so that large-ratio cases (e.g., 100:1) are not clipped out of view.
        if str(args.domain) == "output" and ratios:
            eta = float(args.eta)
            torque_margin = float(args.torque_margin)
            r_min = float(min(ratios))
            r_max = float(max(ratios))

            # Include limits from both base and spec-up (if requested) so lines don't get clipped.
            for variant in variants:
                spec: tuple[float, float] | None = None
                if variant == "spec_up":
                    spec = _p73_spec_up_motor_spec_for_name(names[j])
                else:
                    if str(args.spec_source) == "p73_py" and p73_limits is not None:
                        effort_map, vel_map = p73_limits
                        spec = _motor_spec_from_p73_py_limits(
                            joint_name=str(names[j]),
                            effort_limit_sim=effort_map,
                            velocity_limit_sim=vel_map,
                            base_ratio=float(args.p73_base_ratio),
                        )
                    else:
                        spec = _p73_default_motor_spec_for_name(names[j])

                if spec is None:
                    continue
                tau_m_cont, omega_m_max = spec

                omega_out_lim = abs(float(omega_m_max) / r_min) if r_min > 0.0 else 0.0
                tau_out_cont_lim = abs(float(tau_m_cont) * r_max * eta) if r_max > 0.0 else 0.0
                tau_out_allow_lim = abs(torque_margin * tau_out_cont_lim)

                x_lim = max(float(x_lim or 0.0), float(omega_out_lim)) if omega_out_lim > 0.0 else x_lim
                y_lim = (
                    max(float(y_lim or 0.0), float(tau_out_allow_lim), float(tau_out_cont_lim))
                    if (tau_out_cont_lim > 0.0)
                    else y_lim
                )

        x_lim_per_joint.append(x_lim)
        y_lim_per_joint.append(y_lim)

    for j in range(12):
        ax = axs[j // 3, j % 3]
        ax.grid(True, alpha=0.3)
        ax.set_title(names[j])
        if str(args.domain) == "output":
            ax.set_xlabel("Output speed (rad/s)")
            ax.set_ylabel("Output torque (N·m)")
        else:
            ax.set_xlabel("Motor speed (rad/s)")
            ax.set_ylabel("Motor torque (N·m)")

        if str(args.domain) == "output":
            # Scatter is fixed in output-side coordinates: plot once (avoid duplicating points per ratio/spec).
            omega_out = vel_dof[idx, j]
            tau_out = torque_dof[idx, j]
            scatter_label = "required" if j == 0 else "_nolegend_"
            ax.scatter(
                omega_out,
                tau_out,
                s=float(args.s),
                alpha=float(args.alpha),
                c="black",
                label=scatter_label,
            )

            # Overlay limit sets: ratios × variants (base/spec_up)
            case_idx = 0
            for r in ratios:
                for variant in variants:
                    c = color_cycle[case_idx % len(color_cycle)]
                    case_idx += 1

                    spec: tuple[float, float] | None = None
                    if variant == "spec_up":
                        spec = _p73_spec_up_motor_spec_for_name(names[j])
                    else:
                        if str(args.spec_source) == "p73_py" and p73_limits is not None:
                            effort_map, vel_map = p73_limits
                            spec = _motor_spec_from_p73_py_limits(
                                joint_name=str(names[j]),
                                effort_limit_sim=effort_map,
                                velocity_limit_sim=vel_map,
                                base_ratio=float(args.p73_base_ratio),
                            )
                        else:
                            spec = _p73_default_motor_spec_for_name(names[j])

                    if spec is None:
                        continue

                    tau_m_cont, omega_m_max = spec
                    _overlay_output_limit_box(
                        ax,
                        ratio=float(r),
                        tau_m_cont=float(tau_m_cont),
                        omega_m_max=float(omega_m_max),
                        eta=float(args.eta),
                        torque_margin=float(args.torque_margin),
                        color=c,
                    )

                    # Legend entries only on the first subplot
                    if j == 0:
                        label_suffix = "spec_up" if variant == "spec_up" else "base"
                        ax.plot([], [], color=c, linewidth=2.0, label=f"g={float(r):g}:1 {label_suffix}")
        else:
            for k, r in enumerate(ratios):
                c = color_cycle[k % len(color_cycle)]
                omega_m = vel_dof[idx, j] * r
                tau_m = torque_dof[idx, j] / r
                ax.scatter(omega_m, tau_m, s=float(args.s), alpha=float(args.alpha), c=c, label=f"g={r:g}:1")

        xl = x_lim_per_joint[j]
        yl = y_lim_per_joint[j]
        if xl is not None:
            ax.set_xlim((-1.05 * xl, 1.05 * xl))
        if yl is not None:
            ax.set_ylim((-1.05 * yl, 1.05 * yl))

    # One legend
    axs[0, 0].legend(loc="upper right", fontsize="small")
    plt.tight_layout()

    _ensure_dir(args.out)
    fig.savefig(args.out, dpi=200)
    print(f"[SCATTER] saved: {args.out}", flush=True)


if __name__ == "__main__":
    main()

