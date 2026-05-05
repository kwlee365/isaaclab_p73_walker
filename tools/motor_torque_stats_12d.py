"""
Per-motor torque utilization report (% of continuous output-side limit = 점선).

Reference line (100%):
  점선 (:)  = tau_out_cont = tau_m_cont × g × eta   ← 100% = physics effort limit
  실선 (-)  = tau_out_allow = warn_pct × tau_out_cont   ← WARN threshold (default 70%)

Why values can exceed 100%:
  NPZ torque_dof = robot.data.applied_torque = the torque TARGET sent to PhysX.
  PhysX internally caps the actual applied force at effort_limit_sim (= 100%),
  but the logged "applied_torque" is the commanded value (before physics clamp).
  Therefore values > 100% in the raw data indicate torque SATURATION (physics capped).

  To keep RMS/P95/MAX bounded in [0%, 100%], the tool CLIPS |tau_dof| at tau_out_cont
  before computing statistics. The SAT% column separately reports the fraction of
  time steps where the raw torque exceeded the limit.

Statistics (per motor):
  |tau_clipped| = min(|tau_dof|, tau_out_cont)     ← clipped at 점선 (100%)
  RMS  = sqrt(mean(tau_clipped^2)) / tau_out_cont × 100
  P95  = percentile(|tau_clipped|, 95)  / tau_out_cont × 100
  MAX  = max(|tau_clipped|)             / tau_out_cont × 100   (= 100% if any saturation)
  SAT% = mean(|tau_dof| >= tau_out_cont) × 100    ← % of time steps saturated

Status (P95-based):
  [   OK   ]  P95 <  warn_pct  (default 70%)
  [  WARN  ]  P95 >= warn_pct  but SAT% = 0%
  [SAT x.x%]  SAT% > 0  (saturation occurred; x.x% of time at physical limit)

OVERALL row: all 12 joints' clipped % values pooled → single RMS / P95 / MAX / SAT%.

Motor spec (QTR105-25-Z / QTR78-25 @63V):
  HipRoll, HipPitch, KneeUpper  →  tau_m_cont = 3.30 Nm
  HipYaw,  AnkleM1,  AnkleM2   →  tau_m_cont = 1.38 Nm

Usage:
  python tools/motor_torque_stats_12d.py \\
      --npz  logs/motor_scatter_1ms_run1.npz \\
      --ratios 50 80 100 \\
      --warn_pct 70 \\
      --out  logs/torque_stats_1ms.txt
"""

from __future__ import annotations

import argparse
import ast
import math
import os
import re
import sys
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# ANSI colour helpers
# ---------------------------------------------------------------------------

def _ansi(code: str, text: str, on: bool) -> str:
    return f"\033[{code}m{text}\033[0m" if on else text


def _green(t: str, on: bool) -> str:
    return _ansi("32", t, on)


def _yellow(t: str, on: bool) -> str:
    return _ansi("33", t, on)


def _red(t: str, on: bool) -> str:
    return _ansi("31;1", t, on)


def _bold(t: str, on: bool) -> str:
    return _ansi("1", t, on)


def _strip_ansi(s: str) -> str:
    return re.sub(r"\033\[[0-9;]*m", "", s)


# ---------------------------------------------------------------------------
# Motor spec  (motor-side continuous torque and max speed)
# ---------------------------------------------------------------------------

_QTR105_CONT_NM: float = 3.30       # Nm  QTR105-25-Z continuous torque
_QTR78_CONT_NM:  float = 1.38       # Nm  QTR78-25  continuous torque

# Spec-up motors (one grade up)
_QTR105_34_CONT_NM: float = 5.2    # Nm  QTR105-34-Z continuous torque
_QTR78_34_CONT_NM:  float = 2.10   # Nm  QTR78-34   continuous torque


def _rpm_to_rad_s(rpm: float) -> float:
    return rpm * (2.0 * math.pi / 60.0)


def _motor_tau_cont(name: str) -> float | None:
    """Return tau_m_cont [Nm] for this DOF, or None if unknown."""
    n = str(name)
    if any(k in n for k in ("HipRoll", "HipPitch", "KneeUpper")):
        return _QTR105_CONT_NM
    if any(k in n for k in ("HipYaw", "AnkleM1", "AnkleM2")):
        return _QTR78_CONT_NM
    return None


def _spec_up_tau_m_cont(name: str) -> float | None:
    """Return spec-up tau_m_cont [Nm] (QTR105-34 / QTR78-34)."""
    n = str(name)
    if any(k in n for k in ("HipRoll", "HipPitch", "KneeUpper")):
        return _QTR105_34_CONT_NM
    if any(k in n for k in ("HipYaw", "AnkleM1", "AnkleM2")):
        return _QTR78_34_CONT_NM
    return None


def _motor_type_str(name: str, spec_up: bool = False) -> str:
    n = str(name)
    if any(k in n for k in ("HipRoll", "HipPitch", "KneeUpper")):
        return "QTR105-34" if spec_up else "QTR105"
    if any(k in n for k in ("HipYaw", "AnkleM1", "AnkleM2")):
        return "QTR78-34 " if spec_up else "QTR78  "
    return "??       " if spec_up else "??     "


# ---------------------------------------------------------------------------
# p73.py AST reader  (no import/exec — avoids Isaac Sim dependencies)
# ---------------------------------------------------------------------------

def _extract_effort_limits_from_p73_py(
    p73_py_path: str,
) -> dict[str, float] | None:
    """Parse effort_limit_sim dict from p73.py via AST (no import/exec).

    Returns pattern→Nm mapping, or None if parsing fails.
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

    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        func_name = None
        if isinstance(n.func, ast.Name):
            func_name = n.func.id
        elif isinstance(n.func, ast.Attribute):
            func_name = n.func.attr
        if func_name != "ImplicitActuatorCfg":
            continue
        for kw in n.keywords:
            if kw.arg == "effort_limit_sim":
                m = _literal_dict_to_float_map(kw.value)
                if m is not None:
                    return m

    return None


def _regex_lookup(name: str, mapping: dict[str, float]) -> float | None:
    """Match joint name against mapping keys that may be regex patterns."""
    if name in mapping:
        return float(mapping[name])
    for pat, val in mapping.items():
        try:
            if re.match(pat + r"\Z", name) is not None:
                return float(val)
        except re.error:
            if pat == name:
                return float(val)
    return None


def _get_tau_m_cont(
    name: str,
    *,
    effort_map: dict[str, float] | None,
    base_ratio: float,
) -> float | None:
    """Return tau_m_cont [Nm] for this DOF.

    If effort_map is provided (from p73.py), derives tau_m_cont = effort_limit_sim / base_ratio.
    Falls back to the hardcoded QTR105/QTR78 table if lookup fails or effort_map is None.
    """
    if effort_map is not None:
        tau_out = _regex_lookup(name, effort_map)
        if tau_out is not None:
            return float(tau_out) / float(base_ratio)
    # Hardcoded fallback
    return _motor_tau_cont(name)


# ---------------------------------------------------------------------------
# Ratio parsing
# ---------------------------------------------------------------------------

def _parse_ratios(tokens: Sequence[str]) -> tuple[list[float], bool]:
    """Return (numeric_ratios, want_spec_up). 'spec'/'spec_up' token sets the flag."""
    out: list[float] = []
    spec_up = False
    for t in tokens:
        s = t.strip().lower()
        if s in ("spec", "spec_up", "specup"):
            spec_up = True
            continue
        try:
            r = float(s)
        except Exception:
            raise ValueError(f"Invalid ratio token: {t!r}. Use numbers e.g. 50 80 100.") from None
        if r <= 0.0:
            raise ValueError(f"Ratio must be > 0, got {r}.")
        out.append(r)
    if not out:
        raise ValueError("No numeric ratios found.")
    return out, spec_up


# ---------------------------------------------------------------------------
# Core statistics  (inputs are absolute % values ≥ 0)
# ---------------------------------------------------------------------------

def _stats(pct: np.ndarray) -> tuple[float, float, float]:
    """Return (rms_pct, p95_pct, max_pct)."""
    if pct.size == 0:
        return (0.0, 0.0, 0.0)
    rms = float(np.sqrt(np.mean(pct ** 2)))
    p95 = float(np.percentile(pct, 95))
    mx  = float(np.max(pct))
    return rms, p95, mx


# ---------------------------------------------------------------------------
# Formatting  — values are %, width fixed before colouring
# ---------------------------------------------------------------------------

_NAME_W = 24
_SEP = "─" * 100


def _fmt_pct(v: float, warn_pct: float, on: bool, width: int = 8) -> str:
    """Format a % value as fixed-width, coloured by threshold."""
    s = f"{v:6.1f}%".rjust(width)   # e.g. " 62.5%"  always same width
    if v >= 100.0:
        return _red(s, on)
    if v >= warn_pct:
        return _yellow(s, on)
    return s


def _fmt_sat_pct(v: float, on: bool, width: int = 8) -> str:
    """Format SAT% — red if any saturation, plain otherwise."""
    s = f"{v:6.1f}%".rjust(width)
    if v > 0.0:
        return _red(s, on)
    return s


def _status(p95_pct: float, sat_pct: float, warn_pct: float, on: bool) -> str:
    """
    [SAT x.x%] : SAT% > 0  (physics capped x.x% of time steps)
    [  WARN   ] : P95 >= warn_pct  but no saturation
    [   OK    ] : P95 < warn_pct  and no saturation
    """
    if sat_pct > 0.0:
        return _red(f"[SAT{sat_pct:5.1f}%]", on)
    if p95_pct >= warn_pct:
        return _yellow("[   WARN  ]", on)
    return _green("[    OK   ]", on)


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------

def _print_table(
    *,
    names: list[str],
    # per-joint: (rms_pct, p95_pct, max_pct, sat_pct, tau_out_cont)
    rows: list[tuple[float, float, float, float, float | None]],
    overall: tuple[float, float, float, float],
    ratio: float,
    warn_pct: float,
    on: bool,
    lines: list[str],
    spec_up: bool = False,
) -> None:

    def emit(s: str) -> None:
        print(s)
        lines.append(_strip_ansi(s))

    spec_tag = "  [SPEC-UP: QTR105-34 / QTR78-34]" if spec_up else ""
    emit("")
    emit(_bold(
        f"  Gear ratio  g = {ratio:g}:1"
        f"   (점선 = tau_out_cont = tau_m_cont × {ratio:g}  →  100%){spec_tag}",
        on,
    ))
    emit(_SEP)
    emit(
        f"  {'Joint':<{_NAME_W}} {'Type':<7} {'점선 Limit':>10}"
        f"  {'RMS':>8}  {'P95':>8}  {'MAX':>8}  {'SAT%':>8}  Status"
    )
    emit(f"  {'':<{_NAME_W}} {'':7} {'(Nm)':>10}"
         f"  {'(% of 점선, clipped)':>8}  {'':>8}  {'':>8}  {'(sat)':>8}")
    emit(_SEP)

    for name, (rms_p, p95_p, max_p, sat_p, tau_out_cont) in zip(names, rows):
        mtype = _motor_type_str(name, spec_up=spec_up)
        lim_s = f"{tau_out_cont:.1f}".rjust(10) if tau_out_cont is not None else "         ?"
        rms_s = _fmt_pct(rms_p, warn_pct, on)
        p95_s = _fmt_pct(p95_p, warn_pct, on)
        max_s = _fmt_pct(max_p, warn_pct, on)
        sat_s = _fmt_sat_pct(sat_p, on)
        st = _status(p95_p, sat_p, warn_pct, on)
        emit(f"  {name:<{_NAME_W}} {mtype:<7} {lim_s}  {rms_s}  {p95_s}  {max_s}  {sat_s}  {st}")

    emit(_SEP)

    rms_p, p95_p, max_p, sat_p = overall
    rms_s = _fmt_pct(rms_p, warn_pct, on)
    p95_s = _fmt_pct(p95_p, warn_pct, on)
    max_s = _fmt_pct(max_p, warn_pct, on)
    sat_s = _fmt_sat_pct(sat_p, on)
    st = _status(p95_p, sat_p, warn_pct, on)
    overall_label = "OVERALL (all 12 joints)"
    emit(_bold(
        f"  {overall_label:<{_NAME_W}} {'':7} {'—':>10}  {rms_s}  {p95_s}  {max_s}  {sat_s}  {st}",
        on,
    ))
    emit(_SEP)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-motor torque utilization: RMS / P95 / MAX as % of continuous output torque (점선).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--npz", required=True, help="NPZ path from collect_motor_torque_vel_12d.py.")
    p.add_argument(
        "--ratios",
        nargs="+",
        default=["50", "80", "100"],
        help="Gear ratios to analyse. Numeric tokens; 'spec' token silently ignored.",
    )
    p.add_argument(
        "--warn_pct",
        type=float,
        default=None,
        help=(
            "P95 threshold for WARN status in %% of continuous limit (default 70.0). "
            "Matches --torque_margin 0.70 in the scatter plot."
        ),
    )
    p.add_argument(
        "--torque_margin",
        type=float,
        default=None,
        help=(
            "Alias for --warn_pct but in 0–1 fraction (e.g. 0.70 → 70%%). "
            "If both are given, --warn_pct takes precedence."
        ),
    )
    p.add_argument(
        "--eta",
        type=float,
        default=1.0,
        help="Gear efficiency η (default 1.0 = lossless). tau_out_cont = tau_m_cont × g × η.",
    )
    p.add_argument(
        "--spec_source",
        choices=["p73_py", "p73_default"],
        default="p73_py",
        help=(
            "Torque spec source. "
            "'p73_py' (default): read effort_limit_sim from p73.py and derive "
            "tau_m_cont = effort_limit_sim / --p73_base_ratio. "
            "'p73_default': use hardcoded QTR105=3.30 Nm / QTR78=1.38 Nm table."
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
        help=(
            "Base gear ratio assumed for effort_limit_sim in p73.py "
            "(used to derive tau_m_cont = effort_limit_sim / p73_base_ratio). "
            "Default 100.0."
        ),
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional path to save the plain-text report.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    on = sys.stdout.isatty()

    if not os.path.isfile(args.npz):
        raise FileNotFoundError(f"NPZ not found: {args.npz}")
    data = np.load(args.npz, allow_pickle=True)

    names: list[str] = [str(x) for x in data["names"].tolist()][:12]
    torque_dof = np.asarray(data["torque_dof"], dtype=np.float64)  # (T, >=12)

    if torque_dof.ndim != 2 or torque_dof.shape[1] < 12:
        raise ValueError(f"Unexpected torque_dof shape: {torque_dof.shape}. Expected (T, >=12).")
    torque_dof = torque_dof[:, :12]
    T = torque_dof.shape[0]

    while len(names) < 12:
        names.append(f"joint_{len(names)}")

    ratios, want_spec_up = _parse_ratios(args.ratios)
    eta        = float(args.eta)
    base_ratio = float(args.p73_base_ratio)

    # Resolve warn_pct: --warn_pct (%) takes precedence over --torque_margin (0-1 fraction)
    if args.warn_pct is not None:
        warn_pct = float(args.warn_pct)
    elif args.torque_margin is not None:
        warn_pct = float(args.torque_margin) * 100.0   # 0.70 → 70.0
    else:
        warn_pct = 70.0

    # Load effort limits from p73.py (AST, no import) for dynamic tau_m_cont
    effort_map: dict[str, float] | None = None
    spec_label: str
    if str(args.spec_source) == "p73_py":
        effort_map = _extract_effort_limits_from_p73_py(str(args.p73_py_path))
        if effort_map is not None:
            spec_label = (
                f"p73_py  ({os.path.abspath(args.p73_py_path)}, "
                f"base_ratio={base_ratio:g})"
            )
        else:
            spec_label = (
                f"p73_py  [LOAD FAILED: {args.p73_py_path}] → fallback to p73_default"
            )
    else:
        spec_label = "p73_default  (QTR105=3.30 Nm, QTR78=1.38 Nm hardcoded)"

    lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    emit("=" * 100)
    emit("  Motor Torque Utilization  (% of continuous output torque = 점선)")
    emit(f"  NPZ      : {os.path.abspath(args.npz)}")
    emit(f"  Samples  : T = {T}")
    emit(f"  η (eta)  : {eta:.3f}")
    emit(f"  Spec     : {spec_label}")
    emit(f"  WARN at  : P95 >= {warn_pct:.0f}%  (= 실선, {warn_pct/100:.0%} of 점선)")
    emit(f"  SAT at   : |tau_dof| >= 점선  → PhysX capped; MAX clipped to 100%")
    emit(f"  Ratios   : {' | '.join(f'g={r:g}' for r in ratios)}"
         + ("  +spec-up" if want_spec_up else ""))
    emit("=" * 100)

    for ratio in ratios:
        # Output-side continuous torque limit for this ratio (the dotted line)
        # tau_out_cont = tau_m_cont * g * eta
        tau_dof_abs = np.abs(torque_dof)  # (T,12), output-side [Nm]

        rows: list[tuple[float, float, float, float, float | None]] = []
        all_pct_clipped: list[np.ndarray] = []
        all_sat_flags: list[np.ndarray] = []

        for j in range(12):
            tau_m = _get_tau_m_cont(names[j], effort_map=effort_map, base_ratio=base_ratio)
            if tau_m is not None:
                tau_out_cont = tau_m * ratio * eta        # Nm, output-side = dotted line
                pct_raw = tau_dof_abs[:, j] / tau_out_cont * 100.0  # can exceed 100%
                sat_flag = pct_raw >= 100.0               # bool: saturated at this step
                pct_clipped = np.minimum(pct_raw, 100.0) # bounded [0%, 100%]
            else:
                tau_out_cont = None
                pct_clipped = tau_dof_abs[:, j]  # raw, no normalisation
                sat_flag = np.zeros(T, dtype=bool)

            sat_p = float(np.mean(sat_flag) * 100.0)
            rms_p, p95_p, max_p = _stats(pct_clipped)
            rows.append((rms_p, p95_p, max_p, sat_p, tau_out_cont))
            all_pct_clipped.append(pct_clipped)
            all_sat_flags.append(sat_flag)

        # Overall: pool all clipped % values + sat flags across all 12 joints
        pooled_clipped = np.concatenate(all_pct_clipped, axis=0)
        pooled_sat     = np.concatenate(all_sat_flags, axis=0)
        rms_ov, p95_ov, max_ov = _stats(pooled_clipped)
        sat_ov = float(np.mean(pooled_sat) * 100.0)
        overall = (rms_ov, p95_ov, max_ov, sat_ov)

        _print_table(
            names=names,
            rows=rows,
            overall=overall,
            ratio=ratio,
            warn_pct=warn_pct,
            on=on,
            lines=lines,
            spec_up=False,
        )

        # Spec-up table: same ratio, but tau_m_cont from spec-up motors
        if want_spec_up:
            rows_su: list[tuple[float, float, float, float, float | None]] = []
            all_pct_su: list[np.ndarray] = []
            all_sat_su: list[np.ndarray] = []

            for j in range(12):
                tau_m = _spec_up_tau_m_cont(names[j])
                if tau_m is not None:
                    tau_out_cont = tau_m * ratio * eta
                    pct_raw = tau_dof_abs[:, j] / tau_out_cont * 100.0
                    sat_flag = pct_raw >= 100.0
                    pct_clipped = np.minimum(pct_raw, 100.0)
                else:
                    tau_out_cont = None
                    pct_clipped = tau_dof_abs[:, j]
                    sat_flag = np.zeros(T, dtype=bool)

                sat_p = float(np.mean(sat_flag) * 100.0)
                rms_p, p95_p, max_p = _stats(pct_clipped)
                rows_su.append((rms_p, p95_p, max_p, sat_p, tau_out_cont))
                all_pct_su.append(pct_clipped)
                all_sat_su.append(sat_flag)

            pooled_su  = np.concatenate(all_pct_su, axis=0)
            pooled_sat_su = np.concatenate(all_sat_su, axis=0)
            rms_ov, p95_ov, max_ov = _stats(pooled_su)
            sat_ov = float(np.mean(pooled_sat_su) * 100.0)

            _print_table(
                names=names,
                rows=rows_su,
                overall=(rms_ov, p95_ov, max_ov, sat_ov),
                ratio=ratio,
                warn_pct=warn_pct,
                on=on,
                lines=lines,
                spec_up=True,
            )

    if args.out:
        out_path = os.path.abspath(args.out)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\n[STATS] report saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
