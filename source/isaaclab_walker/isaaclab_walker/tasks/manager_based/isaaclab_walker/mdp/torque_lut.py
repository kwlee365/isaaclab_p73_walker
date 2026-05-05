"""Torque LUT: CSV-based angle-dependent effort limits for 4-bar linkage joints."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch


class KneeTorqueLUT:
    """1-DOF LUT: joint_angle_rad -> torque_limit_nm via linear interpolation."""

    def __init__(self, csv_path: str, device: str = "cuda"):
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        angles = np.array([float(r["joint_angle_rad"]) for r in rows])
        limits = np.array([float(r["torque_limit_nm"]) for r in rows])
        # sort by angle (ascending) for searchsorted
        order = np.argsort(angles)
        self.angles = torch.tensor(angles[order], dtype=torch.float32, device=device)
        self.limits = torch.tensor(limits[order], dtype=torch.float32, device=device)

    def query(self, joint_angle: torch.Tensor) -> torch.Tensor:
        """Batched linear interpolation. Input/output shape: (num_envs,)."""
        q = torch.clamp(joint_angle, self.angles[0], self.angles[-1])
        idx = torch.searchsorted(self.angles, q).clamp(1, len(self.angles) - 1)
        a0, a1 = self.angles[idx - 1], self.angles[idx]
        t0, t1 = self.limits[idx - 1], self.limits[idx]
        alpha = (q - a0) / (a1 - a0 + 1e-8)
        return t0 + alpha * (t1 - t0)


class AnkleTorqueLUT:
    """2-DOF LUT: (pitch_rad, roll_rad) -> (tau_pitch_nm, tau_roll_nm) via bilinear interpolation."""

    def __init__(self, csv_path: str, device: str = "cuda"):
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        pitches = sorted(set(float(r["pitch_rad"]) for r in rows))
        rolls = sorted(set(float(r["roll_rad"]) for r in rows))
        np_, nr = len(pitches), len(rolls)

        # build 2D grids indexed by (pitch_idx, roll_idx)
        p_map = {v: i for i, v in enumerate(pitches)}
        r_map = {v: i for i, v in enumerate(rolls)}
        tau_p = np.zeros((np_, nr))
        tau_r = np.zeros((np_, nr))
        for row in rows:
            pi = p_map[float(row["pitch_rad"])]
            ri = r_map[float(row["roll_rad"])]
            tau_p[pi, ri] = float(row["tau_pitch_nm"])
            tau_r[pi, ri] = float(row["tau_roll_nm"])

        self.pitch_arr = torch.tensor(pitches, dtype=torch.float32, device=device)
        self.roll_arr = torch.tensor(rolls, dtype=torch.float32, device=device)
        self.tau_pitch = torch.tensor(tau_p, dtype=torch.float32, device=device)
        self.tau_roll = torch.tensor(tau_r, dtype=torch.float32, device=device)

    def query(self, pitch: torch.Tensor, roll: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Batched bilinear interpolation. Input/output shape: (num_envs,)."""
        p = torch.clamp(pitch, self.pitch_arr[0], self.pitch_arr[-1])
        r = torch.clamp(roll, self.roll_arr[0], self.roll_arr[-1])

        ip = torch.searchsorted(self.pitch_arr, p).clamp(1, len(self.pitch_arr) - 1)
        ir = torch.searchsorted(self.roll_arr, r).clamp(1, len(self.roll_arr) - 1)

        p0, p1 = self.pitch_arr[ip - 1], self.pitch_arr[ip]
        r0, r1 = self.roll_arr[ir - 1], self.roll_arr[ir]
        ap = (p - p0) / (p1 - p0 + 1e-8)
        ar = (r - r0) / (r1 - r0 + 1e-8)

        def _bilinear(grid: torch.Tensor) -> torch.Tensor:
            v00 = grid[ip - 1, ir - 1]
            v01 = grid[ip - 1, ir]
            v10 = grid[ip, ir - 1]
            v11 = grid[ip, ir]
            return (1 - ap) * ((1 - ar) * v00 + ar * v01) + ap * ((1 - ar) * v10 + ar * v11)

        return _bilinear(self.tau_pitch), _bilinear(self.tau_roll)
