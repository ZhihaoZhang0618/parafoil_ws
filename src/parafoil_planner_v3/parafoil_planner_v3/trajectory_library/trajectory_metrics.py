from __future__ import annotations

from typing import Dict

import numpy as np

from parafoil_planner_v3.types import Trajectory
from parafoil_planner_v3.utils.quaternion_utils import quat_to_rpy, wrap_pi


def _heading_deltas(xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    if xy.shape[0] < 3:
        return np.zeros((0,), dtype=float)
    d = np.diff(xy, axis=0)
    headings = np.arctan2(d[:, 1], d[:, 0])
    dpsi = [wrap_pi(float(headings[i + 1] - headings[i])) for i in range(len(headings) - 1)]
    return np.asarray(dpsi, dtype=float).reshape(-1)


def _turn_sign_changes(dpsi: np.ndarray, eps_rad: float) -> tuple[int, int, float]:
    dpsi = np.asarray(dpsi, dtype=float).reshape(-1)
    if dpsi.size == 0:
        return 0, 0, 0.0

    total_abs = float(np.sum(np.abs(dpsi)))
    pos = float(np.sum(np.clip(dpsi, 0.0, None)))
    neg = float(np.sum(np.clip(-dpsi, 0.0, None)))
    dominant_fraction = 0.0 if total_abs <= 1e-9 else float(max(pos, neg) / total_abs)

    last_sign = 0
    sign_changes = 0
    cluster_sign = 0
    clusters = 0
    for d in dpsi:
        if abs(d) < eps_rad:
            cluster_sign = 0
            continue
        sign = 1 if d > 0.0 else -1
        if last_sign != 0 and sign != last_sign:
            sign_changes += 1
        last_sign = sign
        if cluster_sign == 0 or sign != cluster_sign:
            clusters += 1
            cluster_sign = sign

    return int(sign_changes), int(clusters), float(dominant_fraction)


def compute_trajectory_metrics(traj: Trajectory, turn_eps_deg: float = 3.0) -> Dict[str, float]:
    if not traj.waypoints:
        return {
            "duration_s": 0.0,
            "altitude_loss_m": 0.0,
            "path_length_m": 0.0,
            "max_bank_deg": 0.0,
            "max_yaw_rate_deg_s": 0.0,
            "max_brake": 0.0,
            "max_delta_a": 0.0,
            "turn_total_rad": 0.0,
            "turn_net_rad": 0.0,
            "turn_total_turns": 0.0,
            "turn_net_turns": 0.0,
            "turn_sign_changes": 0.0,
            "turn_clusters": 0.0,
            "turn_dominant_fraction": 0.0,
        }

    xy = np.stack([wp.state.position_xy for wp in traj.waypoints], axis=0)
    if xy.shape[0] >= 2:
        path_length = float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))
    else:
        path_length = 0.0

    alt0 = float(traj.waypoints[0].state.altitude)
    alt1 = float(traj.waypoints[-1].state.altitude)
    altitude_loss = float(max(alt0 - alt1, 0.0))

    rolls = []
    yaw_rates = []
    for wp in traj.waypoints:
        roll, _, _ = quat_to_rpy(wp.state.q_IB)
        rolls.append(abs(float(roll)))
        yaw_rates.append(abs(float(wp.state.w_B[2])))

    max_bank = float(np.rad2deg(max(rolls))) if rolls else 0.0
    max_yaw_rate = float(np.rad2deg(max(yaw_rates))) if yaw_rates else 0.0

    max_brake = 0.0
    max_delta_a = 0.0
    if traj.controls:
        brakes = [(c.delta_L + c.delta_R) * 0.5 for c in traj.controls]
        delta_a = [abs(c.delta_L - c.delta_R) for c in traj.controls]
        max_brake = float(max(brakes)) if brakes else 0.0
        max_delta_a = float(max(delta_a)) if delta_a else 0.0

    dpsi = _heading_deltas(xy)
    total_turn = float(np.sum(np.abs(dpsi))) if dpsi.size else 0.0
    net_turn = float(np.sum(dpsi)) if dpsi.size else 0.0
    sign_changes, clusters, dominant_fraction = _turn_sign_changes(dpsi, eps_rad=float(np.deg2rad(turn_eps_deg)))

    return {
        "duration_s": float(traj.duration),
        "altitude_loss_m": altitude_loss,
        "path_length_m": path_length,
        "max_bank_deg": max_bank,
        "max_yaw_rate_deg_s": max_yaw_rate,
        "max_brake": max_brake,
        "max_delta_a": max_delta_a,
        "turn_total_rad": total_turn,
        "turn_net_rad": net_turn,
        "turn_total_turns": float(total_turn / (2.0 * np.pi)) if total_turn > 0.0 else 0.0,
        "turn_net_turns": float(net_turn / (2.0 * np.pi)) if abs(net_turn) > 0.0 else 0.0,
        "turn_sign_changes": float(sign_changes),
        "turn_clusters": float(clusters),
        "turn_dominant_fraction": float(dominant_fraction),
    }
