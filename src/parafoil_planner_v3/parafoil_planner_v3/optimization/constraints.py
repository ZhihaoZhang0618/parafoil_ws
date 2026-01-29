from __future__ import annotations

import numpy as np

from parafoil_planner_v3.utils.quaternion_utils import quat_to_rpy


def horizontal_speed_constraints(v_xy: np.ndarray, Vmin: float, Vmax: float) -> np.ndarray:
    Vh = float(np.linalg.norm(np.asarray(v_xy, dtype=float).reshape(2)))
    return np.array([Vh - float(Vmin), float(Vmax) - Vh], dtype=float)


def roll_constraint(q_wxyz: np.ndarray, roll_max_rad: float) -> float:
    roll, _, _ = quat_to_rpy(np.asarray(q_wxyz, dtype=float).reshape(4))
    return float(float(roll_max_rad) - abs(float(roll)))


def yaw_rate_constraint(w_z: float, yaw_rate_max: float) -> float:
    return float(float(yaw_rate_max) - abs(float(w_z)))

