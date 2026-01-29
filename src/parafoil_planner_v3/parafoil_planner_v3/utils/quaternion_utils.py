from __future__ import annotations

import numpy as np

from parafoil_dynamics.math3d import normalize_quaternion, quat_to_euler


def quat_normalize(q_wxyz: np.ndarray) -> np.ndarray:
    return normalize_quaternion(np.asarray(q_wxyz, dtype=float).reshape(4))


def quat_to_rpy(q_wxyz: np.ndarray) -> tuple[float, float, float]:
    q = quat_normalize(q_wxyz)
    roll, pitch, yaw = quat_to_euler(q)
    return float(roll), float(pitch), float(yaw)


def wrap_pi(angle_rad: float) -> float:
    return float((angle_rad + np.pi) % (2.0 * np.pi) - np.pi)

