from __future__ import annotations

import numpy as np


def terminal_position_cost(p_final: np.ndarray, p_target: np.ndarray, weight: float) -> float:
    p_final = np.asarray(p_final, dtype=float).reshape(3)
    p_target = np.asarray(p_target, dtype=float).reshape(3)
    e = p_final - p_target
    return float(weight * float(np.dot(e, e)))


def terminal_velocity_cost(v_final: np.ndarray, weight: float) -> float:
    v_final = np.asarray(v_final, dtype=float).reshape(3)
    return float(weight * float(np.dot(v_final, v_final)))


def control_effort_cost(u: np.ndarray, weight: float) -> float:
    u = np.asarray(u, dtype=float).reshape(-1)
    return float(weight * float(np.dot(u, u)))

