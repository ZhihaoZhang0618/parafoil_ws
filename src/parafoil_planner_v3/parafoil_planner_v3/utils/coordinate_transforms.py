from __future__ import annotations

import numpy as np


def ned_to_enu(p_ned: np.ndarray) -> np.ndarray:
    """Convert a vector from NED to ENU by component swap/sign flip."""
    p_ned = np.asarray(p_ned, dtype=float).reshape(3)
    # NED: [N, E, D] -> ENU: [E, N, U]
    return np.array([p_ned[1], p_ned[0], -p_ned[2]], dtype=float)


def enu_to_ned(p_enu: np.ndarray) -> np.ndarray:
    p_enu = np.asarray(p_enu, dtype=float).reshape(3)
    # ENU: [E, N, U] -> NED: [N, E, D]
    return np.array([p_enu[1], p_enu[0], -p_enu[2]], dtype=float)


def ned_to_enu_xy(p_ned_xy: np.ndarray) -> np.ndarray:
    p_ned_xy = np.asarray(p_ned_xy, dtype=float).reshape(2)
    return np.array([p_ned_xy[1], p_ned_xy[0]], dtype=float)


def enu_to_ned_xy(p_enu_xy: np.ndarray) -> np.ndarray:
    p_enu_xy = np.asarray(p_enu_xy, dtype=float).reshape(2)
    return np.array([p_enu_xy[1], p_enu_xy[0]], dtype=float)

