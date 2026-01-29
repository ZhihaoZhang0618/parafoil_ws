from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from parafoil_planner_v3.types import Control, State
from parafoil_planner_v3.utils.quaternion_utils import quat_to_rpy, wrap_pi


@dataclass(frozen=True)
class LateralControlConfig:
    turn_rate_per_delta: float = 1.70  # rad/s per (delta_L - delta_R)
    K_heading: float = 1.0
    yaw_rate_max: float = np.deg2rad(90.0)
    max_delta_a: float = 0.6
    max_brake: float = 1.0


def heading_rad(state: State) -> float:
    v_xy = state.v_I[:2]
    speed_xy = float(np.linalg.norm(v_xy))
    if speed_xy > 0.3:
        return float(np.arctan2(v_xy[1], v_xy[0]))  # atan2(E, N)
    _, _, yaw = quat_to_rpy(state.q_IB)
    return float(yaw)


def track_point_control(
    state: State,
    target_xy: np.ndarray,
    brake_sym: float,
    cfg: LateralControlConfig,
) -> Control:
    target_xy = np.asarray(target_xy, dtype=float).reshape(2)
    p = state.position_xy
    desired_heading = float(np.arctan2(target_xy[1] - p[1], target_xy[0] - p[0]))
    err = wrap_pi(desired_heading - heading_rad(state))

    yaw_rate_cmd = float(np.clip(cfg.K_heading * err, -cfg.yaw_rate_max, cfg.yaw_rate_max))

    # yaw_rate ≈ -k * (delta_L - delta_R)
    delta_a = float(np.clip(-yaw_rate_cmd / max(cfg.turn_rate_per_delta, 1e-6), -cfg.max_delta_a, cfg.max_delta_a))

    b = float(np.clip(brake_sym, 0.0, cfg.max_brake))
    left = float(np.clip(b + 0.5 * delta_a, 0.0, cfg.max_brake))
    right = float(np.clip(b - 0.5 * delta_a, 0.0, cfg.max_brake))
    return Control(delta_L=left, delta_R=right)

