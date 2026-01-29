from __future__ import annotations

import numpy as np

from parafoil_planner_v3.types import State, Target, Wind
from parafoil_planner_v3.utils.quaternion_utils import wrap_pi


def compute_scenario_features(state: State, target: Target, wind: Wind) -> np.ndarray:
    """
    Feature vector for library matching (5D):
      [altitude, distance_to_target, bearing_to_target, wind_speed, relative_wind_angle]
    """
    rel = target.position_xy - state.position_xy
    distance = float(np.linalg.norm(rel))
    bearing = float(np.arctan2(rel[1], rel[0]))  # atan2(E, N)

    wind_xy = wind.v_I[:2]
    wind_speed = float(np.linalg.norm(wind_xy))
    wind_angle = float(np.arctan2(wind_xy[1], wind_xy[0]))
    relative_wind = wrap_pi(wind_angle - bearing)

    return np.array([state.altitude, distance, bearing, wind_speed, relative_wind], dtype=float)

