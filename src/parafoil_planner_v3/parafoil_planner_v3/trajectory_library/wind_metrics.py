from __future__ import annotations

from typing import Dict

import numpy as np

from parafoil_planner_v3.types import Scenario, Trajectory


def compute_wind_drift_metrics(traj: Trajectory, *, scenario: Scenario, wind_I: np.ndarray) -> Dict[str, float]:
    """
    Wind-related metrics for debugging strong-wind / backward-drift behavior.

    Notes:
    - Assumes wind_I is constant and uses internal standard NED + wind-to.
    - Uses v_air ≈ v_ground - wind as an approximation of air-relative velocity.
    """
    if not traj.waypoints:
        return {
            "v_g_along_to_target_min_mps": float("nan"),
            "v_g_along_to_target_max_mps": float("nan"),
            "v_g_along_to_target_mean_mps": float("nan"),
            "v_g_along_wind_min_mps": float("nan"),
            "v_g_along_wind_max_mps": float("nan"),
            "v_g_along_wind_mean_mps": float("nan"),
            "backward_drift_fraction": 0.0,
        }

    wind_I = np.asarray(wind_I, dtype=float).reshape(3)
    wind_xy = wind_I[:2]
    wind_speed = float(np.linalg.norm(wind_xy))
    wind_hat = wind_xy / wind_speed if wind_speed > 1e-9 else None

    bearing = float(np.deg2rad(scenario.target_bearing_deg))
    d_hat = np.array([np.cos(bearing), np.sin(bearing)], dtype=float)

    # Time-weighted stats across segments.
    v_to = []
    v_wind = []
    backward_time = 0.0
    total_time = 0.0

    wps = traj.waypoints
    for i in range(len(wps) - 1):
        t0 = float(wps[i].t)
        t1 = float(wps[i + 1].t)
        dt = float(max(t1 - t0, 0.0))
        if dt <= 1e-9:
            continue

        v_g = np.asarray(wps[i].state.v_I[:2], dtype=float)
        v_air = v_g - wind_xy
        v_air_norm = float(np.linalg.norm(v_air))
        if v_air_norm > 1e-9:
            v_air_hat = v_air / v_air_norm
            if float(np.dot(v_g, v_air_hat)) < 0.0:
                backward_time += dt

        v_to.append((float(np.dot(v_g, d_hat)), dt))
        if wind_hat is not None:
            v_wind.append((float(np.dot(v_g, wind_hat)), dt))
        total_time += dt

    def _weighted_stats(samples: list[tuple[float, float]]) -> tuple[float, float, float]:
        if not samples:
            return float("nan"), float("nan"), float("nan")
        vals = np.array([s[0] for s in samples], dtype=float)
        w = np.array([s[1] for s in samples], dtype=float)
        wsum = float(np.sum(w))
        mean = float(np.sum(vals * w) / wsum) if wsum > 1e-9 else float(np.mean(vals))
        return float(np.min(vals)), float(np.max(vals)), float(mean)

    to_min, to_max, to_mean = _weighted_stats(v_to)
    w_min, w_max, w_mean = _weighted_stats(v_wind)

    frac = 0.0 if total_time <= 1e-9 else float(backward_time / total_time)
    return {
        "v_g_along_to_target_min_mps": float(to_min),
        "v_g_along_to_target_max_mps": float(to_max),
        "v_g_along_to_target_mean_mps": float(to_mean),
        "v_g_along_wind_min_mps": float(w_min),
        "v_g_along_wind_max_mps": float(w_max),
        "v_g_along_wind_mean_mps": float(w_mean),
        "backward_drift_fraction": float(frac),
    }

