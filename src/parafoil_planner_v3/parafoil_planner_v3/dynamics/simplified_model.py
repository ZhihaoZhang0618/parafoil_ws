from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from parafoil_dynamics.math3d import normalize_quaternion, quat_derivative

from parafoil_planner_v3.types import Control, State, Wind

from .aerodynamics import PolarTable


@dataclass(frozen=True)
class SimplifiedGlideModel:
    """
    Lightweight kinematic model for quick planning / initial guess.

    State (NED):
      p_I (3,), heading (rad), altitude = -p_D

    Control:
      symmetric brake b in [0,1] and yaw_rate_cmd (rad/s).
    """

    polar: PolarTable = PolarTable()
    turn_rate_per_delta: float = 1.70  # rad/s per delta_a (approx, from README)

    def speed_sink(self, brake_sym: float) -> Tuple[float, float]:
        return self.polar.interpolate(brake_sym)

    def yaw_rate_from_delta_a(self, delta_a: float) -> float:
        # Convention: delta_a = left - right; left brake -> negative yaw (turn left).
        return float(-self.turn_rate_per_delta * float(delta_a))


def _yaw_from_quat_wxyz(q_wxyz: np.ndarray) -> float:
    """
    Yaw angle (rad) from quaternion [w,x,y,z] using ZYX convention.

    For planning we only need yaw; roll/pitch are handled elsewhere.
    """
    q = normalize_quaternion(np.asarray(q_wxyz, dtype=float).reshape(4))
    w, x, y, z = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    # ZYX yaw
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def yaw_only_quat_wxyz(yaw_rad: float) -> np.ndarray:
    yaw = float(yaw_rad)
    return np.array([np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)], dtype=float)


@dataclass(frozen=True)
class KinematicYawGlideDynamics:
    """
    Fast kinematic-ish dynamics for offline library generation and quick warm-starts.

    - Uses polar table to map symmetric brake -> (airspeed, sink)
    - Uses README yaw-rate fit to map delta_a -> yaw rate
    - Approximates v and w as 1st-order tracking of commanded values

    State (13D, NED):
      x = [p(3), v(3), q(4), w(3)]
    Control:
      u = [delta_L, delta_R] in [0,1]
    """

    polar: PolarTable = PolarTable()
    turn_rate_per_delta: float = 1.70  # rad/s per (delta_L - delta_R)
    tau_v: float = 0.5  # seconds (velocity tracking)
    tau_w: float = 0.3  # seconds (yaw-rate tracking)

    def yaw_rate_from_delta_a(self, delta_a: float) -> float:
        # Convention from simulator README: yaw_rate ≈ -1.7 * delta_a (rad/s)
        return float(-self.turn_rate_per_delta * float(delta_a))

    def f_vector(self, x: np.ndarray, u: np.ndarray, t: float, wind_I: Optional[np.ndarray] = None) -> np.ndarray:
        _ = float(t)  # unused, but kept for interface compatibility
        x = np.asarray(x, dtype=float).reshape(13)
        u = np.asarray(u, dtype=float).reshape(2)
        wind = np.zeros(3, dtype=float) if wind_I is None else np.asarray(wind_I, dtype=float).reshape(3)

        p = x[0:3]
        v = x[3:6]
        q = x[6:10]
        w = x[10:13]

        delta_L = float(u[0])
        delta_R = float(u[1])
        brake_sym = 0.5 * (delta_L + delta_R)
        delta_a = delta_L - delta_R

        V, sink = self.polar.interpolate(brake_sym)
        yaw = _yaw_from_quat_wxyz(q)

        v_air_cmd = np.array([V * np.cos(yaw), V * np.sin(yaw), float(sink)], dtype=float)
        v_cmd = v_air_cmd + wind

        r_cmd = self.yaw_rate_from_delta_a(delta_a)
        w_cmd = np.array([0.0, 0.0, float(r_cmd)], dtype=float)

        tau_v = float(max(self.tau_v, 1e-3))
        tau_w = float(max(self.tau_w, 1e-3))

        p_dot = v
        v_dot = (v_cmd - v) / tau_v
        q_dot = quat_derivative(normalize_quaternion(q), w)
        w_dot = (w_cmd - w) / tau_w

        return np.concatenate([p_dot, v_dot, q_dot, w_dot], axis=0).astype(float).reshape(13)

    def step(self, state: State, control: Control, wind: Wind, dt: float) -> State:
        x = state.to_vector()
        x_dot = self.f_vector(x, control.clipped().as_array, t=float(state.t), wind_I=wind.v_I if wind is not None else None)
        x_next = x + float(dt) * x_dot
        # Normalize quaternion for numerical hygiene
        x_next[6:10] = normalize_quaternion(x_next[6:10])
        return State.from_vector(x_next, t=float(state.t + dt))
