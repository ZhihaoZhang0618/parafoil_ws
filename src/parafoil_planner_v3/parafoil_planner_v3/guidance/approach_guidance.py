from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from parafoil_planner_v3.dynamics.aerodynamics import PolarTable
from parafoil_planner_v3.types import Control, State, Target, Wind

from .control_laws import LateralControlConfig, track_point_control


@dataclass(frozen=True)
class ApproachGuidanceConfig:
    brake_min: float = 0.0
    brake_max: float = 0.6
    wind_correction_gain: float = 1.0
    wind_max_drift_m: float = 80.0
    min_ground_speed: float = 0.5
    strong_wind_ratio: float = 1.05  # wind_speed / airspeed trigger
    strong_wind_strategy: str = "wind_line_projection"  # wind_line_projection|downwind_bias|target|ground_speed
    strong_wind_downwind_bias_m: float = 40.0
    ground_speed_lookahead_m: float = 40.0
    ground_speed_min_time_s: float = 5.0
    ground_speed_max_time_s: float = 120.0
    lateral: LateralControlConfig = LateralControlConfig()


class ApproachGuidance:
    def __init__(self, config: ApproachGuidanceConfig | None = None) -> None:
        self.config = config or ApproachGuidanceConfig()
        self.polar = PolarTable()

    def _required_slope(self, state: State, target: Target) -> float:
        alt_agl = float(state.altitude - target.altitude)
        dist = float(np.linalg.norm(target.position_xy - state.position_xy))
        if dist < 1e-3:
            return 0.0
        return float(max(alt_agl, 0.0) / dist)

    def _is_strong_wind(self, wind: Wind, brake: float) -> bool:
        wind_speed = float(np.linalg.norm(wind.v_I[:2]))
        V_air, _ = self.polar.interpolate(float(brake))
        if V_air <= 1e-6:
            return False
        ratio = wind_speed / float(V_air)
        return bool(ratio >= float(self.config.strong_wind_ratio))

    def _strong_wind_aimpoint(self, state: State, target: Target, wind: Wind) -> np.ndarray:
        wind_xy = wind.v_I[:2]
        n = float(np.linalg.norm(wind_xy))
        if n < 1e-6:
            return target.position_xy.copy()
        wind_hat = wind_xy / n
        mode = str(self.config.strong_wind_strategy).strip().lower()
        if mode == "downwind_bias":
            return (target.position_xy + wind_hat * float(self.config.strong_wind_downwind_bias_m)).astype(float)
        if mode == "target":
            return target.position_xy.copy()
        # Default: project target onto wind line through current position to minimize crosswind error.
        d = target.position_xy - state.position_xy
        s = float(np.dot(d, wind_hat))
        return (state.position_xy + wind_hat * s).astype(float)

    def _select_brake_strong_wind(self, wind: Wind) -> float:
        wind_speed = float(np.linalg.norm(wind.v_I[:2]))
        best_b = float(self.config.brake_min)
        best_metric = float("inf")
        for b in self.polar.brake:
            V_air, sink = self.polar.interpolate(float(b))
            sink = max(float(sink), 1e-3)
            metric = max(wind_speed - float(V_air), 0.0) / sink
            if metric < best_metric:
                best_metric = metric
                best_b = float(b)
        return float(np.clip(best_b, self.config.brake_min, self.config.brake_max))

    def _ground_speed_aimpoint(self, state: State, target: Target, wind: Wind, brake: float) -> np.ndarray:
        wind_xy = wind.v_I[:2]
        dist = float(np.linalg.norm(target.position_xy - state.position_xy))
        if dist < 1e-3:
            return target.position_xy.copy()
        V_air, sink = self.polar.interpolate(float(brake))
        alt_agl = float(max(state.altitude - target.altitude, 0.0))
        t_desc = alt_agl / max(float(sink), 1e-3)
        t_desc = float(np.clip(t_desc, self.config.ground_speed_min_time_s, self.config.ground_speed_max_time_s))
        v_req = (target.position_xy - state.position_xy) / max(t_desc, 1e-3)
        air_req = v_req - wind_xy
        n = float(np.linalg.norm(air_req))
        if n < 1e-6:
            heading_hat = np.array([1.0, 0.0], dtype=float)
        else:
            heading_hat = air_req / n
        lookahead = float(min(dist, max(self.config.ground_speed_lookahead_m, 1.0)))
        return (state.position_xy + heading_hat * lookahead).astype(float)

    def compute_aimpoint(self, state: State, target: Target, wind: Wind, brake: float) -> np.ndarray:
        wind_xy = wind.v_I[:2]
        dist = float(np.linalg.norm(target.position_xy - state.position_xy))
        if dist < 1e-3 or float(np.linalg.norm(wind_xy)) < 1e-6:
            return target.position_xy.copy()

        V_air, _ = self.polar.interpolate(float(brake))
        dir_hat = (target.position_xy - state.position_xy) / max(dist, 1e-6)
        v_ground = float(V_air + np.dot(wind_xy, dir_hat))
        strong_wind = self._is_strong_wind(wind, brake) or v_ground <= float(self.config.min_ground_speed)
        if strong_wind:
            mode = str(self.config.strong_wind_strategy).strip().lower()
            if mode == "ground_speed":
                return self._ground_speed_aimpoint(state, target, wind, brake)
            return self._strong_wind_aimpoint(state, target, wind)

        v_ground = max(v_ground, float(self.config.min_ground_speed))
        t_go = float(dist / v_ground)
        drift = wind_xy * t_go
        drift_mag = float(np.linalg.norm(drift))
        if drift_mag > float(self.config.wind_max_drift_m):
            drift = drift * (float(self.config.wind_max_drift_m) / max(drift_mag, 1e-6))
        return (target.position_xy - float(self.config.wind_correction_gain) * drift).astype(float)

    def should_override_path(self, state: State, target: Target, wind: Wind, brake: float) -> bool:
        wind_xy = wind.v_I[:2]
        dist = float(np.linalg.norm(target.position_xy - state.position_xy))
        if dist < 1e-3:
            return True
        V_air, _ = self.polar.interpolate(float(brake))
        dir_hat = (target.position_xy - state.position_xy) / max(dist, 1e-6)
        v_ground = float(V_air + np.dot(wind_xy, dir_hat))
        return bool(self._is_strong_wind(wind, brake) or v_ground <= float(self.config.min_ground_speed))

    def compute_control(self, state: State, target: Target, wind: Wind, dt: float) -> Control:
        k_req = self._required_slope(state, target)
        b = self.polar.select_brake_for_required_slope(k_req)
        b = float(np.clip(b, self.config.brake_min, self.config.brake_max))
        if self._is_strong_wind(wind, b):
            b = self._select_brake_strong_wind(wind)
        aim = self.compute_aimpoint(state, target, wind, brake=b)
        return track_point_control(state, aim, brake_sym=b, cfg=self.config.lateral)
