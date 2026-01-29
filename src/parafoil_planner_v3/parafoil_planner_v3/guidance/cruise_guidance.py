from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from parafoil_planner_v3.dynamics.aerodynamics import PolarTable
from parafoil_planner_v3.types import Control, State, Target, Wind

from .control_laws import LateralControlConfig, track_point_control


@dataclass(frozen=True)
class CruiseGuidanceConfig:
    final_leg_length: float = 180.0  # m (entry point upwind distance)
    hold_threshold_alt_excess: float = 25.0  # m
    racetrack_threshold_alt_excess: float = 60.0  # m
    hold_radius: float = 40.0  # m (circle radius)
    hold_direction: int = 1  # +1 CCW, -1 CW (in NED yaw convention)
    racetrack_length: float = 120.0  # m (distance between turn centers)
    racetrack_radius: float = 40.0  # m
    s_turn_amplitude: float = 45.0  # m (lateral swing)
    s_turn_period_s: float = 30.0  # s
    strategy: str = "auto"  # auto|circle|s_turn|racetrack|direct
    brake_cruise: float = 0.2
    lateral: LateralControlConfig = LateralControlConfig()
    glide_ratio_nominal: float = 3.0
    strong_wind_ratio: float = 1.05  # wind_speed / airspeed trigger
    strong_wind_entry_mode: str = "downwind"  # downwind|upwind|adaptive|target
    strong_wind_force_direct: bool = True


class CruiseGuidance:
    def __init__(self, config: CruiseGuidanceConfig | None = None) -> None:
        self.config = config or CruiseGuidanceConfig()
        self.polar = PolarTable()

    def _entry_point_xy(self, target: Target, wind: Wind) -> np.ndarray:
        # Approach into wind: entry is upwind of target by final_leg_length.
        # Wind vector uses wind-to convention (points where wind goes), so upwind is -wind_hat.
        wind_hat = wind.direction_hat_xy
        return target.position_xy - wind_hat * float(self.config.final_leg_length)

    def _entry_point_xy_strong_wind(self, state: State, target: Target, wind: Wind) -> np.ndarray:
        wind_hat = wind.direction_hat_xy
        mode = str(self.config.strong_wind_entry_mode).strip().lower()
        if mode == "target":
            return target.position_xy.copy()
        if mode == "upwind":
            return target.position_xy - wind_hat * float(self.config.final_leg_length)
        if mode == "downwind":
            return target.position_xy + wind_hat * float(self.config.final_leg_length)
        # adaptive: if target is downwind of current state, aim upwind of target to reduce drift; else go downwind.
        d = target.position_xy - state.position_xy
        if float(np.dot(d, wind_hat)) >= 0.0:
            return target.position_xy - wind_hat * float(self.config.final_leg_length)
        return target.position_xy + wind_hat * float(self.config.final_leg_length)

    def _entry_axis_hat(self, state: State, target: Target, wind: Wind) -> np.ndarray:
        wind_xy = wind.v_I[:2]
        if float(np.linalg.norm(wind_xy)) > 1e-6:
            return wind.direction_hat_xy
        # Fallback: axis from state to target
        d = target.position_xy - state.position_xy
        n = float(np.linalg.norm(d))
        if n < 1e-6:
            return np.array([1.0, 0.0], dtype=float)
        return (d / n).astype(float)

    def _required_altitude(self, distance_to_entry: float) -> float:
        return float(distance_to_entry / max(self.config.glide_ratio_nominal, 1e-3))

    def _is_strong_wind(self, wind: Wind) -> bool:
        wind_speed = float(np.linalg.norm(wind.v_I[:2]))
        V_air, _ = self.polar.interpolate(float(self.config.brake_cruise))
        if V_air <= 1e-6:
            return False
        ratio = wind_speed / float(V_air)
        return bool(ratio >= float(self.config.strong_wind_ratio))

    def _altitude_excess(self, state: State, target: Target, wind: Wind) -> float:
        entry = self._entry_point_xy(target, wind)
        dist = float(np.linalg.norm(entry - state.position_xy))
        alt_agl = float(state.altitude - target.altitude)
        return alt_agl - self._required_altitude(dist)

    def _hold_track_point(self, state: State, center_xy: np.ndarray, radius: float | None = None, direction: int | None = None) -> np.ndarray:
        # Pick a point tangentially ahead on a circle around center.
        p = state.position_xy
        r = p - center_xy
        norm = float(np.linalg.norm(r))
        if norm < 1e-6:
            r = np.array([radius or self.config.hold_radius, 0.0], dtype=float)
            norm = float(np.linalg.norm(r))
        r_hat = r / norm
        # Rotate +/- 90deg to get tangent direction, then step ahead.
        turn_dir = self.config.hold_direction if direction is None else int(direction)
        if turn_dir >= 0:
            t_hat = np.array([-r_hat[1], r_hat[0]], dtype=float)
        else:
            t_hat = np.array([r_hat[1], -r_hat[0]], dtype=float)
        use_radius = float(radius if radius is not None else self.config.hold_radius)
        return center_xy + r_hat * use_radius + t_hat * float(use_radius * 0.5)

    def _s_turn_track_point(self, state: State, entry_xy: np.ndarray, wind: Wind) -> np.ndarray:
        axis = self._entry_axis_hat(state, Target(p_I=np.array([entry_xy[0], entry_xy[1], 0.0])), wind)
        perp = np.array([-axis[1], axis[0]], dtype=float)
        period = float(max(self.config.s_turn_period_s, 1e-3))
        phase = 2.0 * np.pi * float(state.t) / period
        offset = float(self.config.s_turn_amplitude) * float(np.sin(phase))
        return entry_xy + perp * offset

    def _racetrack_track_point(self, state: State, entry_xy: np.ndarray, wind: Wind) -> np.ndarray:
        axis = self._entry_axis_hat(state, Target(p_I=np.array([entry_xy[0], entry_xy[1], 0.0])), wind)
        half = float(self.config.racetrack_length) * 0.5
        center_a = entry_xy + axis * half
        center_b = entry_xy - axis * half
        da = float(np.linalg.norm(state.position_xy - center_a))
        db = float(np.linalg.norm(state.position_xy - center_b))
        center = center_a if da <= db else center_b
        return self._hold_track_point(state, center, radius=float(self.config.racetrack_radius))

    def _select_strategy(self, alt_excess: float) -> str:
        strategy = str(self.config.strategy).strip().lower()
        if strategy in {"circle", "s_turn", "racetrack", "direct"}:
            return strategy
        # auto
        if alt_excess > float(self.config.racetrack_threshold_alt_excess):
            return "racetrack"
        if alt_excess > float(self.config.hold_threshold_alt_excess):
            return "s_turn"
        return "direct"

    def should_override_path(self, wind: Wind) -> bool:
        return bool(self._is_strong_wind(wind))

    def compute_control(self, state: State, target: Target, wind: Wind, dt: float) -> Control:
        alt_excess = self._altitude_excess(state, target, wind)
        strong_wind = self._is_strong_wind(wind)
        entry_xy = self._entry_point_xy_strong_wind(state, target, wind) if strong_wind else self._entry_point_xy(target, wind)

        strategy = self._select_strategy(alt_excess)
        if strong_wind and bool(self.config.strong_wind_force_direct):
            strategy = "direct"
        if strategy == "racetrack":
            track_xy = self._racetrack_track_point(state, entry_xy, wind)
        elif strategy == "s_turn":
            track_xy = self._s_turn_track_point(state, entry_xy, wind)
        elif strategy == "circle":
            track_xy = self._hold_track_point(state, entry_xy)
        else:
            track_xy = entry_xy

        return track_point_control(state, track_xy, brake_sym=float(self.config.brake_cruise), cfg=self.config.lateral)
