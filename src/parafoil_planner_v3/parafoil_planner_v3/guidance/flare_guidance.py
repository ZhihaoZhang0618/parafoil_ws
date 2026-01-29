from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from parafoil_planner_v3.types import Control, State, Target, Wind

from .control_laws import LateralControlConfig, track_point_control


@dataclass(frozen=True)
class FlareGuidanceConfig:
    mode: str = "spec_full_brake"  # spec_full_brake|touchdown_brake
    flare_initial_brake: float = 0.2
    flare_max_brake: float = 1.0
    flare_ramp_time: float = 0.5  # s
    flare_full_brake_duration_s: float = 3.0  # 2~5s full brake per spec
    touchdown_brake_altitude_m: float = 0.2
    lateral: LateralControlConfig = LateralControlConfig(max_brake=1.0)


class FlareGuidance:
    def __init__(self, config: FlareGuidanceConfig | None = None) -> None:
        self.config = config or FlareGuidanceConfig()
        self._flare_start_time: float | None = None
        self._touchdown_brake_start_time: float | None = None

    def on_enter(self, state: State) -> None:
        self._flare_start_time = float(state.t)
        self._touchdown_brake_start_time = None

    def _ramp_up_brake(self, t_since: float) -> float:
        if self.config.flare_ramp_time <= 1e-6:
            return float(self.config.flare_max_brake)
        progress = float(np.clip(t_since / float(self.config.flare_ramp_time), 0.0, 1.0))
        smooth = 3.0 * progress**2 - 2.0 * progress**3  # smoothstep
        return float(
            float(self.config.flare_initial_brake)
            + (float(self.config.flare_max_brake) - float(self.config.flare_initial_brake)) * smooth
        )

    def brake_command(self, state: State, target: Target) -> float:
        """
        Flare brake schedule.

        Two modes:
          - spec_full_brake: ramp to full brake and hold for 2~5s
          - touchdown_brake: mild brake until near-ground, then ramp to full brake
        """
        altitude_agl = float(state.altitude - target.altitude)
        mode = str(self.config.mode).strip().lower()

        if mode == "spec_full_brake":
            if self._flare_start_time is None:
                self._flare_start_time = float(state.t)
            t_since = float(state.t - self._flare_start_time)
            if t_since <= float(self.config.flare_full_brake_duration_s):
                return float(np.clip(self._ramp_up_brake(t_since), 0.0, 1.0))
            return float(np.clip(self.config.flare_max_brake, 0.0, 1.0))

        if altitude_agl > float(self.config.touchdown_brake_altitude_m):
            self._touchdown_brake_start_time = None
            return float(np.clip(self.config.flare_initial_brake, 0.0, 1.0))

        if self._touchdown_brake_start_time is None:
            self._touchdown_brake_start_time = float(state.t)

        t_since = float(state.t - self._touchdown_brake_start_time)
        return float(np.clip(self._ramp_up_brake(t_since), 0.0, 1.0))

    def compute_control(self, state: State, target: Target, wind: Wind, dt: float) -> Control:
        if self._flare_start_time is None:
            self._flare_start_time = float(state.t)
        brake = self.brake_command(state, target)

        # Keep some heading correction during flare to reduce lateral drift.
        return track_point_control(state, target.position_xy, brake_sym=brake, cfg=self.config.lateral)
