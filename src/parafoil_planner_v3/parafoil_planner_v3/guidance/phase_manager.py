from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from parafoil_planner_v3.types import GuidancePhase, PhaseTransition, State, Target, Wind


@dataclass(frozen=True)
class PhaseConfig:
    approach_entry_distance: float = 180.0  # m
    altitude_margin: float = 1.1
    glide_ratio_nominal: float = 3.0  # conservative L/D proxy
    approach_altitude_extra: float = 5.0  # m

    flare_altitude: float = 10.0  # m AGL
    flare_distance: float = 30.0  # m

    # Abort logic (safety)
    abort_min_altitude_m: float = 2.0
    abort_min_altitude_distance_m: float = 60.0
    abort_max_glide_ratio: float = 10.0
    abort_max_time_s: float = 300.0
    abort_max_phase_time_s: float = 180.0


class PhaseManager:
    def __init__(self, config: PhaseConfig | None = None) -> None:
        self.config = config or PhaseConfig()
        self.current_phase = GuidancePhase.CRUISE
        self.phase_start_time: float | None = None
        self.phase_start_state: State | None = None

    def _transition_to(self, to_phase: GuidancePhase, state: State, reason: str) -> PhaseTransition:
        from_phase = self.current_phase
        self.current_phase = to_phase
        self.phase_start_time = float(state.t)
        self.phase_start_state = state.copy()
        return PhaseTransition(from_phase=from_phase, to_phase=to_phase, triggered=True, reason=reason)

    def update(self, state: State, target: Target, wind: Wind) -> PhaseTransition:
        transition = PhaseTransition(
            from_phase=self.current_phase,
            to_phase=self.current_phase,
            triggered=False,
            reason="",
        )

        if self.current_phase not in (GuidancePhase.LANDED, GuidancePhase.ABORT):
            should_abort, reason = self._should_abort(state, target, wind)
            if should_abort:
                return self._transition_to(GuidancePhase.ABORT, state, reason=reason)

        if self.current_phase == GuidancePhase.CRUISE:
            if self._should_enter_approach(state, target, wind):
                return self._transition_to(GuidancePhase.APPROACH, state, reason="entry_conditions_met")

        elif self.current_phase == GuidancePhase.APPROACH:
            if self._should_enter_flare(state, target):
                return self._transition_to(GuidancePhase.FLARE, state, reason="flare_trigger")

        elif self.current_phase == GuidancePhase.FLARE:
            if self._has_landed(state, target):
                return self._transition_to(GuidancePhase.LANDED, state, reason="touchdown")

        elif self.current_phase == GuidancePhase.ABORT:
            if self._has_landed(state, target):
                return self._transition_to(GuidancePhase.LANDED, state, reason="touchdown_abort")

        return transition

    def _estimate_approach_altitude(self, distance_to_target: float, wind: Wind) -> float:
        # Conservative altitude requirement based on nominal glide ratio.
        # This ignores wind; wind can be folded into glide_ratio_nominal if needed.
        req = float(distance_to_target / max(self.config.glide_ratio_nominal, 1e-3))
        return req + float(self.config.approach_altitude_extra)

    def _should_enter_approach(self, state: State, target: Target, wind: Wind) -> bool:
        distance_to_target = float(np.linalg.norm(state.position_xy - target.position_xy))
        altitude_agl = float(state.altitude - target.altitude)

        distance_ok = distance_to_target <= float(self.config.approach_entry_distance)
        required_altitude = self._estimate_approach_altitude(distance_to_target, wind)
        altitude_ok = altitude_agl <= required_altitude * float(self.config.altitude_margin)
        return bool(distance_ok and altitude_ok)

    def _should_enter_flare(self, state: State, target: Target) -> bool:
        altitude_agl = float(state.altitude - target.altitude)
        distance = float(np.linalg.norm(state.position_xy - target.position_xy))

        altitude_trigger = altitude_agl <= float(self.config.flare_altitude)
        distance_trigger = distance <= float(self.config.flare_distance)
        return bool(altitude_trigger or distance_trigger)

    def _should_abort(self, state: State, target: Target, wind: Wind) -> tuple[bool, str]:
        if self.current_phase == GuidancePhase.FLARE:
            return False, ""
        distance = float(np.linalg.norm(state.position_xy - target.position_xy))
        altitude_agl = float(state.altitude - target.altitude)

        if altitude_agl <= float(self.config.abort_min_altitude_m) and distance > float(self.config.abort_min_altitude_distance_m):
            return True, "low_altitude_far"

        if altitude_agl > 1e-3:
            required_glide = float(distance / altitude_agl)
            if required_glide > float(self.config.abort_max_glide_ratio):
                return True, "glide_ratio_exceeded"

        if self.phase_start_time is not None:
            if float(state.t - self.phase_start_time) > float(self.config.abort_max_phase_time_s):
                return True, "phase_timeout"

        if float(self.config.abort_max_time_s) > 0.0 and float(state.t) > float(self.config.abort_max_time_s):
            return True, "mission_timeout"

        return False, ""

    @staticmethod
    def _has_landed(state: State, target: Target) -> bool:
        # NED down: landing when down >= target_down (often 0).
        return bool(state.p_I[2] >= target.p_I[2])
