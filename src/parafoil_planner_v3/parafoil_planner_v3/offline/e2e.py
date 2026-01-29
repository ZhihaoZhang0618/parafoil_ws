from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from parafoil_planner_v3.dynamics.aerodynamics import PolarTable
from parafoil_planner_v3.dynamics.parafoil_6dof import SixDOFDynamics
from parafoil_planner_v3.dynamics.simplified_model import KinematicYawGlideDynamics, yaw_only_quat_wxyz
from parafoil_planner_v3.guidance.control_laws import LateralControlConfig, track_point_control
from parafoil_planner_v3.guidance.cruise_guidance import CruiseGuidance, CruiseGuidanceConfig
from parafoil_planner_v3.guidance.approach_guidance import ApproachGuidance, ApproachGuidanceConfig
from parafoil_planner_v3.guidance.flare_guidance import FlareGuidance, FlareGuidanceConfig
from parafoil_planner_v3.guidance.phase_manager import PhaseConfig, PhaseManager
from parafoil_planner_v3.offline.simulator import OfflineSimulator
from parafoil_planner_v3.planner_core import PlannerConfig, PlannerCore
from parafoil_planner_v3.types import Control, GuidancePhase, State, Target, Trajectory, Wind


@dataclass(frozen=True)
class Scenario:
    altitude_m: float
    distance_m: float
    bearing_deg: float
    wind_speed_mps: float
    wind_direction_deg: float


def _lookahead_point(path_xy: List[np.ndarray], p_xy: np.ndarray, L1: float) -> Optional[np.ndarray]:
    if len(path_xy) < 2:
        return None
    d2 = [float(np.dot(w - p_xy, w - p_xy)) for w in path_xy]
    i0 = int(np.argmin(d2))
    dist = 0.0
    for i in range(i0, len(path_xy) - 1):
        p1 = path_xy[i]
        p2 = path_xy[i + 1]
        ds = float(np.linalg.norm(p2 - p1))
        if dist + ds >= L1:
            t = 0.0 if ds < 1e-6 else (L1 - dist) / ds
            return p1 + t * (p2 - p1)
        dist += ds
    return path_xy[-1].copy()


def _remaining_distance(path_xy: List[np.ndarray], p_xy: np.ndarray) -> float:
    if len(path_xy) < 2:
        return 0.0
    d2 = [float(np.dot(w - p_xy, w - p_xy)) for w in path_xy]
    i0 = int(np.argmin(d2))
    S = float(np.linalg.norm(path_xy[i0] - p_xy))
    for i in range(i0, len(path_xy) - 1):
        S += float(np.linalg.norm(path_xy[i + 1] - path_xy[i]))
    return float(S)


def make_initial_state(s: Scenario) -> Tuple[State, Target, Wind]:
    bearing = float(np.deg2rad(s.bearing_deg))
    # Convention: bearing_deg is bearing from state -> target (same as library generator).
    # Target is at origin; therefore the initial position (target->state) is the opposite direction.
    p0 = np.array([-s.distance_m * np.cos(bearing), -s.distance_m * np.sin(bearing), -s.altitude_m], dtype=float)
    v0 = np.array([4.5 * np.cos(bearing), 4.5 * np.sin(bearing), 0.9], dtype=float)
    state = State(p_I=p0, v_I=v0, q_IB=yaw_only_quat_wxyz(bearing), w_B=np.zeros(3), t=0.0)

    target = Target(p_I=np.array([0.0, 0.0, 0.0], dtype=float))

    wd = float(np.deg2rad(s.wind_direction_deg))
    wind = Wind(v_I=np.array([s.wind_speed_mps * np.cos(wd), s.wind_speed_mps * np.sin(wd), 0.0], dtype=float))
    return state, target, wind


def simulate_one(
    scenario: Scenario,
    planner_rate_hz: float = 1.0,
    control_rate_hz: float = 20.0,
    max_time_s: float = 200.0,
    L1_distance: float = 20.0,
    use_gpm: bool = False,
    dynamics_mode: str = "6dof",  # 6dof | simplified
    record_history: bool = True,
    flare_touchdown_altitude_m: float | None = None,
    flare_ramp_time_s: float = 0.1,
    flare_mode: str = "spec_full_brake",
    abort_brake: float = 0.3,
    planner_config: PlannerConfig | None = None,
    lateral_config: LateralControlConfig | None = None,
    flare_config: FlareGuidanceConfig | None = None,
    cruise_config: CruiseGuidanceConfig | None = None,
    approach_config: ApproachGuidanceConfig | None = None,
    phase_config: PhaseConfig | None = None,
    library: "TrajectoryLibrary | None" = None,
) -> Dict[str, Any]:
    dynamics_mode = str(dynamics_mode).strip().lower()
    if dynamics_mode in {"simplified", "simple"}:
        polar = PolarTable()
        dynamics = KinematicYawGlideDynamics(polar=polar)
        dyn_tag = "simplified"
    else:
        dynamics = SixDOFDynamics()
        dyn_tag = "6dof"

    if flare_touchdown_altitude_m is None:
        # The simplified polar model does not capture flare physics; keep it disabled by default.
        flare_touchdown_altitude_m = 0.2 if dyn_tag == "6dof" else -1.0
    state, target, wind = make_initial_state(scenario)
    sim = OfflineSimulator(dynamics=dynamics, state=state, wind=wind)

    cfg = planner_config or PlannerConfig(gpm_num_nodes=10, gpm_scheme="LGL", tf_guess=30.0, use_library=False)
    planner = PlannerCore(dynamics=dynamics, config=cfg, library=library)
    phase_mgr = PhaseManager(config=phase_config)
    polar = PolarTable()

    planned: Optional[Trajectory] = None
    planner_logs: Optional[List[Dict[str, Any]]] = [] if record_history else None
    states: Optional[List[Dict[str, Any]]] = [] if record_history else None
    controls: Optional[List[Dict[str, Any]]] = [] if record_history else None

    dt_ctl = 1.0 / max(control_rate_hz, 1.0)
    dt_plan = 1.0 / max(planner_rate_hz, 0.1)
    next_plan_t = 0.0

    flare_start: Optional[float] = None
    lateral_cfg = lateral_config or LateralControlConfig()
    if flare_config is None:
        flare_cfg = FlareGuidanceConfig(
            mode=str(flare_mode),
            touchdown_brake_altitude_m=float(flare_touchdown_altitude_m),
            flare_ramp_time=float(flare_ramp_time_s),
            lateral=lateral_cfg,
        )
    else:
        flare_cfg = flare_config
        lateral_cfg = flare_cfg.lateral
    flare_guidance = FlareGuidance(config=flare_cfg)
    cruise_guidance = CruiseGuidance(config=cruise_config)
    approach_guidance = ApproachGuidance(config=approach_config)

    phase_durations: Dict[str, float] = {p.value: 0.0 for p in GuidancePhase}
    events: Optional[List[Dict[str, Any]]] = [] if record_history else None
    replan_count = 0
    control_effort_integral = 0.0
    touchdown_before: Optional[State] = None
    touchdown_after: Optional[State] = None

    while sim.get_state().t < max_time_s:
        st = sim.get_state()
        if st.p_I[2] >= target.p_I[2]:
            trans = phase_mgr.update(st, target, wind)
            if trans.triggered and events is not None:
                events.append(
                    {
                        "t": float(st.t),
                        "event": "phase_transition",
                        "from": trans.from_phase.value,
                        "to": trans.to_phase.value,
                        "reason": trans.reason,
                    }
                )
            if events is not None:
                events.append({"t": float(st.t), "event": "touchdown"})
            touchdown_after = st.copy()
            break
        if states is not None:
            states.append(st.to_dict())

        # Replan
        if st.t >= next_plan_t:
            next_plan_t += dt_plan
            if use_gpm:
                traj, info = planner.plan(st, target, wind)
                solver_info = info.__dict__
            else:
                traj = planner._fallback_direct(st, target, wind)  # type: ignore[attr-defined]
                solver_info = None
            planned = traj
            replan_count += 1
            if events is not None:
                events.append({"t": float(st.t), "event": "replan", "planned_waypoints": int(len(traj.waypoints))})
            if planner_logs is not None:
                planner_logs.append({"t": st.t, "planned_waypoints": len(traj.waypoints), "solver": solver_info})

        # Phase update
        trans = phase_mgr.update(st, target, wind)
        if trans.triggered and events is not None:
            events.append(
                {
                    "t": float(st.t),
                    "event": "phase_transition",
                    "from": trans.from_phase.value,
                    "to": trans.to_phase.value,
                    "reason": trans.reason,
                }
            )
        if trans.triggered and trans.to_phase == GuidancePhase.FLARE:
            flare_start = st.t
            flare_guidance.on_enter(st)

        # Build path for lookahead
        if planned is not None and planned.waypoints:
            path_xy = [wp.state.position_xy for wp in planned.waypoints]
            track_xy = _lookahead_point(path_xy, st.position_xy, L1_distance)
            if track_xy is None:
                track_xy = target.position_xy
            S_rem = _remaining_distance(path_xy, st.position_xy)
        else:
            track_xy = target.position_xy
            S_rem = float(np.linalg.norm(target.position_xy - st.position_xy))

        H_rem = float(max(st.altitude - target.altitude, 0.0))
        k_req = 0.0 if S_rem < 1e-3 else float(H_rem / S_rem)
        b_slope = float(polar.select_brake_for_required_slope(k_req))

        # Control by phase
        if phase_mgr.current_phase == GuidancePhase.CRUISE:
            if planned is None or not planned.waypoints or cruise_guidance.should_override_path(wind):
                control = cruise_guidance.compute_control(st, target, wind, dt=dt_ctl)
            else:
                brake = 0.2
                control = track_point_control(st, track_xy, brake_sym=brake, cfg=lateral_cfg)
        elif phase_mgr.current_phase == GuidancePhase.APPROACH:
            if planned is None or not planned.waypoints:
                control = approach_guidance.compute_control(st, target, wind, dt=dt_ctl)
            else:
                brake = float(np.clip(b_slope, 0.0, 0.6))
                if approach_guidance.should_override_path(st, target, wind, brake=brake):
                    control = approach_guidance.compute_control(st, target, wind, dt=dt_ctl)
                else:
                    control = track_point_control(st, track_xy, brake_sym=brake, cfg=lateral_cfg)
        elif phase_mgr.current_phase == GuidancePhase.FLARE:
            brake = float(flare_guidance.brake_command(st, target))
            control = track_point_control(st, track_xy, brake_sym=brake, cfg=lateral_cfg)
        elif phase_mgr.current_phase == GuidancePhase.ABORT:
            control = track_point_control(st, track_xy, brake_sym=float(np.clip(abort_brake, 0.0, 1.0)), cfg=lateral_cfg)
        else:
            control = Control(0.0, 0.0)
        phase_durations[phase_mgr.current_phase.value] += float(dt_ctl)
        control_effort_integral += float(dt_ctl) * float(control.delta_L * control.delta_L + control.delta_R * control.delta_R)
        if controls is not None:
            controls.append({"t": float(st.t), "delta_L": float(control.delta_L), "delta_R": float(control.delta_R)})
        sim.step(control, dt=dt_ctl)
        st_next = sim.get_state()
        if st_next.p_I[2] >= target.p_I[2]:
            touchdown_before = st.copy()
            touchdown_after = st_next.copy()
            trans = phase_mgr.update(st_next, target, wind)
            if trans.triggered and events is not None:
                events.append(
                    {
                        "t": float(st_next.t),
                        "event": "phase_transition",
                        "from": trans.from_phase.value,
                        "to": trans.to_phase.value,
                        "reason": trans.reason,
                    }
                )
            if events is not None:
                events.append({"t": float(st_next.t), "event": "touchdown"})
            break

    final = sim.get_state()
    initial = state

    touchdown_vz = float(abs(final.v_I[2]))
    touchdown_t = float(final.t)
    touchdown_xy = final.position_xy.copy()
    if touchdown_before is not None and touchdown_after is not None:
        z0 = float(touchdown_before.p_I[2])
        z1 = float(touchdown_after.p_I[2])
        zt = float(target.p_I[2])
        dz = float(z1 - z0)
        if abs(dz) > 1e-9 and (z0 <= zt <= z1 or z1 <= zt <= z0):
            a = float(np.clip((zt - z0) / dz, 0.0, 1.0))
            v_I = touchdown_before.v_I + a * (touchdown_after.v_I - touchdown_before.v_I)
            p_xy = touchdown_before.position_xy + a * (touchdown_after.position_xy - touchdown_before.position_xy)
            touchdown_vz = float(abs(v_I[2]))
            touchdown_xy = p_xy.astype(float)
            touchdown_t = float(touchdown_before.t + a * (touchdown_after.t - touchdown_before.t))

    landing_error = float(np.linalg.norm(final.position_xy - target.position_xy))
    touchdown_landing_error = float(np.linalg.norm(touchdown_xy - target.position_xy))
    altitude_used = float(max(initial.altitude - final.altitude, 0.0))
    flare_time = None if flare_start is None else float(flare_start)
    metrics = {
        "landing_error_m": landing_error,
        "vertical_velocity_mps": float(abs(final.v_I[2])),
        "touchdown_landing_error_m": float(touchdown_landing_error),
        "touchdown_vertical_velocity_mps": float(touchdown_vz),
        "touchdown_time_s": float(touchdown_t),
        "final_position_xy": [float(final.position_xy[0]), float(final.position_xy[1])],
        "touchdown_position_xy": [float(touchdown_xy[0]), float(touchdown_xy[1])],
        "time_s": float(final.t),
        "altitude_used_m": altitude_used,
        "replan_count": int(replan_count),
        "control_effort_integral": float(control_effort_integral),
        "control_effort_mean": float(control_effort_integral / max(float(final.t), 1e-6)),
        "phase_durations_s": {k: float(v) for k, v in phase_durations.items() if float(v) > 0.0},
        "flare_start_time_s": flare_time,
        "success": bool(landing_error < 10.0),
        "final_phase": phase_mgr.current_phase.value,
        "dynamics_mode": dyn_tag,
    }

    out: Dict[str, Any] = {"scenario": asdict(scenario), "metrics": metrics}
    if events is not None:
        out["events"] = events
    if planner_logs is not None:
        out["planner_logs"] = planner_logs
    if states is not None:
        out["state_history"] = states
    if controls is not None:
        out["control_history"] = controls
    return out
