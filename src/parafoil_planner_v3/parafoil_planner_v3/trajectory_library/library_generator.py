from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import time

from parafoil_planner_v3.dynamics.aerodynamics import PolarTable
from parafoil_planner_v3.dynamics.parafoil_6dof import SixDOFDynamics
from parafoil_planner_v3.dynamics.simplified_model import KinematicYawGlideDynamics, yaw_only_quat_wxyz
from parafoil_planner_v3.optimization.gpm_collocation import GPMCollocation
from parafoil_planner_v3.optimization.solver_interface import GPMSolver, SolverConfig
from parafoil_planner_v3.guidance.control_laws import LateralControlConfig, track_point_control
from parafoil_planner_v3.types import Control, Scenario, State, Target, Trajectory, TrajectoryType, Waypoint, Wind

from .library_manager import LibraryTrajectory, TrajectoryLibrary
from .trajectory_metrics import compute_trajectory_metrics


@dataclass(frozen=True)
class ScenarioConfig:
    wind_speeds: Sequence[float]
    wind_directions_deg: Sequence[float]
    initial_altitudes_m: Sequence[float]
    target_distances_m: Sequence[float]
    target_bearings_deg: Sequence[float]


@dataclass(frozen=True)
class GPMGenerationConfig:
    method: str = "SLSQP"
    num_nodes: int = 20
    scheme: str = "LGL"
    maxiter: int = 400
    ftol: float = 1e-6
    w_u_ref: float = 2.0
    dynamics_mode: str = "simplified"  # simplified | 6dof

    # Reference control shaping
    delta_a_amp: float = 0.4
    s_turn_cycles: float = 2.0
    spiral_delta_a_sign: float = 1.0
    rollout_dt: float = 0.2

    # Symmetric brake baseline
    brake_min: float = 0.0
    brake_max: float = 0.6
    brake_fallback: float = 0.2

    # Acceptance
    max_violation_accept: float = 0.5

    # Solve mode
    solve_mode: str = "least_squares"  # least_squares | minimize
    lsq_max_nfev: int = 300
    lsq_w_dynamics: float = 1.0
    lsq_w_boundary: float = 10.0
    lsq_w_ineq: float = 10.0

    # Shape constraints (post-check)
    shape_enforce: bool = True
    shape_turn_eps_deg: float = 3.0
    direct_max_total_turn_deg: float = 60.0
    direct_max_net_turn_deg: float = 45.0
    s_turn_min_total_turn_deg: float = 90.0
    s_turn_max_net_turn_deg: float = 45.0
    s_turn_min_sign_changes: int = 1
    racetrack_min_total_turn_deg: float = 150.0
    racetrack_max_net_turn_deg: float = 60.0
    racetrack_min_turn_clusters: int = 2
    racetrack_min_turn_fraction: float = 0.75
    spiral_min_turns: float = 1.0
    spiral_min_turn_fraction: float = 0.9


def _scenario_to_wind_I(s: Scenario) -> np.ndarray:
    ang = float(np.deg2rad(s.wind_direction_deg))
    return np.array([s.wind_speed * np.cos(ang), s.wind_speed * np.sin(ang), 0.0], dtype=float)


def _scenario_initial_state_target_centered(s: Scenario, polar: PolarTable, brake_sym: float) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Returns:
      x0 (13,), p_target (3,), tf_guess
    """
    bearing = float(np.deg2rad(s.target_bearing_deg))
    # Target is at origin; scenario.target_bearing_deg is bearing from state -> target.
    # Therefore state position (target->state) is the opposite direction.
    p0 = np.array(
        [
            -s.target_distance_m * np.cos(bearing),
            -s.target_distance_m * np.sin(bearing),
            -s.initial_altitude_m,
        ],
        dtype=float,
    )
    V, sink = polar.interpolate(brake_sym)
    # Initial velocity points toward target (state->target bearing).
    v0 = np.array([V * np.cos(bearing), V * np.sin(bearing), sink], dtype=float)
    q0 = yaw_only_quat_wxyz(bearing)
    x0 = State(p_I=p0, v_I=v0, q_IB=q0, w_B=np.zeros(3), t=0.0).to_vector()

    p_target = np.array([0.0, 0.0, 0.0], dtype=float)
    tf_guess = float(max(5.0, s.target_distance_m / max(V, 0.5), s.initial_altitude_m / max(sink, 0.05)))
    return x0, p_target, tf_guess


def _u_ref_pattern(
    traj_type: TrajectoryType,
    tau: np.ndarray,
    brake_sym: float,
    delta_a_amp: float,
    s_turn_cycles: float,
    spiral_sign: float,
) -> np.ndarray:
    """
    Build per-node reference controls (delta_L, delta_R) for shaping.
    """
    tau = np.asarray(tau, dtype=float).reshape(-1)
    s = 0.5 * (tau + 1.0)  # normalized [0,1]

    if traj_type == TrajectoryType.DIRECT:
        delta_a = np.zeros_like(s)
    elif traj_type == TrajectoryType.S_TURN:
        delta_a = float(delta_a_amp) * np.sin(2.0 * np.pi * float(s_turn_cycles) * s)
    elif traj_type == TrajectoryType.SPIRAL:
        delta_a = float(delta_a_amp) * float(np.sign(spiral_sign)) * np.ones_like(s)
    elif traj_type == TrajectoryType.RACETRACK:
        # Two same-direction turns with straights in-between.
        delta_a = np.zeros_like(s)
        delta_a[(s >= 0.20) & (s <= 0.35)] = float(delta_a_amp)
        delta_a[(s >= 0.65) & (s <= 0.80)] = float(delta_a_amp)
    else:
        delta_a = np.zeros_like(s)

    left = brake_sym + 0.5 * delta_a
    right = brake_sym - 0.5 * delta_a
    U_ref = np.stack([left, right], axis=1).astype(float)
    return np.clip(U_ref, 0.0, 1.0)


def _interp_state_vector(traj: Trajectory, t: float) -> np.ndarray:
    """Linear interpolation over trajectory.waypoints by time."""
    if not traj.waypoints:
        raise ValueError("Empty trajectory")
    times = np.array([wp.t for wp in traj.waypoints], dtype=float)
    X = np.stack([wp.state.to_vector() for wp in traj.waypoints], axis=0).astype(float)

    if t <= float(times[0]):
        return X[0].copy()
    if t >= float(times[-1]):
        return X[-1].copy()
    i = int(np.searchsorted(times, t, side="right") - 1)
    t0 = float(times[i])
    t1 = float(times[i + 1])
    a = 0.0 if abs(t1 - t0) < 1e-9 else float((t - t0) / (t1 - t0))
    return (1.0 - a) * X[i] + a * X[i + 1]


def _interp_array(t_grid: np.ndarray, Y: np.ndarray, t: float) -> np.ndarray:
    """Linear interpolation for array-valued samples Y[k,:] defined at t_grid[k]."""
    t_grid = np.asarray(t_grid, dtype=float).reshape(-1)
    Y = np.asarray(Y, dtype=float)
    if t <= float(t_grid[0]):
        return Y[0].copy()
    if t >= float(t_grid[-1]):
        return Y[-1].copy()
    i = int(np.searchsorted(t_grid, t, side="right") - 1)
    t0 = float(t_grid[i])
    t1 = float(t_grid[i + 1])
    a = 0.0 if abs(t1 - t0) < 1e-9 else float((t - t0) / (t1 - t0))
    return (1.0 - a) * Y[i] + a * Y[i + 1]


def _pack_decision(X: np.ndarray, U: np.ndarray, tf: float) -> np.ndarray:
    return np.concatenate([np.asarray(X, dtype=float).reshape(-1), np.asarray(U, dtype=float).reshape(-1), np.array([tf], dtype=float)])


def _warm_start_from_template(gpm: GPMCollocation, template: Trajectory, U_ref: np.ndarray) -> np.ndarray:
    if not template.waypoints:
        raise ValueError("Template trajectory empty")
    tf = float(template.waypoints[-1].t - template.waypoints[0].t)
    tf = float(max(tf, 5.0))
    X_nodes = np.zeros((gpm.N, 13), dtype=float)
    for k in range(gpm.N):
        t_k = gpm.tau_to_time(float(gpm.tau[k]), 0.0, tf)
        X_nodes[k, :] = _interp_state_vector(template, t_k)
    return _pack_decision(X_nodes, U_ref, tf)


def _warm_start_from_rollout(
    dyn: SixDOFDynamics,
    gpm: GPMCollocation,
    x0: np.ndarray,
    U_ref: np.ndarray,
    tf: float,
    wind_I: np.ndarray,
    rollout_dt: float,
) -> np.ndarray:
    """
    Build a dynamically consistent warm-start by rolling out the 6-DOF model
    using the reference control profile.
    """
    tf = float(max(tf, 5.0))
    rollout_dt = float(max(rollout_dt, 0.02))
    node_times = np.array([gpm.tau_to_time(float(tau), 0.0, tf) for tau in gpm.tau], dtype=float)

    # Rollout samples
    times = [0.0]
    Xs = [np.asarray(x0, dtype=float).reshape(13)]
    state = State.from_vector(x0, t=0.0)
    wind = Wind(v_I=np.asarray(wind_I, dtype=float).reshape(3))

    t = 0.0
    while t < tf - 1e-9:
        u = _interp_array(node_times, U_ref, t)
        ctrl = Control(float(u[0]), float(u[1])).clipped()
        state = dyn.step(state, ctrl, wind, dt=rollout_dt)
        t = float(state.t)
        times.append(t)
        Xs.append(state.to_vector())

        if len(times) > 20000:  # safety
            break

    t_grid = np.array(times, dtype=float)
    Y = np.stack(Xs, axis=0).astype(float)

    X_nodes = np.zeros((gpm.N, 13), dtype=float)
    for k in range(gpm.N):
        X_nodes[k, :] = _interp_array(t_grid, Y, float(node_times[k]))

    return _pack_decision(X_nodes, U_ref, tf)


def _lookahead_point(path_xy: np.ndarray, p_xy: np.ndarray, L1: float) -> np.ndarray:
    path_xy = np.asarray(path_xy, dtype=float).reshape(-1, 2)
    p_xy = np.asarray(p_xy, dtype=float).reshape(2)
    if path_xy.shape[0] == 0:
        return p_xy.copy()
    if path_xy.shape[0] == 1:
        return path_xy[0].copy()

    d2 = np.sum((path_xy - p_xy[None, :]) ** 2, axis=1)
    i0 = int(np.argmin(d2))

    dist = 0.0
    for i in range(i0, path_xy.shape[0] - 1):
        p1 = path_xy[i]
        p2 = path_xy[i + 1]
        ds = float(np.linalg.norm(p2 - p1))
        if dist + ds >= L1:
            a = 0.0 if ds < 1e-6 else (L1 - dist) / ds
            return p1 + float(a) * (p2 - p1)
        dist += ds
    return path_xy[-1].copy()


def _guided_rollout(
    dyn: SixDOFDynamics,
    x0: np.ndarray,
    wind_I: np.ndarray,
    path_xy: np.ndarray,
    brake_sym: float,
    dt: float,
    max_time: float,
    L1: float,
    lateral_cfg: LateralControlConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    state = State.from_vector(x0, t=0.0)
    wind = Wind(v_I=np.asarray(wind_I, dtype=float).reshape(3))
    t = 0.0

    times = [0.0]
    Xs = [state.to_vector()]
    Us: List[np.ndarray] = []

    while t < max_time - 1e-9:
        track = _lookahead_point(path_xy, state.position_xy, L1=float(L1))
        ctrl = track_point_control(state, track, brake_sym=float(brake_sym), cfg=lateral_cfg)
        Us.append(ctrl.as_array)
        state = dyn.step(state, ctrl, wind, dt=float(dt))
        t = float(state.t)
        times.append(t)
        Xs.append(state.to_vector())

        # stop on ground contact (target down=0)
        if state.p_I[2] >= 0.0:
            break

        # Stop early only if we're both near target and near ground.
        if float(np.linalg.norm(state.position_xy)) < 3.0 and float(-state.p_I[2]) < 2.0:
            break

    t_grid = np.asarray(times, dtype=float)
    X_grid = np.stack(Xs, axis=0).astype(float)
    if Us:
        U_grid = np.stack(Us, axis=0).astype(float)
        # U is defined for each interval start; align to state times except last.
        U_times = t_grid[:-1]
    else:
        U_grid = np.zeros((1, 2), dtype=float)
        U_times = np.array([0.0], dtype=float)
    return t_grid, X_grid, np.column_stack([U_times, U_grid])


def _shape_constraints_ok(traj_type: TrajectoryType, metrics: Dict[str, float], gpm_cfg: GPMGenerationConfig) -> Tuple[bool, str]:
    if not bool(getattr(gpm_cfg, "shape_enforce", True)):
        return True, ""

    total_deg = float(np.rad2deg(metrics.get("turn_total_rad", 0.0)))
    net_deg = float(abs(np.rad2deg(metrics.get("turn_net_rad", 0.0))))
    sign_changes = int(metrics.get("turn_sign_changes", 0))
    clusters = int(metrics.get("turn_clusters", 0))
    dom = float(metrics.get("turn_dominant_fraction", 0.0))
    total_turns = float(metrics.get("turn_total_turns", 0.0))

    if traj_type == TrajectoryType.DIRECT:
        if total_deg > float(gpm_cfg.direct_max_total_turn_deg):
            return False, "direct_total_turn_excess"
        if net_deg > float(gpm_cfg.direct_max_net_turn_deg):
            return False, "direct_net_turn_excess"
        return True, ""

    if traj_type == TrajectoryType.S_TURN:
        if total_deg < float(gpm_cfg.s_turn_min_total_turn_deg):
            return False, "s_turn_insufficient_turn"
        if net_deg > float(gpm_cfg.s_turn_max_net_turn_deg):
            return False, "s_turn_net_turn_excess"
        if sign_changes < int(gpm_cfg.s_turn_min_sign_changes):
            return False, "s_turn_missing_sign_change"
        return True, ""

    if traj_type == TrajectoryType.RACETRACK:
        if total_deg < float(gpm_cfg.racetrack_min_total_turn_deg):
            return False, "racetrack_insufficient_turn"
        if net_deg > float(gpm_cfg.racetrack_max_net_turn_deg):
            return False, "racetrack_net_turn_excess"
        if clusters < int(gpm_cfg.racetrack_min_turn_clusters):
            return False, "racetrack_missing_turn_clusters"
        if dom < float(gpm_cfg.racetrack_min_turn_fraction):
            return False, "racetrack_low_turn_fraction"
        return True, ""

    if traj_type == TrajectoryType.SPIRAL:
        if total_turns < float(gpm_cfg.spiral_min_turns):
            return False, "spiral_insufficient_turns"
        if dom < float(gpm_cfg.spiral_min_turn_fraction):
            return False, "spiral_low_turn_fraction"
        return True, ""

    return True, ""


def _warm_start_from_guided_rollout(
    dyn: SixDOFDynamics,
    gpm: GPMCollocation,
    x0: np.ndarray,
    wind_I: np.ndarray,
    path_xy: np.ndarray,
    brake_sym: float,
    tf_guess: float,
    rollout_dt: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    tf_max = float(max(10.0, 1.5 * tf_guess))
    L1 = float(max(10.0, 0.15 * float(np.linalg.norm(np.asarray(path_xy[0], dtype=float))) if path_xy.size else 20.0))
    lateral_cfg = LateralControlConfig(max_brake=1.0, max_delta_a=0.6)

    t_grid, X_grid, U_pack = _guided_rollout(
        dyn=dyn,
        x0=x0,
        wind_I=wind_I,
        path_xy=path_xy,
        brake_sym=brake_sym,
        dt=float(rollout_dt),
        max_time=tf_max,
        L1=L1,
        lateral_cfg=lateral_cfg,
    )

    tf = float(max(t_grid[-1], 5.0))
    node_times = np.array([gpm.tau_to_time(float(tau), 0.0, tf) for tau in gpm.tau], dtype=float)

    # Interpolate states and controls at node times
    X_nodes = np.zeros((gpm.N, 13), dtype=float)
    for k in range(gpm.N):
        X_nodes[k, :] = _interp_array(t_grid, X_grid, float(node_times[k]))
    # Enforce initial state exactly and terminal position at target (origin in library frame).
    X_nodes[0, :] = np.asarray(x0, dtype=float).reshape(13)
    # NOTE: do not hard-enforce terminal position here; leave it near-feasible
    # and let the NLP enforce boundary constraints.

    U_times = U_pack[:, 0]
    U_grid = U_pack[:, 1:3]
    U_nodes = np.zeros((gpm.N, 2), dtype=float)
    for k in range(gpm.N):
        U_nodes[k, :] = _interp_array(U_times, U_grid, float(node_times[k]))

    return _pack_decision(X_nodes, U_nodes, tf), U_nodes, tf

# ----------------------------------------------------------------------
# Parallel GPM worker (module-level for multiprocessing pickling)
# ----------------------------------------------------------------------
_GPM_WORKER: dict = {}


def _init_gpm_worker(gpm_cfg: GPMGenerationConfig) -> None:
    polar = PolarTable()
    dyn_6dof = SixDOFDynamics()
    dyn_simplified = KinematicYawGlideDynamics(polar=polar)
    gpm = GPMCollocation(N=int(gpm_cfg.num_nodes), scheme=str(gpm_cfg.scheme))
    cfg = SolverConfig(method=str(gpm_cfg.method), maxiter=int(gpm_cfg.maxiter), ftol=float(gpm_cfg.ftol), w_u_ref=float(gpm_cfg.w_u_ref))
    solver = GPMSolver(f=lambda x, u, t: np.zeros_like(x), gpm=gpm, config=cfg)
    _GPM_WORKER["dyn_6dof"] = dyn_6dof
    _GPM_WORKER["dyn_simplified"] = dyn_simplified
    _GPM_WORKER["gpm"] = gpm
    _GPM_WORKER["solver"] = solver
    _GPM_WORKER["polar"] = polar
    _GPM_WORKER["gpm_cfg"] = gpm_cfg


def _solve_gpm_task(task: Tuple[Scenario, TrajectoryType]) -> Optional[LibraryTrajectory]:
    dyn_6dof: SixDOFDynamics = _GPM_WORKER["dyn_6dof"]
    dyn_simplified: KinematicYawGlideDynamics = _GPM_WORKER["dyn_simplified"]
    gpm: GPMCollocation = _GPM_WORKER["gpm"]
    solver: GPMSolver = _GPM_WORKER["solver"]
    polar: PolarTable = _GPM_WORKER["polar"]
    gpm_cfg: GPMGenerationConfig = _GPM_WORKER["gpm_cfg"]

    scenario, traj_type = task

    template_gen = TrajectoryLibraryGenerator(polar=polar)
    bearing = float(np.deg2rad(scenario.target_bearing_deg))
    path_xy = template_gen._path_xy(traj_type, float(scenario.target_distance_m), bearing)

    # Baseline symmetric brake based on required slope over the template path length.
    if path_xy.shape[0] >= 2:
        ds = np.linalg.norm(np.diff(path_xy, axis=0), axis=1)
        path_length = float(np.sum(ds))
    else:
        path_length = float(scenario.target_distance_m)
    k_req = 0.0 if path_length < 1e-6 else float(scenario.initial_altitude_m / path_length)
    try:
        brake_sym = float(polar.select_brake_for_required_slope(k_req))
    except Exception:
        brake_sym = float(gpm_cfg.brake_fallback)
    brake_sym = float(np.clip(brake_sym, float(gpm_cfg.brake_min), float(gpm_cfg.brake_max)))

    x0, p_target, tf_guess = _scenario_initial_state_target_centered(scenario, polar, brake_sym=brake_sym)
    wind_I = _scenario_to_wind_I(scenario)

    dynamics_mode = str(getattr(gpm_cfg, "dynamics_mode", "simplified")).strip().lower()
    if dynamics_mode in {"6dof", "6-dof", "sixdof"}:
        dyn = dyn_6dof
        dyn_tag = "6dof"
    else:
        dyn = dyn_simplified
        dyn_tag = "simplified"

    def f(x: np.ndarray, u: np.ndarray, t: float) -> np.ndarray:
        return dyn.f_vector(x, u, t, wind_I=wind_I)

    solver.f = f

    U_ref = _u_ref_pattern(
        traj_type,
        tau=gpm.tau,
        brake_sym=brake_sym,
        delta_a_amp=float(gpm_cfg.delta_a_amp),
        s_turn_cycles=float(gpm_cfg.s_turn_cycles),
        spiral_sign=float(gpm_cfg.spiral_delta_a_sign),
    )

    if dyn_tag == "6dof":
        # Warm-start using a closed-loop rollout tracking a geometric template.
        z0, _, tf_warm = _warm_start_from_guided_rollout(
            dyn=dyn_6dof,
            gpm=gpm,
            x0=x0,
            wind_I=wind_I,
            path_xy=path_xy,
            brake_sym=brake_sym,
            tf_guess=tf_guess,
            rollout_dt=float(gpm_cfg.rollout_dt),
        )
    else:
        template = template_gen._trajectory_from_xy(scenario, traj_type, path_xy, brake=float(brake_sym))
        z0 = _warm_start_from_template(gpm=gpm, template=template, U_ref=U_ref)
        X0, U0, tf_warm = solver._unpack(z0)
        X0[0, :] = np.asarray(x0, dtype=float).reshape(13)
        z0 = solver._pack(X0, U0, float(tf_warm))

    # Ensure warm-start final time is within solver bounds (least_squares requires feasible x0).
    tf_warm = float(np.clip(float(tf_warm), solver.config.tf_min, solver.config.tf_max))
    z0 = np.asarray(z0, dtype=float).copy()
    z0[-1] = tf_warm

    solve_mode = str(gpm_cfg.solve_mode).strip().lower()
    if solve_mode == "least_squares":
        import scipy.optimize
        import time

        # Convert list bounds -> arrays for least_squares.
        b_list = solver._bounds()
        lb = np.array([-np.inf if b[0] is None else float(b[0]) for b in b_list], dtype=float)
        ub = np.array([np.inf if b[1] is None else float(b[1]) for b in b_list], dtype=float)

        # Keep initial state tightly bounded (least_squares requires strict lb < ub).
        eps_fix = 1e-9
        for i in range(solver.n_x):
            xi = float(x0[i])
            lb[i] = xi - eps_fix
            ub[i] = xi + eps_fix

        w_dyn = float(gpm_cfg.lsq_w_dynamics)
        w_bnd = float(gpm_cfg.lsq_w_boundary)
        w_ineq = float(gpm_cfg.lsq_w_ineq)
        w_u = float(max(gpm_cfg.w_u_ref, 0.0))

        def residual(z: np.ndarray) -> np.ndarray:
            # Equality constraints
            r_dyn = solver._constraint_dynamics(z)
            r_bnd = solver._constraint_boundary(z, x0, p_target)
            # Inequality constraints g(z) >= 0 => penalize negative part.
            g = solver._constraint_path(z)
            r_ineq = np.minimum(0.0, g)

            X, U, _tf = solver._unpack(z)
            r_u = (U - U_ref).reshape(-1)

            parts = [
                w_dyn * r_dyn,
                w_bnd * r_bnd,
                w_ineq * r_ineq,
                np.sqrt(w_u) * r_u,
            ]
            return np.concatenate(parts, axis=0)

        t0 = time.perf_counter()
        lsq = scipy.optimize.least_squares(
            residual,
            z0,
            bounds=(lb, ub),
            max_nfev=int(gpm_cfg.lsq_max_nfev),
        )
        solve_time = float(time.perf_counter() - t0)
        z_sol = np.asarray(lsq.x, dtype=float)
        solver.last_solution_z = z_sol.copy()

        # Extract trajectory and build info (reuse existing metrics helpers).
        X, U, tf_sol = solver._unpack(z_sol)
        tau = gpm.tau
        waypoints = []
        for k in range(gpm.N):
            t_k = gpm.tau_to_time(float(tau[k]), 0.0, tf_sol)
            waypoints.append(Waypoint(t=t_k, state=State.from_vector(X[k], t=t_k)))
        controls = [Control(float(U[k, 0]), float(U[k, 1])).clipped() for k in range(gpm.N)]
        traj = Trajectory(waypoints=waypoints, controls=controls, trajectory_type=traj_type)

        dyn_violation = float(np.max(np.abs(solver._constraint_dynamics(z_sol))))
        bnd_violation = float(np.max(np.abs(solver._constraint_boundary(z_sol, x0, p_target))))
        path_violation = float(np.min(solver._constraint_path(z_sol)))
        max_violation = float(max(dyn_violation, bnd_violation, max(0.0, -path_violation)))

        terminal_error = float(np.linalg.norm(X[-1, 0:3] - p_target))
        solver._u_ref = U_ref
        objective_cost = float(solver._objective(z_sol, x0, p_target))
        solver._u_ref = None

        info = {
            "success": bool(lsq.success),
            "status": int(lsq.status),
            "message": str(lsq.message),
            "iterations": int(getattr(lsq, "nfev", -1)),
            "lsq_cost": float(lsq.cost),
            "objective_cost": objective_cost,
            "solve_time": solve_time,
            "max_violation": max_violation,
            "terminal_error_m": terminal_error,
            "solve_mode": "least_squares",
            "dynamics_mode": dyn_tag,
        }

        if (max_violation > float(gpm_cfg.max_violation_accept)) or (terminal_error > 5.0):
            return None

        traj_metrics = compute_trajectory_metrics(traj, turn_eps_deg=float(gpm_cfg.shape_turn_eps_deg))
        shape_ok, shape_reason = _shape_constraints_ok(traj_type, traj_metrics, gpm_cfg)
        traj.metadata = {**(traj.metadata or {}), **traj_metrics, "shape_ok": bool(shape_ok), "shape_reason": shape_reason}

        if bool(gpm_cfg.shape_enforce) and not shape_ok:
            return None

        meta = {
            "generator": "gpm_offline",
            "solver": info,
            "brake_sym_ref": brake_sym,
            "dynamics_mode": dyn_tag,
            "trajectory_metrics": traj_metrics,
            "shape_ok": bool(shape_ok),
            "shape_reason": shape_reason,
        }
        return LibraryTrajectory(scenario=scenario, trajectory_type=traj_type, trajectory=traj, cost=objective_cost, metadata=meta)

    # Default: constrained minimize (may be slow/fragile for full 6-DOF)
    traj, info_obj = solver.solve(x0=x0, p_target=p_target, tf_guess=tf_warm, warm_start=z0, u_ref=U_ref)

    if (not info_obj.success) or (info_obj.max_violation > float(gpm_cfg.max_violation_accept)):
        return None

    traj_metrics = compute_trajectory_metrics(traj, turn_eps_deg=float(gpm_cfg.shape_turn_eps_deg))
    shape_ok, shape_reason = _shape_constraints_ok(traj_type, traj_metrics, gpm_cfg)
    traj.metadata = {**(traj.metadata or {}), **traj_metrics, "shape_ok": bool(shape_ok), "shape_reason": shape_reason}

    if bool(gpm_cfg.shape_enforce) and not shape_ok:
        return None

    meta = {
        "generator": "gpm_offline",
        "solver": {
            "success": info_obj.success,
            "status": info_obj.status,
            "message": info_obj.message,
            "iterations": info_obj.iterations,
            "cost": info_obj.cost,
            "solve_time": info_obj.solve_time,
            "max_violation": info_obj.max_violation,
            "terminal_error_m": info_obj.terminal_error_m,
            "solve_mode": "minimize",
            "dynamics_mode": dyn_tag,
        },
        "brake_sym_ref": brake_sym,
        "dynamics_mode": dyn_tag,
        "trajectory_metrics": traj_metrics,
        "shape_ok": bool(shape_ok),
        "shape_reason": shape_reason,
    }

    return LibraryTrajectory(scenario=scenario, trajectory_type=traj_type, trajectory=traj, cost=float(info_obj.cost), metadata=meta)


class TrajectoryLibraryGenerator:
    """
    Offline trajectory library generator.

    This implementation keeps it lightweight: it produces kinematic DIRECT trajectories
    using the current polar table, suitable for warm-start / pattern selection.
    """

    def __init__(self, polar: PolarTable | None = None) -> None:
        self.polar = polar or PolarTable()

    def enumerate_scenarios(self, config: ScenarioConfig) -> List[Scenario]:
        scenarios: List[Scenario] = []
        for ws in config.wind_speeds:
            for wd in config.wind_directions_deg:
                for alt in config.initial_altitudes_m:
                    for dist in config.target_distances_m:
                        for brg in config.target_bearings_deg:
                            scenarios.append(
                                Scenario(
                                    wind_speed=float(ws),
                                    wind_direction_deg=float(wd),
                                    initial_altitude_m=float(alt),
                                    target_distance_m=float(dist),
                                    target_bearing_deg=float(brg),
                                )
                            )
        return scenarios

    @staticmethod
    def _unit(v: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(v))
        if n < 1e-9:
            return np.array([1.0, 0.0], dtype=float)
        return v / n

    def _path_xy(self, traj_type: TrajectoryType, distance: float, bearing_rad: float) -> np.ndarray:
        """
        Return (M,2) waypoints in target-centered NED xy.
        Start at distance/bearing, end at (0,0).
        """
        d = float(max(distance, 0.0))
        # Target-centered frame (target at origin). bearing_rad is state->target,
        # so state position (target->state) is opposite direction.
        start = np.array([-d * np.cos(bearing_rad), -d * np.sin(bearing_rad)], dtype=float)
        if d < 1e-6:
            return np.array([[0.0, 0.0]], dtype=float)

        u = self._unit(start)  # from target -> start
        v = np.array([-u[1], u[0]], dtype=float)

        if traj_type == TrajectoryType.DIRECT:
            a = np.linspace(0.0, 1.0, num=30)
            pts = (1.0 - a)[:, None] * start[None, :]
            return pts.astype(float)

        if traj_type == TrajectoryType.S_TURN:
            a = np.linspace(0.0, 1.0, num=60)
            A = 0.18 * d
            cycles = 2.0
            offset = (A * np.sin(2.0 * np.pi * cycles * a))[:, None] * v[None, :]
            pts = (1.0 - a)[:, None] * start[None, :] + offset
            return pts.astype(float)

        if traj_type == TrajectoryType.RACETRACK:
            L = 0.45 * d
            W = min(0.25 * d, 60.0)
            p0 = start
            p1 = p0 + v * W
            p2 = p1 - u * L
            p3 = p2 - v * 2.0 * W
            p4 = p3 + u * L
            p5 = p4 + v * W  # back to start
            corners = np.stack([p0, p1, p2, p3, p4, p5, np.zeros(2)], axis=0)

            # Densify line segments.
            pts = []
            for i in range(corners.shape[0] - 1):
                a = np.linspace(0.0, 1.0, num=15, endpoint=False)
                seg = (1.0 - a)[:, None] * corners[i][None, :] + a[:, None] * corners[i + 1][None, :]
                pts.append(seg)
            pts.append(corners[-1][None, :])
            return np.concatenate(pts, axis=0).astype(float)

        if traj_type == TrajectoryType.SPIRAL:
            turns = 1.5
            a = np.linspace(0.0, 1.0, num=80)
            r = d * (1.0 - a)
            ang = bearing_rad + 2.0 * np.pi * turns * a
            pts = np.stack([r * np.cos(ang), r * np.sin(ang)], axis=1)
            return pts.astype(float)

        raise ValueError(f"Unsupported trajectory type: {traj_type}")

    def _trajectory_from_xy(self, scenario: Scenario, traj_type: TrajectoryType, xy: np.ndarray, brake: float) -> Trajectory:
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        V, sink = self.polar.interpolate(brake)

        # Compute arc length for time allocation.
        if xy.shape[0] < 2:
            S_total = 0.0
            s_cum = np.array([0.0], dtype=float)
        else:
            ds = np.linalg.norm(np.diff(xy, axis=0), axis=1)
            s_cum = np.concatenate([[0.0], np.cumsum(ds)], axis=0).astype(float)
            S_total = float(s_cum[-1])

        alt = float(max(scenario.initial_altitude_m, 0.0))
        tf_by_sink = alt / max(sink, 0.05)
        tf_by_dist = S_total / max(V, 0.5)
        tf = float(max(5.0, tf_by_sink, tf_by_dist))

        if S_total < 1e-6:
            s_norm = np.zeros_like(s_cum)
        else:
            s_norm = s_cum / S_total

        # Down coordinate: from -alt (start) to 0 (target)
        down = -alt * (1.0 - s_norm)
        v_down = alt / tf if tf > 1e-6 else sink

        waypoints: List[Waypoint] = []
        controls: List[Control] = []
        for i in range(xy.shape[0]):
            t = float(tf * s_norm[i])
            p = np.array([xy[i, 0], xy[i, 1], down[i]], dtype=float)

            # Approx velocity from next point (or previous for last).
            if xy.shape[0] >= 2:
                if i < xy.shape[0] - 1:
                    j = i + 1
                else:
                    j = i - 1
                dt = float(abs(tf * (s_norm[j] - s_norm[i])) + 1e-6)
                v_xy = (xy[j] - xy[i]) / dt
            else:
                v_xy = np.array([V, 0.0], dtype=float)

            v = np.array([v_xy[0], v_xy[1], float(v_down)], dtype=float)

            yaw = float(np.arctan2(v[1], v[0]))
            q = yaw_only_quat_wxyz(yaw)
            s = State(p_I=p, v_I=v, q_IB=q, w_B=np.zeros(3), t=t)
            waypoints.append(Waypoint(t=t, state=s))
            controls.append(Control(delta_L=brake, delta_R=brake))

        return Trajectory(waypoints=waypoints, controls=controls, trajectory_type=traj_type, metadata={"template": True})

    def _generate_trajectory(self, scenario: Scenario, traj_type: TrajectoryType, brake: float = 0.2) -> Trajectory:
        bearing = float(np.deg2rad(scenario.target_bearing_deg))
        xy = self._path_xy(traj_type, float(scenario.target_distance_m), bearing)
        return self._trajectory_from_xy(scenario, traj_type, xy, brake=brake)

    def generate_library(self, scenario_cfg: ScenarioConfig, output_path: str) -> TrajectoryLibrary:
        scenarios = self.enumerate_scenarios(scenario_cfg)
        trajectories: List[LibraryTrajectory] = []

        for s in scenarios:
            for traj_type in (TrajectoryType.DIRECT, TrajectoryType.S_TURN, TrajectoryType.RACETRACK, TrajectoryType.SPIRAL):
                traj = self._generate_trajectory(s, traj_type)
                traj_metrics = compute_trajectory_metrics(traj)
                traj.metadata = {**(traj.metadata or {}), **traj_metrics}
                trajectories.append(
                    LibraryTrajectory(
                        scenario=s,
                        trajectory_type=traj_type,
                        trajectory=traj,
                        cost=0.0,
                        metadata={"generator": "kinematic_templates", "trajectory_metrics": traj_metrics},
                    )
                )

        lib = TrajectoryLibrary(trajectories)
        lib.build_index()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        lib.save(output_path)
        return lib


class GPMTrajectoryLibraryGenerator:
    """
    Offline library generator using GPM trajectory optimization (parallel, multi-scenario).

    Notes:
    - Uses kinematic templates as warm-starts.
    - Uses control reference tracking (u_ref) to bias trajectory shapes per TrajectoryType.
    - Stores only successful solutions (configurable via acceptance thresholds).
    """

    def __init__(self, gpm_config: GPMGenerationConfig, polar: Optional[PolarTable] = None) -> None:
        self.gpm_config = gpm_config
        self.polar = polar or PolarTable()

    def generate_library(
        self,
        scenario_cfg: ScenarioConfig,
        output_path: str,
        num_workers: int = 1,
        trajectory_types: Optional[Sequence[TrajectoryType]] = None,
    ) -> TrajectoryLibrary:
        scenarios = TrajectoryLibraryGenerator(polar=self.polar).enumerate_scenarios(scenario_cfg)
        types = list(trajectory_types or (TrajectoryType.DIRECT, TrajectoryType.S_TURN, TrajectoryType.RACETRACK, TrajectoryType.SPIRAL))

        tasks: List[Tuple[Scenario, TrajectoryType]] = [(s, t) for s in scenarios for t in types]

        results: List[LibraryTrajectory] = []

        total = len(tasks)
        start = time.perf_counter()
        last_time = start
        last_count = 0

        def _print_progress(done: int) -> None:
            nonlocal last_time, last_count
            now = time.perf_counter()
            step = max(1, total // 100)
            if done != total and (done - last_count) < step and (now - last_time) < 1.0:
                return
            frac = 0.0 if total <= 0 else float(done) / float(total)
            bar_len = 30
            filled = int(round(bar_len * frac))
            bar = "#" * filled + "-" * (bar_len - filled)
            elapsed = now - start
            eta = (elapsed / frac - elapsed) if frac > 1e-9 else 0.0
            eta_str = "?" if frac <= 1e-9 else f"{eta:5.0f}s"
            print(f"\r[{bar}] {done}/{total} {frac*100:5.1f}% eta {eta_str}", end="", flush=True)
            if done >= total:
                print()
            last_time = now
            last_count = done

        if int(num_workers) <= 1:
            _init_gpm_worker(self.gpm_config)
            done = 0
            for task in tasks:
                item = _solve_gpm_task(task)
                if item is not None:
                    results.append(item)
                done += 1
                _print_progress(done)
        else:
            import multiprocessing as mp

            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=int(num_workers), initializer=_init_gpm_worker, initargs=(self.gpm_config,)) as pool:
                done = 0
                for item in pool.imap_unordered(_solve_gpm_task, tasks, chunksize=1):
                    if item is not None:
                        results.append(item)
                    done += 1
                    _print_progress(done)

        lib = TrajectoryLibrary(results)
        lib.build_index()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        lib.save(output_path)
        return lib
