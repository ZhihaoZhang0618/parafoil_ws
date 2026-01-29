#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np

from parafoil_planner_v3.dynamics.parafoil_6dof import SixDOFDynamics
from parafoil_planner_v3.dynamics.simplified_model import KinematicYawGlideDynamics
from parafoil_planner_v3.offline.e2e import Scenario, make_initial_state
from parafoil_planner_v3.planner_core import PlannerConfig, PlannerCore
from parafoil_planner_v3.trajectory_library.library_manager import TrajectoryLibrary
from parafoil_planner_v3.types import Control, State, Trajectory, Wind
from parafoil_planner_v3.utils.quaternion_utils import quat_to_rpy, wrap_pi
from parafoil_planner_v3.reporting.report_utils import histogram_svg, multi_line_svg, render_report, xy_path_svg


def _stats(values: list[float]) -> dict:
    if not values:
        return {"mean": 0.0, "std": 0.0, "p50": 0.0, "p95": 0.0}
    arr = np.asarray(values, dtype=float).reshape(-1)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


def _integrate_open_loop(
    dynamics: SixDOFDynamics | KinematicYawGlideDynamics,
    trajectory: Trajectory,
    initial_state: State,
    wind: Wind,
    dt_max: float,
) -> list[State]:
    if not trajectory.waypoints:
        return [initial_state.copy()]
    state = initial_state.copy()
    state.t = float(trajectory.waypoints[0].t)
    states = [state.copy()]
    n_seg = max(len(trajectory.waypoints) - 1, 0)
    for i in range(n_seg):
        t0 = float(trajectory.waypoints[i].t)
        t1 = float(trajectory.waypoints[i + 1].t)
        dt = float(max(t1 - t0, 0.0))
        if dt <= 1e-9:
            states.append(state.copy())
            continue
        ctrl = trajectory.controls[i] if i < len(trajectory.controls) else trajectory.controls[-1]
        ctrl = Control(float(ctrl.delta_L), float(ctrl.delta_R)).clipped()

        n = max(int(np.ceil(dt / max(float(dt_max), 1e-6))), 1)
        dt_sub = float(dt / n)
        state.t = float(t0)
        for _ in range(n):
            state = dynamics.step(state, ctrl, wind, dt=dt_sub)
        # Snap time to the planned node time to avoid numerical drift in long integrations.
        state.t = float(t1)
        states.append(state.copy())
    return states


def _run_one(task: tuple) -> dict:
    (
        scenario_dict,
        N,
        scheme,
        tf_guess,
        enable_warm_start,
        dynamics_mode,
        library_path,
        sim_dt_max,
        record_history,
        terminal_upwind,
        terminal_heading_tol_deg,
        terrain_type,
        terrain_height0_m,
        terrain_slope_n,
        terrain_slope_e,
        terrain_clearance_m,
        no_fly_circles,
    ) = task
    scenario = Scenario(**scenario_dict)
    state, target, wind = make_initial_state(scenario)
    lib = None
    use_library = bool(library_path)
    if use_library:
        lib = TrajectoryLibrary.load(str(library_path))
    cfg = PlannerConfig(
        gpm_num_nodes=int(N),
        gpm_scheme=str(scheme),
        tf_guess=float(tf_guess),
        enable_warm_start=bool(enable_warm_start),
        use_library=use_library,
        enforce_terminal_upwind_heading=bool(terminal_upwind),
        terminal_heading_tol_deg=float(terminal_heading_tol_deg),
        terrain_type=str(terrain_type),
        terrain_height0_m=float(terrain_height0_m),
        terrain_slope_n=float(terrain_slope_n),
        terrain_slope_e=float(terrain_slope_e),
        terrain_clearance_m=float(terrain_clearance_m),
        no_fly_circles=tuple(tuple(float(x) for x in c) for c in (no_fly_circles or ())),
    )
    # PlannerCore expects library inside config; pass via library param.
    dynamics_mode = str(dynamics_mode)
    if dynamics_mode.strip().lower() in {"simplified", "simple"}:
        dynamics = KinematicYawGlideDynamics()
    else:
        dynamics = SixDOFDynamics()
    planner = PlannerCore(dynamics=dynamics, config=cfg, library=lib)
    traj, info = planner.plan(state, target, wind)

    sim_states = _integrate_open_loop(dynamics, traj, state, wind, dt_max=float(sim_dt_max))

    # Compare planned vs simulated
    planned_pos = np.stack([wp.state.p_I for wp in traj.waypoints], axis=0) if traj.waypoints else np.zeros((0, 3))
    planned_vel = np.stack([wp.state.v_I for wp in traj.waypoints], axis=0) if traj.waypoints else np.zeros((0, 3))
    sim_pos = np.stack([s.p_I for s in sim_states], axis=0) if sim_states else np.zeros((0, 3))
    sim_vel = np.stack([s.v_I for s in sim_states], axis=0) if sim_states else np.zeros((0, 3))

    n = int(min(planned_pos.shape[0], sim_pos.shape[0]))
    pos_err = planned_pos[:n] - sim_pos[:n]
    vel_err = planned_vel[:n] - sim_vel[:n]
    pos_err_norm = np.linalg.norm(pos_err, axis=1) if n > 0 else np.zeros((0,), dtype=float)
    vel_err_norm = np.linalg.norm(vel_err, axis=1) if n > 0 else np.zeros((0,), dtype=float)

    position_rmse = float(np.sqrt(np.mean(pos_err_norm**2))) if pos_err_norm.size else 0.0
    velocity_rmse = float(np.sqrt(np.mean(vel_err_norm**2))) if vel_err_norm.size else 0.0
    max_position_error = float(np.max(pos_err_norm)) if pos_err_norm.size else 0.0

    sim_final = sim_states[-1] if sim_states else state
    terminal_position_error_xy = float(np.linalg.norm(sim_final.position_xy - target.position_xy))
    terminal_position_error = float(np.linalg.norm(sim_final.p_I[0:3] - target.p_I))

    # Basic feasibility
    deltas = np.array([[c.delta_L, c.delta_R] for c in traj.controls], dtype=float) if traj.controls else np.zeros((0, 2))
    delta_min = float(np.min(deltas)) if deltas.size else 0.0
    delta_max = float(np.max(deltas)) if deltas.size else 0.0
    max_control_bound_violation = float(max(0.0, -delta_min, delta_max - 1.0))

    times = np.array([wp.t for wp in traj.waypoints], dtype=float) if traj.waypoints else np.zeros((0,), dtype=float)
    max_delta_rate = 0.0
    u_dot_norms: list[float] = []
    if deltas.shape[0] >= 2 and times.size >= 2:
        for k in range(min(deltas.shape[0], times.size) - 1):
            dt = float(times[k + 1] - times[k])
            if dt <= 1e-9:
                continue
            du = (deltas[k + 1] - deltas[k]) / dt
            max_delta_rate = max(max_delta_rate, float(np.max(np.abs(du))))
            u_dot_norms.append(float(np.linalg.norm(du)))

    control_smoothness_mean = float(np.mean(u_dot_norms)) if u_dot_norms else 0.0

    # State constraints
    from parafoil_planner_v3.optimization.solver_interface import SolverConfig

    s_cfg = SolverConfig()
    Vh_list = [float(np.linalg.norm(wp.state.v_I[:2])) for wp in traj.waypoints]
    roll_list = [float(quat_to_rpy(wp.state.q_IB)[0]) for wp in traj.waypoints]
    yaw_rate_list = [float(wp.state.w_B[2]) for wp in traj.waypoints]
    max_Vh_min_violation = float(max(0.0, s_cfg.Vh_min - (min(Vh_list) if Vh_list else s_cfg.Vh_min)))
    max_Vh_max_violation = float(max(0.0, (max(Vh_list) if Vh_list else s_cfg.Vh_max) - s_cfg.Vh_max))
    max_roll_violation = float(max(0.0, (max(abs(r) for r in roll_list) if roll_list else 0.0) - float(s_cfg.roll_max_rad)))
    max_yaw_rate_violation = float(max(0.0, (max(abs(r) for r in yaw_rate_list) if yaw_rate_list else 0.0) - float(s_cfg.yaw_rate_max)))
    max_delta_rate_violation = float(max(0.0, float(max_delta_rate) - float(s_cfg.delta_rate_max)))

    # Optimality
    target_alt = float(target.altitude)
    planned_final_alt = float(traj.waypoints[-1].state.altitude) if traj.waypoints else float(state.altitude)
    planned_alt_used = float(max(state.altitude - planned_final_alt, 0.0))
    sim_alt_used = float(max(state.altitude - sim_final.altitude, 0.0))
    alt_available = float(max(state.altitude - target_alt, 1e-6))
    altitude_efficiency_planned = float(planned_alt_used / alt_available)
    altitude_efficiency_sim = float(sim_alt_used / alt_available)

    control_effort_integral = 0.0
    if deltas.shape[0] >= 1 and times.size >= 2:
        n_seg_eff = min(deltas.shape[0] - 1, times.size - 1)
        for k in range(n_seg_eff):
            dt = float(max(times[k + 1] - times[k], 0.0))
            if dt <= 1e-9:
                continue
            u2 = float(np.dot(deltas[k], deltas[k]))
            control_effort_integral += dt * u2
    total_time = float(times[-1] - times[0]) if times.size >= 2 else float(traj.duration)
    control_effort_mean = float(control_effort_integral / max(total_time, 1e-6))

    # Heading error at terminal (optional quick check)
    heading_final = float(np.arctan2(sim_final.v_I[1], sim_final.v_I[0])) if np.linalg.norm(sim_final.v_I[:2]) > 0.3 else float(quat_to_rpy(sim_final.q_IB)[2])
    desired_heading = float(np.arctan2(target.position_xy[1] - sim_final.position_xy[1], target.position_xy[0] - sim_final.position_xy[0]))
    terminal_heading_error_deg = float(abs(np.rad2deg(wrap_pi(desired_heading - heading_final))))

    metrics = {
        "planner_success": bool(info.success),
        "planner_status": int(info.status),
        "planner_message": str(info.message),
        "solve_time_s": float(info.solve_time),
        "planner_max_violation": float(info.max_violation),
        "planner_terminal_error_m": float(info.terminal_error_m),
        "planned_vs_simulated": {
            "position_rmse_m": float(position_rmse),
            "velocity_rmse_mps": float(velocity_rmse),
            "max_position_error_m": float(max_position_error),
        },
        "terminal_accuracy": {
            "terminal_position_error_m": float(terminal_position_error),
            "terminal_position_error_xy_m": float(terminal_position_error_xy),
            "terminal_heading_error_deg": float(terminal_heading_error_deg),
        },
        "feasibility": {
            "max_control_bound_violation": float(max_control_bound_violation),
            "max_delta_rate_1ps": float(max_delta_rate),
            "max_delta_rate_violation": float(max_delta_rate_violation),
            "max_Vh_min_violation": float(max_Vh_min_violation),
            "max_Vh_max_violation": float(max_Vh_max_violation),
            "max_roll_violation": float(max_roll_violation),
            "max_yaw_rate_violation": float(max_yaw_rate_violation),
        },
        "optimality": {
            "altitude_efficiency_planned": float(altitude_efficiency_planned),
            "altitude_efficiency_simulated": float(altitude_efficiency_sim),
            "control_effort_integral": float(control_effort_integral),
            "control_effort_mean": float(control_effort_mean),
            "control_smoothness_mean": float(control_smoothness_mean),
        },
        "dynamics_mode": "simplified" if dynamics_mode.strip().lower() in {"simplified", "simple"} else "6dof",
    }

    out: dict = {"scenario": asdict(scenario), "metrics": metrics}
    if record_history:
        out["planned"] = traj.to_dict()
        out["simulated_states"] = [s.to_dict() for s in sim_states]
    return out


def _write_html(path: Path, summary: dict, results: list[dict]) -> None:
    payload = {"summary": summary, "runs": results}
    pos_rmse = [r["metrics"]["planned_vs_simulated"]["position_rmse_m"] for r in results]
    term_xy = [r["metrics"]["terminal_accuracy"]["terminal_position_error_xy_m"] for r in results]
    solve_time = [r["metrics"]["solve_time_s"] for r in results]
    smooth = [r["metrics"]["optimality"]["control_smoothness_mean"] for r in results]

    summary_rows = [
        ("n_runs", summary.get("n_runs")),
        ("planner_success_rate", summary.get("planner_success_rate")),
        ("pos_rmse_mean", summary.get("planned_vs_simulated", {}).get("position_rmse_m", {}).get("mean")),
        ("pos_rmse_p95", summary.get("planned_vs_simulated", {}).get("position_rmse_m", {}).get("p95")),
        ("terminal_xy_p95", summary.get("terminal_position_error_xy_m", {}).get("p95")),
        ("solve_time_p95", summary.get("solve_time_s", {}).get("p95")),
    ]
    charts = [
        {"title": "Position RMSE (m)", "svg": histogram_svg(pos_rmse, bins=20, x_label="m")},
        {"title": "Terminal XY Error (m)", "svg": histogram_svg(term_xy, bins=20, x_label="m")},
        {"title": "Solve Time (s)", "svg": histogram_svg(solve_time, bins=20, x_label="s")},
        {"title": "Control Smoothness (mean)", "svg": histogram_svg(smooth, bins=20, x_label="")},
    ]

    # Add detailed plots for the first run if history is recorded
    if results:
        first = results[0]
        planned = first.get("planned", {}) if isinstance(first, dict) else {}
        sim_states = first.get("simulated_states", []) if isinstance(first, dict) else []

        if planned and sim_states:
            planned_pts = [w.get("state", {}).get("position", [0.0, 0.0, 0.0]) for w in planned.get("waypoints", [])]
            sim_pts = [s.get("position", [0.0, 0.0, 0.0]) for s in sim_states]
            planned_xy = [(float(p[0]), float(p[1])) for p in planned_pts]
            sim_xy = [(float(p[0]), float(p[1])) for p in sim_pts]
            charts.append(
                {
                    "title": "XY Path (Planned vs Simulated)",
                    "svg": xy_path_svg(
                        [
                            {"label": "planned", "xy": planned_xy, "color": "#4C78A8"},
                            {"label": "simulated", "xy": sim_xy, "color": "#F28E2B"},
                        ],
                        target_xy=(0.0, 0.0),
                    ),
                }
            )

            planned_t = [float(w.get("t", 0.0)) for w in planned.get("waypoints", [])]
            planned_alt = [float(-w.get("state", {}).get("position", [0.0, 0.0, 0.0])[2]) for w in planned.get("waypoints", [])]
            sim_t = [float(s.get("t", 0.0)) for s in sim_states]
            sim_alt = [float(-s.get("position", [0.0, 0.0, 0.0])[2]) for s in sim_states]
            charts.append(
                {
                    "title": "Altitude Profile (m)",
                    "svg": multi_line_svg(
                        [
                            {"label": "planned", "x": planned_t, "y": planned_alt, "color": "#4C78A8"},
                            {"label": "simulated", "x": sim_t, "y": sim_alt, "color": "#F28E2B"},
                        ],
                        x_label="t (s)",
                        y_label="altitude (m)",
                    ),
                }
            )

            controls = planned.get("controls", []) if isinstance(planned, dict) else []
            if controls:
                u_t = planned_t[: len(controls)]
                u_l = [float(c.get("delta_L", 0.0)) for c in controls]
                u_r = [float(c.get("delta_R", 0.0)) for c in controls]
                charts.append(
                    {
                        "title": "Control Time Series (planned)",
                        "svg": multi_line_svg(
                            [
                                {"label": "delta_L", "x": u_t, "y": u_l, "color": "#59A14F"},
                                {"label": "delta_R", "x": u_t, "y": u_r, "color": "#E15759"},
                            ],
                            x_label="t (s)",
                            y_label="brake",
                        ),
                    }
                )
    html = render_report(
        title="parafoil_planner_v3 - Planner Open-loop Report",
        summary_rows=summary_rows,
        charts=charts,
        payload=payload,
        subtitle="Planned vs open-loop simulated comparison with constraint and timing stats.",
    )
    path.write_text(html)


def main() -> None:
    parser = argparse.ArgumentParser(description="Planner open-loop verification (planned vs open-loop simulated RMSE).")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--parallel", type=int, default=1, help="Number of worker processes (OFFLINE).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--N", type=int, default=20)
    parser.add_argument("--scheme", type=str, default="LGL")
    parser.add_argument("--tf", type=float, default=30.0)
    parser.add_argument("--no-warm-start", action="store_true")
    parser.add_argument("--library", type=str, default="", help="Optional trajectory library pickle path.")
    parser.add_argument("--dynamics-mode", type=str, default="simplified", help="simplified|6dof (OFFLINE sim & planning)")
    parser.add_argument("--sim-dt-max", type=float, default=0.2, help="Max integration step for open-loop rollout.")
    parser.add_argument("--no-history", action="store_true", help="Do not include planned/sim states in report.")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--terminal-upwind", action="store_true", help="Hard constraint: terminal ground-track into wind.")
    parser.add_argument("--terminal-heading-tol-deg", type=float, default=5.0)
    parser.add_argument("--terrain-type", type=str, default="flat", choices=["flat", "plane"])
    parser.add_argument("--terrain-height0", type=float, default=0.0)
    parser.add_argument("--terrain-slope-n", type=float, default=0.0)
    parser.add_argument("--terrain-slope-e", type=float, default=0.0)
    parser.add_argument("--terrain-clearance", type=float, default=0.0)
    parser.add_argument(
        "--no-fly-circle",
        action="append",
        default=[],
        help="Repeatable. Format: center_n,center_e,radius_m[,clearance_m]",
    )
    args = parser.parse_args()

    def _parse_no_fly(specs: Sequence[str]) -> list[tuple[float, float, float, float]]:
        out: list[tuple[float, float, float, float]] = []
        for s in specs:
            parts = [p.strip() for p in str(s).split(",") if p.strip()]
            if len(parts) < 3:
                raise ValueError(f"Bad --no-fly-circle '{s}' (need at least 3 numbers)")
            cn = float(parts[0])
            ce = float(parts[1])
            r = float(parts[2])
            c = float(parts[3]) if len(parts) >= 4 else 0.0
            out.append((cn, ce, r, c))
        return out

    no_fly = _parse_no_fly(args.no_fly_circle)

    rng = np.random.default_rng(int(args.seed))
    scenarios = [
        Scenario(
            altitude_m=float(rng.uniform(50.0, 200.0)),
            distance_m=float(rng.uniform(50.0, 250.0)),
            bearing_deg=float(rng.uniform(-180.0, 180.0)),
            wind_speed_mps=float(rng.uniform(0.0, 6.0)),
            wind_direction_deg=float(rng.uniform(0.0, 360.0)),
        )
        for _ in range(int(args.runs))
    ]

    tasks = [
        (
            asdict(s),
            int(args.N),
            str(args.scheme),
            float(args.tf),
            (not bool(args.no_warm_start)),
            str(args.dynamics_mode),
            str(args.library),
            float(args.sim_dt_max),
            (not bool(args.no_history)),
            bool(args.terminal_upwind),
            float(args.terminal_heading_tol_deg),
            str(args.terrain_type),
            float(args.terrain_height0),
            float(args.terrain_slope_n),
            float(args.terrain_slope_e),
            float(args.terrain_clearance),
            no_fly,
        )
        for s in scenarios
    ]

    results: list[dict] = []
    if int(args.parallel) <= 1:
        for task in tasks:
            results.append(_run_one(task))
    else:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=int(args.parallel)) as pool:
            for item in pool.imap_unordered(_run_one, tasks, chunksize=1):
                results.append(item)

    # Summary metrics
    pos_rmse = [r["metrics"]["planned_vs_simulated"]["position_rmse_m"] for r in results]
    vel_rmse = [r["metrics"]["planned_vs_simulated"]["velocity_rmse_mps"] for r in results]
    max_pos_err = [r["metrics"]["planned_vs_simulated"]["max_position_error_m"] for r in results]
    term_xy = [r["metrics"]["terminal_accuracy"]["terminal_position_error_xy_m"] for r in results]
    term_3d = [r["metrics"]["terminal_accuracy"]["terminal_position_error_m"] for r in results]
    solve_time = [r["metrics"]["solve_time_s"] for r in results]
    smooth = [r["metrics"]["optimality"]["control_smoothness_mean"] for r in results]
    alt_eff_sim = [r["metrics"]["optimality"]["altitude_efficiency_simulated"] for r in results]
    ctrl_eff_mean = [r["metrics"]["optimality"]["control_effort_mean"] for r in results]
    max_violation = [r["metrics"]["planner_max_violation"] for r in results]

    summary = {
        "n_runs": int(len(results)),
        "planner_success_rate": float(np.mean([1.0 if r["metrics"]["planner_success"] else 0.0 for r in results])) if results else 0.0,
        "goal_pass_rates": {
            "position_rmse_lt_1m": float(np.mean([1.0 if float(v) < 1.0 else 0.0 for v in pos_rmse])) if results else 0.0,
            "terminal_xy_lt_5m": float(np.mean([1.0 if float(v) < 5.0 else 0.0 for v in term_xy])) if results else 0.0,
            "solve_time_lt_1s": float(np.mean([1.0 if float(v) < 1.0 else 0.0 for v in solve_time])) if results else 0.0,
        },
        "planned_vs_simulated": {
            "position_rmse_m": _stats([float(x) for x in pos_rmse]),
            "velocity_rmse_mps": _stats([float(x) for x in vel_rmse]),
            "max_position_error_m": _stats([float(x) for x in max_pos_err]),
        },
        "terminal_position_error_xy_m": _stats([float(x) for x in term_xy]),
        "terminal_position_error_m": _stats([float(x) for x in term_3d]),
        "solve_time_s": _stats([float(x) for x in solve_time]),
        "control_smoothness_mean": _stats([float(x) for x in smooth]),
        "altitude_efficiency_simulated": _stats([float(x) for x in alt_eff_sim]),
        "control_effort_mean": _stats([float(x) for x in ctrl_eff_mean]),
        "planner_max_violation": _stats([float(x) for x in max_violation]),
    }

    out_obj = {"summary": summary, "runs": results}

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix.lower() == ".html":
            _write_html(out, summary, results)
        else:
            out.write_text(json.dumps(out_obj, indent=2))
        print(f"Wrote report: {out}")
    else:
        print(json.dumps(out_obj["summary"], indent=2))


if __name__ == "__main__":
    main()
