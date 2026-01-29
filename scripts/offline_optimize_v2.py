#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import yaml

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR / "src" / "parafoil_plannerv2"))
sys.path.append(str(ROOT_DIR / "src" / "parafoil_dynamics"))

from parafoil_dynamics.params import Params
from parafoil_dynamics.state import State, ControlCmd
from parafoil_dynamics.dynamics import dynamics
from parafoil_dynamics.integrators import rk4_step, semi_implicit_step, euler_step
from parafoil_dynamics.math3d import quat_from_euler
from parafoil_dynamics.wind import WindConfig, WindModel

from parafoil_plannerv2.models import PolarTable, select_brake_for_required_slope
from parafoil_plannerv2.planner_core import ParafoilPlannerV2
from parafoil_plannerv2.types import Plan, PlanDebug, PlanMode, PlannerConstraints, PlannerState, Waypoint, WindEstimate


@dataclass
class Scenario:
    steady_speed: float
    steady_dir: float  # radians, 0=N, CW positive (NED)
    gust_magnitude: float
    gust_interval: float
    gust_duration: float
    altitude: float
    target_x: float
    target_y: float
    seed: int


@dataclass
class ControllerParams:
    L1: float
    K_yaw: float
    max_delta: float
    capture_radius: float
    lock_radius: float
    terminal_radius: float
    terminal_brake: float
    terminal_alt: float
    distance_blend: float
    altitude_reserve: float
    altitude_safety: float
    max_brake: float
    drift_alt: float
    drift_brake: float
    drift_scale: float
    wind_est_sigma: float
    gust_scale: float
    gust_noise_sigma: float
    steady_bias: np.ndarray


def load_params_from_yaml(path: str) -> Params:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    params = data.get("parafoil_simulator", {}).get("ros__parameters", {})

    kwargs: Dict[str, object] = {}
    for key in [
        "rho", "g", "m", "S", "b", "c", "S_pd", "c_D_pd",
        "c_L0", "c_La", "c_Lds", "c_D0", "c_Da2", "c_Dds",
        "alpha_stall", "alpha_stall_brake", "alpha_stall_width", "c_D_stall",
        "c_Yb", "c_lp", "c_lda", "c_m0", "c_ma", "c_mq",
        "c_nr", "c_nda", "c_nb", "c_n_weath",
        "tau_act", "eps", "V_min", "pendulum_arm",
    ]:
        if key in params:
            kwargs[key if key != "V_min" else "V_min"] = params[key]

    if "I_B_diag" in params:
        kwargs["I_B"] = np.diag(params["I_B_diag"])
    if "r_pd_B" in params:
        kwargs["r_pd_B"] = np.array(params["r_pd_B"], dtype=float)
    if "r_canopy_B" in params:
        kwargs["r_canopy_B"] = np.array(params["r_canopy_B"], dtype=float)

    return Params(**kwargs)


def integrator_step(name: str):
    if name == "euler":
        return euler_step
    if name == "semi_implicit":
        return semi_implicit_step
    return rk4_step


def unit_from_angle(theta: float) -> np.ndarray:
    return np.array([math.cos(theta), math.sin(theta)])


def build_constraints(p: ControllerParams) -> PlannerConstraints:
    return PlannerConstraints(
        final_leg_length=40.0,
        entry_leg_length=120.0,
        hold_distance=100.0,
        racetrack_radius=20.0,
        racetrack_leg_length=80.0,
        max_hold_loops=8,
        min_turn_radius=8.0,
        step_m=2.0,
        capture_radius=p.capture_radius,
        max_brake=p.max_brake,
        altitude_reserve=p.altitude_reserve,
        altitude_safety_margin=p.altitude_safety,
        Vmin=0.5,
        Vg_par_min=0.8,
        z_p=1.64,
    )


def effective_target(
    target: np.ndarray,
    altitude: float,
    target_alt: float,
    wind_est: WindEstimate,
    polar: PolarTable,
    z_p: float,
    drift_alt: float,
    drift_brake: float,
    drift_scale: float,
) -> np.ndarray:
    if drift_scale <= 0.0:
        return target
    _, sink = polar.interpolate(drift_brake)
    sink = max(float(sink), 0.1)
    alt_for_drift = min(altitude, drift_alt)
    t_drift = max((alt_for_drift - target_alt) / sink, 0.0)
    wind_vec = wind_est.wind_vector_robust(z_p)
    drift = wind_vec * t_drift * drift_scale
    return target - drift


def guidance_cmd(
    pos: np.ndarray,
    vel: np.ndarray,
    heading: float,
    path: List[np.ndarray],
    altitude: float,
    wind_est: WindEstimate,
    params: ControllerParams,
    polar: PolarTable,
) -> Tuple[float, float]:
    if len(path) < 2:
        return (0.0, 0.0)

    # Closest index
    d2 = [float(np.dot(w - pos, w - pos)) for w in path]
    i0 = int(np.argmin(d2))

    # Lookahead point
    dist = 0.0
    L1 = max(params.L1, 1e-3)
    p_l1 = path[-1]
    for i in range(i0, len(path) - 1):
        p1, p2 = path[i], path[i + 1]
        ds = float(np.linalg.norm(p2 - p1))
        if dist + ds >= L1:
            t = 0.0 if ds < 1e-6 else (L1 - dist) / ds
            p_l1 = p1 + t * (p2 - p1)
            break
        dist += ds

    # Lateral control
    speed = float(np.linalg.norm(vel))
    if speed < 0.3:
        speed = 0.3
        vel = np.array([speed * math.cos(heading), speed * math.sin(heading)])
    v_hat = vel / (np.linalg.norm(vel) + 1e-9)
    r = p_l1 - pos
    r_norm = float(np.linalg.norm(r))
    if r_norm < 1e-6:
        return (0.0, 0.0)
    r_hat = r / r_norm
    cross = float(v_hat[0] * r_hat[1] - v_hat[1] * r_hat[0])
    dot = float(np.clip(np.dot(v_hat, r_hat), -1.0, 1.0))
    eta = float(np.arctan2(cross, dot))
    kappa = 2.0 * np.sin(eta) / L1
    yaw_rate_cmd = params.K_yaw * speed * kappa
    # Match parafoil_plannerv2 guidance_node default.
    delta_a_raw = -yaw_rate_cmd / 1.70
    delta_a_cmd = float(np.clip(delta_a_raw, -params.max_delta, params.max_delta))

    # Remaining distance (blend path vs direct)
    S_path = 0.0
    S_path += float(np.linalg.norm(path[i0] - pos))
    for i in range(i0, len(path) - 1):
        S_path += float(np.linalg.norm(path[i + 1] - path[i]))
    S_direct = float(np.linalg.norm(path[-1] - pos))
    blend = float(np.clip(params.distance_blend, 0.0, 1.0))
    S_rem = (1.0 - blend) * S_path + blend * S_direct

    H_rem = max(altitude - 0.0 - params.altitude_reserve, 0.0)
    k_req = max((H_rem - params.altitude_safety) / max(S_rem, 5.0), 0.0)

    direction = r_hat
    b_sym, _, _, _ = select_brake_for_required_slope(
        polar,
        direction,
        wind_est.wind_vector_robust(1.64),
        k_req,
        0.5,
        0.8,
        params.max_brake,
    )

    # Terminal braking
    d_target = float(np.linalg.norm(path[-1] - pos))
    if d_target <= params.terminal_radius and altitude <= params.terminal_alt:
        b_sym = max(b_sym, params.terminal_brake)

    b_sym = float(np.clip(b_sym, 0.0, params.max_brake))
    abs_delta = abs(float(delta_a_cmd))
    mean_min = 0.5 * abs_delta
    mean_max = float(params.max_brake) - 0.5 * abs_delta
    b_used = b_sym
    if mean_min <= mean_max:
        b_used = float(np.clip(b_used, mean_min, mean_max))
    else:
        b_used = 0.5 * float(params.max_brake)
        delta_a_cmd = float(np.clip(delta_a_cmd, -float(params.max_brake), float(params.max_brake)))

    left = float(np.clip(b_used + 0.5 * delta_a_cmd, 0.0, params.max_brake))
    right = float(np.clip(b_used - 0.5 * delta_a_cmd, 0.0, params.max_brake))
    return (left, right)


def simulate_episode(
    scenario: Scenario,
    ctrl: ControllerParams,
    params: Params,
    dt: float,
    plan_dt: float,
    max_time: float,
    integrator: str,
) -> Dict[str, float]:
    # Wind config (true)
    steady_vec = scenario.steady_speed * np.array([
        math.cos(scenario.steady_dir),
        math.sin(scenario.steady_dir),
        0.0,
    ])
    wind_cfg = WindConfig(
        enable_steady=True,
        enable_gust=scenario.gust_magnitude > 1e-3,
        enable_colored=False,
        steady_wind=steady_vec,
        gust_interval=scenario.gust_interval,
        gust_duration=scenario.gust_duration,
        gust_magnitude=scenario.gust_magnitude,
        seed=scenario.seed,
    )
    wind_model = WindModel(wind_cfg)
    wind_model.reset()
    rng = np.random.default_rng(scenario.seed + 1)

    # Initial state
    q_IB = quat_from_euler(0.0, 0.0, 0.0)
    # Initial velocity (NED). User requested x=0, y=0, z=-3 m/s.
    state = State(
        p_I=np.array([0.0, 0.0, -scenario.altitude]),
        v_I=np.array([0.0, 0.0, -3.0]),
        q_IB=q_IB,
        w_B=np.zeros(3),
        delta=np.zeros(2),
        t=0.0,
    )

    polar = PolarTable()
    planner = ParafoilPlannerV2(polar=polar, constraints=build_constraints(ctrl))

    target = np.array([scenario.target_x, scenario.target_y])
    target_alt = 0.0

    last_plan_t = -1e9
    path: List[np.ndarray] = []
    target_locked = False
    unreachable_detected = False
    initial_dist = float(np.linalg.norm(target - state.p_I[:2]))

    step_fn = integrator_step(integrator)

    min_dist = 1e9
    t_reach_20 = None

    while state.t < max_time:
        alt = -state.p_I[2]
        pos_xy = state.p_I[:2].copy()
        vel_xy = state.v_I[:2].copy()
        heading = math.atan2(vel_xy[1], vel_xy[0]) if np.linalg.norm(vel_xy) > 0.3 else 0.0

        true_wind = wind_model.get_wind(state.t, dt)
        gust_true = true_wind - steady_vec
        est = steady_vec + ctrl.steady_bias + ctrl.gust_scale * gust_true + rng.normal(0.0, ctrl.gust_noise_sigma, 3)
        est_speed = float(np.linalg.norm(est[:2]))
        if est_speed < 1e-6:
            est_dir = np.array([1.0, 0.0])
        else:
            est_dir = est[:2] / est_speed
        wind_est = WindEstimate(
            speed_mean=est_speed,
            speed_sigma=ctrl.wind_est_sigma,
            direction_hat=est_dir,
        )

        # Plan update
        if state.t - last_plan_t >= plan_dt or not path:
            target_eff = effective_target(
                target, alt, target_alt, wind_est, polar, 1.64,
                ctrl.drift_alt, ctrl.drift_brake, ctrl.drift_scale,
            )
            d_target = float(np.linalg.norm(target_eff - pos_xy))
            if not target_locked and d_target <= ctrl.lock_radius:
                target_locked = True

            plan_state = PlannerState(
                position_xy=pos_xy,
                altitude=alt,
                ground_velocity=vel_xy,
                heading=heading,
                timestamp=state.t,
            )
            plan = planner.plan(plan_state, target_eff, target_alt, wind_est)
            if plan.mode == PlanMode.UNREACHABLE:
                unreachable_detected = True
            if target_locked and plan.mode == PlanMode.UNREACHABLE:
                path = [pos_xy, target_eff]
            else:
                path = [np.array([wp.x, wp.y]) for wp in plan.waypoints]
            last_plan_t = state.t

        left, right = guidance_cmd(pos_xy, vel_xy, heading, path, alt, wind_est, ctrl, polar)
        cmd = ControlCmd.from_left_right(left, right)

        state = step_fn(dynamics, state, cmd, params, dt, true_wind)

        d = float(np.linalg.norm(pos_xy - target))
        min_dist = min(min_dist, d)
        if t_reach_20 is None and d < 20.0:
            t_reach_20 = state.t

        if state.p_I[2] > 0.0:
            landing_xy = state.p_I[:2].copy()
            landing_dist = float(np.linalg.norm(landing_xy - target))
            return {
                "landing_dist": landing_dist,
                "min_dist": min_dist,
                "t_reach_20": -1.0 if t_reach_20 is None else t_reach_20,
                "landing_x": float(landing_xy[0]),
                "landing_y": float(landing_xy[1]),
                "time": state.t,
                "initial_dist": initial_dist,
                "unreachable_detected": 1.0 if unreachable_detected else 0.0,
            }

    # timeout
    landing_xy = state.p_I[:2].copy()
    landing_dist = float(np.linalg.norm(landing_xy - target))
    return {
        "landing_dist": landing_dist,
        "min_dist": min_dist,
        "t_reach_20": -1.0 if t_reach_20 is None else t_reach_20,
        "landing_x": float(landing_xy[0]),
        "landing_y": float(landing_xy[1]),
        "time": state.t,
        "initial_dist": initial_dist,
        "unreachable_detected": 1.0 if unreachable_detected else 0.0,
    }


def sample_scenarios(n: int, seed: int, altitude_min: float, altitude_max: float) -> List[Scenario]:
    rng = random.Random(seed)
    scenarios = []
    for i in range(n):
        steady_speed = rng.uniform(0.0, 6.0)
        steady_dir = rng.uniform(-math.pi, math.pi)
        gust_mag = rng.uniform(0.0, 6.0)
        gust_interval = rng.uniform(6.0, 14.0)
        gust_duration = rng.uniform(1.0, 3.0)
        altitude = rng.uniform(altitude_min, altitude_max)
        target_x = rng.uniform(0.0, 300.0)
        target_y = rng.uniform(0.0, 300.0)
        scenarios.append(Scenario(
            steady_speed=steady_speed,
            steady_dir=steady_dir,
            gust_magnitude=gust_mag,
            gust_interval=gust_interval,
            gust_duration=gust_duration,
            altitude=altitude,
            target_x=target_x,
            target_y=target_y,
            seed=seed * 1000 + i,
        ))
    return scenarios


def sample_controller(rng: random.Random, max_brake: float) -> ControllerParams:
    return ControllerParams(
        L1=rng.uniform(8.0, 25.0),
        K_yaw=rng.uniform(0.8, 1.8),
        max_delta=min(rng.uniform(0.2, 0.5), max_brake),
        capture_radius=rng.uniform(20.0, 50.0),
        lock_radius=rng.uniform(20.0, 60.0),
        terminal_radius=rng.uniform(20.0, 60.0),
        terminal_brake=min(rng.uniform(0.3, max_brake), max_brake),
        terminal_alt=rng.uniform(40.0, 120.0),
        distance_blend=rng.uniform(0.7, 1.0),
        altitude_reserve=rng.uniform(0.0, 10.0),
        altitude_safety=rng.uniform(0.0, 10.0),
        max_brake=max_brake,
        drift_alt=rng.uniform(40.0, 120.0),
        drift_brake=min(rng.uniform(0.3, max_brake), max_brake),
        drift_scale=rng.uniform(0.5, 1.5),
        wind_est_sigma=rng.uniform(0.2, 1.0),
        gust_scale=rng.uniform(0.7, 1.3),
        gust_noise_sigma=rng.uniform(0.2, 1.0),
        steady_bias=np.array([
            rng.uniform(-0.5, 0.5),
            rng.uniform(-0.5, 0.5),
            0.0,
        ]),
    )


def evaluate_controller(
    scenarios: List[Scenario],
    ctrl: ControllerParams,
    params: Params,
    dt: float,
    plan_dt: float,
    max_time: float,
    integrator: str,
    attempt_ratio: float,
) -> Dict[str, float]:
    dists = []
    success_attempt = []
    for sc in scenarios:
        out = simulate_episode(sc, ctrl, params, dt, plan_dt, max_time, integrator)
        dists.append(out["landing_dist"])
        unreachable = out.get("unreachable_detected", 0.0) > 0.5
        initial_dist = float(out.get("initial_dist", 0.0))
        min_dist = float(out.get("min_dist", out["landing_dist"]))
        attempt_ok = unreachable and initial_dist > 1e-6 and min_dist <= attempt_ratio * initial_dist
        success_attempt.append((out["landing_dist"] < 20.0) or attempt_ok)
    d = np.array(dists, dtype=float)
    return {
        "mean": float(np.mean(d)),
        "median": float(np.median(d)),
        "p90": float(np.percentile(d, 90)),
        "success_20m": float(np.mean(d < 20.0)),
        "success_attempt": float(np.mean(success_attempt)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", type=int, default=30)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-brake", type=float, default=0.5)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--plan-dt", type=float, default=1.0)
    ap.add_argument("--max-time", type=float, default=200.0)
    ap.add_argument("--integrator", type=str, default="rk4", choices=["rk4", "semi_implicit", "euler"])
    ap.add_argument("--params-yaml", type=str, default="src/parafoil_simulator_ros/config/params.yaml")
    ap.add_argument("--attempt-ratio", type=float, default=0.7)
    ap.add_argument("--altitude", type=float, default=None)
    ap.add_argument("--altitude-min", type=float, default=15.0)
    ap.add_argument("--altitude-max", type=float, default=300.0)
    args = ap.parse_args()

    params = load_params_from_yaml(args.params_yaml)
    if args.altitude is not None:
        altitude_min = altitude_max = float(args.altitude)
    else:
        altitude_min = float(args.altitude_min)
        altitude_max = float(args.altitude_max)
    scenarios = sample_scenarios(args.scenarios, args.seed, altitude_min, altitude_max)

    rng = random.Random(args.seed + 123)
    best = None
    best_ctrl = None

    for i in range(args.trials):
        ctrl = sample_controller(rng, args.max_brake)
        metrics = evaluate_controller(
            scenarios, ctrl, params, args.dt, args.plan_dt, args.max_time, args.integrator, args.attempt_ratio
        )
        score = metrics["mean"] + 50.0 * (1.0 - metrics["success_attempt"])
        if best is None or score < best:
            best = score
            best_ctrl = (ctrl, metrics)
        print(
            f"[trial {i+1}/{args.trials}] mean={metrics['mean']:.2f} "
            f"median={metrics['median']:.2f} p90={metrics['p90']:.2f} "
            f"succ20={metrics['success_20m']:.2f} succ_attempt={metrics['success_attempt']:.2f}"
        )

    if best_ctrl is None:
        print("No result.")
        return

    ctrl, metrics = best_ctrl
    print("=== best controller ===")
    print(
        f"L1={ctrl.L1:.2f} K={ctrl.K_yaw:.2f} max_delta={ctrl.max_delta:.2f} "
        f"cap={ctrl.capture_radius:.1f} lock={ctrl.lock_radius:.1f} term_r={ctrl.terminal_radius:.1f} "
        f"term_brake={ctrl.terminal_brake:.2f} term_alt={ctrl.terminal_alt:.1f} "
        f"blend={ctrl.distance_blend:.2f} alt_res={ctrl.altitude_reserve:.1f} alt_safe={ctrl.altitude_safety:.1f} "
        f"drift_alt={ctrl.drift_alt:.1f} drift_brake={ctrl.drift_brake:.2f} drift_scale={ctrl.drift_scale:.2f}"
    )
    print(
        f"wind_est_sigma={ctrl.wind_est_sigma:.2f} gust_scale={ctrl.gust_scale:.2f} "
        f"gust_noise_sigma={ctrl.gust_noise_sigma:.2f} bias={ctrl.steady_bias.tolist()}"
    )
    print(
        f"metrics: mean={metrics['mean']:.2f} median={metrics['median']:.2f} "
        f"p90={metrics['p90']:.2f} succ20={metrics['success_20m']:.2f} "
        f"succ_attempt={metrics['success_attempt']:.2f}"
    )


if __name__ == "__main__":
    main()
