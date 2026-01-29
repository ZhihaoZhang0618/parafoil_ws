#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path

import yaml

from parafoil_planner_v3.trajectory_library.library_generator import (
    GPMGenerationConfig,
    GPMTrajectoryLibraryGenerator,
    ScenarioConfig,
    TrajectoryLibraryGenerator,
)
from parafoil_planner_v3.types import TrajectoryType


def _load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    # Handle ROS2-style nesting
    if "parafoil_planner_v3" in data:
        data = data["parafoil_planner_v3"]
    if "ros__parameters" in data:
        data = data["ros__parameters"]
    return data


def _read_cpu_times() -> tuple[int, int] | None:
    try:
        with open("/proc/stat", "r") as f:
            line = f.readline()
        if not line.startswith("cpu "):
            return None
        parts = line.split()[1:]
        vals = [int(p) for p in parts[:8]]
        idle = vals[3] + vals[4]
        total = sum(vals)
        return total, idle
    except Exception:
        return None


def _estimate_cpu_usage(interval_s: float = 0.2) -> float | None:
    first = _read_cpu_times()
    if first is None:
        return None
    time.sleep(max(0.05, float(interval_s)))
    second = _read_cpu_times()
    if second is None:
        return None
    total = second[0] - first[0]
    idle = second[1] - first[1]
    if total <= 0:
        return None
    return max(0.0, min(1.0, 1.0 - idle / total))


def _resolve_num_workers(raw_value, task_count: int) -> tuple[int, float | None]:
    cpu_count = int(os.cpu_count() or 1)
    reserve_frac = 0.2
    max_workers = max(1, int(cpu_count * (1.0 - reserve_frac)))

    if isinstance(raw_value, str):
        raw = raw_value.strip().lower()
        if raw in {"auto", "max", "dynamic"}:
            raw_value = 0
        else:
            try:
                raw_value = int(raw)
            except Exception:
                raw_value = 0

    if raw_value is None:
        raw_value = 0

    try:
        requested = int(raw_value)
    except Exception:
        requested = 0

    if requested <= 0:
        usage = _estimate_cpu_usage()
        if usage is None:
            workers = max_workers
        else:
            free = max(0.0, 1.0 - usage - reserve_frac)
            workers = max(1, int(round(cpu_count * free)))
            workers = min(workers, max_workers)
        return max(1, min(workers, max(task_count, 1))), usage

    return max(1, min(requested, max_workers, max(task_count, 1))), None


def _build_tasks(scenario_cfg: ScenarioConfig, types: list[TrajectoryType]) -> list[tuple]:
    scenarios = TrajectoryLibraryGenerator().enumerate_scenarios(scenario_cfg)
    return [(s, t) for s in scenarios for t in types]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate parafoil_planner_v3 trajectory library (offline).")
    parser.add_argument("--config", type=str, required=True, help="Path to library_params.yaml")
    parser.add_argument("--output", type=str, required=True, help="Output pickle path")
    parser.add_argument("--method", type=str, default="", help="Override generation method: template|gpm")
    parser.add_argument("--estimate", action="store_true", help="Sample a subset and estimate full runtime (GPM only).")
    parser.add_argument("--sample-fraction", type=float, default=0.0, help="Sample fraction for estimate (e.g. 0.01).")
    parser.add_argument("--sample-count", type=int, default=0, help="Sample count for estimate (overrides fraction).")
    parser.add_argument("--sample-seed", type=int, default=7, help="Random seed for estimate sampling.")
    args = parser.parse_args()

    cfg = _load_yaml(args.config)
    gen = cfg.get("generation", {})

    scenario_cfg = ScenarioConfig(
        wind_speeds=gen.get("wind_speeds", [0.0, 2.0, 4.0]),
        wind_directions_deg=gen.get("wind_directions", [0, 90, 180, 270]),
        initial_altitudes_m=gen.get("initial_altitudes", [50, 80, 120]),
        target_distances_m=gen.get("target_distances", [50, 100, 150]),
        target_bearings_deg=gen.get("target_bearings", [0, 90, 180, 270]),
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    method = str(args.method).strip() or str(gen.get("method", "template")).strip()
    method = method.lower()
    if method not in {"template", "gpm"}:
        raise SystemExit(f"Unknown method '{method}', expected template|gpm")

    traj_types = gen.get("trajectory_types", ["DIRECT", "S_TURN", "RACETRACK", "SPIRAL"])
    types = [TrajectoryType(t) for t in traj_types]

    if method == "template":
        if args.estimate:
            print("Estimate mode is only supported for GPM generation (template is already fast).")
            return
        lib = TrajectoryLibraryGenerator().generate_library(scenario_cfg, str(out))
        print(f"Generated TEMPLATE library: {len(lib)} trajectories -> {out}")
        return

    gpm = gen.get("gpm", {})
    gpm_cfg = GPMGenerationConfig(
        method=str(gpm.get("method", "SLSQP")),
        num_nodes=int(gpm.get("num_nodes", 20)),
        scheme=str(gpm.get("scheme", "LGL")),
        maxiter=int(gpm.get("max_iterations", 400)),
        ftol=float(gpm.get("tolerance", 1e-6)),
        w_u_ref=float(gpm.get("w_u_ref", 2.0)),
        dynamics_mode=str(gpm.get("dynamics_mode", "simplified")),
        sixdof_ratio=float(gpm.get("sixdof_ratio", 0.0)),
        sixdof_seed=int(gpm.get("sixdof_seed", 0)),
        delta_a_amp=float(gpm.get("delta_a_amp", 0.4)),
        s_turn_cycles=float(gpm.get("s_turn_cycles", 2.0)),
        spiral_delta_a_sign=float(gpm.get("spiral_delta_a_sign", 1.0)),
        rollout_dt=float(gpm.get("rollout_dt", 0.2)),
        brake_min=float(gpm.get("brake_min", 0.0)),
        brake_max=float(gpm.get("brake_max", 0.6)),
        brake_fallback=float(gpm.get("brake_fallback", 0.2)),
        max_violation_accept=float(gpm.get("max_violation_accept", 0.5)),
        solve_mode=str(gpm.get("solve_mode", "least_squares")),
        lsq_max_nfev=int(gpm.get("lsq_max_nfev", 300)),
        lsq_w_dynamics=float(gpm.get("lsq_w_dynamics", 1.0)),
        lsq_w_boundary=float(gpm.get("lsq_w_boundary", 10.0)),
        lsq_w_ineq=float(gpm.get("lsq_w_ineq", 10.0)),
        shape_enforce=bool(gpm.get("shape_enforce", True)),
        shape_turn_eps_deg=float(gpm.get("shape_turn_eps_deg", 3.0)),
        direct_max_total_turn_deg=float(gpm.get("direct_max_total_turn_deg", 60.0)),
        direct_max_net_turn_deg=float(gpm.get("direct_max_net_turn_deg", 45.0)),
        s_turn_min_total_turn_deg=float(gpm.get("s_turn_min_total_turn_deg", 90.0)),
        s_turn_max_net_turn_deg=float(gpm.get("s_turn_max_net_turn_deg", 45.0)),
        s_turn_min_sign_changes=int(gpm.get("s_turn_min_sign_changes", 1)),
        racetrack_min_total_turn_deg=float(gpm.get("racetrack_min_total_turn_deg", 150.0)),
        racetrack_max_net_turn_deg=float(gpm.get("racetrack_max_net_turn_deg", 60.0)),
        racetrack_min_turn_clusters=int(gpm.get("racetrack_min_turn_clusters", 2)),
        racetrack_min_turn_fraction=float(gpm.get("racetrack_min_turn_fraction", 0.75)),
        spiral_min_turns=float(gpm.get("spiral_min_turns", 1.0)),
        spiral_min_turn_fraction=float(gpm.get("spiral_min_turn_fraction", 0.9)),
    )

    task_count = len(scenario_cfg.wind_speeds) * len(scenario_cfg.wind_directions_deg) * len(scenario_cfg.initial_altitudes_m) * len(scenario_cfg.target_distances_m) * len(scenario_cfg.target_bearings_deg) * len(types)
    num_workers, usage = _resolve_num_workers(gen.get("num_workers", "auto"), task_count)
    if usage is None:
        print(f"Using workers={num_workers} (cpu_count={os.cpu_count()})")
    else:
        print(f"Auto workers={num_workers} (cpu_count={os.cpu_count()}, cpu_usage={usage:.0%}, tasks={task_count})")

    if args.estimate:
        tasks = _build_tasks(scenario_cfg, types)
        total = len(tasks)
        if total == 0:
            print("[estimate] No tasks found; check config.")
            return
        sample_n = int(args.sample_count or 0)
        if sample_n <= 0:
            frac = float(args.sample_fraction or 0.0)
            sample_n = int(round(total * frac)) if frac > 0.0 else max(1, total // 100)
        sample_n = max(1, min(sample_n, total))
        if sample_n < total:
            rng = random.Random(int(args.sample_seed))
            tasks = rng.sample(tasks, sample_n)
        print(f"[estimate] Sampling {sample_n}/{total} tasks (seed={int(args.sample_seed)})...")
        start = time.perf_counter()
        GPMTrajectoryLibraryGenerator(gpm_cfg).generate_library_from_tasks(
            tasks=tasks,
            output_path=str(out),
            num_workers=num_workers,
            save=False,
        )
        elapsed = time.perf_counter() - start
        rate = sample_n / max(elapsed, 1e-9)
        est = float(total) / max(rate, 1e-9)
        print(
            "[estimate] "
            f"elapsed={elapsed:0.1f}s avg={elapsed/sample_n:0.3f}s/task "
            f"throughput={rate:0.2f} task/s -> est_total={est/60.0:0.1f} min"
        )
        return

    lib = GPMTrajectoryLibraryGenerator(gpm_cfg).generate_library(
        scenario_cfg=scenario_cfg,
        output_path=str(out),
        num_workers=num_workers,
        trajectory_types=types,
    )
    print(f"Generated GPM library: {len(lib)} trajectories -> {out} (workers={num_workers})")


if __name__ == "__main__":
    main()
