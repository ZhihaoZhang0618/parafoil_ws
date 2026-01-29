#!/usr/bin/env python3
"""
Performance benchmark for parafoil_planner_v3.

Targets (from CLAUDE_PATH_PLANNING_README.md Section 9):
- GPM solve time (N=30): < 1.0 s
- Library match time: < 10 ms
- Trajectory adaptation time: < 50 ms
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from parafoil_planner_v3.dynamics.parafoil_6dof import SixDOFDynamics
from parafoil_planner_v3.dynamics.simplified_model import KinematicYawGlideDynamics
from parafoil_planner_v3.dynamics.aerodynamics import PolarTable
from parafoil_planner_v3.optimization import GPMCollocation, GPMSolver
from parafoil_planner_v3.planner_core import PlannerConfig, PlannerCore
from parafoil_planner_v3.trajectory_library.library_manager import TrajectoryLibrary
from parafoil_planner_v3.trajectory_library.scenario_features import compute_scenario_features
from parafoil_planner_v3.trajectory_library.trajectory_adapter import adapt_trajectory
from parafoil_planner_v3.types import State, Target, Wind


@dataclass
class BenchmarkResult:
    name: str
    target_ms: float
    samples: int
    mean_ms: float
    std_ms: float
    p50_ms: float
    p95_ms: float
    max_ms: float
    min_ms: float
    pass_rate: float  # percentage of samples under target
    status: str  # PASS / FAIL / WARN


def _stats(times_s: List[float], target_s: float) -> Dict[str, float]:
    arr = np.array(times_s, dtype=float) * 1000  # to ms
    target_ms = target_s * 1000
    under = np.sum(arr < target_ms)
    return {
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "max_ms": float(np.max(arr)),
        "min_ms": float(np.min(arr)),
        "pass_rate": float(under / len(arr)) if len(arr) > 0 else 0.0,
    }


def benchmark_gpm_solve(
    N: int = 30,
    scheme: str = "LGL",
    runs: int = 10,
    dynamics_mode: str = "simplified",
    seed: int = 42,
) -> BenchmarkResult:
    """Benchmark GPM solve time for N nodes."""
    rng = np.random.default_rng(seed)

    if dynamics_mode == "6dof":
        dyn = SixDOFDynamics()
    else:
        polar = PolarTable()
        dyn = KinematicYawGlideDynamics(polar=polar)

    wind_I = np.array([0.0, 2.0, 0.0], dtype=float)

    def f(x: np.ndarray, u: np.ndarray, t: float) -> np.ndarray:
        return dyn.f_vector(x, u, t, wind_I=wind_I)

    gpm = GPMCollocation(N=N, scheme=scheme)
    solver = GPMSolver(f=f, gpm=gpm)

    times: List[float] = []
    for _ in range(runs):
        alt = float(rng.uniform(50, 150))
        dist = float(rng.uniform(50, 200))
        bearing = float(rng.uniform(-np.pi, np.pi))
        x0 = State(
            p_I=np.array([-dist * np.cos(bearing), -dist * np.sin(bearing), -alt]),
            v_I=np.array([4.5 * np.cos(bearing), 4.5 * np.sin(bearing), 0.9]),
            q_IB=np.array([1.0, 0.0, 0.0, 0.0]),
            w_B=np.zeros(3),
            t=0.0,
        ).to_vector()
        p_target = np.array([0.0, 0.0, 0.0], dtype=float)

        t0 = time.perf_counter()
        try:
            solver.solve(x0=x0, p_target=p_target, tf_guess=max(30.0, dist / 4.0))
        except Exception:
            pass
        times.append(time.perf_counter() - t0)

    target_s = 1.0  # 1 second target
    stats = _stats(times, target_s)
    status = "PASS" if stats["p95_ms"] < target_s * 1000 else ("WARN" if stats["mean_ms"] < target_s * 1000 else "FAIL")

    return BenchmarkResult(
        name=f"GPM_solve_N{N}_{scheme}_{dynamics_mode}",
        target_ms=target_s * 1000,
        samples=runs,
        status=status,
        **stats,
    )


def benchmark_library_match(
    library_path: str,
    runs: int = 100,
    seed: int = 42,
) -> BenchmarkResult:
    """Benchmark KNN library match time."""
    lib = TrajectoryLibrary.load(library_path)
    rng = np.random.default_rng(seed)

    times: List[float] = []
    for _ in range(runs):
        alt = float(rng.uniform(50, 150))
        dist = float(rng.uniform(50, 200))
        bearing = float(rng.uniform(-np.pi, np.pi))
        ws = float(rng.uniform(0, 4))
        wd = float(rng.uniform(0, 2 * np.pi))

        state = State(
            p_I=np.array([-dist * np.cos(bearing), -dist * np.sin(bearing), -alt]),
            v_I=np.array([4.5, 0.0, 0.9]),
            q_IB=np.array([1.0, 0.0, 0.0, 0.0]),
            w_B=np.zeros(3),
            t=0.0,
        )
        target = Target(p_I=np.zeros(3))
        wind = Wind(v_I=np.array([ws * np.cos(wd), ws * np.sin(wd), 0.0]))

        feats = compute_scenario_features(state, target, wind)

        t0 = time.perf_counter()
        lib.query_knn(feats, k=5)
        times.append(time.perf_counter() - t0)

    target_s = 0.010  # 10 ms target
    stats = _stats(times, target_s)
    status = "PASS" if stats["p95_ms"] < target_s * 1000 else ("WARN" if stats["mean_ms"] < target_s * 1000 else "FAIL")

    return BenchmarkResult(
        name="library_knn_match",
        target_ms=target_s * 1000,
        samples=runs,
        status=status,
        **stats,
    )


def benchmark_trajectory_adapt(
    library_path: str,
    runs: int = 100,
    seed: int = 42,
) -> BenchmarkResult:
    """Benchmark trajectory adaptation time."""
    lib = TrajectoryLibrary.load(library_path)
    rng = np.random.default_rng(seed)

    times: List[float] = []
    for _ in range(runs):
        alt = float(rng.uniform(50, 150))
        dist = float(rng.uniform(50, 200))
        bearing = float(rng.uniform(-np.pi, np.pi))
        ws = float(rng.uniform(0, 4))
        wd = float(rng.uniform(0, 2 * np.pi))

        state = State(
            p_I=np.array([-dist * np.cos(bearing), -dist * np.sin(bearing), -alt]),
            v_I=np.array([4.5, 0.0, 0.9]),
            q_IB=np.array([1.0, 0.0, 0.0, 0.0]),
            w_B=np.zeros(3),
            t=0.0,
        )
        target = Target(p_I=np.zeros(3))
        wind = Wind(v_I=np.array([ws * np.cos(wd), ws * np.sin(wd), 0.0]))

        feats = compute_scenario_features(state, target, wind)
        _, idx = lib.query_knn(feats, k=1)
        lib_traj = lib[int(np.atleast_1d(idx)[0])]

        t0 = time.perf_counter()
        adapt_trajectory(lib_traj, state, target, wind)
        times.append(time.perf_counter() - t0)

    target_s = 0.050  # 50 ms target
    stats = _stats(times, target_s)
    status = "PASS" if stats["p95_ms"] < target_s * 1000 else ("WARN" if stats["mean_ms"] < target_s * 1000 else "FAIL")

    return BenchmarkResult(
        name="trajectory_adapt",
        target_ms=target_s * 1000,
        samples=runs,
        status=status,
        **stats,
    )


def benchmark_planner_plan(
    library_path: Optional[str] = None,
    runs: int = 20,
    seed: int = 42,
    use_library: bool = True,
    dynamics_mode: str = "simplified",
) -> BenchmarkResult:
    """Benchmark full PlannerCore.plan() call."""
    rng = np.random.default_rng(seed)

    if dynamics_mode == "6dof":
        dyn = SixDOFDynamics()
    else:
        polar = PolarTable()
        dyn = KinematicYawGlideDynamics(polar=polar)

    lib = None
    if use_library and library_path:
        try:
            lib = TrajectoryLibrary.load(library_path)
        except Exception:
            pass

    cfg = PlannerConfig(
        gpm_num_nodes=20,
        gpm_scheme="LGL",
        use_library=use_library and lib is not None,
        enable_gpm_fine_tuning=False,  # Disable fine-tuning for benchmark
    )
    planner = PlannerCore(dynamics=dyn, config=cfg, library=lib)

    times: List[float] = []
    for _ in range(runs):
        alt = float(rng.uniform(50, 150))
        dist = float(rng.uniform(50, 200))
        bearing = float(rng.uniform(-np.pi, np.pi))
        ws = float(rng.uniform(0, 4))
        wd = float(rng.uniform(0, 2 * np.pi))

        state = State(
            p_I=np.array([-dist * np.cos(bearing), -dist * np.sin(bearing), -alt]),
            v_I=np.array([4.5 * np.cos(bearing), 4.5 * np.sin(bearing), 0.9]),
            q_IB=np.array([1.0, 0.0, 0.0, 0.0]),
            w_B=np.zeros(3),
            t=0.0,
        )
        target = Target(p_I=np.zeros(3))
        wind = Wind(v_I=np.array([ws * np.cos(wd), ws * np.sin(wd), 0.0]))

        t0 = time.perf_counter()
        try:
            planner.plan(state, target, wind)
        except Exception:
            pass
        times.append(time.perf_counter() - t0)

    # Target depends on mode: library ~50ms, GPM ~1s
    target_s = 0.1 if use_library else 1.0
    stats = _stats(times, target_s)
    status = "PASS" if stats["p95_ms"] < target_s * 1000 else ("WARN" if stats["mean_ms"] < target_s * 1000 else "FAIL")

    name = f"planner_plan_{'lib' if use_library else 'gpm'}_{dynamics_mode}"
    return BenchmarkResult(
        name=name,
        target_ms=target_s * 1000,
        samples=runs,
        status=status,
        **stats,
    )


def _write_html_report(results: List[BenchmarkResult], output_path: Path) -> None:
    """Generate HTML benchmark report."""
    html = """<!DOCTYPE html>
<html>
<head>
    <title>parafoil_planner_v3 Performance Benchmark</title>
    <style>
        body { font-family: 'Segoe UI', sans-serif; margin: 40px; background: #f5f5f5; }
        h1 { color: #333; }
        table { border-collapse: collapse; width: 100%; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        th, td { padding: 12px 15px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #4a90d9; color: white; }
        tr:hover { background: #f1f1f1; }
        .pass { color: #28a745; font-weight: bold; }
        .fail { color: #dc3545; font-weight: bold; }
        .warn { color: #ffc107; font-weight: bold; }
        .summary { margin: 20px 0; padding: 15px; background: #e8f4fd; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>Performance Benchmark Report</h1>
    <div class="summary">
        <strong>Targets (Section 9):</strong>
        GPM solve (N=30) &lt; 1.0s |
        Library match &lt; 10ms |
        Trajectory adapt &lt; 50ms
    </div>
    <table>
        <tr>
            <th>Benchmark</th>
            <th>Samples</th>
            <th>Target (ms)</th>
            <th>Mean (ms)</th>
            <th>P50 (ms)</th>
            <th>P95 (ms)</th>
            <th>Max (ms)</th>
            <th>Pass Rate</th>
            <th>Status</th>
        </tr>
"""
    for r in results:
        status_class = r.status.lower()
        html += f"""        <tr>
            <td>{r.name}</td>
            <td>{r.samples}</td>
            <td>{r.target_ms:.1f}</td>
            <td>{r.mean_ms:.2f}</td>
            <td>{r.p50_ms:.2f}</td>
            <td>{r.p95_ms:.2f}</td>
            <td>{r.max_ms:.2f}</td>
            <td>{r.pass_rate*100:.1f}%</td>
            <td class="{status_class}">{r.status}</td>
        </tr>
"""
    html += """    </table>
    <p style="color: #666; margin-top: 20px;">
        Generated by parafoil_planner_v3/scripts/benchmark_solver.py
    </p>
</body>
</html>
"""
    output_path.write_text(html)


def main() -> None:
    parser = argparse.ArgumentParser(description="Performance benchmark for parafoil_planner_v3.")
    parser.add_argument("--runs", type=int, default=20, help="Number of runs per benchmark")
    parser.add_argument("--library", type=str, default="", help="Path to trajectory library for match/adapt benchmarks")
    parser.add_argument("--output", type=str, default="reports/benchmark_report.html", help="Output file (html or json)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dynamics-mode", type=str, default="simplified", choices=["simplified", "6dof"])
    parser.add_argument("--skip-gpm", action="store_true", help="Skip GPM benchmarks (slow)")
    args = parser.parse_args()

    results: List[BenchmarkResult] = []

    # GPM solve benchmarks
    if not args.skip_gpm:
        print("[benchmark] Running GPM solve N=20...")
        results.append(benchmark_gpm_solve(N=20, runs=args.runs, dynamics_mode=args.dynamics_mode, seed=args.seed))

        print("[benchmark] Running GPM solve N=30...")
        results.append(benchmark_gpm_solve(N=30, runs=args.runs, dynamics_mode=args.dynamics_mode, seed=args.seed))

    # Library benchmarks (if library provided)
    if args.library:
        print("[benchmark] Running library KNN match...")
        results.append(benchmark_library_match(args.library, runs=args.runs * 5, seed=args.seed))

        print("[benchmark] Running trajectory adaptation...")
        results.append(benchmark_trajectory_adapt(args.library, runs=args.runs * 5, seed=args.seed))

        print("[benchmark] Running planner with library...")
        results.append(benchmark_planner_plan(
            library_path=args.library,
            runs=args.runs,
            use_library=True,
            dynamics_mode=args.dynamics_mode,
            seed=args.seed,
        ))

    # Planner without library (GPM only)
    if not args.skip_gpm:
        print("[benchmark] Running planner without library (GPM)...")
        results.append(benchmark_planner_plan(
            library_path=None,
            runs=min(args.runs, 10),
            use_library=False,
            dynamics_mode=args.dynamics_mode,
            seed=args.seed,
        ))

    # Print summary
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    for r in results:
        status_icon = "✅" if r.status == "PASS" else ("⚠️" if r.status == "WARN" else "❌")
        print(f"{status_icon} {r.name}: mean={r.mean_ms:.2f}ms p95={r.p95_ms:.2f}ms target={r.target_ms:.0f}ms [{r.status}]")

    # Write output
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".json":
        out.write_text(json.dumps([asdict(r) for r in results], indent=2))
    else:
        _write_html_report(results, out)
    print(f"\nReport written to: {out}")


if __name__ == "__main__":
    main()
