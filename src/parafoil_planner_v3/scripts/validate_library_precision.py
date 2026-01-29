#!/usr/bin/env python3
"""
Validate trajectory library precision vs online GPM.

This script compares:
1. Library-only (no fine-tuning) landing precision
2. Library + GPM fine-tuning precision
3. GPM-only precision (baseline)

Outputs a report showing whether the library degrades to a "discrete lookup table"
with poor precision, or maintains acceptable accuracy through adaptation.

Usage:
  python3 validate_library_precision.py --library /tmp/parafoil_library.pkl --runs 50 --output reports/precision.html
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import numpy as np

from parafoil_planner_v3.dynamics.aerodynamics import PolarTable
from parafoil_planner_v3.dynamics.simplified_model import KinematicYawGlideDynamics
from parafoil_planner_v3.offline.e2e import Scenario, simulate_one
from parafoil_planner_v3.planner_core import PlannerConfig, PlannerCore
from parafoil_planner_v3.trajectory_library.library_manager import TrajectoryLibrary
from parafoil_planner_v3.trajectory_library.scenario_features import compute_scenario_features
from parafoil_planner_v3.types import State, Target, Wind


@dataclass
class PrecisionResult:
    mode: str  # library_only, library_fine_tune, gpm_only
    n_runs: int
    landing_error_mean: float
    landing_error_std: float
    landing_error_p50: float
    landing_error_p95: float
    success_rate: float
    fine_tune_rate: float  # fraction of runs that triggered fine-tuning
    solve_time_mean_ms: float


def _generate_scenarios(n: int, seed: int) -> List[Scenario]:
    rng = np.random.default_rng(seed)
    scenarios = []
    for _ in range(n):
        scenarios.append(Scenario(
            altitude_m=float(rng.uniform(50, 150)),
            distance_m=float(rng.uniform(50, 200)),
            bearing_deg=float(rng.uniform(-180, 180)),
            wind_speed_mps=float(rng.uniform(0, 5)),
            wind_direction_deg=float(rng.uniform(0, 360)),
        ))
    return scenarios


def _run_mode(
    scenarios: List[Scenario],
    mode: str,
    library_path: Optional[str] = None,
) -> PrecisionResult:
    """Run a batch of scenarios with a specific planner mode."""

    # Load library if needed
    lib = None
    if library_path and mode in ("library_only", "library_fine_tune"):
        lib = TrajectoryLibrary.load(library_path)

    errors: List[float] = []
    successes: List[bool] = []
    fine_tune_triggered: List[bool] = []
    solve_times: List[float] = []

    for s in scenarios:
        use_library = mode in ("library_only", "library_fine_tune")
        use_gpm = mode == "gpm_only" or mode == "library_fine_tune"

        # Configure planner
        cfg = PlannerConfig(
            use_library=use_library and lib is not None,
            enable_gpm_fine_tuning=(mode == "library_fine_tune"),
            fine_tuning_trigger_m=5.0,  # trigger fine-tuning if terminal error > 5m
        )

        result = simulate_one(
            scenario=s,
            planner_rate_hz=1.0,
            control_rate_hz=20.0,
            max_time_s=200.0,
            L1_distance=20.0,
            use_gpm=use_gpm,
            dynamics_mode="simplified",
            record_history=False,
            planner_config=cfg,
            library=lib if use_library else None,
        )

        errors.append(float(result["metrics"]["landing_error_m"]))
        successes.append(bool(result["metrics"]["success"]))

        # Check if fine-tuning was triggered (look for GPM solve time in message)
        msg = result["metrics"].get("planner_message", "") or ""
        fine_tune_triggered.append("gpm" in msg.lower() or "solve" in msg.lower())

        solve_times.append(float(result["metrics"].get("plan_time_ms", 0.0)))

    errors_arr = np.array(errors, dtype=float)

    return PrecisionResult(
        mode=mode,
        n_runs=len(scenarios),
        landing_error_mean=float(np.mean(errors_arr)),
        landing_error_std=float(np.std(errors_arr)),
        landing_error_p50=float(np.percentile(errors_arr, 50)),
        landing_error_p95=float(np.percentile(errors_arr, 95)),
        success_rate=float(np.mean([1.0 if s else 0.0 for s in successes])),
        fine_tune_rate=float(np.mean([1.0 if f else 0.0 for f in fine_tune_triggered])),
        solve_time_mean_ms=float(np.mean(solve_times)),
    )


def _analyze_library_coverage(library_path: str) -> dict:
    """Analyze the library coverage and feature distribution."""
    lib = TrajectoryLibrary.load(library_path)

    features = []
    for i in range(len(lib)):
        lib_traj = lib[i]
        traj = lib_traj.trajectory
        if traj.waypoints:
            start = traj.waypoints[0].state
            end = traj.waypoints[-1].state
            alt = float(start.altitude)
            dist = float(np.linalg.norm(end.position_xy - start.position_xy))
            features.append({
                "altitude": alt,
                "distance": dist,
                "type": lib_traj.trajectory_type.value,
            })

    altitudes = [f["altitude"] for f in features]
    distances = [f["distance"] for f in features]
    types = {}
    for f in features:
        t = f["type"]
        types[t] = types.get(t, 0) + 1

    return {
        "n_trajectories": len(lib),
        "altitude_range": [float(min(altitudes)), float(max(altitudes))] if altitudes else [0, 0],
        "distance_range": [float(min(distances)), float(max(distances))] if distances else [0, 0],
        "type_distribution": types,
    }


def _write_html_report(
    results: List[PrecisionResult],
    coverage: dict,
    output_path: Path,
) -> None:
    """Generate HTML report."""
    html = """<!DOCTYPE html>
<html>
<head>
    <title>Trajectory Library Precision Analysis</title>
    <style>
        body { font-family: 'Segoe UI', sans-serif; margin: 40px; background: #f5f5f5; }
        h1, h2 { color: #333; }
        table { border-collapse: collapse; width: 100%; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin: 20px 0; }
        th, td { padding: 12px 15px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #4a90d9; color: white; }
        tr:hover { background: #f1f1f1; }
        .good { color: #28a745; font-weight: bold; }
        .warn { color: #ffc107; font-weight: bold; }
        .bad { color: #dc3545; font-weight: bold; }
        .summary { margin: 20px 0; padding: 15px; background: #e8f4fd; border-radius: 5px; }
        .conclusion { margin: 20px 0; padding: 15px; background: #fff3cd; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>Trajectory Library Precision Analysis</h1>

    <div class="summary">
        <strong>Question:</strong> Does the offline trajectory library degrade to a "discrete lookup table" with poor precision?
    </div>

    <h2>Library Coverage</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
"""
    html += f"        <tr><td>Number of trajectories</td><td>{coverage['n_trajectories']}</td></tr>\n"
    html += f"        <tr><td>Altitude range</td><td>{coverage['altitude_range'][0]:.1f} - {coverage['altitude_range'][1]:.1f} m</td></tr>\n"
    html += f"        <tr><td>Distance range</td><td>{coverage['distance_range'][0]:.1f} - {coverage['distance_range'][1]:.1f} m</td></tr>\n"
    html += f"        <tr><td>Type distribution</td><td>{json.dumps(coverage['type_distribution'])}</td></tr>\n"
    html += """    </table>

    <h2>Precision Comparison</h2>
    <table>
        <tr>
            <th>Mode</th>
            <th>Runs</th>
            <th>Mean Error (m)</th>
            <th>P50 Error (m)</th>
            <th>P95 Error (m)</th>
            <th>Success Rate</th>
            <th>Fine-tune Rate</th>
            <th>Solve Time (ms)</th>
        </tr>
"""
    for r in results:
        # Color coding based on p95 error
        if r.landing_error_p95 < 10:
            status_class = "good"
        elif r.landing_error_p95 < 20:
            status_class = "warn"
        else:
            status_class = "bad"

        html += f"""        <tr>
            <td>{r.mode}</td>
            <td>{r.n_runs}</td>
            <td class="{status_class}">{r.landing_error_mean:.2f}</td>
            <td>{r.landing_error_p50:.2f}</td>
            <td>{r.landing_error_p95:.2f}</td>
            <td>{r.success_rate*100:.1f}%</td>
            <td>{r.fine_tune_rate*100:.1f}%</td>
            <td>{r.solve_time_mean_ms:.1f}</td>
        </tr>
"""

    # Determine conclusion
    lib_only = next((r for r in results if r.mode == "library_only"), None)
    gpm_only = next((r for r in results if r.mode == "gpm_only"), None)

    if lib_only and gpm_only:
        degradation = lib_only.landing_error_mean / max(gpm_only.landing_error_mean, 0.1)
        if degradation < 1.5:
            conclusion = f"<span class='good'>Library precision is acceptable</span> (degradation ratio: {degradation:.2f}x)"
            recommendation = "The trajectory library with adaptation provides precision comparable to online GPM. Safe to use in production."
        elif degradation < 3.0:
            conclusion = f"<span class='warn'>Library precision is moderate</span> (degradation ratio: {degradation:.2f}x)"
            recommendation = "Consider: (1) Denser library grid, (2) Lower fine_tuning_trigger_m, or (3) Accept the precision trade-off for speed."
        else:
            conclusion = f"<span class='bad'>Library precision is poor</span> (degradation ratio: {degradation:.2f}x)"
            recommendation = "Library is too sparse. Either generate a denser library or enable fine-tuning more aggressively."
    else:
        conclusion = "Unable to compare (missing data)"
        recommendation = ""

    html += f"""    </table>

    <div class="conclusion">
        <h3>Conclusion</h3>
        <p>{conclusion}</p>
        <p><strong>Recommendation:</strong> {recommendation}</p>
    </div>

    <h2>Interpretation</h2>
    <ul>
        <li><strong>library_only</strong>: Uses trajectory library with adaptation, no GPM fine-tuning</li>
        <li><strong>library_fine_tune</strong>: Uses library first, falls back to GPM if terminal error exceeds threshold</li>
        <li><strong>gpm_only</strong>: Always uses online GPM optimization (slow but accurate baseline)</li>
    </ul>

    <p style="color: #666; margin-top: 20px;">
        Generated by parafoil_planner_v3/scripts/validate_library_precision.py
    </p>
</body>
</html>
"""
    output_path.write_text(html)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate trajectory library precision.")
    parser.add_argument("--library", type=str, required=True, help="Path to trajectory library")
    parser.add_argument("--runs", type=int, default=30, help="Number of test scenarios")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="reports/library_precision.html")
    parser.add_argument("--skip-gpm", action="store_true", help="Skip GPM-only baseline (slow)")
    args = parser.parse_args()

    scenarios = _generate_scenarios(args.runs, args.seed)

    print(f"[precision] Analyzing library: {args.library}")
    coverage = _analyze_library_coverage(args.library)
    print(f"[precision] Library has {coverage['n_trajectories']} trajectories")

    results: List[PrecisionResult] = []

    print(f"[precision] Running {args.runs} scenarios with library_only...")
    results.append(_run_mode(scenarios, "library_only", args.library))

    print(f"[precision] Running {args.runs} scenarios with library_fine_tune...")
    results.append(_run_mode(scenarios, "library_fine_tune", args.library))

    if not args.skip_gpm:
        print(f"[precision] Running {args.runs} scenarios with gpm_only (slow)...")
        results.append(_run_mode(scenarios, "gpm_only", None))

    # Print summary
    print("\n" + "=" * 80)
    print("PRECISION ANALYSIS RESULTS")
    print("=" * 80)
    for r in results:
        print(f"{r.mode:20s}: mean={r.landing_error_mean:.2f}m p95={r.landing_error_p95:.2f}m success={r.success_rate*100:.0f}%")

    # Write report
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_html_report(results, coverage, out)
    print(f"\nReport written to: {out}")


if __name__ == "__main__":
    main()
