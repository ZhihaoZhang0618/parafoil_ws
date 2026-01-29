#!/usr/bin/env python3
"""
Trajectory visualization script for parafoil_planner_v3.

Generates comprehensive visualization reports including:
- XY trajectory plots
- Altitude profiles
- Control time series
- Landing scatter plots

Usage:
  # Visualize a single simulation
  python3 visualize_trajectory.py --runs 1 --output /tmp/single_vis.html

  # Monte Carlo visualization with scatter plot
  python3 visualize_trajectory.py --runs 50 --parallel 4 --output /tmp/mc_vis.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from parafoil_planner_v3.offline.e2e import Scenario, simulate_one
from parafoil_planner_v3.reporting.report_utils import (
    altitude_profile_svg,
    histogram_svg,
    render_report,
    scatter_svg,
    timeseries_svg,
    trajectory_svg,
)


def _run_scenario(args: tuple) -> Dict[str, Any]:
    """Run a single scenario and return results with history."""
    scenario, dynamics_mode = args
    return simulate_one(
        scenario=scenario,
        planner_rate_hz=1.0,
        control_rate_hz=20.0,
        max_time_s=200.0,
        L1_distance=20.0,
        use_gpm=False,
        dynamics_mode=dynamics_mode,
        record_history=True,
    )


def _generate_scenarios(n: int, seed: int) -> List[Scenario]:
    """Generate random scenarios."""
    rng = np.random.default_rng(seed)
    return [
        Scenario(
            altitude_m=float(rng.uniform(50, 150)),
            distance_m=float(rng.uniform(50, 200)),
            bearing_deg=float(rng.uniform(-180, 180)),
            wind_speed_mps=float(rng.uniform(0, 5)),
            wind_direction_deg=float(rng.uniform(0, 360)),
        )
        for _ in range(n)
    ]


def _extract_trajectory_data(result: Dict[str, Any]) -> Dict[str, List[float]]:
    """Extract time series data from simulation result."""
    states = result.get("states") or []
    controls = result.get("controls") or []

    t = [s.get("t", 0.0) for s in states]
    x = [s.get("p_I", [0, 0, 0])[0] for s in states]  # N
    y = [s.get("p_I", [0, 0, 0])[1] for s in states]  # E
    z = [s.get("p_I", [0, 0, 0])[2] for s in states]  # D (negative = up)
    altitude = [-zi for zi in z]  # Convert to altitude

    delta_L = [c.get("delta_L", 0.0) for c in controls]
    delta_R = [c.get("delta_R", 0.0) for c in controls]
    t_ctrl = [c.get("t", 0.0) for c in controls]

    # Compute ground distance from target (assumed at origin)
    ground_dist = [float(np.sqrt(xi * xi + yi * yi)) for xi, yi in zip(x, y)]

    return {
        "t": t,
        "x": x,  # North
        "y": y,  # East
        "altitude": altitude,
        "ground_dist": ground_dist,
        "t_ctrl": t_ctrl,
        "delta_L": delta_L,
        "delta_R": delta_R,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Trajectory visualization for parafoil_planner_v3.")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--dynamics-mode", type=str, default="simplified", choices=["simplified", "6dof"])
    parser.add_argument("--output", type=str, default="reports/trajectory_vis.html")
    args = parser.parse_args()

    scenarios = _generate_scenarios(args.runs, args.seed)
    tasks = [(s, args.dynamics_mode) for s in scenarios]

    print(f"[visualize] Running {args.runs} scenarios...")
    results: List[Dict[str, Any]] = []

    if args.parallel <= 1:
        for i, task in enumerate(tasks, 1):
            results.append(_run_scenario(task))
            if i % 10 == 0 or i == len(tasks):
                print(f"  [{i}/{len(tasks)}]")
    else:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.parallel) as pool:
            for i, item in enumerate(pool.imap_unordered(_run_scenario, tasks, chunksize=1), start=1):
                results.append(item)
                if i % 10 == 0 or i == len(tasks):
                    print(f"  [{i}/{len(tasks)}]")

    # Extract metrics
    errors = [r["metrics"]["landing_error_m"] for r in results]
    vz_values = [r["metrics"]["vertical_velocity_mps"] for r in results]
    successes = [r["metrics"]["success"] for r in results]

    # Landing positions (relative to target at origin)
    landing_x: List[float] = []
    landing_y: List[float] = []
    for r in results:
        states = r.get("states") or []
        if states:
            final = states[-1]
            p_I = final.get("p_I", [0, 0, 0])
            landing_x.append(float(p_I[0]))  # North
            landing_y.append(float(p_I[1]))  # East

    # Build charts
    charts = []

    # 1. Landing scatter plot
    if len(landing_x) > 0:
        charts.append({
            "title": "Landing Scatter (NED)",
            "svg": scatter_svg(
                x=landing_y,  # East
                y=landing_x,  # North
                x_label="East (m)",
                y_label="North (m)",
                title="Landing Positions",
            ),
        })

    # 2. Landing error histogram
    charts.append({
        "title": "Landing Error Distribution",
        "svg": histogram_svg(errors, bins=15, x_label="Error (m)"),
    })

    # 3. Vertical velocity histogram
    charts.append({
        "title": "Touchdown Vertical Speed",
        "svg": histogram_svg(vz_values, bins=15, x_label="Vz (m/s)"),
    })

    # 4-7. Single trajectory visualizations (first run with history)
    if results and results[0].get("states"):
        data = _extract_trajectory_data(results[0])

        # XY trajectory
        charts.append({
            "title": "XY Trajectory (Run 1)",
            "svg": trajectory_svg(
                x=data["y"],  # East
                y=data["x"],  # North
                x_label="East (m)",
                y_label="North (m)",
                target_xy=(0.0, 0.0),
            ),
        })

        # Altitude profile
        if data["ground_dist"] and data["altitude"]:
            charts.append({
                "title": "Altitude Profile (Run 1)",
                "svg": altitude_profile_svg(
                    distance=data["ground_dist"],
                    altitude=data["altitude"],
                    target_altitude=0.0,
                ),
            })

        # Control time series
        if data["t_ctrl"] and data["delta_L"]:
            charts.append({
                "title": "Left Brake (Run 1)",
                "svg": timeseries_svg(
                    t=data["t_ctrl"],
                    values=data["delta_L"],
                    y_label="delta_L",
                    y_min=0.0,
                    y_max=1.0,
                    color="#E45756",
                ),
            })
            charts.append({
                "title": "Right Brake (Run 1)",
                "svg": timeseries_svg(
                    t=data["t_ctrl"],
                    values=data["delta_R"],
                    y_label="delta_R",
                    y_min=0.0,
                    y_max=1.0,
                    color="#4a90d9",
                ),
            })

        # Altitude vs time
        if data["t"] and data["altitude"]:
            charts.append({
                "title": "Altitude vs Time (Run 1)",
                "svg": timeseries_svg(
                    t=data["t"],
                    values=data["altitude"],
                    y_label="Altitude (m)",
                    color="#59A14F",
                ),
            })

    # Summary
    summary_rows = [
        ("n_runs", len(results)),
        ("success_rate", f"{100 * sum(1 for s in successes if s) / max(len(successes), 1):.1f}%"),
        ("landing_error_mean", f"{float(np.mean(errors)):.2f} m"),
        ("landing_error_p50", f"{float(np.percentile(errors, 50)):.2f} m"),
        ("landing_error_p95", f"{float(np.percentile(errors, 95)):.2f} m"),
        ("CEP50", f"{float(np.percentile(errors, 50)):.2f} m"),
        ("CEP95", f"{float(np.percentile(errors, 95)):.2f} m"),
        ("vz_mean", f"{float(np.mean(vz_values)):.2f} m/s"),
        ("dynamics_mode", args.dynamics_mode),
    ]

    payload = {
        "summary": dict(summary_rows),
        "n_runs": len(results),
    }

    html = render_report(
        title="parafoil_planner_v3 - Trajectory Visualization",
        summary_rows=summary_rows,
        charts=charts,
        payload=payload,
        subtitle="XY trajectory, altitude profile, control time series, and landing scatter analysis.",
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)

    print(f"\nReport written to: {out}")
    print(f"CEP50: {float(np.percentile(errors, 50)):.2f}m, CEP95: {float(np.percentile(errors, 95)):.2f}m")


if __name__ == "__main__":
    main()
