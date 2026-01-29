#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from parafoil_planner_v3.offline.e2e import Scenario, simulate_one
from parafoil_planner_v3.reporting.report_utils import histogram_svg, line_svg, render_report, scatter_svg, xy_path_svg


def _run_one(task: tuple[dict, float, float, float, float, bool, str, bool]) -> dict:
    scenario_dict, planner_rate_hz, control_rate_hz, max_time_s, L1_distance, use_gpm, dynamics_mode, record_history = task
    s = Scenario(**scenario_dict)
    return simulate_one(
        scenario=s,
        planner_rate_hz=float(planner_rate_hz),
        control_rate_hz=float(control_rate_hz),
        max_time_s=float(max_time_s),
        L1_distance=float(L1_distance),
        use_gpm=bool(use_gpm),
        dynamics_mode=str(dynamics_mode),
        record_history=bool(record_history),
    )


def _write_html(path: Path, summary: dict, results: list[dict]) -> None:
    payload = {"summary": summary, "runs": results}
    landing_errors = [r["metrics"]["landing_error_m"] for r in results]
    vertical_velocities = [r["metrics"]["vertical_velocity_mps"] for r in results]
    times = [r["metrics"]["time_s"] for r in results]
    landing_xy = []
    for r in results:
        pos = r["metrics"].get("touchdown_position_xy") or r["metrics"].get("final_position_xy")
        if isinstance(pos, (list, tuple)) and len(pos) >= 2:
            landing_xy.append((float(pos[0]), float(pos[1])))

    summary_rows = [
        ("n_runs", summary.get("n_runs")),
        ("success_rate", summary.get("success_rate")),
        ("landing_error_mean", summary.get("landing_error", {}).get("mean")),
        ("landing_error_p95", summary.get("landing_error", {}).get("p95")),
        ("vertical_velocity_p95", summary.get("vertical_velocity_mps", {}).get("p95")),
        ("flare_success_rate", summary.get("flare_success_rate")),
        ("mission_time_p95", summary.get("mission_time_s", {}).get("p95")),
    ]
    charts = [
        {"title": "Landing Error (m)", "svg": histogram_svg(landing_errors, bins=20, x_label="m")},
        {"title": "Touchdown Vertical Speed (m/s)", "svg": histogram_svg(vertical_velocities, bins=20, x_label="m/s")},
        {"title": "Mission Time (s)", "svg": histogram_svg(times, bins=20, x_label="s")},
    ]
    if landing_xy:
        charts.append({"title": "Landing Scatter (XY)", "svg": scatter_svg(landing_xy, target_xy=(0.0, 0.0))})

    # Example path from first run (if history present)
    if results:
        first = results[0]
        history = first.get("state_history", [])
        if history:
            xy = [(float(s.get("position", [0.0, 0.0, 0.0])[0]), float(s.get("position", [0.0, 0.0, 0.0])[1])) for s in history]
            alt = [float(-s.get("position", [0.0, 0.0, 0.0])[2]) for s in history]
            t = [float(s.get("t", 0.0)) for s in history]
            charts.append({"title": "XY Path (example)", "svg": xy_path_svg([{"label": "trajectory", "xy": xy, "color": "#4C78A8"}])})
            charts.append({"title": "Altitude Profile (example)", "svg": line_svg(t, alt, x_label="t (s)", y_label="altitude (m)")})
    html = render_report(
        title="parafoil_planner_v3 - End-to-End Report",
        summary_rows=summary_rows,
        charts=charts,
        payload=payload,
        subtitle="Offline end-to-end verification summary with distributions.",
    )
    path.write_text(html)


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end offline verification for parafoil_planner_v3.")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--parallel", type=int, default=1, help="Number of worker processes (OFFLINE).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--planner-rate", type=float, default=1.0)
    parser.add_argument("--control-rate", type=float, default=20.0)
    parser.add_argument("--max-time", type=float, default=200.0)
    parser.add_argument("--L1", type=float, default=20.0)
    parser.add_argument("--use-gpm", action="store_true")
    parser.add_argument("--dynamics-mode", type=str, default="simplified", help="simplified|6dof (OFFLINE sim)")
    parser.add_argument("--no-history", action="store_true", help="Do not record full state history (smaller reports).")
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

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
            float(args.planner_rate),
            float(args.control_rate),
            float(args.max_time),
            float(args.L1),
            bool(args.use_gpm),
            str(args.dynamics_mode),
            (not bool(args.no_history)),
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

    landing_errors = [r["metrics"]["landing_error_m"] for r in results]
    vertical_velocities = [r["metrics"]["vertical_velocity_mps"] for r in results]
    replans = [r["metrics"].get("replan_count", 0) for r in results]
    flare_success = [1.0 if float(v) < 2.0 else 0.0 for v in vertical_velocities]
    times = [r["metrics"]["time_s"] for r in results]
    control_effort = [r["metrics"].get("control_effort_mean", 0.0) for r in results]

    # Phase duration aggregation (mean across runs)
    phase_sum: dict[str, float] = {}
    for r in results:
        pd = r["metrics"].get("phase_durations_s", {}) or {}
        for k, v in pd.items():
            phase_sum[k] = phase_sum.get(k, 0.0) + float(v)
    phase_mean = {k: float(v / max(len(results), 1)) for k, v in phase_sum.items()}
    summary = {
        "n_runs": len(results),
        "success_rate": float(np.mean([1.0 if r["metrics"]["success"] else 0.0 for r in results])),
        "landing_error": {
            "mean": float(np.mean(landing_errors)),
            "std": float(np.std(landing_errors)),
            "p50": float(np.percentile(landing_errors, 50)),
            "p95": float(np.percentile(landing_errors, 95)),
        },
        "vertical_velocity_mps": {
            "mean": float(np.mean(vertical_velocities)),
            "std": float(np.std(vertical_velocities)),
            "p50": float(np.percentile(vertical_velocities, 50)),
            "p95": float(np.percentile(vertical_velocities, 95)),
        },
        "flare_success_rate": float(np.mean(flare_success)),
        "replan_count": {
            "mean": float(np.mean(replans)),
            "p50": float(np.percentile(replans, 50)),
            "p95": float(np.percentile(replans, 95)),
        },
        "mission_time_s": {
            "mean": float(np.mean(times)),
            "p50": float(np.percentile(times, 50)),
            "p95": float(np.percentile(times, 95)),
        },
        "control_effort_mean": {
            "mean": float(np.mean(control_effort)),
            "p50": float(np.percentile(control_effort, 50)),
            "p95": float(np.percentile(control_effort, 95)),
        },
        "phase_durations_s_mean": phase_mean,
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
