#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from parafoil_planner_v3.offline.e2e import Scenario, simulate_one
from parafoil_planner_v3.reporting.report_utils import histogram_svg, render_report


def _run_one(task: tuple[dict, float, float, float, float, bool, str]) -> dict:
    scenario_dict, planner_rate_hz, control_rate_hz, max_time_s, L1_distance, use_gpm, dynamics_mode = task
    s = Scenario(**scenario_dict)
    return simulate_one(
        scenario=s,
        planner_rate_hz=float(planner_rate_hz),
        control_rate_hz=float(control_rate_hz),
        max_time_s=float(max_time_s),
        L1_distance=float(L1_distance),
        use_gpm=bool(use_gpm),
        dynamics_mode=str(dynamics_mode),
        record_history=False,
    )


def _write_html(path: Path, summary: dict, results: list[dict]) -> None:
    payload = {"summary": summary, "runs": results}
    errors = np.array([r["metrics"]["landing_error_m"] for r in results], dtype=float).tolist()
    vertical_velocities = np.array([r["metrics"]["vertical_velocity_mps"] for r in results], dtype=float).tolist()
    times = np.array([r["metrics"].get("time_s", 0.0) for r in results], dtype=float).tolist()

    summary_rows = [
        ("n_runs", summary.get("n_runs")),
        ("success_rate", summary.get("success_rate")),
        ("CEP", summary.get("CEP")),
        ("CEP95", summary.get("CEP95")),
        ("mean_error", summary.get("mean_error")),
        ("std_error", summary.get("std_error")),
        ("flare_success_rate", summary.get("flare_success_rate")),
        ("vertical_velocity_p95", summary.get("vertical_velocity_mps", {}).get("p95")),
        ("mission_time_p95", summary.get("mission_time_s", {}).get("p95")),
    ]
    charts = [
        {"title": "Landing Error (m)", "svg": histogram_svg(errors, bins=20, x_label="m")},
        {"title": "Touchdown Vertical Speed (m/s)", "svg": histogram_svg(vertical_velocities, bins=20, x_label="m/s")},
        {"title": "Mission Time (s)", "svg": histogram_svg(times, bins=20, x_label="s")},
    ]
    html = render_report(
        title="parafoil_planner_v3 - Monte Carlo Report",
        summary_rows=summary_rows,
        charts=charts,
        payload=payload,
        subtitle="Monte Carlo distribution summary for landing and flare performance.",
    )
    path.write_text(html)


def main() -> None:
    parser = argparse.ArgumentParser(description="Monte Carlo offline evaluation wrapper.")
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--parallel", type=int, default=1, help="Number of worker processes (OFFLINE).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="reports/monte_carlo_results.json")
    parser.add_argument("--use-gpm", action="store_true")
    parser.add_argument("--planner-rate", type=float, default=1.0)
    parser.add_argument("--control-rate", type=float, default=20.0)
    parser.add_argument("--max-time", type=float, default=200.0)
    parser.add_argument("--L1", type=float, default=20.0)
    parser.add_argument("--dynamics-mode", type=str, default="simplified", help="simplified|6dof (OFFLINE sim)")
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
        (asdict(s), float(args.planner_rate), float(args.control_rate), float(args.max_time), float(args.L1), bool(args.use_gpm), str(args.dynamics_mode))
        for s in scenarios
    ]

    results: list[dict] = []
    if int(args.parallel) <= 1:
        for i, task in enumerate(tasks, start=1):
            results.append(_run_one(task))
            if i % 20 == 0 or i == len(tasks):
                print(f"[monte_carlo] {i}/{len(tasks)}")
    else:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=int(args.parallel)) as pool:
            for i, item in enumerate(pool.imap_unordered(_run_one, tasks, chunksize=1), start=1):
                results.append(item)
                if i % 20 == 0 or i == len(tasks):
                    print(f"[monte_carlo] {i}/{len(tasks)}")

    errors = np.array([r["metrics"]["landing_error_m"] for r in results], dtype=float)
    vertical_velocities = np.array([r["metrics"]["vertical_velocity_mps"] for r in results], dtype=float)
    replans = np.array([r["metrics"].get("replan_count", 0) for r in results], dtype=float)
    times = np.array([r["metrics"].get("time_s", 0.0) for r in results], dtype=float)
    control_effort = np.array([r["metrics"].get("control_effort_mean", 0.0) for r in results], dtype=float)

    flare_success = np.array([1.0 if float(v) < 2.0 else 0.0 for v in vertical_velocities], dtype=float)

    # Phase duration aggregation (mean across runs)
    phase_sum: dict[str, float] = {}
    for r in results:
        pd = r["metrics"].get("phase_durations_s", {}) or {}
        for k, v in pd.items():
            phase_sum[k] = phase_sum.get(k, 0.0) + float(v)
    phase_mean = {k: float(v / max(len(results), 1)) for k, v in phase_sum.items()}

    final_phase_counts: dict[str, int] = {}
    for r in results:
        ph = str(r["metrics"].get("final_phase", ""))
        final_phase_counts[ph] = final_phase_counts.get(ph, 0) + 1
    summary = {
        "n_runs": int(len(results)),
        "success_rate": float(np.mean([1.0 if r["metrics"]["success"] else 0.0 for r in results])),
        "CEP": float(np.percentile(errors, 50)),
        "CEP95": float(np.percentile(errors, 95)),
        "mean_error": float(np.mean(errors)),
        "std_error": float(np.std(errors)),
        "landing_error_m": {
            "mean": float(np.mean(errors)),
            "std": float(np.std(errors)),
            "p50": float(np.percentile(errors, 50)),
            "p95": float(np.percentile(errors, 95)),
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
        "final_phase_counts": final_phase_counts,
    }

    out_obj = {"summary": summary, "runs": results}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".html":
        _write_html(out, summary, results)
    else:
        out.write_text(json.dumps(out_obj, indent=2))
    print(f"Wrote {out} (CEP={summary['CEP']:.2f}m, parallel={int(args.parallel)})")


if __name__ == "__main__":
    main()
