#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from parafoil_planner_v3.offline.e2e import Scenario, simulate_one
from parafoil_planner_v3.reporting.report_utils import histogram_svg, render_report


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


def _touchdown_v(metrics: dict) -> float:
    if "touchdown_vertical_velocity_mps" in metrics:
        return float(metrics["touchdown_vertical_velocity_mps"])
    return float(metrics.get("vertical_velocity_mps", 0.0))


def _run_one(task: tuple) -> dict:
    scenario_dict, dynamics_mode, touchdown_alt_m, flare_ramp_time_s, record_history = task
    s = Scenario(**scenario_dict)

    base = simulate_one(
        scenario=s,
        dynamics_mode=str(dynamics_mode),
        record_history=bool(record_history),
        flare_touchdown_altitude_m=-1.0,
        flare_ramp_time_s=float(flare_ramp_time_s),
        flare_mode="touchdown_brake",
    )
    flare = simulate_one(
        scenario=s,
        dynamics_mode=str(dynamics_mode),
        record_history=bool(record_history),
        flare_touchdown_altitude_m=float(touchdown_alt_m),
        flare_ramp_time_s=float(flare_ramp_time_s),
        flare_mode="touchdown_brake",
    )

    v_base = _touchdown_v(base["metrics"])
    v_flare = _touchdown_v(flare["metrics"])

    return {
        "scenario": scenario_dict,
        "baseline": {
            "touchdown_vertical_velocity_mps": float(v_base),
            "landing_error_m": float(base["metrics"].get("landing_error_m", 0.0)),
            "final_phase": str(base["metrics"].get("final_phase", "")),
        },
        "flare": {
            "touchdown_vertical_velocity_mps": float(v_flare),
            "landing_error_m": float(flare["metrics"].get("landing_error_m", 0.0)),
            "final_phase": str(flare["metrics"].get("final_phase", "")),
        },
        "delta": {
            "touchdown_vertical_velocity_mps": float(v_flare - v_base),
            "landing_error_m": float(float(flare["metrics"].get("landing_error_m", 0.0)) - float(base["metrics"].get("landing_error_m", 0.0))),
        },
    }


def _write_html(path: Path, summary: dict, results: list[dict]) -> None:
    payload = {"summary": summary, "runs": results}
    dv = [float(r["delta"]["touchdown_vertical_velocity_mps"]) for r in results]

    summary_rows = [
        ("n_runs", summary.get("n_runs")),
        ("dynamics_mode", summary.get("dynamics_mode")),
        ("touchdown_alt_m", summary.get("touchdown_alt_m")),
        ("flare_ramp_time_s", summary.get("flare_ramp_time_s")),
        ("improvement_rate", summary.get("improvement_rate")),
        ("v_touchdown_base_mean", summary.get("touchdown_vertical_velocity_baseline_mps", {}).get("mean")),
        ("v_touchdown_flare_mean", summary.get("touchdown_vertical_velocity_flare_mps", {}).get("mean")),
        ("delta_v_mean", summary.get("delta_touchdown_vertical_velocity_mps", {}).get("mean")),
        ("delta_v_p95", summary.get("delta_touchdown_vertical_velocity_mps", {}).get("p95")),
    ]
    charts = [
        {"title": "Touchdown Vertical Speed Delta (m/s)", "svg": histogram_svg(dv, bins=20, x_label="m/s")},
    ]
    html = render_report(
        title="parafoil_planner_v3 - Flare (6DOF) Verification",
        summary_rows=summary_rows,
        charts=charts,
        payload=payload,
        subtitle="Compare touchdown vertical speed with vs without touchdown-brake flare.",
    )
    path.write_text(html)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify flare effect: touchdown vertical speed reduction (6DOF).")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--parallel", type=int, default=1, help="Number of worker processes (OFFLINE).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dynamics-mode", type=str, default="6dof", help="6dof|simplified (OFFLINE sim)")
    parser.add_argument("--touchdown-alt", type=float, default=0.2, help="Touchdown-brake start altitude AGL (m).")
    parser.add_argument("--flare-ramp-time", type=float, default=0.1, help="Ramp time to max brake after touchdown trigger (s).")
    parser.add_argument("--alt-min", type=float, default=5.0)
    parser.add_argument("--alt-max", type=float, default=20.0)
    parser.add_argument("--dist-min", type=float, default=5.0)
    parser.add_argument("--dist-max", type=float, default=60.0)
    parser.add_argument("--wind-max", type=float, default=3.0)
    parser.add_argument("--no-history", action="store_true", help="Do not record full state history (smaller reports).")
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    rng = np.random.default_rng(int(args.seed))

    scenarios = [
        Scenario(
            altitude_m=float(rng.uniform(float(args.alt_min), float(args.alt_max))),
            distance_m=float(rng.uniform(float(args.dist_min), float(args.dist_max))),
            bearing_deg=float(rng.uniform(-180.0, 180.0)),
            wind_speed_mps=float(rng.uniform(0.0, float(args.wind_max))),
            wind_direction_deg=float(rng.uniform(0.0, 360.0)),
        )
        for _ in range(int(args.runs))
    ]

    tasks = [
        (asdict(s), str(args.dynamics_mode), float(args.touchdown_alt), float(args.flare_ramp_time), (not bool(args.no_history)))
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

    v_base = [float(r["baseline"]["touchdown_vertical_velocity_mps"]) for r in results]
    v_flare = [float(r["flare"]["touchdown_vertical_velocity_mps"]) for r in results]
    dv = [float(r["delta"]["touchdown_vertical_velocity_mps"]) for r in results]
    improvement_rate = float(np.mean([1.0 if d < 0.0 else 0.0 for d in dv])) if dv else 0.0

    summary = {
        "n_runs": int(len(results)),
        "dynamics_mode": str(args.dynamics_mode),
        "touchdown_alt_m": float(args.touchdown_alt),
        "flare_ramp_time_s": float(args.flare_ramp_time),
        "improvement_rate": float(improvement_rate),
        "touchdown_vertical_velocity_baseline_mps": _stats(v_base),
        "touchdown_vertical_velocity_flare_mps": _stats(v_flare),
        "delta_touchdown_vertical_velocity_mps": _stats(dv),
    }

    out_obj = {"summary": summary, "runs": results}

    if args.output:
        out = Path(str(args.output))
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
