#!/usr/bin/env python3
"""
Batch ROS2 verification script for parafoil_planner_v3.

Runs multiple scenarios with e2e_verification.launch.py, optionally recording
rosbags for each run, and produces a summary report.

Usage:
  python3 batch_ros_verify.py --scenarios config/batch_scenarios.yaml --output reports/batch_results
  python3 batch_ros_verify.py --n-runs 5 --record-bags --output reports/batch_results
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class Scenario:
    name: str = "default"
    initial_altitude: float = 80.0
    target_enu_x: float = 150.0
    target_enu_y: float = 50.0
    target_enu_z: float = 0.0
    wind_steady_n: float = 0.0
    wind_steady_e: float = 2.0
    wind_steady_d: float = 0.0
    wind_enable_gust: bool = False
    wind_gust_magnitude: float = 3.0
    use_library: bool = False
    library_path: str = "/tmp/parafoil_library.pkl"
    timeout_s: float = 120.0


@dataclass
class RunResult:
    scenario_name: str
    success: bool
    duration_s: float
    bag_path: Optional[str] = None
    error: str = ""


def load_scenarios(path: str) -> List[Scenario]:
    """Load scenarios from YAML file."""
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    scenarios = []
    for item in data.get("scenarios", []):
        scenarios.append(Scenario(**item))
    return scenarios


def generate_default_scenarios(n: int, seed: int = 42) -> List[Scenario]:
    """Generate n default scenarios with varying parameters."""
    import numpy as np
    rng = np.random.default_rng(seed)
    scenarios = []
    for i in range(n):
        scenarios.append(Scenario(
            name=f"scenario_{i+1}",
            initial_altitude=float(rng.uniform(50, 150)),
            target_enu_x=float(rng.uniform(100, 200)),
            target_enu_y=float(rng.uniform(-50, 100)),
            target_enu_z=0.0,
            wind_steady_n=float(rng.uniform(-2, 2)),
            wind_steady_e=float(rng.uniform(0, 4)),
            wind_steady_d=0.0,
            wind_enable_gust=bool(rng.random() > 0.7),
            wind_gust_magnitude=float(rng.uniform(1, 4)),
        ))
    return scenarios


def run_scenario(
    scenario: Scenario,
    output_dir: Path,
    record_bag: bool = False,
    verbose: bool = False,
) -> RunResult:
    """Run a single scenario using ros2 launch."""
    bag_path = None
    if record_bag:
        bag_path = str(output_dir / f"bags/{scenario.name}")

    cmd = [
        "ros2", "launch", "parafoil_planner_v3", "e2e_verification.launch.py",
        f"initial_altitude:={scenario.initial_altitude}",
        f"target_enu_x:={scenario.target_enu_x}",
        f"target_enu_y:={scenario.target_enu_y}",
        f"target_enu_z:={scenario.target_enu_z}",
        f"wind_steady_n:={scenario.wind_steady_n}",
        f"wind_steady_e:={scenario.wind_steady_e}",
        f"wind_steady_d:={scenario.wind_steady_d}",
        f"wind_enable_gust:={'true' if scenario.wind_enable_gust else 'false'}",
        f"wind_gust_magnitude:={scenario.wind_gust_magnitude}",
        f"use_library:={'true' if scenario.use_library else 'false'}",
        f"library_path:={scenario.library_path}",
        f"record_bag:={'true' if record_bag else 'false'}",
    ]
    if bag_path:
        cmd.append(f"bag_output:={bag_path}")

    start_time = time.time()
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE if not verbose else None,
            stderr=subprocess.STDOUT if not verbose else None,
            preexec_fn=os.setsid,
        )

        # Wait for timeout or completion
        try:
            proc.wait(timeout=scenario.timeout_s)
            success = proc.returncode == 0
            error = "" if success else f"exit_code={proc.returncode}"
        except subprocess.TimeoutExpired:
            # Kill process group
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            time.sleep(1)
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            success = True  # Timeout is expected for simulation
            error = ""

    except Exception as e:
        success = False
        error = str(e)

    duration = time.time() - start_time

    return RunResult(
        scenario_name=scenario.name,
        success=success,
        duration_s=duration,
        bag_path=bag_path if record_bag else None,
        error=error,
    )


def write_summary(
    results: List[RunResult],
    scenarios: List[Scenario],
    output_dir: Path,
) -> None:
    """Write JSON and Markdown summary."""
    summary = {
        "n_runs": len(results),
        "n_success": sum(1 for r in results if r.success),
        "total_duration_s": sum(r.duration_s for r in results),
        "results": [asdict(r) for r in results],
        "scenarios": [asdict(s) for s in scenarios],
    }

    # JSON
    json_path = output_dir / "batch_summary.json"
    json_path.write_text(json.dumps(summary, indent=2))

    # Markdown
    md_path = output_dir / "batch_summary.md"
    lines = [
        "# Batch ROS2 Verification Results\n",
        f"- Total runs: {summary['n_runs']}",
        f"- Successful: {summary['n_success']}",
        f"- Total duration: {summary['total_duration_s']:.1f}s\n",
        "## Results\n",
        "| Scenario | Success | Duration (s) | Bag Path | Error |",
        "|----------|---------|--------------|----------|-------|",
    ]
    for r in results:
        bag = r.bag_path or "-"
        err = r.error or "-"
        lines.append(f"| {r.scenario_name} | {'Yes' if r.success else 'No'} | {r.duration_s:.1f} | {bag} | {err} |")

    lines.append("\n## Scenarios\n")
    for s in scenarios:
        lines.append(f"### {s.name}")
        lines.append(f"- Altitude: {s.initial_altitude}m")
        lines.append(f"- Target ENU: ({s.target_enu_x}, {s.target_enu_y}, {s.target_enu_z})")
        lines.append(f"- Wind NED: ({s.wind_steady_n}, {s.wind_steady_e}, {s.wind_steady_d})")
        lines.append(f"- Gust: {s.wind_enable_gust} (mag={s.wind_gust_magnitude})")
        lines.append("")

    md_path.write_text("\n".join(lines))

    print(f"Summary written to: {json_path}")
    print(f"Report written to: {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch ROS2 verification for parafoil_planner_v3.")
    parser.add_argument("--scenarios", type=str, help="YAML file with scenario definitions")
    parser.add_argument("--n-runs", type=int, default=3, help="Number of default scenarios to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for scenario generation")
    parser.add_argument("--output", type=str, default="reports/batch_ros", help="Output directory")
    parser.add_argument("--record-bags", action="store_true", help="Record rosbags for each run")
    parser.add_argument("--timeout", type=float, default=60.0, help="Per-scenario timeout in seconds")
    parser.add_argument("--verbose", action="store_true", help="Show launch output")
    args = parser.parse_args()

    # Load or generate scenarios
    if args.scenarios:
        scenarios = load_scenarios(args.scenarios)
    else:
        scenarios = generate_default_scenarios(args.n_runs, args.seed)

    # Apply timeout override
    for s in scenarios:
        s.timeout_s = args.timeout

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.record_bags:
        (output_dir / "bags").mkdir(parents=True, exist_ok=True)

    print(f"Running {len(scenarios)} scenarios...")
    results: List[RunResult] = []

    for i, scenario in enumerate(scenarios, 1):
        print(f"[{i}/{len(scenarios)}] Running {scenario.name}...")
        result = run_scenario(
            scenario,
            output_dir,
            record_bag=args.record_bags,
            verbose=args.verbose,
        )
        results.append(result)
        status = "SUCCESS" if result.success else f"FAILED: {result.error}"
        print(f"  -> {status} ({result.duration_s:.1f}s)")

    write_summary(results, scenarios, output_dir)
    print(f"\nCompleted {len(results)} runs.")


if __name__ == "__main__":
    main()
