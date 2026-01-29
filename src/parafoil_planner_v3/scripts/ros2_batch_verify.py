#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import yaml


def _load_scenarios(path: Path | None, n_runs: int, seed: int) -> list[dict]:
    if path:
        data = yaml.safe_load(path.read_text()) if path.suffix.lower() in {".yaml", ".yml"} else json.loads(path.read_text())
        if isinstance(data, dict):
            data = data.get("scenarios", [])
        return list(data)
    rng = np.random.default_rng(int(seed))
    scenarios = []
    for _ in range(int(n_runs)):
        scenarios.append(
            {
                "target_enu_x": float(rng.uniform(80.0, 220.0)),
                "target_enu_y": float(rng.uniform(-80.0, 80.0)),
                "target_enu_z": 0.0,
                "wind_steady_n": float(rng.uniform(-3.0, 3.0)),
                "wind_steady_e": float(rng.uniform(0.0, 4.0)),
                "wind_steady_d": 0.0,
            }
        )
    return scenarios


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch ROS2 verification runs (launch + optional rosbag).")
    parser.add_argument("--launch", type=str, default="parafoil_planner_v3 e2e_verification.launch.py")
    parser.add_argument("--n-runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration", type=float, default=120.0, help="Seconds to let each run execute before shutdown.")
    parser.add_argument("--scenario-file", type=str, default="", help="YAML/JSON list of scenarios.")
    parser.add_argument("--output-dir", type=str, default="/tmp/ros2_batch")
    parser.add_argument("--record-bag", action="store_true")
    parser.add_argument("--record-mission-log", action="store_true")
    parser.add_argument("--mission-log-dir", type=str, default="/tmp/mission_logs")
    parser.add_argument("--library-path", type=str, default="")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scenarios = _load_scenarios(Path(args.scenario_file) if args.scenario_file else None, args.n_runs, args.seed)
    if args.n_runs and len(scenarios) > int(args.n_runs):
        scenarios = scenarios[: int(args.n_runs)]

    launch_pkg, launch_file = args.launch.split()
    for i, sc in enumerate(scenarios):
        bag_output = out_dir / f"run_{i:02d}"
        log_dir = Path(args.mission_log_dir) / f"run_{i:02d}"
        cmd = [
            "ros2",
            "launch",
            launch_pkg,
            launch_file,
            f"target_enu_x:={sc.get('target_enu_x', 150.0)}",
            f"target_enu_y:={sc.get('target_enu_y', 50.0)}",
            f"target_enu_z:={sc.get('target_enu_z', 0.0)}",
            f"wind_steady_n:={sc.get('wind_steady_n', 0.0)}",
            f"wind_steady_e:={sc.get('wind_steady_e', 2.0)}",
            f"wind_steady_d:={sc.get('wind_steady_d', 0.0)}",
            f"record_bag:={'true' if args.record_bag else 'false'}",
            f"bag_output:={str(bag_output)}",
            f"record_mission_log:={'true' if args.record_mission_log else 'false'}",
            f"mission_log_dir:={str(log_dir)}",
        ]
        if args.library_path:
            cmd.append("use_library:=true")
            cmd.append(f"library_path:={args.library_path}")

        print(f"[batch] Run {i+1}/{len(scenarios)}: {' '.join(cmd)}")
        proc = subprocess.Popen(cmd)
        time.sleep(float(args.duration))
        proc.terminate()
        try:
            proc.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            proc.kill()
        time.sleep(1.0)


if __name__ == "__main__":
    main()
