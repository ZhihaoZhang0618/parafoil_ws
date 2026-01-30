#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass
class BagSafetySummary:
    bag: str
    has_safety: bool
    reachable: bool
    reason: str
    risk: float
    distance_to_desired_m: float
    reach_margin_mps: float


def _stats(values: list[float]) -> dict:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0}
    arr = sorted(values)
    import numpy as np

    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


def _iter_bags(paths: Iterable[str]) -> list[Path]:
    bags: list[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            # rosbag2 directory contains metadata.yaml
            if (path / "metadata.yaml").exists():
                bags.append(path)
            else:
                # search for bag directories
                bags.extend([d for d in path.iterdir() if d.is_dir() and (d / "metadata.yaml").exists()])
        elif path.exists():
            bags.append(path)
    return bags


def _load_planner_status_messages(bag_path: Path) -> list[str]:
    try:
        from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"rosbag2_py not available: {e}")

    reader = SequentialReader()
    storage_options = StorageOptions(uri=str(bag_path), storage_id="sqlite3")
    converter_options = ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr")
    reader.open(storage_options, converter_options)

    topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if "/planner_status" not in topic_types:
        return []
    msg_type = get_message(topic_types["/planner_status"])

    msgs: list[str] = []
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic != "/planner_status":
            continue
        msg = deserialize_message(data, msg_type)
        msgs.append(str(msg.data))
    return msgs


def _parse_safety_status(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("{"):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            safety = payload.get("safety")
            if isinstance(safety, dict):
                return {
                    "reason": str(safety.get("reason", "")),
                    "risk": float(safety.get("risk", float("inf"))),
                    "distance_to_desired_m": float(safety.get("distance_to_desired_m", float("inf"))),
                    "reach_margin_mps": float(safety.get("reach_margin_mps", float("inf"))),
                }
    pattern = re.compile(
        r"safety=([\w\-]+)\s+risk=([0-9eE+\-.]+)\s+dist=([0-9eE+\-.]+)m\s+margin=([0-9eE+\-.]+)mps"
    )
    match = pattern.search(text)
    if not match:
        return None
    reason = match.group(1)
    return {
        "reason": reason,
        "risk": float(match.group(2)),
        "distance_to_desired_m": float(match.group(3)),
        "reach_margin_mps": float(match.group(4)),
    }


def _summarize_bag(bag_path: Path) -> BagSafetySummary:
    msgs = _load_planner_status_messages(bag_path)
    safety_msgs = [m for m in ([_parse_safety_status(t) for t in msgs]) if m is not None]
    if not safety_msgs:
        return BagSafetySummary(
            bag=str(bag_path),
            has_safety=False,
            reachable=False,
            reason="no_safety_status",
            risk=float("inf"),
            distance_to_desired_m=float("inf"),
            reach_margin_mps=float("inf"),
        )

    last = safety_msgs[-1]
    reachable = bool(last.get("reason") == "ok")
    return BagSafetySummary(
        bag=str(bag_path),
        has_safety=True,
        reachable=reachable,
        reason=str(last.get("reason")),
        risk=float(last.get("risk", float("inf"))),
        distance_to_desired_m=float(last.get("distance_to_desired_m", float("inf"))),
        reach_margin_mps=float(last.get("reach_margin_mps", float("inf"))),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze ROS2 bags for safety metrics (/planner_status).")
    parser.add_argument("--bags", type=str, nargs="+", required=True, help="Bag dir(s) or parent dir(s)")
    parser.add_argument("--output", type=str, default="reports/safety_ros2", help="Output directory")
    args = parser.parse_args()

    bag_paths = _iter_bags(args.bags)
    if not bag_paths:
        raise SystemExit("No rosbag2 directories found")

    summaries = [_summarize_bag(p) for p in bag_paths]
    reachable = [s for s in summaries if s.reachable]
    reach_rate = float(len(reachable) / max(len(summaries), 1))
    risk_vals = [s.risk for s in reachable if s.has_safety and s.risk != float("inf")]
    dist_vals = [s.distance_to_desired_m for s in reachable if s.has_safety]
    margin_vals = [s.reach_margin_mps for s in reachable if s.has_safety]

    summary = {
        "n_bags": len(summaries),
        "reachable_rate": reach_rate,
        "risk": _stats(risk_vals),
        "distance_to_desired_m": _stats(dist_vals),
        "reach_margin_mps": _stats(margin_vals),
    }

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "safety_ros2_summary.json"
    json_path.write_text(json.dumps({"summary": summary, "bags": [asdict(s) for s in summaries]}, indent=2))

    md_path = output_dir / "safety_ros2_summary.md"
    lines = [
        "# ROS2 Safety Summary",
        f"- Total bags: {summary['n_bags']}",
        f"- Reachable rate: {summary['reachable_rate']:.3f}",
        "",
        "## Metrics (reachable only)",
        "| Metric | Mean | P50 | P95 |",
        "|--------|------|-----|-----|",
        f"| Risk | {summary['risk']['mean']:.3f} | {summary['risk']['p50']:.3f} | {summary['risk']['p95']:.3f} |",
        f"| Distance to Desired (m) | {summary['distance_to_desired_m']['mean']:.2f} | {summary['distance_to_desired_m']['p50']:.2f} | {summary['distance_to_desired_m']['p95']:.2f} |",
        f"| Reach Margin (m/s) | {summary['reach_margin_mps']['mean']:.2f} | {summary['reach_margin_mps']['p50']:.2f} | {summary['reach_margin_mps']['p95']:.2f} |",
        "",
        "## Bags",
        "| Bag | Has Safety | Reachable | Reason | Risk | Dist (m) | Margin (m/s) |",
        "|-----|------------|-----------|--------|------|----------|--------------|",
    ]
    for s in summaries:
        lines.append(
            f"| {s.bag} | {'Yes' if s.has_safety else 'No'} | {'Yes' if s.reachable else 'No'} | {s.reason} | {s.risk:.3f} | {s.distance_to_desired_m:.2f} | {s.reach_margin_mps:.2f} |"
        )
    md_path.write_text("\n".join(lines))

    print(f"Summary written to {json_path}")
    print(f"Report written to {md_path}")


if __name__ == "__main__":
    main()
