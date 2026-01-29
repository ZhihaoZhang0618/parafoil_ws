#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from typing import Dict, List, Tuple, Optional

import numpy as np

from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message


def read_bag(path: str) -> Dict[str, List[Tuple[float, object]]]:
    reader = SequentialReader()
    storage_options = StorageOptions(uri=path, storage_id="sqlite3")
    converter_options = ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr")
    reader.open(storage_options, converter_options)

    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    msg_cache: Dict[str, List[Tuple[float, object]]] = {}

    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic not in type_map:
            continue
        msg_type = get_message(type_map[topic])
        msg = deserialize_message(data, msg_type)
        # t is in nanoseconds
        msg_cache.setdefault(topic, []).append((t * 1e-9, msg))

    return msg_cache


def pos_series(msgs: List[Tuple[float, object]]) -> np.ndarray:
    arr = []
    for t, m in msgs:
        arr.append([t, float(m.vector.x), float(m.vector.y), float(m.vector.z)])
    return np.array(arr)


def cmd_series(msgs: List[Tuple[float, object]]) -> np.ndarray:
    arr = []
    for t, m in msgs:
        arr.append([t, float(m.vector.x), float(m.vector.y)])
    return np.array(arr)


def path_points_from_msg(msg) -> np.ndarray:
    pts = []
    for pose in msg.poses:
        # Path is published in ENU (x=E, y=N); convert to NED [xN, yE]
        x_n = float(pose.pose.position.y)
        y_e = float(pose.pose.position.x)
        pts.append([x_n, y_e])
    return np.array(pts)


def min_dist_to_path(pos_xy: np.ndarray, path_xy: np.ndarray) -> np.ndarray:
    if pos_xy.size == 0 or path_xy.size == 0:
        return np.array([])
    # Broadcasted distance: [N,2] vs [M,2] -> [N,M]
    diffs = pos_xy[:, None, :] - path_xy[None, :, :]
    d2 = np.sum(diffs * diffs, axis=2)
    return np.sqrt(np.min(d2, axis=1))


def parse_planner_status(msg) -> Dict[str, str]:
    data = getattr(msg, "data", "") or ""
    out: Dict[str, str] = {}
    for line in data.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        out[key.strip()] = value.strip()
    return out


def compute_metrics(pos: np.ndarray, target_x: float, target_y: float, landing_alt: float) -> dict:
    if pos.size == 0:
        return {"error": "no position data"}

    times = pos[:, 0]
    xs = pos[:, 1]
    ys = pos[:, 2]
    alts = -pos[:, 3]

    dists = np.sqrt((xs - target_x) ** 2 + (ys - target_y) ** 2)
    min_idx = int(np.argmin(dists))

    # landing detection (first time altitude <= landing_alt)
    landing_idx = None
    for i, alt in enumerate(alts):
        if alt <= landing_alt:
            landing_idx = i
            break
    if landing_idx is None:
        landing_idx = len(alts) - 1

    # Approx heading from position deltas
    headings = []
    for i in range(1, len(xs)):
        dx = xs[i] - xs[i - 1]
        dy = ys[i] - ys[i - 1]
        if math.hypot(dx, dy) < 1e-6:
            continue
        headings.append(math.degrees(math.atan2(dy, dx)))

    metrics = {
        "t_start": float(times[0]),
        "t_end": float(times[-1]),
        "t_min_dist": float(times[min_idx]),
        "min_dist": float(dists[min_idx]),
        "final_dist": float(dists[-1]),
        "final_alt": float(alts[-1]),
        "mean_alt": float(np.mean(alts)),
        "mean_ground_speed": float(np.mean(np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2) / np.diff(times))),
        "heading_mean": float(np.mean(headings)) if headings else 0.0,
        "landing_time": float(times[landing_idx]),
        "landing_x": float(xs[landing_idx]),
        "landing_y": float(ys[landing_idx]),
        "landing_alt": float(alts[landing_idx]),
        "landing_dist": float(dists[landing_idx]),
    }

    # time to reach within 20m
    within = np.where(dists < 20.0)[0]
    metrics["t_reach_20m"] = float(times[within[0]]) if within.size > 0 else None
    return metrics


def compute_cmd_metrics(cmd: np.ndarray) -> dict:
    if cmd.size == 0:
        return {"cmd_error": "no command data"}
    left = cmd[:, 1]
    right = cmd[:, 2]
    sym = 0.5 * (left + right)
    diff = left - right
    return {
        "cmd_sym_mean": float(np.mean(sym)),
        "cmd_sym_min": float(np.min(sym)),
        "cmd_sym_max": float(np.max(sym)),
        "cmd_diff_mean": float(np.mean(diff)),
        "cmd_diff_abs_mean": float(np.mean(np.abs(diff))),
        "cmd_zero_ratio": float(np.mean((left < 0.02) & (right < 0.02))),
    }


def compute_path_metrics(pos: np.ndarray, path_xy: np.ndarray) -> dict:
    if pos.size == 0 or path_xy.size == 0:
        return {"path_error": "no path or position data"}
    pos_xy = pos[:, 1:3]
    dists = min_dist_to_path(pos_xy, path_xy)
    return {
        "path_pts": int(path_xy.shape[0]),
        "path_min_x": float(np.min(path_xy[:, 0])),
        "path_max_x": float(np.max(path_xy[:, 0])),
        "path_min_y": float(np.min(path_xy[:, 1])),
        "path_max_y": float(np.max(path_xy[:, 1])),
        "path_err_mean": float(np.mean(dists)),
        "path_err_max": float(np.max(dists)),
        "path_err_final": float(dists[-1]),
        "path_end_x": float(path_xy[-1, 0]),
        "path_end_y": float(path_xy[-1, 1]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bag", help="bag directory path (ros2 bag record output)")
    ap.add_argument("--target-x", type=float, default=150.0)
    ap.add_argument("--target-y", type=float, default=50.0)
    ap.add_argument("--landing-alt", type=float, default=1.0)
    args = ap.parse_args()

    msgs = read_bag(args.bag)

    pos = pos_series(msgs.get("/position", [])) if "/position" in msgs else np.array([])
    cmd = cmd_series(msgs.get("/rockpara_actuators_node/auto_commands", [])) if "/rockpara_actuators_node/auto_commands" in msgs else np.array([])

    path_msgs = msgs.get("/planned_path", [])
    path_xy = path_points_from_msg(path_msgs[-1][1]) if path_msgs else np.array([])

    status_msgs = msgs.get("/planner_status", [])
    status = parse_planner_status(status_msgs[-1][1]) if status_msgs else {}

    metrics = compute_metrics(pos, args.target_x, args.target_y, args.landing_alt) if pos.size else {"error": "no position data"}
    cmd_metrics = compute_cmd_metrics(cmd) if cmd.size else {"cmd_error": "no command data"}
    path_metrics = compute_path_metrics(pos, path_xy) if path_xy.size else {"path_error": "no path data"}

    print("=== parafoil_plannerv2 bag summary ===")
    for k, v in metrics.items():
        print(f"{k:>18}: {v}")
    for k, v in cmd_metrics.items():
        print(f"{k:>18}: {v}")
    for k, v in path_metrics.items():
        print(f"{k:>18}: {v}")
    if status:
        print("=== planner_status (last) ===")
        for k, v in status.items():
            print(f"{k:>18}: {v}")


if __name__ == "__main__":
    main()
