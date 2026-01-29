#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from parafoil_planner_v3.reporting.report_utils import line_svg, render_report, scatter_svg, xy_path_svg


def _load_logs(path: Path) -> list[dict]:
    if path.is_dir():
        return [json.loads(p.read_text()) for p in sorted(path.rglob("*.json"))]
    return [json.loads(path.read_text())]


def _extract_state(entry: dict) -> dict | None:
    if "state" in entry:
        return entry.get("state")
    return entry if "position" in entry else None


def _metrics_from_log(log: dict) -> dict:
    cfg = log.get("config", {}) if isinstance(log, dict) else {}
    scenario = cfg.get("scenario", {}) if isinstance(cfg, dict) else {}
    target = scenario.get("target_ned") or scenario.get("target") or [0.0, 0.0, 0.0]
    target = np.asarray(target, dtype=float).reshape(3)

    state_hist = log.get("state_history", []) or []
    last_state = _extract_state(state_hist[-1]) if state_hist else {}
    pos = np.asarray(last_state.get("position", [0.0, 0.0, 0.0]), dtype=float).reshape(3) if isinstance(last_state, dict) else np.zeros(3)
    vel = np.asarray(last_state.get("velocity", [0.0, 0.0, 0.0]), dtype=float).reshape(3) if isinstance(last_state, dict) else np.zeros(3)

    landing_error = float(np.linalg.norm(pos[:2] - target[:2]))
    vertical_v = float(abs(vel[2]))
    time_s = float(last_state.get("t", 0.0)) if isinstance(last_state, dict) else 0.0

    return {
        "landing_error_m": landing_error,
        "vertical_velocity_mps": vertical_v,
        "time_s": time_s,
        "final_xy": [float(pos[0]), float(pos[1])],
    }


def _write_html(path: Path, logs: list[dict]) -> None:
    metrics = [_metrics_from_log(log) for log in logs]
    landing_xy = [tuple(m["final_xy"]) for m in metrics]

    summary_rows = [
        ("n_logs", len(metrics)),
        ("landing_error_p95", float(np.percentile([m["landing_error_m"] for m in metrics], 95)) if metrics else 0.0),
        ("vertical_velocity_p95", float(np.percentile([m["vertical_velocity_mps"] for m in metrics], 95)) if metrics else 0.0),
    ]
    charts = [
        {"title": "Landing Scatter (XY)", "svg": scatter_svg(landing_xy, target_xy=(0.0, 0.0))},
    ]

    # Add path/altitude from first log if available
    if logs:
        hist = logs[0].get("state_history", []) or []
        if hist:
            xy = []
            alt = []
            t = []
            for entry in hist:
                st = _extract_state(entry) or {}
                pos = st.get("position", [0.0, 0.0, 0.0])
                xy.append((float(pos[0]), float(pos[1])))
                alt.append(float(-pos[2]))
                t.append(float(st.get("t", entry.get("timestamp", 0.0))))
            charts.append({"title": "XY Path (example)", "svg": xy_path_svg([{"label": "trajectory", "xy": xy, "color": "#4C78A8"}])})
            charts.append({"title": "Altitude Profile (example)", "svg": line_svg(t, alt, x_label="t (s)", y_label="altitude (m)")})

    payload = {"metrics": metrics, "logs": logs}
    html = render_report(
        title="parafoil_planner_v3 - Mission Log Analysis",
        summary_rows=summary_rows,
        charts=charts,
        payload=payload,
        subtitle="ROS2 mission logs summarized into landing accuracy and flight profiles.",
    )
    path.write_text(html)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze ROS2 mission logs and render a report.")
    parser.add_argument("--input", type=str, required=True, help="Path to a log JSON or directory")
    parser.add_argument("--output", type=str, default="mission_report.html")
    args = parser.parse_args()

    logs = _load_logs(Path(args.input))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_html(out, logs)
    print(f"Wrote report: {out}")


if __name__ == "__main__":
    main()
