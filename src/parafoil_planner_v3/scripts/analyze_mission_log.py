#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from parafoil_planner_v3.environment import load_no_fly_polygons
from parafoil_planner_v3.landing_site_selector import RiskGrid
from parafoil_planner_v3.reporting.report_utils import line_svg, render_report, scatter_svg, xy_path_svg


def _load_logs(path: Path) -> list[dict]:
    if path.is_dir():
        return [json.loads(p.read_text()) for p in sorted(path.rglob("*.json"))]
    return [json.loads(path.read_text())]


def _extract_state(entry: dict) -> dict | None:
    if "state" in entry:
        return entry.get("state")
    return entry if "position" in entry else None


def _extract_control(entry: dict) -> dict | None:
    if "control" in entry:
        return entry.get("control")
    return entry if "delta_L" in entry or "delta_R" in entry else None


def _risk_grid_oob(grid: RiskGrid, north_m: float, east_m: float) -> bool:
    res = float(grid.resolution_m)
    if res <= 1e-9:
        return False
    u = (float(north_m) - float(grid.origin_n)) / res
    v = (float(east_m) - float(grid.origin_e)) / res
    return bool(u < 0.0 or v < 0.0 or u > (grid.risk_map.shape[0] - 1) or v > (grid.risk_map.shape[1] - 1))


def _compute_risk_metrics(path_xy: list[tuple[float, float]], risk_grid: RiskGrid | None) -> dict:
    if risk_grid is None or len(path_xy) < 2:
        return {"risk_available": False}

    integral = 0.0
    max_risk = float("-inf")
    total_len = 0.0
    oob_samples = 0
    n_samples = 0

    p_prev = np.asarray(path_xy[0], dtype=float)
    r_prev = float(risk_grid.risk_at(float(p_prev[0]), float(p_prev[1])))
    max_risk = max(max_risk, r_prev)
    if _risk_grid_oob(risk_grid, float(p_prev[0]), float(p_prev[1])):
        oob_samples += 1
    n_samples += 1

    for p in path_xy[1:]:
        p_cur = np.asarray(p, dtype=float)
        seg = p_cur - p_prev
        seg_len = float(np.linalg.norm(seg))
        if seg_len > 1e-9:
            r_cur = float(risk_grid.risk_at(float(p_cur[0]), float(p_cur[1])))
            integral += 0.5 * (r_prev + r_cur) * seg_len
            total_len += seg_len
            max_risk = max(max_risk, r_cur)
            if _risk_grid_oob(risk_grid, float(p_cur[0]), float(p_cur[1])):
                oob_samples += 1
            n_samples += 1
            r_prev = r_cur
        p_prev = p_cur

    if max_risk == float("-inf"):
        max_risk = 0.0
    mean = float(integral / total_len) if total_len > 1e-9 else 0.0
    return {
        "risk_available": True,
        "risk_integral": float(integral),
        "risk_max": float(max_risk),
        "risk_mean": float(mean),
        "oob_rate": float(oob_samples / max(n_samples, 1)),
    }


def _compute_nofly_metrics(path_xy: list[tuple[float, float]], no_fly_polygons) -> dict:
    if not no_fly_polygons or not path_xy:
        return {"nofly_available": False}

    violation = 0
    total = 0
    min_margin = float("inf")
    for n, e in path_xy:
        total += 1
        d_min = float("inf")
        for poly in no_fly_polygons:
            d_min = min(d_min, float(poly.signed_distance_m(float(n), float(e))))
        min_margin = min(min_margin, d_min)
        if d_min < 0.0:
            violation += 1
    rate = float(violation / max(total, 1))
    return {
        "nofly_available": True,
        "violation_samples": int(violation),
        "total_samples": int(total),
        "violation_rate": float(rate),
        "min_nofly_margin_m": float(min_margin) if np.isfinite(min_margin) else None,
    }


def _parse_planner_status(entry: Any) -> dict | None:
    if isinstance(entry, dict):
        return entry
    if isinstance(entry, str):
        text = entry.strip()
        if text.startswith("{"):
            try:
                payload = json.loads(text)
                return payload if isinstance(payload, dict) else None
            except json.JSONDecodeError:
                return None
        # Legacy key=value format
        out: dict = {}
        for token in text.split():
            if "=" in token:
                k, v = token.split("=", 1)
                out[k.strip()] = v.strip()
        return out if out else None
    return None


def _compute_tracking_ratio(tracking_history: list[dict]) -> float | None:
    if not tracking_history:
        return None
    items = sorted(
        [h for h in tracking_history if isinstance(h, dict) and "timestamp" in h],
        key=lambda x: float(x.get("timestamp", 0.0)),
    )
    if not items:
        return None
    if len(items) == 1:
        return 1.0 if str(items[0].get("mode", "")) == "strong_wind_l1" else 0.0
    strong_time = 0.0
    total_time = 0.0
    for i in range(len(items) - 1):
        t0 = float(items[i].get("timestamp", 0.0))
        t1 = float(items[i + 1].get("timestamp", 0.0))
        dt = float(max(t1 - t0, 0.0))
        total_time += dt
        if str(items[i].get("mode", "")) == "strong_wind_l1":
            strong_time += dt
    if total_time <= 1e-9:
        return None
    return float(strong_time / total_time)


def _compute_control_peaks(control_hist: list[dict], controller_logs: list[dict]) -> dict:
    brake_max = 0.0
    delta_a_max = 0.0
    for entry in control_hist:
        ctrl = _extract_control(entry) if isinstance(entry, dict) else None
        if not isinstance(ctrl, dict):
            continue
        dl = float(ctrl.get("delta_L", 0.0))
        dr = float(ctrl.get("delta_R", 0.0))
        brake = 0.5 * (dl + dr)
        delta_a = abs(dl - dr)
        brake_max = max(brake_max, brake)
        delta_a_max = max(delta_a_max, delta_a)

    yaw_rate_cmd_max = None
    for item in controller_logs:
        if not isinstance(item, dict):
            continue
        ctrl = item.get("control") if isinstance(item.get("control"), dict) else None
        if ctrl and "yaw_rate_cmd" in ctrl:
            val = abs(float(ctrl.get("yaw_rate_cmd", 0.0)))
            yaw_rate_cmd_max = max(yaw_rate_cmd_max or 0.0, val)

    return {
        "yaw_rate_cmd_max": float(yaw_rate_cmd_max) if yaw_rate_cmd_max is not None else None,
        "delta_a_max": float(delta_a_max),
        "brake_max": float(brake_max),
    }


def _metrics_from_log(log: dict, risk_grid: RiskGrid | None, no_fly_polygons) -> dict:
    cfg = log.get("config", {}) if isinstance(log, dict) else {}
    scenario = cfg.get("scenario", {}) if isinstance(cfg, dict) else {}
    target = scenario.get("target_ned") or scenario.get("target") or [0.0, 0.0, 0.0]
    target = np.asarray(target, dtype=float).reshape(3)

    state_hist = log.get("state_history", []) or []
    states = []
    times = []
    for entry in state_hist:
        st = _extract_state(entry) or {}
        pos = st.get("position")
        if pos is None:
            continue
        p = np.asarray(pos, dtype=float).reshape(3)
        states.append(p)
        times.append(float(st.get("t", entry.get("timestamp", 0.0))))

    path_xy = [(float(p[0]), float(p[1])) for p in states]
    last_state = states[-1] if states else np.zeros(3)
    final_xy = [float(last_state[0]), float(last_state[1])]

    landing_error = float(np.linalg.norm(last_state[:2] - target[:2])) if states else float("inf")
    touched_down = False
    events = log.get("events", []) or []
    for ev in events:
        if isinstance(ev, dict) and str(ev.get("type", "")) == "touchdown":
            touched_down = True
            break
    if not touched_down and states:
        touched_down = bool(float(-last_state[2]) <= 0.0)

    risk_metrics = _compute_risk_metrics(path_xy, risk_grid)
    nofly_metrics = _compute_nofly_metrics(path_xy, no_fly_polygons)

    tracking_ratio = _compute_tracking_ratio(log.get("tracking_history", []) or [])
    control_peaks = _compute_control_peaks(log.get("control_history", []) or [], log.get("controller_logs", []) or [])

    planner_status_hist = log.get("planner_status_history", []) or []
    reasons = Counter()
    library_hits = []
    for item in planner_status_hist:
        status = None
        if isinstance(item, dict) and "status" in item:
            status = _parse_planner_status(item.get("status"))
        else:
            status = _parse_planner_status(item)
        if not isinstance(status, dict):
            continue
        reason = status.get("reason")
        if reason is not None:
            reasons[str(reason)] += 1
        if "library_hit" in status:
            library_hits.append(bool(status.get("library_hit")))

    library_hit_rate = float(np.mean([1.0 if h else 0.0 for h in library_hits])) if library_hits else None

    return {
        "touchdown": {
            "touched_down": bool(touched_down),
            "position_xy": final_xy,
            "distance_to_target_m": float(landing_error),
        },
        "no_fly_violation_rate": nofly_metrics,
        "risk": risk_metrics,
        "strong_wind_l1_ratio": float(tracking_ratio) if tracking_ratio is not None else None,
        "control_peaks": control_peaks,
        "planner": {
            "library_hit_rate": float(library_hit_rate) if library_hit_rate is not None else None,
            "reason_distribution": dict(reasons),
        },
        "final_xy": final_xy,
    }


def _write_html(path: Path, logs: list[dict], metrics: list[dict]) -> None:
    landing_xy = [tuple(m.get("final_xy", [0.0, 0.0])) for m in metrics]

    summary_rows = [
        ("n_logs", len(metrics)),
        (
            "landing_error_p95",
            float(np.percentile([m["touchdown"]["distance_to_target_m"] for m in metrics], 95)) if metrics else 0.0,
        ),
    ]
    charts = [
        {"title": "Landing Scatter (XY)", "svg": scatter_svg(landing_xy, target_xy=(0.0, 0.0))},
    ]

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
        subtitle="E2E mission logs summarized into landing accuracy, safety, and control metrics.",
    )
    path.write_text(html)


def _summary(metrics: list[dict]) -> dict:
    if not metrics:
        return {}
    landing_errors = [m["touchdown"]["distance_to_target_m"] for m in metrics]
    risk_integrals = [m.get("risk", {}).get("risk_integral") for m in metrics if m.get("risk", {}).get("risk_integral") is not None]
    risk_max = [m.get("risk", {}).get("risk_max") for m in metrics if m.get("risk", {}).get("risk_max") is not None]
    nofly_rates = [
        m.get("no_fly_violation_rate", {}).get("violation_rate")
        for m in metrics
        if m.get("no_fly_violation_rate", {}).get("violation_rate") is not None
    ]
    return {
        "n_logs": int(len(metrics)),
        "landing_error_p95": float(np.percentile(landing_errors, 95)) if landing_errors else None,
        "risk_integral_mean": float(np.mean(risk_integrals)) if risk_integrals else None,
        "risk_max_mean": float(np.mean(risk_max)) if risk_max else None,
        "nofly_violation_rate_mean": float(np.mean(nofly_rates)) if nofly_rates else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze ROS2 mission logs and output safety/quality metrics.")
    parser.add_argument("--log", type=str, required=True, help="Path to a log JSON or directory")
    parser.add_argument("--risk-grid", type=str, default="", help="Risk grid file (.npz/.json/.yaml)")
    parser.add_argument("--no-fly", type=str, default="", help="No-fly polygon file (.json/.yaml/.geojson)")
    parser.add_argument("--out", type=str, default="metrics.json", help="Output metrics JSON file")
    parser.add_argument("--html", type=str, default="", help="Optional HTML report output")
    args = parser.parse_args()

    logs = _load_logs(Path(args.log))
    risk_grid = RiskGrid.from_file(args.risk_grid) if args.risk_grid else None
    no_fly_polygons = load_no_fly_polygons(args.no_fly) if args.no_fly else []

    metrics = [_metrics_from_log(log, risk_grid, no_fly_polygons) for log in logs]
    summary = _summary(metrics)

    payload = {"summary": summary, "metrics": metrics}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    if args.html:
        html_path = Path(args.html)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        _write_html(html_path, logs, metrics)

    print("Mission log metrics summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"Wrote metrics: {out}")
    if args.html:
        print(f"Wrote HTML report: {args.html}")


if __name__ == "__main__":
    main()
