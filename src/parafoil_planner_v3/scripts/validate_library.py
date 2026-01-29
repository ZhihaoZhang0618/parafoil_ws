#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from parafoil_planner_v3.trajectory_library.library_manager import TrajectoryLibrary
from parafoil_planner_v3.reporting.report_utils import bar_svg, histogram_svg, render_report


METRIC_KEYS = [
    "duration_s",
    "altitude_loss_m",
    "path_length_m",
    "max_bank_deg",
    "max_yaw_rate_deg_s",
    "max_brake",
    "max_delta_a",
    "turn_total_turns",
    "turn_net_turns",
    "turn_sign_changes",
    "turn_clusters",
    "turn_dominant_fraction",
]


def _stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "p50": 0.0, "p95": 0.0}
    arr = np.asarray(values, dtype=float).reshape(-1)
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


def _hist(values: list[float], bins: int) -> dict:
    if not values:
        return {"bins": [], "counts": []}
    arr = np.asarray(values, dtype=float).reshape(-1)
    counts, edges = np.histogram(arr, bins=int(bins))
    return {"bins": [float(x) for x in edges.tolist()], "counts": [int(c) for c in counts.tolist()]}


def _traj_type_name(entry) -> str:
    t = getattr(entry, "trajectory_type", None)
    return getattr(t, "value", str(t))


def _extract_metrics(entry) -> dict:
    metrics: dict = {}
    traj_meta = (entry.trajectory.metadata or {}) if getattr(entry, "trajectory", None) is not None else {}
    if isinstance(traj_meta, dict):
        for k in METRIC_KEYS:
            if k in traj_meta:
                metrics[k] = float(traj_meta[k])

    entry_meta = entry.metadata or {}
    if isinstance(entry_meta, dict):
        tm = entry_meta.get("trajectory_metrics", {})
        if isinstance(tm, dict):
            for k in METRIC_KEYS:
                if k not in metrics and k in tm:
                    metrics[k] = float(tm[k])
    return metrics


def _write_html(path: Path, summary: dict, results: dict, metric_values: dict) -> None:
    payload = {"summary": summary, "report": results}
    metrics = results.get("metrics", {})
    traj_counts = results.get("trajectory_type_counts", {})
    shape = results.get("shape", {})

    summary_rows = [
        ("n_total", summary.get("n_total")),
        ("n_with_metrics", summary.get("n_with_metrics")),
        ("shape_ok_rate", shape.get("shape_ok_rate")),
    ]

    charts = [
        {"title": "Altitude Loss (m)", "svg": histogram_svg(metric_values.get("altitude_loss_m", []), bins=20, x_label="m")},
        {"title": "Max Bank (deg)", "svg": histogram_svg(metric_values.get("max_bank_deg", []), bins=20, x_label="deg")},
        {"title": "Duration (s)", "svg": histogram_svg(metric_values.get("duration_s", []), bins=20, x_label="s")},
    ]
    if traj_counts:
        labels = list(traj_counts.keys())
        values = [traj_counts[k] for k in labels]
        charts.append({"title": "Trajectory Type Counts", "svg": bar_svg(labels, values)})

    html = render_report(
        title="parafoil_planner_v3 - Library Quality Report",
        summary_rows=summary_rows,
        charts=charts,
        payload=payload,
        subtitle="Library metadata distribution and shape constraint statistics.",
    )
    path.write_text(html)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a generated trajectory library and output a quality report.")
    parser.add_argument("--library", type=str, required=True, help="Path to library pickle")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--bins", type=int, default=12)
    parser.add_argument("--output", type=str, default="", help="Optional report path (.json or .html)")
    args = parser.parse_args()

    lib = TrajectoryLibrary.load(args.library)
    print(f"Loaded library: {len(lib)} trajectories")

    # Query with random features (sanity)
    features = np.array([80.0, 120.0, 0.0, 2.0, 0.0], dtype=float)
    dist, idx = lib.query_knn(features, k=args.k)
    print("KNN distances:", dist)
    print("KNN indices:", idx)

    by_metric: dict[str, list[float]] = {k: [] for k in METRIC_KEYS}
    by_type: dict[str, dict[str, list[float]]] = {}
    type_counts: dict[str, int] = {}
    generator_counts: dict[str, int] = {}
    shape_ok = 0
    shape_total = 0
    shape_reasons: dict[str, int] = {}
    n_with_metrics = 0

    for entry in lib:
        t_name = _traj_type_name(entry)
        type_counts[t_name] = type_counts.get(t_name, 0) + 1
        meta = entry.metadata or {}
        gen = meta.get("generator", "unknown") if isinstance(meta, dict) else "unknown"
        generator_counts[gen] = generator_counts.get(gen, 0) + 1

        if isinstance(meta, dict) and "shape_ok" in meta:
            shape_total += 1
            if bool(meta.get("shape_ok")):
                shape_ok += 1
            else:
                reason = str(meta.get("shape_reason", "unknown"))
                shape_reasons[reason] = shape_reasons.get(reason, 0) + 1

        metrics = _extract_metrics(entry)
        if metrics:
            n_with_metrics += 1
        for k, v in metrics.items():
            by_metric[k].append(float(v))
            by_type.setdefault(t_name, {}).setdefault(k, []).append(float(v))

    metrics_summary = {k: _stats(v) for k, v in by_metric.items()}
    metrics_hist = {k: _hist(v, bins=args.bins) for k, v in by_metric.items() if v}

    metrics_by_type = {
        t: {k: _stats(v) for k, v in vals.items()} for t, vals in by_type.items()
    }

    report = {
        "metrics": metrics_summary,
        "metrics_hist": metrics_hist,
        "metrics_by_type": metrics_by_type,
        "trajectory_type_counts": type_counts,
        "generator_counts": generator_counts,
        "shape": {
            "shape_total": int(shape_total),
            "shape_ok": int(shape_ok),
            "shape_ok_rate": float(shape_ok / max(shape_total, 1)) if shape_total else 0.0,
            "shape_fail_reasons": shape_reasons,
        },
    }
    summary = {
        "n_total": int(len(lib)),
        "n_with_metrics": int(n_with_metrics),
        "shape": report["shape"],
    }

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix.lower() == ".html":
            _write_html(out, summary, report, by_metric)
        else:
            out.write_text(json.dumps({"summary": summary, "report": report}, indent=2))
        print(f"Wrote report: {out}")
    else:
        print(json.dumps({"summary": summary, "report": report}, indent=2))


if __name__ == "__main__":
    main()
