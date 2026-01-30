#!/usr/bin/env python3

from __future__ import annotations

"""
Evaluate trajectory-library "hit rate" (library match success) across sampled scenarios.

This script does *not* run the online GPM solver. It measures whether the library
retrieval + adapt_trajectory + feasibility checks can return a usable trajectory.

Outputs:
  - Overall hit/skip rates
  - Failure-reason counts
  - Hit rate binned by (wind_speed, relative_wind_angle) by default

Example:
  python3 evaluate_library_hit_rate.py \\
    --fine /tmp/parafoil_library_strongwind_headwind.pkl \\
    --coarse /tmp/parafoil_library_strongwind_headwind_coarse.pkl \\
    --runs 5000 --wind-speed-range 6,12 --alt-range 20,80 --dist-range 50,300 \\
    --output /tmp/library_hit_rate.json
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from parafoil_planner_v3.planner_core import PlannerConfig, PlannerCore
from parafoil_planner_v3.trajectory_library.library_manager import TrajectoryLibrary
from parafoil_planner_v3.trajectory_library.scenario_features import compute_scenario_features
from parafoil_planner_v3.trajectory_library.trajectory_adapter import adapt_trajectory
from parafoil_planner_v3.types import State, Target, Wind


@dataclass(frozen=True)
class SampleConfig:
    runs: int
    seed: int
    alt_range: tuple[float, float]
    dist_range: tuple[float, float]
    bearing_range_deg: tuple[float, float]
    wind_speed_range: tuple[float, float]
    wind_dir_range_deg: tuple[float, float]


def _parse_range(spec: str, *, name: str) -> tuple[float, float]:
    parts = [p.strip() for p in str(spec).split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"--{name} expects 'min,max' (got '{spec}')")
    a = float(parts[0])
    b = float(parts[1])
    lo = min(a, b)
    hi = max(a, b)
    return (lo, hi)


def _make_state_target_wind(*, altitude_m: float, distance_m: float, bearing_deg: float, wind_speed_mps: float, wind_direction_deg: float) -> tuple[State, Target, Wind]:
    bearing = float(np.deg2rad(bearing_deg))
    # Place state at origin, target at (distance, bearing) in NED.
    state = State(
        p_I=np.array([0.0, 0.0, -float(altitude_m)], dtype=float),
        v_I=np.zeros(3, dtype=float),
        q_IB=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
        w_B=np.zeros(3, dtype=float),
        t=0.0,
    )
    target = Target(
        p_I=np.array(
            [float(distance_m) * float(np.cos(bearing)), float(distance_m) * float(np.sin(bearing)), 0.0],
            dtype=float,
        )
    )
    wd = float(np.deg2rad(wind_direction_deg))
    wind = Wind(v_I=np.array([wind_speed_mps * np.cos(wd), wind_speed_mps * np.sin(wd), 0.0], dtype=float))
    return state, target, wind


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0.0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=float).reshape(-1)
    return {
        "n": float(arr.size),
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _match_stage(
    planner: PlannerCore,
    library: TrajectoryLibrary,
    features: np.ndarray,
    target: Target,
    state: State,
    wind: Wind,
    *,
    k: int,
    stage: str,
) -> tuple[bool, dict[str, Any]]:
    """
    Return (hit, meta) where `hit` indicates a feasible trajectory was found.
    meta includes:
      - best_* fields (if hit)
      - nn_distance stats
      - reject_reason_counts over the KNN candidates
    """
    dist, idx = library.query_knn(features, k=max(int(k), 1))
    idx_arr = np.atleast_1d(idx)
    dist_arr = np.atleast_1d(dist)

    reject_reason_counts: dict[str, int] = {}
    best_cost = float("inf")
    best_idx = None
    best_dist = None
    best_term_err = float("inf")

    # For this offline evaluation, ignore terrain/nofly. We only evaluate intrinsic
    # constraints and terminal tolerance.
    terrain = None
    no_fly_circles: list = []
    no_fly_polygons: list = []

    for i, cand_idx in enumerate(idx_arr.tolist()):
        lib = library[int(cand_idx)]
        adapted = adapt_trajectory(lib, state, target, wind)
        knn_dist = float(dist_arr[i]) if i < int(dist_arr.size) else None

        match_ok, match_reason = planner._library_match_ok(lib, features, knn_dist)
        scale_ok, scale_reason = planner._library_scale_ok(adapted)
        if not match_ok:
            feasible, reason = False, f"match:{match_reason}"
        elif not scale_ok:
            feasible, reason = False, f"scale:{scale_reason}"
        else:
            feasible, reason = planner._check_trajectory_feasible(adapted, target, wind, terrain, no_fly_circles, no_fly_polygons)

        reject_reason_counts[str(reason)] = int(reject_reason_counts.get(str(reason), 0) + 1)

        if feasible:
            cost = float(planner._evaluate_library_cost(adapted, target, wind))
            term_err = float(np.linalg.norm(adapted.waypoints[-1].state.position_xy - target.position_xy)) if adapted.waypoints else float("inf")
            if cost < best_cost:
                best_cost = float(cost)
                best_idx = int(cand_idx)
                best_dist = float(knn_dist) if knn_dist is not None else None
                best_term_err = float(term_err)

    nn_dists = [float(x) for x in dist_arr.reshape(-1).tolist()]
    meta: dict[str, Any] = {
        "stage": str(stage),
        "k": int(k),
        "nn_distance": _stats(nn_dists),
        "reject_reason_counts": reject_reason_counts,
    }
    hit = best_idx is not None
    if hit:
        meta.update(
            {
                "best_idx": int(best_idx),
                "best_knn_distance": best_dist,
                "best_cost": float(best_cost),
                "best_terminal_error_m": float(best_term_err),
            }
        )
    return bool(hit), meta


def _bin_index(x: float, edges: np.ndarray) -> int:
    # edges are ascending bin edges; returns bin id in [0, n_bins-1], or -1 if OOB.
    if not np.isfinite(float(x)):
        return -1
    i = int(np.searchsorted(edges, float(x), side="right") - 1)
    if i < 0 or i >= int(edges.size) - 1:
        return -1
    return i


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate library match hit rate without running online GPM.")
    parser.add_argument("--fine", type=str, required=True, help="Fine library .pkl path")
    parser.add_argument("--coarse", type=str, default="", help="Optional coarse library .pkl path")
    parser.add_argument("--runs", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--k-fine", type=int, default=5)
    parser.add_argument("--k-coarse", type=int, default=5)
    parser.add_argument("--require-coarse-match", action="store_true", help="Require coarse hit before trying fine.")
    parser.add_argument("--fallback-to-coarse", action="store_true", help="If fine misses, allow selecting coarse.")

    parser.add_argument("--terminal-pos-tol", type=float, default=30.0)
    parser.add_argument("--skip-unreachable-wind", action="store_true", help="Skip library if wind makes target line untrackable.")
    parser.add_argument("--min-track-ground-speed", type=float, default=0.2)

    parser.add_argument("--alt-range", type=str, default="20,60")
    parser.add_argument("--dist-range", type=str, default="50,300")
    parser.add_argument("--bearing-range-deg", type=str, default="-180,180")
    parser.add_argument("--wind-speed-range", type=str, default="6,12")
    parser.add_argument("--wind-dir-range-deg", type=str, default="0,360")

    parser.add_argument(
        "--wind-speed-bins",
        type=str,
        default="0,2,4,6,8,10,12,14",
        help="Comma-separated bin edges for wind speed (m/s).",
    )
    parser.add_argument(
        "--rel-wind-bins-deg",
        type=str,
        default="-180,-120,-60,-30,-10,10,30,60,120,180",
        help="Comma-separated bin edges for relative wind angle (deg).",
    )

    parser.add_argument("--output", type=str, default="", help="Optional output path (.json).")
    args = parser.parse_args()

    fine = TrajectoryLibrary.load(str(args.fine))
    coarse = TrajectoryLibrary.load(str(args.coarse)) if str(args.coarse).strip() else None

    cfg = PlannerConfig(
        use_library=True,
        library_terminal_pos_tol_m=float(args.terminal_pos_tol),
        library_coarse_k_neighbors=int(args.k_coarse),
        library_fine_k_neighbors=int(args.k_fine),
        library_require_coarse_match=bool(args.require_coarse_match),
        library_fallback_to_coarse=bool(args.fallback_to_coarse),
        library_skip_if_unreachable_wind=bool(args.skip_unreachable_wind),
        library_min_track_ground_speed_mps=float(args.min_track_ground_speed),
        headwind_enable=False,  # evaluation is on touchdown target
    )
    # The dynamics object is unused for matching/feasibility checks here; pass None-like stub.
    from parafoil_planner_v3.dynamics.parafoil_6dof import SixDOFDynamics

    planner = PlannerCore(dynamics=SixDOFDynamics(), config=cfg, library_coarse=coarse, library_fine=fine)

    sample_cfg = SampleConfig(
        runs=int(args.runs),
        seed=int(args.seed),
        alt_range=_parse_range(args.alt_range, name="alt-range"),
        dist_range=_parse_range(args.dist_range, name="dist-range"),
        bearing_range_deg=_parse_range(args.bearing_range_deg, name="bearing-range-deg"),
        wind_speed_range=_parse_range(args.wind_speed_range, name="wind-speed-range"),
        wind_dir_range_deg=_parse_range(args.wind_dir_range_deg, name="wind-dir-range-deg"),
    )

    wind_speed_edges = np.asarray([float(x) for x in args.wind_speed_bins.split(",") if x.strip()], dtype=float)
    rel_wind_edges_deg = np.asarray([float(x) for x in args.rel_wind_bins_deg.split(",") if x.strip()], dtype=float)
    if wind_speed_edges.size < 2 or rel_wind_edges_deg.size < 2:
        raise ValueError("Need at least 2 edges for wind-speed-bins and rel-wind-bins-deg")
    n_ws = int(wind_speed_edges.size) - 1
    n_rw = int(rel_wind_edges_deg.size) - 1

    # Accumulators
    total = int(sample_cfg.runs)
    n_skip = 0
    skip_reason_counts: dict[str, int] = {}

    n_hit_coarse = 0
    n_hit_fine = 0
    n_hit_final = 0
    n_gate_block = 0  # coarse miss blocks fine (only meaningful when require_coarse_match=true)

    final_fail_reason_counts: dict[str, int] = {}

    # Binned stats
    bin_total = np.zeros((n_ws, n_rw), dtype=int)
    bin_hit = np.zeros((n_ws, n_rw), dtype=int)
    bin_skip = np.zeros((n_ws, n_rw), dtype=int)

    rng = np.random.default_rng(int(sample_cfg.seed))
    for _ in range(total):
        alt = float(rng.uniform(sample_cfg.alt_range[0], sample_cfg.alt_range[1]))
        dist = float(rng.uniform(sample_cfg.dist_range[0], sample_cfg.dist_range[1]))
        bearing_deg = float(rng.uniform(sample_cfg.bearing_range_deg[0], sample_cfg.bearing_range_deg[1]))
        wind_speed = float(rng.uniform(sample_cfg.wind_speed_range[0], sample_cfg.wind_speed_range[1]))
        wind_dir_deg = float(rng.uniform(sample_cfg.wind_dir_range_deg[0], sample_cfg.wind_dir_range_deg[1]))

        state, target, wind = _make_state_target_wind(
            altitude_m=alt,
            distance_m=dist,
            bearing_deg=bearing_deg,
            wind_speed_mps=wind_speed,
            wind_direction_deg=wind_dir_deg,
        )
        feats = compute_scenario_features(state, target, wind)

        # bin indices use wind speed and rel-wind angle (deg)
        wind_speed_mps = float(feats[3])
        rel_wind_deg = float(np.rad2deg(float(feats[4])))
        bi_ws = _bin_index(wind_speed_mps, wind_speed_edges)
        bi_rw = _bin_index(rel_wind_deg, rel_wind_edges_deg)
        if bi_ws >= 0 and bi_rw >= 0:
            bin_total[bi_ws, bi_rw] += 1

        # Optional "unreachable wind" gate (matches PlannerCore.plan)
        if bool(cfg.library_skip_if_unreachable_wind):
            track = planner._wind_trackability_diag(state, target, wind)
            if not bool(track.get("cross_ok", True)):
                n_skip += 1
                r = "wind_crosswind_exceeds_V_air_max"
                skip_reason_counts[r] = int(skip_reason_counts.get(r, 0) + 1)
                if bi_ws >= 0 and bi_rw >= 0:
                    bin_skip[bi_ws, bi_rw] += 1
                continue
            v_track_max = float(track.get("v_track_max_mps", 0.0))
            if v_track_max <= float(cfg.library_min_track_ground_speed_mps):
                n_skip += 1
                r = "wind_no_progress_to_target"
                skip_reason_counts[r] = int(skip_reason_counts.get(r, 0) + 1)
                if bi_ws >= 0 and bi_rw >= 0:
                    bin_skip[bi_ws, bi_rw] += 1
                continue

        coarse_hit = False
        coarse_meta: dict[str, Any] = {}
        if coarse is not None and len(coarse) > 0:
            coarse_hit, coarse_meta = _match_stage(planner, coarse, feats, target, state, wind, k=int(cfg.library_coarse_k_neighbors), stage="coarse")
        if coarse_hit:
            n_hit_coarse += 1

        # If coarse is required and misses, we consider it gate-blocked.
        if bool(cfg.library_require_coarse_match) and coarse is not None and not coarse_hit:
            n_gate_block += 1
            # For diagnostics, also compute whether fine *would* have hit (to quantify gating harm).
            fine_hit, fine_meta = _match_stage(planner, fine, feats, target, state, wind, k=int(cfg.library_fine_k_neighbors), stage="fine")
            if fine_hit:
                # gate blocked a possible success
                final_fail_reason_counts["gate_blocked_fine_hit"] = int(final_fail_reason_counts.get("gate_blocked_fine_hit", 0) + 1)
            else:
                # propagate dominant reject reason from fine stage (mode over KNN candidates)
                rc = fine_meta.get("reject_reason_counts", {}) or {}
                if rc:
                    dom = max(rc.items(), key=lambda kv: int(kv[1]))[0]
                    final_fail_reason_counts[f"fine:{dom}"] = int(final_fail_reason_counts.get(f"fine:{dom}", 0) + 1)
                else:
                    final_fail_reason_counts["fine:no_candidate"] = int(final_fail_reason_counts.get("fine:no_candidate", 0) + 1)
            if bi_ws >= 0 and bi_rw >= 0:
                # gate-blocked counts as miss
                pass
            continue

        fine_hit, fine_meta = _match_stage(planner, fine, feats, target, state, wind, k=int(cfg.library_fine_k_neighbors), stage="fine")
        if fine_hit:
            n_hit_fine += 1

        final_hit = bool(fine_hit or (bool(cfg.library_fallback_to_coarse) and coarse_hit))
        if final_hit:
            n_hit_final += 1
            if bi_ws >= 0 and bi_rw >= 0:
                bin_hit[bi_ws, bi_rw] += 1
        else:
            # Record a lightweight reason for "why did we miss?"
            if coarse is not None and coarse_hit:
                final_fail_reason_counts["fine_miss_but_coarse_hit_fallback_disabled"] = int(
                    final_fail_reason_counts.get("fine_miss_but_coarse_hit_fallback_disabled", 0) + 1
                )
            else:
                rc = fine_meta.get("reject_reason_counts", {}) or {}
                if rc:
                    dom = max(rc.items(), key=lambda kv: int(kv[1]))[0]
                    final_fail_reason_counts[f"fine:{dom}"] = int(final_fail_reason_counts.get(f"fine:{dom}", 0) + 1)
                else:
                    final_fail_reason_counts["fine:no_candidate"] = int(final_fail_reason_counts.get("fine:no_candidate", 0) + 1)

    # Build binned hit-rate table
    bins_out: list[dict[str, Any]] = []
    for i_ws in range(n_ws):
        for i_rw in range(n_rw):
            n = int(bin_total[i_ws, i_rw])
            if n <= 0:
                continue
            bins_out.append(
                {
                    "wind_speed_bin": [float(wind_speed_edges[i_ws]), float(wind_speed_edges[i_ws + 1])],
                    "rel_wind_bin_deg": [float(rel_wind_edges_deg[i_rw]), float(rel_wind_edges_deg[i_rw + 1])],
                    "n": int(n),
                    "hit_rate": float(bin_hit[i_ws, i_rw] / max(n, 1)),
                    "skip_rate": float(bin_skip[i_ws, i_rw] / max(n, 1)),
                }
            )

    summary = {
        "sample_config": asdict(sample_cfg),
        "libraries": {
            "fine": {"path": str(args.fine), "n": int(len(fine))},
            "coarse": {"path": str(args.coarse), "n": int(len(coarse))} if coarse is not None else None,
        },
        "planner_config": {
            "k_fine": int(cfg.library_fine_k_neighbors),
            "k_coarse": int(cfg.library_coarse_k_neighbors),
            "require_coarse_match": bool(cfg.library_require_coarse_match),
            "fallback_to_coarse": bool(cfg.library_fallback_to_coarse),
            "terminal_pos_tol_m": float(cfg.library_terminal_pos_tol_m),
            "skip_unreachable_wind": bool(cfg.library_skip_if_unreachable_wind),
            "min_track_ground_speed_mps": float(cfg.library_min_track_ground_speed_mps),
        },
        "rates": {
            "skip_rate": float(n_skip / max(total, 1)),
            "coarse_hit_rate": float(n_hit_coarse / max(total - n_skip, 1)),
            "fine_hit_rate": float(n_hit_fine / max(total - n_skip, 1)),
            "final_library_hit_rate": float(n_hit_final / max(total - n_skip, 1)),
            "gate_block_rate": float(n_gate_block / max(total - n_skip, 1)) if coarse is not None else 0.0,
        },
        "counts": {
            "n_total": int(total),
            "n_skip": int(n_skip),
            "n_eval": int(total - n_skip),
            "n_hit_coarse": int(n_hit_coarse),
            "n_hit_fine": int(n_hit_fine),
            "n_hit_final": int(n_hit_final),
            "n_gate_block": int(n_gate_block),
        },
        "skip_reason_counts": skip_reason_counts,
        "final_fail_reason_counts": final_fail_reason_counts,
        "bins": bins_out,
    }

    out_text = json.dumps(summary, indent=2)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(out_text)
        print(f"Wrote: {out}")
    else:
        print(out_text)


if __name__ == "__main__":
    main()

