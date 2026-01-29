#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from parafoil_planner_v3.dynamics.aerodynamics import PolarTable
from parafoil_planner_v3.dynamics.parafoil_6dof import SixDOFDynamics
from parafoil_planner_v3.dynamics.simplified_model import KinematicYawGlideDynamics, yaw_only_quat_wxyz
from parafoil_planner_v3.guidance.control_laws import LateralControlConfig, heading_rad, track_point_control
from parafoil_planner_v3.offline.simulator import OfflineSimulator
from parafoil_planner_v3.types import Control, State, Wind
from parafoil_planner_v3.utils.quaternion_utils import wrap_pi
from parafoil_planner_v3.reporting.report_utils import histogram_svg, line_svg, multi_line_svg, render_report


@dataclass(frozen=True)
class CaseConfig:
    name: str
    path_xy: np.ndarray  # (M,2)
    brake_profile: str  # constant|flare
    brake_base: float
    flare_duration_s: float


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


def _polyline_cumulative_s(path_xy: np.ndarray) -> np.ndarray:
    path_xy = np.asarray(path_xy, dtype=float).reshape(-1, 2)
    if path_xy.shape[0] == 0:
        return np.zeros((0,), dtype=float)
    s = np.zeros((path_xy.shape[0],), dtype=float)
    for i in range(1, path_xy.shape[0]):
        s[i] = s[i - 1] + float(np.linalg.norm(path_xy[i] - path_xy[i - 1]))
    return s


def _interp_on_polyline(path_xy: np.ndarray, s_cum: np.ndarray, s: float) -> Tuple[np.ndarray, np.ndarray]:
    path_xy = np.asarray(path_xy, dtype=float).reshape(-1, 2)
    s_cum = np.asarray(s_cum, dtype=float).reshape(-1)
    if path_xy.shape[0] == 0:
        return np.zeros((2,), dtype=float), np.array([1.0, 0.0], dtype=float)
    if path_xy.shape[0] == 1:
        return path_xy[0].copy(), np.array([1.0, 0.0], dtype=float)

    s = float(np.clip(s, float(s_cum[0]), float(s_cum[-1])))
    i = int(np.searchsorted(s_cum, s, side="right") - 1)
    i = int(np.clip(i, 0, path_xy.shape[0] - 2))
    s0 = float(s_cum[i])
    s1 = float(s_cum[i + 1])
    p0 = path_xy[i]
    p1 = path_xy[i + 1]
    ds = float(max(s1 - s0, 1e-9))
    a = float(np.clip((s - s0) / ds, 0.0, 1.0))
    p = (1.0 - a) * p0 + a * p1
    t_hat = p1 - p0
    n = float(np.linalg.norm(t_hat))
    t_hat = t_hat / n if n > 1e-9 else np.array([1.0, 0.0], dtype=float)
    return p.astype(float), t_hat.astype(float)


def _project_point_to_polyline(path_xy: np.ndarray, s_cum: np.ndarray, p_xy: np.ndarray) -> Tuple[float, float]:
    """
    Returns:
      s_proj (m), cross_track_signed (m)
    """
    path_xy = np.asarray(path_xy, dtype=float).reshape(-1, 2)
    s_cum = np.asarray(s_cum, dtype=float).reshape(-1)
    p_xy = np.asarray(p_xy, dtype=float).reshape(2)
    if path_xy.shape[0] < 2:
        return 0.0, float(np.linalg.norm(p_xy - (path_xy[0] if path_xy.shape[0] else 0.0)))

    best_d2 = float("inf")
    best_s = 0.0
    best_sign = 0.0
    for i in range(path_xy.shape[0] - 1):
        a = path_xy[i]
        b = path_xy[i + 1]
        ab = b - a
        ab2 = float(np.dot(ab, ab))
        if ab2 < 1e-12:
            continue
        t = float(np.clip(np.dot(p_xy - a, ab) / ab2, 0.0, 1.0))
        proj = a + t * ab
        d = p_xy - proj
        d2 = float(np.dot(d, d))
        if d2 < best_d2:
            best_d2 = d2
            best_s = float(s_cum[i] + t * np.sqrt(ab2))
            cross = float(ab[0] * d[1] - ab[1] * d[0])  # z-component of 2D cross
            best_sign = float(np.sign(cross)) if abs(cross) > 1e-12 else 0.0
    return float(best_s), float(best_sign * np.sqrt(best_d2))


def _smoothstep01(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return float(3.0 * x * x - 2.0 * x * x * x)


def _build_cases() -> list[CaseConfig]:
    # Paths are defined in ENU-ish XY (consistent with State.position_xy).
    x = np.linspace(-200.0, 0.0, 121, dtype=float)
    straight = np.stack([x, np.zeros_like(x)], axis=1)

    s_turn = np.stack([x, 35.0 * np.sin(2.0 * np.pi * (x - x[0]) / (x[-1] - x[0]))], axis=1)

    # Half-circle turn: start left, end right
    r = 70.0
    ang = np.linspace(np.pi, 0.0, 121, dtype=float)
    turn = np.stack([r * np.cos(ang), r * np.sin(ang)], axis=1)
    # Shift so final is near origin
    turn[:, 0] -= float(turn[-1, 0])
    turn[:, 1] -= float(turn[-1, 1])

    return [
        CaseConfig(name="straight", path_xy=straight, brake_profile="constant", brake_base=0.2, flare_duration_s=0.0),
        CaseConfig(name="s_turn", path_xy=s_turn, brake_profile="constant", brake_base=0.2, flare_duration_s=0.0),
        CaseConfig(name="turn", path_xy=turn, brake_profile="constant", brake_base=0.2, flare_duration_s=0.0),
        CaseConfig(name="flare_straight", path_xy=straight, brake_profile="flare", brake_base=0.2, flare_duration_s=2.0),
    ]


def _brake_at(t: float, duration: float, cfg: CaseConfig) -> float:
    if cfg.brake_profile == "flare":
        t0 = float(max(duration - cfg.flare_duration_s, 0.0))
        a = 0.0 if cfg.flare_duration_s <= 1e-6 else float((t - t0) / cfg.flare_duration_s)
        return float(cfg.brake_base + (1.0 - cfg.brake_base) * _smoothstep01(a))
    return float(cfg.brake_base)


def _run_one(task: tuple[dict, str, float, float, float, float, int, bool]) -> dict:
    """
    Task tuple:
      case_dict, dynamics_mode, dt, L1, wind_speed, wind_dir_deg, seed, record_history
    """
    case_dict, dynamics_mode, dt, L1, wind_speed, wind_dir_deg, seed, record_history = task
    case = CaseConfig(
        name=str(case_dict["name"]),
        path_xy=np.asarray(case_dict["path_xy"], dtype=float),
        brake_profile=str(case_dict["brake_profile"]),
        brake_base=float(case_dict["brake_base"]),
        flare_duration_s=float(case_dict["flare_duration_s"]),
    )
    dynamics_mode = str(dynamics_mode).strip().lower()
    if dynamics_mode in {"simplified", "simple"}:
        dynamics = KinematicYawGlideDynamics()
        dyn_tag = "simplified"
    else:
        dynamics = SixDOFDynamics()
        dyn_tag = "6dof"

    dt = float(max(dt, 1e-3))
    L1 = float(max(L1, 0.1))

    polar = PolarTable()
    s_cum = _polyline_cumulative_s(case.path_xy)
    path_len = float(s_cum[-1]) if s_cum.size else 0.0

    # Build wind (constant)
    wd = float(np.deg2rad(wind_dir_deg))
    wind = Wind(v_I=np.array([float(wind_speed) * np.cos(wd), float(wind_speed) * np.sin(wd), 0.0], dtype=float))
    wind_xy = wind.v_I[:2].copy()

    # Estimate duration by integrating expected progress (accounts for wind + flare slowdown).
    V_air0, _ = polar.interpolate(case.brake_base)
    duration = float(path_len / max(V_air0, 0.5))
    for _ in range(4):
        s_ref = 0.0
        for k in range(int(np.ceil(duration / dt))):
            t = float(k * dt)
            _, t_hat = _interp_on_polyline(case.path_xy, s_cum, s_ref)
            b = _brake_at(t, duration, case)
            V_air_k, _ = polar.interpolate(b)
            v_ground = np.array([V_air_k * t_hat[0], V_air_k * t_hat[1]], dtype=float) + wind_xy
            v_along = float(np.dot(v_ground, t_hat))
            s_ref = float(np.clip(s_ref + v_along * dt, 0.0, path_len))
        if s_ref <= 1e-6:
            break
        duration = float(np.clip(duration * (path_len / s_ref), 1.0, 300.0))

    # Reference schedules (s_ref(t), heading_ref(t), altitude_ref(t))
    s_ref = 0.0
    s_ref_hist: list[float] = [s_ref]
    heading_ref_hist: list[float] = []
    alt_loss_hist: list[float] = [0.0]
    for k in range(int(np.ceil(duration / dt))):
        t = float(k * dt)
        _, t_hat = _interp_on_polyline(case.path_xy, s_cum, s_ref)
        heading_ref_hist.append(float(np.arctan2(t_hat[1], t_hat[0])))
        b = _brake_at(t, duration, case)
        V_air_k, sink_k = polar.interpolate(b)
        v_ground = np.array([V_air_k * t_hat[0], V_air_k * t_hat[1]], dtype=float) + wind_xy
        v_along = float(np.dot(v_ground, t_hat))
        s_ref = float(np.clip(s_ref + v_along * dt, 0.0, path_len))
        s_ref_hist.append(s_ref)
        alt_loss_hist.append(float(alt_loss_hist[-1] + sink_k * dt))

    alt0 = float(alt_loss_hist[-1] + 20.0)  # keep altitude positive with margin
    alt_ref_hist = [float(alt0 - loss) for loss in alt_loss_hist]

    n_steps = len(s_ref_hist) - 1

    # Initial condition (start near path start with a small offset)
    p0, t0_hat = _interp_on_polyline(case.path_xy, s_cum, 0.0)
    n_hat = np.array([-t0_hat[1], t0_hat[0]], dtype=float)
    rng = np.random.default_rng(int(seed))
    offset = float(rng.uniform(-8.0, 8.0))
    p0_off = p0 + offset * n_hat
    yaw0 = float(np.arctan2(t0_hat[1], t0_hat[0]) + np.deg2rad(rng.uniform(-15.0, 15.0)))
    V0, sink0 = polar.interpolate(case.brake_base)
    v0 = np.array([V0 * np.cos(yaw0), V0 * np.sin(yaw0), sink0], dtype=float)
    state = State(
        p_I=np.array([p0_off[0], p0_off[1], -alt0], dtype=float),
        v_I=v0,
        q_IB=yaw_only_quat_wxyz(yaw0),
        w_B=np.zeros(3),
        t=0.0,
    )
    sim = OfflineSimulator(dynamics=dynamics, state=state, wind=wind)

    lateral_cfg = LateralControlConfig()

    # Logs
    cross_track: list[float] = []
    along_track: list[float] = []
    altitude_err: list[float] = []
    heading_err: list[float] = []
    controls: list[Control] = []

    for k in range(n_steps):
        st = sim.get_state()
        s_proj, ct = _project_point_to_polyline(case.path_xy, s_cum, st.position_xy)
        s_look = float(np.clip(s_proj + L1, 0.0, path_len))
        lookahead, _ = _interp_on_polyline(case.path_xy, s_cum, s_look)

        t = float(k * dt)
        b = _brake_at(t, duration, case)
        u = track_point_control(st, lookahead, brake_sym=b, cfg=lateral_cfg)

        # Reference values at this step
        s_ref_k = float(s_ref_hist[k])
        alt_ref_k = float(alt_ref_hist[k])
        heading_ref_k = float(heading_ref_hist[min(k, len(heading_ref_hist) - 1)]) if heading_ref_hist else 0.0

        cross_track.append(float(ct))
        along_track.append(float(s_proj - s_ref_k))
        altitude_err.append(float(st.altitude - alt_ref_k))
        heading_err.append(float(np.rad2deg(wrap_pi(heading_rad(st) - heading_ref_k))))

        controls.append(u)
        sim.step(u, dt=dt)

    # Metrics
    ct_arr = np.asarray(cross_track, dtype=float)
    at_arr = np.asarray(along_track, dtype=float)
    alt_arr = np.asarray(altitude_err, dtype=float)
    hdg_arr = np.asarray(heading_err, dtype=float)
    u_arr = np.asarray([[c.delta_L, c.delta_R] for c in controls], dtype=float) if controls else np.zeros((0, 2))
    du = np.diff(u_arr, axis=0) / dt if u_arr.shape[0] >= 2 else np.zeros((0, 2))
    u_dot_norm = np.linalg.norm(du, axis=1) if du.size else np.zeros((0,), dtype=float)

    # Simple response metric: time to settle heading error within 10deg for >=0.5s.
    heading_settle_time_s = None
    if hdg_arr.size:
        hold = max(int(np.ceil(0.5 / dt)), 1)
        hdg_abs = np.abs(hdg_arr)
        for i in range(max(int(hdg_abs.size - hold + 1), 0)):
            if bool(np.all(hdg_abs[i : i + hold] <= 10.0)):
                heading_settle_time_s = float(i * dt)
                break

    metrics = {
        "duration_s": float(duration),
        "path_length_m": float(path_len),
        "wind_speed_mps": float(wind_speed),
        "wind_direction_deg": float(wind_dir_deg),
        "cross_track_rmse_m": float(np.sqrt(np.mean(ct_arr**2))) if ct_arr.size else 0.0,
        "cross_track_max_m": float(np.max(np.abs(ct_arr))) if ct_arr.size else 0.0,
        "along_track_rmse_m": float(np.sqrt(np.mean(at_arr**2))) if at_arr.size else 0.0,
        "along_track_max_m": float(np.max(np.abs(at_arr))) if at_arr.size else 0.0,
        "altitude_rmse_m": float(np.sqrt(np.mean(alt_arr**2))) if alt_arr.size else 0.0,
        "altitude_max_m": float(np.max(np.abs(alt_arr))) if alt_arr.size else 0.0,
        "heading_rmse_deg": float(np.sqrt(np.mean(hdg_arr**2))) if hdg_arr.size else 0.0,
        "heading_max_deg": float(np.max(np.abs(hdg_arr))) if hdg_arr.size else 0.0,
        "heading_settle_time_s": heading_settle_time_s,
        "control_rate_std": float(np.std(u_dot_norm)) if u_dot_norm.size else 0.0,
        "control_rate_p95": float(np.percentile(u_dot_norm, 95)) if u_dot_norm.size else 0.0,
        "dynamics_mode": dyn_tag,
    }

    out: dict = {
        "case": case.name,
        "init_offset_m": float(offset),
        "metrics": metrics,
    }
    if record_history:
        out["history"] = {
            "cross_track_m": cross_track,
            "along_track_m": along_track,
            "altitude_error_m": altitude_err,
            "heading_error_deg": heading_err,
            "s_ref_m": s_ref_hist[:-1],
            "alt_ref_m": alt_ref_hist[:-1],
            "controls": [{"delta_L": c.delta_L, "delta_R": c.delta_R} for c in controls],
        }
    return out


def _write_html(path: Path, summary: dict, results: list[dict]) -> None:
    payload = {"summary": summary, "runs": results}
    cross_track = [float(r["metrics"]["cross_track_rmse_m"]) for r in results]
    heading_rmse = [float(r["metrics"]["heading_rmse_deg"]) for r in results]
    altitude_rmse = [float(r["metrics"]["altitude_rmse_m"]) for r in results]

    case_names = sorted(summary.get("cases", {}).keys())
    summary_rows = [
        ("n_runs", summary.get("n_runs")),
        ("n_cases", len(case_names)),
        ("cases", ", ".join(case_names)),
    ]
    charts = [
        {"title": "Cross-track RMSE (m)", "svg": histogram_svg(cross_track, bins=20, x_label="m")},
        {"title": "Heading RMSE (deg)", "svg": histogram_svg(heading_rmse, bins=20, x_label="deg")},
        {"title": "Altitude RMSE (m)", "svg": histogram_svg(altitude_rmse, bins=20, x_label="m")},
    ]
    # Example time series from first run (if recorded)
    if results:
        first = results[0]
        hist = first.get("history", {}) if isinstance(first, dict) else {}
        if hist:
            s_ref = hist.get("s_ref_m", [])
            cross = hist.get("cross_track_m", [])
            alt_err = hist.get("altitude_error_m", [])
            if s_ref and cross:
                charts.append({"title": "Cross-track vs Along-track (example)", "svg": line_svg(s_ref, cross, x_label="s (m)", y_label="cross-track (m)")})
            if s_ref and alt_err:
                charts.append({"title": "Altitude Error vs Along-track (example)", "svg": line_svg(s_ref, alt_err, x_label="s (m)", y_label="altitude err (m)")})
            controls = hist.get("controls", [])
            if controls:
                u_t = list(range(len(controls)))
                u_l = [float(c.get("delta_L", 0.0)) for c in controls]
                u_r = [float(c.get("delta_R", 0.0)) for c in controls]
                charts.append(
                    {
                        "title": "Control Time Series (example)",
                        "svg": multi_line_svg(
                            [
                                {"label": "delta_L", "x": u_t, "y": u_l, "color": "#59A14F"},
                                {"label": "delta_R", "x": u_t, "y": u_r, "color": "#E15759"},
                            ],
                            x_label="step",
                            y_label="brake",
                        ),
                    }
                )
    html = render_report(
        title="parafoil_planner_v3 - Controller Tracking Report",
        summary_rows=summary_rows,
        charts=charts,
        payload=payload,
        subtitle="Tracking performance across standard path cases.",
    )
    path.write_text(html)


def main() -> None:
    parser = argparse.ArgumentParser(description="Controller tracking verification (OFFLINE).")
    parser.add_argument("--runs", type=int, default=5, help="Runs per case (random initial offsets).")
    parser.add_argument("--parallel", type=int, default=1, help="Number of worker processes (OFFLINE).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--L1", type=float, default=20.0)
    parser.add_argument("--dynamics-mode", type=str, default="simplified", help="simplified|6dof (OFFLINE sim)")
    parser.add_argument("--wind-speed", type=float, default=0.0)
    parser.add_argument("--wind-direction", type=float, default=0.0)
    parser.add_argument("--case", type=str, default="all", help="all|straight|s_turn|turn|flare_straight")
    parser.add_argument("--no-history", action="store_true")
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    cases = _build_cases()
    if str(args.case).strip().lower() != "all":
        wanted = str(args.case).strip().lower()
        cases = [c for c in cases if c.name.lower() == wanted]
        if not cases:
            raise SystemExit(f"Unknown case: {args.case}")

    rng = np.random.default_rng(int(args.seed))
    tasks: list[tuple] = []
    for case in cases:
        for i in range(int(args.runs)):
            case_seed = int(rng.integers(0, 2**32 - 1))
            tasks.append(
                (
                    {
                        "name": case.name,
                        "path_xy": case.path_xy.tolist(),
                        "brake_profile": case.brake_profile,
                        "brake_base": float(case.brake_base),
                        "flare_duration_s": float(case.flare_duration_s),
                    },
                    str(args.dynamics_mode),
                    float(args.dt),
                    float(args.L1),
                    float(args.wind_speed),
                    float(args.wind_direction),
                    case_seed,
                    (not bool(args.no_history)),
                )
            )

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

    # Aggregate per-case
    grouped: Dict[str, List[dict]] = {}
    for r in results:
        grouped.setdefault(str(r["case"]), []).append(r)

    summary_cases: Dict[str, dict] = {}
    for name, items in grouped.items():
        ct = [float(x["metrics"]["cross_track_rmse_m"]) for x in items]
        at = [float(x["metrics"]["along_track_rmse_m"]) for x in items]
        alt = [float(x["metrics"]["altitude_rmse_m"]) for x in items]
        hdg = [float(x["metrics"]["heading_rmse_deg"]) for x in items]
        settle = [x["metrics"].get("heading_settle_time_s") for x in items]
        settle_ok = [float(v) for v in settle if isinstance(v, (int, float))]
        jitter = [float(x["metrics"]["control_rate_std"]) for x in items]
        summary_cases[name] = {
            "n_runs": int(len(items)),
            "goal_pass_rates": {
                "cross_track_rmse_lt_3m": float(np.mean([1.0 if v < 3.0 else 0.0 for v in ct])) if ct else 0.0,
                "along_track_rmse_lt_5m": float(np.mean([1.0 if v < 5.0 else 0.0 for v in at])) if at else 0.0,
                "altitude_rmse_lt_2m": float(np.mean([1.0 if v < 2.0 else 0.0 for v in alt])) if alt else 0.0,
                "heading_rmse_lt_10deg": float(np.mean([1.0 if v < 10.0 else 0.0 for v in hdg])) if hdg else 0.0,
                "heading_settle_lt_2s": float(np.mean([1.0 if v < 2.0 else 0.0 for v in settle_ok])) if settle_ok else 0.0,
                "heading_settle_available": float(len(settle_ok) / max(len(items), 1)),
            },
            "cross_track_rmse_m": _stats(ct),
            "along_track_rmse_m": _stats(at),
            "altitude_rmse_m": _stats(alt),
            "heading_rmse_deg": _stats(hdg),
            "heading_settle_time_s": _stats(settle_ok),
            "control_rate_std": _stats(jitter),
        }

    summary = {"n_runs": int(len(results)), "cases": summary_cases}
    out_obj = {"summary": summary, "runs": results}

    if args.output:
        out = Path(args.output)
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
