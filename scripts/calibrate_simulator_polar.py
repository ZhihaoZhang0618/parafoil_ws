#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR / "src" / "parafoil_dynamics"))

from parafoil_dynamics.dynamics import compute_turn_rate, dynamics  # noqa: E402
from parafoil_dynamics.integrators import rk4_step  # noqa: E402
from parafoil_dynamics.math3d import quat_from_euler  # noqa: E402
from parafoil_dynamics.params import Params  # noqa: E402
from parafoil_dynamics.state import ControlCmd, State  # noqa: E402


def _parse_floats_csv(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _linspace_values(spec: str) -> list[float]:
    """
    Parse a value spec:
      - "0,0.25,0.5" -> explicit list
      - "start:step:stop" -> range inclusive-ish (stop included if aligns)
    """
    spec = spec.strip()
    if ":" in spec:
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError(f"Bad range spec: {spec!r}, expected start:step:stop")
        start, step, stop = (float(p) for p in parts)
        if step <= 0:
            raise ValueError("step must be > 0")
        n = int(math.floor((stop - start) / step + 1e-9)) + 1
        vals = [start + i * step for i in range(max(n, 0))]
        if vals and vals[-1] < stop - 1e-9:
            vals.append(stop)
        return [float(v) for v in vals if v <= stop + 1e-9]
    return _parse_floats_csv(spec)


def load_params_from_yaml(path: Path) -> tuple[Params, dict]:
    data = yaml.safe_load(path.read_text())
    ros_params = data.get("parafoil_simulator", {}).get("ros__parameters", {})

    kwargs: dict[str, object] = {}
    for key in [
        "rho",
        "g",
        "m",
        "S",
        "b",
        "c",
        "S_pd",
        "c_D_pd",
        "c_L0",
        "c_La",
        "c_Lds",
        "c_D0",
        "c_Da2",
        "c_Dds",
        "alpha_stall",
        "alpha_stall_brake",
        "alpha_stall_width",
        "c_D_stall",
        "c_Yb",
        "c_lp",
        "c_lda",
        "c_m0",
        "c_ma",
        "c_mq",
        "c_nr",
        "c_nda",
        "c_nb",
        "c_n_weath",
        "tau_act",
        "eps",
        "V_min",
        "pendulum_arm",
    ]:
        if key in ros_params:
            kwargs[key if key != "V_min" else "V_min"] = ros_params[key]

    if "I_B_diag" in ros_params:
        kwargs["I_B"] = np.diag(ros_params["I_B_diag"])
    if "r_pd_B" in ros_params:
        kwargs["r_pd_B"] = np.array(ros_params["r_pd_B"], dtype=float)
    if "r_canopy_B" in ros_params:
        kwargs["r_canopy_B"] = np.array(ros_params["r_canopy_B"], dtype=float)

    return (Params(**kwargs), ros_params)


def initial_state(ros_params: dict, altitude_m: float) -> State:
    pos = np.array(ros_params.get("initial_position", [0.0, 0.0, -500.0]), dtype=float)
    vel = np.array(ros_params.get("initial_velocity", [4.0, 0.0, 1.0]), dtype=float)
    euler = np.array(ros_params.get("initial_euler", [0.0, 0.0, 0.0]), dtype=float)

    pos[2] = -float(altitude_m)
    q_IB = quat_from_euler(float(euler[0]), float(euler[1]), float(euler[2]))
    return State(
        p_I=pos,
        v_I=vel,
        q_IB=q_IB,
        w_B=np.zeros(3),
        delta=np.zeros(2),
        t=0.0,
    )


def simulate_constant_brake(
    params: Params,
    ros_params: dict,
    brake: float,
    dt: float,
    sim_time: float,
    avg_window: float,
    altitude_m: float,
) -> tuple[float, float, float]:
    s = initial_state(ros_params, altitude_m=altitude_m)
    cmd = ControlCmd.from_left_right(brake, brake)
    wind = np.zeros(3)

    times: list[float] = []
    hs: list[float] = []
    vs: list[float] = []
    n_steps = int(max(sim_time / max(dt, 1e-6), 1))
    for _ in range(n_steps):
        s = rk4_step(dynamics, s, cmd, params, dt, wind)
        times.append(float(s.t))
        v = s.v_I
        hs.append(float(np.hypot(v[0], v[1])))
        vs.append(float(v[2]))
        if s.p_I[2] > 0.0:
            break

    t_end = times[-1] if times else 0.0
    t0 = max(t_end - float(avg_window), 0.0)
    idx = [i for i, t in enumerate(times) if t >= t0]
    if not idx:
        return (0.0, 0.0, 0.0)
    Vh = float(np.mean([hs[i] for i in idx]))
    sink = float(np.mean([vs[i] for i in idx]))
    return (Vh, sink, Vh / max(sink, 1e-6))


def estimate_turn_rate_gain(
    params: Params,
    ros_params: dict,
    mean_brake: float,
    deltas: Iterable[float],
    dt: float,
    trim_time: float,
    sample_start: float,
    sample_end: float,
    altitude_m: float,
) -> tuple[float, list[tuple[float, float]]]:
    """Return (gain_mag, points) where points are (delta_a, yaw_rate_mean)."""
    wind = np.zeros(3)

    # Trim at mean brake
    s = initial_state(ros_params, altitude_m=altitude_m)
    cmd_trim = ControlCmd.from_left_right(mean_brake, mean_brake)
    for _ in range(int(max(trim_time / max(dt, 1e-6), 1))):
        s = rk4_step(dynamics, s, cmd_trim, params, dt, wind)

    points: list[tuple[float, float]] = []
    for delta in deltas:
        left = float(np.clip(mean_brake + 0.5 * delta, 0.0, 1.0))
        right = float(np.clip(mean_brake - 0.5 * delta, 0.0, 1.0))
        cmd = ControlCmd.from_left_right(left, right)

        s2 = State(
            p_I=s.p_I.copy(),
            v_I=s.v_I.copy(),
            q_IB=s.q_IB.copy(),
            w_B=s.w_B.copy(),
            delta=s.delta.copy(),
            t=0.0,
        )

        yaw_rates: list[float] = []
        n_steps = int(max(sample_end / max(dt, 1e-6), 1))
        for _ in range(n_steps):
            s2 = rk4_step(dynamics, s2, cmd, params, dt, wind)
            if sample_start <= s2.t <= sample_end:
                yaw_rates.append(float(compute_turn_rate(s2)))

        yaw_mean = float(np.mean(yaw_rates)) if yaw_rates else 0.0
        points.append((float(delta), yaw_mean))

    xs = np.array([d for d, _ in points if d > 1e-9], dtype=float)
    ys = np.array([y for d, y in points if d > 1e-9], dtype=float)
    gain = float(np.dot(xs, ys) / max(np.dot(xs, xs), 1e-9)) if len(xs) else 0.0
    return (abs(gain), points)


def _default_targets() -> dict[float, dict[str, float]]:
    # Targets from user message (flight-log summary). Units: m/s.
    return {
        0.0: {"Vh": 4.5, "sink": 0.90},
        0.25: {"sink": 1.152},
        0.5: {"Vh": 3.47, "sink": 1.30},
        0.9: {"sink": 1.32, "Vg": 2.81},  # Vg from GPS ground speed; wind may exist
    }


def score_against_targets(sim: dict[float, tuple[float, float, float]], targets: dict[float, dict[str, float]]) -> float:
    w = 0.0
    for b, t in targets.items():
        if b not in sim:
            continue
        Vh, sink, _ = sim[b]
        if "Vh" in t:
            w += (Vh - float(t["Vh"])) ** 2
        if "sink" in t:
            w += (sink - float(t["sink"])) ** 2
        # If only Vg is given (GPS ground speed), treat it as a soft hint.
        if "Vg" in t and "Vh" not in t:
            w += 0.25 * (Vh - float(t["Vg"])) ** 2
    return float(w)


def write_params_yaml(template_path: Path, out_path: Path, updates: dict[str, object]) -> None:
    data = yaml.safe_load(template_path.read_text())
    ros_params = data.setdefault("parafoil_simulator", {}).setdefault("ros__parameters", {})
    ros_params.update(updates)
    out_path.write_text(yaml.safe_dump(data, sort_keys=False))


def main() -> int:
    ap = argparse.ArgumentParser(description="Calibrate parafoil_simulator_ros params against flight polar targets.")
    ap.add_argument(
        "--params-yaml",
        default=str(ROOT_DIR / "src" / "parafoil_simulator_ros" / "config" / "params.yaml"),
        help="Simulator params YAML (parafoil_simulator/ros__parameters).",
    )
    ap.add_argument("--altitude", type=float, default=800.0)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--sim-time", type=float, default=260.0)
    ap.add_argument("--avg-window", type=float, default=20.0)
    ap.add_argument("--brakes", default="0:0.1:1.0", help='E.g. "0,0.25,0.5,0.9" or "0:0.1:1.0"')

    ap.add_argument("--show-turn-gain", action="store_true")
    ap.add_argument("--turn-mean", type=float, default=0.25)
    ap.add_argument("--turn-deltas", default="0.05,0.1,0.2,0.3,0.4,0.5")
    ap.add_argument("--turn-trim-time", type=float, default=30.0)
    ap.add_argument("--turn-sample-start", type=float, default=8.0)
    ap.add_argument("--turn-sample-end", type=float, default=18.0)

    ap.add_argument("--targets", action="store_true", help="Score against built-in flight targets.")
    ap.add_argument("--tune", action="store_true", help="Grid-search (m,c_Lds,c_Dds) around current params.")
    ap.add_argument("--m", default="2.30:0.05:2.60")
    ap.add_argument("--c_Lds", default="0.20:0.05:0.35")
    ap.add_argument("--c_Dds", default="0.60:0.05:0.85")
    ap.add_argument("--stall-mild", action="store_true", help="Also set a mild stall model while tuning.")
    ap.add_argument("--write-yaml", default="", help="Write best tuned params to this YAML file.")

    args = ap.parse_args()

    params_path = Path(args.params_yaml)
    params, ros_params = load_params_from_yaml(params_path)

    brakes = [float(np.clip(b, 0.0, 1.0)) for b in _linspace_values(args.brakes)]
    brakes = sorted(set(brakes))

    targets = _default_targets() if args.targets else {}

    def evaluate(p: Params) -> dict[float, tuple[float, float, float]]:
        out: dict[float, tuple[float, float, float]] = {}
        for b in brakes:
            out[b] = simulate_constant_brake(
                p,
                ros_params,
                brake=b,
                dt=float(args.dt),
                sim_time=float(args.sim_time),
                avg_window=float(args.avg_window),
                altitude_m=float(args.altitude),
            )
        return out

    best_params = params
    best_metrics = evaluate(best_params)
    best_score = score_against_targets(best_metrics, targets) if targets else 0.0

    if args.tune:
        m_vals = _linspace_values(args.m)
        cl_vals = _linspace_values(args.c_Lds)
        cd_vals = _linspace_values(args.c_Dds)

        stall_updates = {}
        if args.stall_mild:
            stall_updates = {
                "alpha_stall": 0.35,
                "alpha_stall_brake": 0.02,
                "alpha_stall_width": 0.15,
                "c_D_stall": 0.15,
            }

        for m in m_vals:
            for c_Lds in cl_vals:
                for c_Dds in cd_vals:
                    cand = replace(best_params, m=float(m), c_Lds=float(c_Lds), c_Dds=float(c_Dds), **stall_updates)
                    metrics = evaluate(cand)
                    sc = score_against_targets(metrics, targets) if targets else 0.0
                    if targets and sc < best_score:
                        best_score = sc
                        best_params = cand
                        best_metrics = metrics

    print(f"Params: {params_path}")
    if targets:
        print(f"Score (lower is better): {best_score:.6f}")
    print("")
    print("brake   Vh(m/s)  sink(m/s)  L/D")
    for b in brakes:
        Vh, sink, ld = best_metrics[b]
        print(f"{b:4.2f}   {Vh:7.3f}   {sink:8.3f}   {ld:4.2f}")

    if targets:
        print("")
        print("Targets (flight summary) deltas:")
        for b, t in sorted(targets.items()):
            if b not in best_metrics:
                continue
            Vh, sink, _ = best_metrics[b]
            parts = []
            if "Vh" in t:
                parts.append(f"dVh={Vh-float(t['Vh']):+.3f}")
            if "sink" in t:
                parts.append(f"dsink={sink-float(t['sink']):+.3f}")
            if parts:
                print(f"  brake={b:.2f}: " + " ".join(parts))

    if args.show_turn_gain:
        deltas = _parse_floats_csv(args.turn_deltas)
        gain, pts = estimate_turn_rate_gain(
            best_params,
            ros_params,
            mean_brake=float(args.turn_mean),
            deltas=deltas,
            dt=float(args.dt),
            trim_time=float(args.turn_trim_time),
            sample_start=float(args.turn_sample_start),
            sample_end=float(args.turn_sample_end),
            altitude_m=float(args.altitude),
        )
        print("")
        print(f"Estimated |yaw_rate| gain: ~{gain:.3f} rad/s per delta_a (delta_a = left-right)")
        for d, y in pts:
            print(f"  delta_a={d:.2f}  yaw_rate_mean={y:.3f} rad/s ({math.degrees(y):.1f} deg/s)")

    if args.write_yaml:
        out_path = Path(args.write_yaml)
        updates = {
            "m": float(best_params.m),
            "c_Lds": float(best_params.c_Lds),
            "c_Dds": float(best_params.c_Dds),
            "alpha_stall": float(best_params.alpha_stall),
            "alpha_stall_brake": float(best_params.alpha_stall_brake),
            "alpha_stall_width": float(best_params.alpha_stall_width),
            "c_D_stall": float(best_params.c_D_stall),
        }
        write_params_yaml(params_path, out_path, updates)
        print("")
        print(f"Wrote tuned YAML: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

