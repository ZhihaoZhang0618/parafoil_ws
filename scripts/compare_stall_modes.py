#!/usr/bin/env python3
"""
Compare symmetric (双边) vs asymmetric (单边) stall behavior.

Symmetric stall: both brakes pulled equally -> straight descent
Asymmetric stall: one brake pulled -> spin/spiral
"""

import numpy as np
import sys
sys.path.insert(0, '/home/aims/parafoil_ws/src/parafoil_dynamics')

from parafoil_dynamics.state import State, ControlCmd
from parafoil_dynamics.params import Params
from parafoil_dynamics.dynamics import dynamics
from parafoil_dynamics.math3d import quat_from_euler, quat_to_euler, quat_to_rotmat, compute_aero_angles
from parafoil_dynamics.integrators import integrate_with_substeps, IntegratorType


def simulate_stall(left_brake: float, right_brake: float, label: str, duration: float = 10.0):
    """Simulate stall with given brake configuration."""
    params = Params()

    # Start at steady glide
    p_I = np.array([0.0, 0.0, -500.0])
    v_I = np.array([4.0, 0.0, 1.0])
    q_IB = quat_from_euler(0, 0, 0)

    state = State(
        p_I=p_I, v_I=v_I, q_IB=q_IB,
        w_B=np.zeros(3),
        delta=np.array([0.0, 0.0]),
        t=0.0
    )

    dt = 0.02
    dt_max = 0.005

    # First: stabilize for 5 seconds
    cmd = ControlCmd(delta_cmd=np.array([0.0, 0.0]))
    for _ in range(250):
        state = integrate_with_substeps(
            dynamics, state, cmd, params, dt, dt_max,
            method=IntegratorType.RK4
        )

    # Record initial position and heading
    initial_pos = state.p_I.copy()
    _, _, initial_yaw = quat_to_euler(state.q_IB)

    # Apply brake command
    cmd = ControlCmd(delta_cmd=np.array([left_brake, right_brake]))

    # Record history
    history = []

    n_steps = int(duration / dt)
    for step in range(n_steps):
        state = integrate_with_substeps(
            dynamics, state, cmd, params, dt, dt_max,
            method=IntegratorType.RK4
        )

        roll, pitch, yaw = quat_to_euler(state.q_IB)
        C_IB = quat_to_rotmat(state.q_IB)
        v_B = C_IB.T @ state.v_I
        V, alpha, beta = compute_aero_angles(v_B)

        h_speed = np.sqrt(state.v_I[0]**2 + state.v_I[1]**2)
        sink = state.v_I[2]

        # Check stall
        delta_s = (state.delta[0] + state.delta[1]) / 2.0
        alpha_stall = params.alpha_stall - params.alpha_stall_brake * delta_s

        history.append({
            't': state.t,
            'h_speed': h_speed,
            'sink': sink,
            'roll': np.degrees(roll),
            'pitch': np.degrees(pitch),
            'yaw': np.degrees(yaw),
            'yaw_rate': np.degrees(state.w_B[2]),
            'alpha': np.degrees(alpha),
            'is_stalled': alpha > alpha_stall,
            'pos_N': state.p_I[0],
            'pos_E': state.p_I[1],
            'alt': -state.p_I[2]
        })

    return history


def main():
    print("=" * 70)
    print("STALL MODE COMPARISON: 双边失速 vs 单边失速")
    print("=" * 70)

    # Test cases
    cases = [
        (1.0, 1.0, "双边失速 (100% both)"),
        (1.0, 0.0, "单边失速 - 左刹车 (100% left, 0% right)"),
        (0.0, 1.0, "单边失速 - 右刹车 (0% left, 100% right)"),
        (1.0, 0.5, "非对称 (100% left, 50% right)"),
    ]

    all_results = {}

    for left, right, label in cases:
        print(f"\n{'='*60}")
        print(f"Testing: {label}")
        print(f"{'='*60}")

        history = simulate_stall(left, right, label, duration=8.0)
        all_results[label] = history

        # Print summary at key time points
        print(f"\n{'Time':>6} {'H-Spd':>7} {'Sink':>7} {'Roll':>7} {'Yaw':>8} {'YawRate':>8} {'Stall':>6}")
        print("-" * 58)

        for i, h in enumerate(history):
            if i % 50 == 0 or i == len(history) - 1:  # Every 1 second
                stall_str = "Y" if h['is_stalled'] else "N"
                print(f"{h['t']:>6.1f} {h['h_speed']:>7.2f} {h['sink']:>7.2f} "
                      f"{h['roll']:>7.1f} {h['yaw']:>8.1f} {h['yaw_rate']:>8.1f} {stall_str:>6}")

        # Final summary
        final = history[-1]
        initial = history[0]

        total_yaw_change = final['yaw'] - initial['yaw']
        # Handle wrap-around
        if abs(total_yaw_change) > 180:
            if total_yaw_change > 0:
                total_yaw_change -= 360
            else:
                total_yaw_change += 360

        alt_loss = initial['alt'] - final['alt']

        print(f"\n  Summary:")
        print(f"    Altitude lost: {alt_loss:.1f} m")
        print(f"    Total yaw change: {total_yaw_change:.1f}°")
        print(f"    Final roll: {final['roll']:.1f}°")
        print(f"    Average sink rate: {np.mean([h['sink'] for h in history[-100:]]):.2f} m/s")

    # Comparison summary
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    print(f"\n{'Mode':<35} {'Yaw Change':>12} {'Sink Rate':>12} {'Roll':>10}")
    print("-" * 70)

    for label, history in all_results.items():
        final = history[-1]
        initial = history[0]
        yaw_change = final['yaw'] - initial['yaw']
        if abs(yaw_change) > 180:
            yaw_change = yaw_change - 360 if yaw_change > 0 else yaw_change + 360
        avg_sink = np.mean([h['sink'] for h in history[-100:]])

        print(f"{label:<35} {yaw_change:>10.1f}° {avg_sink:>10.2f} m/s {final['roll']:>8.1f}°")

    print("\n" + "-" * 70)
    print("Analysis:")
    print("-" * 70)

    # Compare symmetric vs asymmetric
    sym = all_results["双边失速 (100% both)"]
    asym_left = all_results["单边失速 - 左刹车 (100% left, 0% right)"]
    asym_right = all_results["单边失速 - 右刹车 (0% left, 100% right)"]

    sym_yaw = sym[-1]['yaw'] - sym[0]['yaw']
    left_yaw = asym_left[-1]['yaw'] - asym_left[0]['yaw']
    right_yaw = asym_right[-1]['yaw'] - asym_right[0]['yaw']

    print(f"\n1. 双边失速: Yaw change = {sym_yaw:.1f}° (should be ~0°)")
    print(f"   -> 直线下降，无旋转")

    print(f"\n2. 单边左刹车: Yaw change = {left_yaw:.1f}°")
    if left_yaw > 10:
        print(f"   -> 向右旋转 (opposite to brake side)")
    elif left_yaw < -10:
        print(f"   -> 向左旋转 (toward brake side)")
    else:
        print(f"   -> 轻微旋转")

    print(f"\n3. 单边右刹车: Yaw change = {right_yaw:.1f}°")
    if right_yaw > 10:
        print(f"   -> 向右旋转")
    elif right_yaw < -10:
        print(f"   -> 向左旋转 (toward brake side)")
    else:
        print(f"   -> 轻微旋转")


if __name__ == '__main__':
    main()
