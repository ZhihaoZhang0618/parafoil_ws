#!/usr/bin/env python3
"""
Validate tuned parameters against flight data.

Tests:
1. Steady glide at different brake ratios
2. Turn performance (yaw rate vs differential brake)
3. Stability (oscillation damping)
"""

import numpy as np
import sys
sys.path.insert(0, '/home/aims/parafoil_ws/src/parafoil_dynamics')

from parafoil_dynamics.state import State, ControlCmd
from parafoil_dynamics.params import Params
from parafoil_dynamics.dynamics import dynamics
from parafoil_dynamics.math3d import quat_from_euler, quat_to_euler, compute_aero_angles, quat_to_rotmat
from parafoil_dynamics.integrators import integrate_with_substeps, IntegratorType


def simulate_steady_glide(brake_ratio: float, duration: float = 30.0):
    """Simulate steady glide at given brake ratio."""
    params = Params()

    p_I = np.array([0.0, 0.0, -500.0])
    v_I = np.array([4.0, 0.0, 1.0])
    q_IB = quat_from_euler(0, 0, 0)

    state = State(
        p_I=p_I, v_I=v_I, q_IB=q_IB,
        w_B=np.zeros(3),
        delta=np.array([brake_ratio, brake_ratio]),
        t=0.0
    )
    cmd = ControlCmd(delta_cmd=np.array([brake_ratio, brake_ratio]))

    dt = 0.02
    dt_max = 0.005
    n_steps = int(duration / dt)

    # Record last 5 seconds for averaging
    history = []

    for step in range(n_steps):
        state = integrate_with_substeps(
            dynamics, state, cmd, params, dt, dt_max,
            method=IntegratorType.RK4
        )

        if step > n_steps - 250:  # Last 5 seconds
            C_IB = quat_to_rotmat(state.q_IB)
            v_B = C_IB.T @ state.v_I
            V, alpha, beta = compute_aero_angles(v_B)
            h_speed = np.sqrt(state.v_I[0]**2 + state.v_I[1]**2)
            sink = state.v_I[2]
            history.append({
                'V': V, 'sink': sink, 'h_speed': h_speed,
                'alpha': alpha
            })

    # Average
    avg = {k: np.mean([h[k] for h in history]) for k in history[0].keys()}
    avg['glide_ratio'] = avg['h_speed'] / avg['sink'] if avg['sink'] > 0.01 else 0
    return avg


def simulate_turn(differential_brake: float, duration: float = 15.0):
    """Simulate turn with differential brake."""
    params = Params()

    p_I = np.array([0.0, 0.0, -500.0])
    v_I = np.array([4.0, 0.0, 1.0])
    q_IB = quat_from_euler(0, 0, 0)

    # Differential brake: positive = more right brake = turn left
    left_brake = max(0, -differential_brake)
    right_brake = max(0, differential_brake)

    state = State(
        p_I=p_I, v_I=v_I, q_IB=q_IB,
        w_B=np.zeros(3),
        delta=np.array([left_brake, right_brake]),
        t=0.0
    )
    cmd = ControlCmd(delta_cmd=np.array([left_brake, right_brake]))

    dt = 0.02
    dt_max = 0.005
    n_steps = int(duration / dt)

    yaw_history = []

    for step in range(n_steps):
        state = integrate_with_substeps(
            dynamics, state, cmd, params, dt, dt_max,
            method=IntegratorType.RK4
        )

        roll, pitch, yaw = quat_to_euler(state.q_IB)
        yaw_history.append(yaw)

    # Compute average yaw rate (last 5 seconds)
    yaw_array = np.array(yaw_history[-250:])
    yaw_unwrapped = np.unwrap(yaw_array)
    yaw_rate = np.mean(np.diff(yaw_unwrapped) / dt)

    return {
        'yaw_rate_deg_s': np.degrees(yaw_rate),
        'final_yaw_deg': np.degrees(yaw_history[-1])
    }


def main():
    print("=" * 70)
    print("PARAFOIL SIMULATION VALIDATION")
    print("=" * 70)
    print("\nComparing tuned simulation against flight log 09_01_06 data")

    # === Test 1: Steady glide at different brake ratios ===
    print("\n" + "-" * 70)
    print("TEST 1: Steady Glide Performance")
    print("-" * 70)
    print(f"\n{'Brake %':<10} {'V (m/s)':<12} {'Sink (m/s)':<12} {'L/D':<10} {'Alpha (°)':<10}")
    print("-" * 54)

    # Flight data targets
    targets = {
        0.0: {'sink': 0.85, 'h_speed': 4.27, 'LD': 5.0},
        0.5: {'sink': 0.90, 'h_speed': 4.46, 'LD': 4.97},
        # 0.9: {'sink': 1.32, 'h_speed': 2.81, 'LD': 2.14},  # Less reliable data
    }

    for brake in [0.0, 0.25, 0.5, 0.75, 1.0]:
        result = simulate_steady_glide(brake)

        # Compare with target if available
        target_str = ""
        if brake in targets:
            t = targets[brake]
            target_str = f" (target: sink={t['sink']}, L/D={t['LD']:.1f})"

        print(f"{brake*100:>6.0f}%    {result['V']:>8.2f}    {result['sink']:>8.2f}    "
              f"{result['glide_ratio']:>6.2f}    {np.degrees(result['alpha']):>6.1f}°{target_str}")

    # === Test 2: Turn performance ===
    print("\n" + "-" * 70)
    print("TEST 2: Turn Performance (Differential Brake)")
    print("-" * 70)
    print("\nNote: Positive diff = right brake = turn left (negative yaw rate)")
    print(f"\n{'Diff Brake':<12} {'Yaw Rate (°/s)':<15} {'Direction':<10}")
    print("-" * 37)

    for diff in [-0.5, -0.25, 0.0, 0.25, 0.5]:
        result = simulate_turn(diff)
        direction = "Left" if result['yaw_rate_deg_s'] < -1 else ("Right" if result['yaw_rate_deg_s'] > 1 else "Straight")
        print(f"{diff:>8.2f}     {result['yaw_rate_deg_s']:>10.1f}       {direction}")

    # === Test 3: Stability check ===
    print("\n" + "-" * 70)
    print("TEST 3: Stability Check")
    print("-" * 70)

    params = Params()

    # Start with perturbed attitude
    p_I = np.array([0.0, 0.0, -500.0])
    v_I = np.array([4.0, 0.0, 1.0])
    q_IB = quat_from_euler(np.radians(15), np.radians(10), 0)  # Perturbed

    state = State(
        p_I=p_I, v_I=v_I, q_IB=q_IB,
        w_B=np.zeros(3),
        delta=np.zeros(2),
        t=0.0
    )
    cmd = ControlCmd(delta_cmd=np.zeros(2))

    dt = 0.02
    dt_max = 0.005

    print("\nPerturbed start: roll=15°, pitch=10°")
    print(f"\n{'Time (s)':<10} {'Roll (°)':<12} {'Pitch (°)':<12} {'V (m/s)':<12}")
    print("-" * 46)

    for t_sec in range(16):
        if t_sec % 2 == 0:
            roll, pitch, yaw = quat_to_euler(state.q_IB)
            C_IB = quat_to_rotmat(state.q_IB)
            v_B = C_IB.T @ state.v_I
            V, _, _ = compute_aero_angles(v_B)
            print(f"{state.t:>6.1f}     {np.degrees(roll):>8.1f}     {np.degrees(pitch):>8.1f}     {V:>8.2f}")

        # Simulate 1 second
        for _ in range(50):
            state = integrate_with_substeps(
                dynamics, state, cmd, params, dt, dt_max,
                method=IntegratorType.RK4
            )

    # Final stability assessment
    roll, pitch, yaw = quat_to_euler(state.q_IB)
    stable = abs(np.degrees(roll)) < 5 and abs(np.degrees(pitch)) < 10
    print(f"\nStability: {'STABLE ✓' if stable else 'UNSTABLE ✗'}")
    print(f"Final attitude: roll={np.degrees(roll):.1f}°, pitch={np.degrees(pitch):.1f}°")

    # === Summary ===
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    # Check zero brake performance
    zero_brake = simulate_steady_glide(0.0)
    v_err = abs(zero_brake['V'] - 4.3) / 4.3 * 100
    sink_err = abs(zero_brake['sink'] - 0.85) / 0.85 * 100
    ld_err = abs(zero_brake['glide_ratio'] - 5.0) / 5.0 * 100

    print(f"\nZero brake performance vs targets:")
    print(f"  Airspeed error: {v_err:.1f}%")
    print(f"  Sink rate error: {sink_err:.1f}%")
    print(f"  Glide ratio error: {ld_err:.1f}%")

    if v_err < 5 and sink_err < 5 and ld_err < 10:
        print("\n✓ VALIDATION PASSED - Simulation matches flight data")
    else:
        print("\n✗ VALIDATION NEEDS IMPROVEMENT")


if __name__ == '__main__':
    main()
