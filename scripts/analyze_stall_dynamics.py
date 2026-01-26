#!/usr/bin/env python3
"""
Analyze stall dynamics: verify that horizontal speed decreases
before vertical speed increases during stall onset.
"""

import numpy as np
import sys
sys.path.insert(0, '/home/aims/parafoil_ws/src/parafoil_dynamics')

from parafoil_dynamics.state import State, ControlCmd
from parafoil_dynamics.params import Params
from parafoil_dynamics.dynamics import dynamics
from parafoil_dynamics.math3d import quat_from_euler, quat_to_rotmat, compute_aero_angles
from parafoil_dynamics.integrators import integrate_with_substeps, IntegratorType


def simulate_stall_entry():
    """Simulate transition into stall with gradual brake application."""
    params = Params()

    # Start at steady glide (50% brake - just before stall)
    p_I = np.array([0.0, 0.0, -500.0])
    v_I = np.array([4.0, 0.0, 1.0])
    q_IB = quat_from_euler(0, 0, 0)

    # Start with 50% brake
    initial_brake = 0.50
    state = State(
        p_I=p_I, v_I=v_I, q_IB=q_IB,
        w_B=np.zeros(3),
        delta=np.array([initial_brake, initial_brake]),
        t=0.0
    )

    dt = 0.02
    dt_max = 0.005

    # First: stabilize at 50% brake for 10 seconds
    print("Stabilizing at 50% brake...")
    cmd = ControlCmd(delta_cmd=np.array([initial_brake, initial_brake]))
    for _ in range(500):
        state = integrate_with_substeps(
            dynamics, state, cmd, params, dt, dt_max,
            method=IntegratorType.RK4
        )

    # Record baseline
    baseline_h_speed = np.sqrt(state.v_I[0]**2 + state.v_I[1]**2)
    baseline_sink = state.v_I[2]

    print(f"\nBaseline at 50% brake:")
    print(f"  Horizontal speed: {baseline_h_speed:.2f} m/s")
    print(f"  Sink rate: {baseline_sink:.2f} m/s")

    # Now ramp up to 100% brake (inducing stall)
    print("\n" + "=" * 60)
    print("Ramping brake to 100% to induce stall")
    print("=" * 60)

    # Record history
    history = []
    target_brake = 1.0
    cmd = ControlCmd(delta_cmd=np.array([target_brake, target_brake]))

    # Simulate 5 seconds of stall entry
    for step in range(250):
        state = integrate_with_substeps(
            dynamics, state, cmd, params, dt, dt_max,
            method=IntegratorType.RK4
        )

        C_IB = quat_to_rotmat(state.q_IB)
        v_B = C_IB.T @ state.v_I
        V, alpha, beta = compute_aero_angles(v_B)

        h_speed = np.sqrt(state.v_I[0]**2 + state.v_I[1]**2)
        sink = state.v_I[2]

        # Check stall condition
        delta_s = (state.delta[0] + state.delta[1]) / 2.0
        alpha_stall = params.alpha_stall - params.alpha_stall_brake * delta_s
        is_stalled = alpha > alpha_stall

        history.append({
            't': state.t,
            'h_speed': h_speed,
            'sink': sink,
            'alpha': alpha,
            'alpha_stall': alpha_stall,
            'is_stalled': is_stalled,
            'brake': delta_s,
            'V': V
        })

    # Analyze results
    print(f"\n{'Time':>6} {'Brake':>6} {'H-Spd':>8} {'Sink':>8} {'Alpha':>8} {'Stall?':>8}")
    print("-" * 54)

    # Find key transition points
    first_stall_idx = None
    h_speed_start_decrease = None
    sink_start_increase = None

    prev_h_speed = baseline_h_speed
    prev_sink = baseline_sink

    for i, h in enumerate(history):
        # Print every 10 steps
        if i % 10 == 0:
            stall_str = "STALL" if h['is_stalled'] else ""
            print(f"{h['t']:>6.2f} {h['brake']*100:>5.0f}% {h['h_speed']:>8.2f} "
                  f"{h['sink']:>8.2f} {np.degrees(h['alpha']):>7.1f}° {stall_str:>8}")

        # Track transitions
        if first_stall_idx is None and h['is_stalled']:
            first_stall_idx = i

        if h_speed_start_decrease is None and h['h_speed'] < baseline_h_speed - 0.1:
            h_speed_start_decrease = i

        if sink_start_increase is None and h['sink'] > baseline_sink + 0.1:
            sink_start_increase = i

    # Summary
    print("\n" + "=" * 60)
    print("STALL DYNAMICS ANALYSIS")
    print("=" * 60)

    if first_stall_idx is not None:
        t_stall = history[first_stall_idx]['t']
        print(f"\nStall onset at t = {t_stall:.2f}s")

    if h_speed_start_decrease is not None:
        t_h = history[h_speed_start_decrease]['t']
        print(f"Horizontal speed decrease at t = {t_h:.2f}s")

    if sink_start_increase is not None:
        t_s = history[sink_start_increase]['t']
        print(f"Sink rate increase at t = {t_s:.2f}s")

    if h_speed_start_decrease is not None and sink_start_increase is not None:
        delta_t = history[sink_start_increase]['t'] - history[h_speed_start_decrease]['t']
        if delta_t > 0:
            print(f"\n✓ CORRECT: Horizontal speed decreases {delta_t:.2f}s BEFORE sink rate increases")
        elif delta_t < 0:
            print(f"\n✗ INCORRECT: Sink rate increases {-delta_t:.2f}s BEFORE horizontal speed decreases")
        else:
            print(f"\n~ SIMULTANEOUS: Both change at the same time")

    # Final state comparison
    print(f"\nFinal state (after 5 seconds):")
    final = history[-1]
    print(f"  Horizontal speed: {final['h_speed']:.2f} m/s (was {baseline_h_speed:.2f})")
    print(f"  Sink rate: {final['sink']:.2f} m/s (was {baseline_sink:.2f})")
    print(f"  Speed change: {final['h_speed'] - baseline_h_speed:.2f} m/s")
    print(f"  Sink change: {final['sink'] - baseline_sink:.2f} m/s")


if __name__ == '__main__':
    simulate_stall_entry()
