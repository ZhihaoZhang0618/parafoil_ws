#!/usr/bin/env python3
"""
Test weathercock effect: parafoil should naturally turn from headwind to tailwind.
"""

import numpy as np
import sys
sys.path.insert(0, '/home/aims/parafoil_ws/src/parafoil_dynamics')

from parafoil_dynamics.state import State, ControlCmd
from parafoil_dynamics.params import Params
from parafoil_dynamics.dynamics import dynamics
from parafoil_dynamics.math3d import quat_from_euler, quat_to_euler, quat_to_rotmat, compute_aero_angles
from parafoil_dynamics.integrators import integrate_with_substeps, IntegratorType


def simulate_weathercock(wind_speed: float, c_nb: float, c_Yb: float, duration: float = 60.0):
    """Simulate flight in steady wind to observe weathercock effect."""
    params = Params()
    params.c_nb = c_nb
    params.c_Yb = c_Yb

    # Start heading north (into the wind from south)
    # Wind blowing FROM south TO north = headwind
    wind_I = np.array([-wind_speed, 0.0, 0.0])  # Wind FROM south (headwind)

    p_I = np.array([0.0, 0.0, -500.0])
    v_I = np.array([4.0, 0.0, 1.0])  # Flying north
    q_IB = quat_from_euler(0, 0, 0)  # Heading north

    state = State(
        p_I=p_I, v_I=v_I, q_IB=q_IB,
        w_B=np.zeros(3),
        delta=np.array([0.0, 0.0]),
        t=0.0
    )
    cmd = ControlCmd(delta_cmd=np.array([0.0, 0.0]))

    dt = 0.02
    dt_max = 0.005
    n_steps = int(duration / dt)

    history = []

    for step in range(n_steps):
        state = integrate_with_substeps(
            dynamics, state, cmd, params, dt, dt_max,
            method=IntegratorType.RK4,
            wind_fn=lambda t: wind_I
        )

        roll, pitch, yaw = quat_to_euler(state.q_IB)

        # Compute beta
        C_IB = quat_to_rotmat(state.q_IB)
        v_rel_I = state.v_I - wind_I
        v_rel_B = C_IB.T @ v_rel_I
        V, alpha, beta = compute_aero_angles(v_rel_B)

        history.append({
            't': state.t,
            'yaw': np.degrees(yaw),
            'beta': np.degrees(beta),
            'yaw_rate': np.degrees(state.w_B[2])
        })

    return history


def main():
    print("=" * 70)
    print("WEATHERCOCK EFFECT TEST")
    print("=" * 70)
    print("\nParafoil starts heading NORTH into HEADWIND (wind from south)")
    print("Physical expectation: should turn to face DOWNWIND (south)\n")

    wind_speed = 3.0  # m/s

    # Test different c_nb and c_Yb combinations
    test_cases = [
        ("Current params", 0.15, -6.8),
        ("Higher c_nb", 0.5, -6.8),
        ("Much higher c_nb", 1.0, -6.8),
        ("Lower c_Yb", 0.15, -2.0),
        ("Combined: high c_nb, low c_Yb", 0.5, -2.0),
    ]

    for name, c_nb, c_Yb in test_cases:
        print(f"\n{'='*60}")
        print(f"Test: {name} (c_nb={c_nb}, c_Yb={c_Yb})")
        print(f"{'='*60}")

        history = simulate_weathercock(wind_speed, c_nb, c_Yb, duration=60.0)

        print(f"\n{'Time':>6} {'Yaw':>10} {'Beta':>10} {'Yaw Rate':>12}")
        print("-" * 40)

        for i, h in enumerate(history):
            if i % 250 == 0 or i == len(history) - 1:
                print(f"{h['t']:>6.1f} {h['yaw']:>9.1f}° {h['beta']:>9.2f}° {h['yaw_rate']:>10.2f}°/s")

        # Summary
        initial_yaw = history[0]['yaw']
        final_yaw = history[-1]['yaw']
        total_turn = final_yaw - initial_yaw

        # Handle wrap
        while total_turn > 180:
            total_turn -= 360
        while total_turn < -180:
            total_turn += 360

        print(f"\nTotal yaw change: {total_turn:.1f}°")

        if abs(total_turn) > 90:
            print("→ STRONG weathercock effect (turned significantly)")
        elif abs(total_turn) > 30:
            print("→ MODERATE weathercock effect")
        elif abs(total_turn) > 10:
            print("→ WEAK weathercock effect")
        else:
            print("→ NO weathercock effect (yaw stable)")


if __name__ == '__main__':
    main()
