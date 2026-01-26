#!/usr/bin/env python3
"""
Test weathercock effect with the new c_n_weath parameter.

This parameter models the destabilizing yaw moment from crosswind
relative to the parafoil's heading, independent of sideslip.
"""

import numpy as np
import sys
sys.path.insert(0, '/home/aims/parafoil_ws/src/parafoil_dynamics')

from parafoil_dynamics.state import State, ControlCmd
from parafoil_dynamics.params import Params
from parafoil_dynamics.dynamics import dynamics
from parafoil_dynamics.math3d import quat_from_euler, quat_to_euler, quat_to_rotmat, compute_aero_angles
from parafoil_dynamics.integrators import integrate_with_substeps, IntegratorType


def simulate_weathercock(initial_yaw_deg: float, wind_speed: float, c_n_weath: float, duration: float = 60.0):
    """Simulate flight with weathercock effect."""
    params = Params()
    params.c_n_weath = c_n_weath

    # Wind blowing FROM south TO north (headwind for northward flight)
    wind_I = np.array([-wind_speed, 0.0, 0.0])

    p_I = np.array([0.0, 0.0, -500.0])
    v_I = np.array([4.0, 0.0, 1.0])  # Flying north

    # Start with small yaw perturbation
    initial_yaw_rad = np.radians(initial_yaw_deg)
    q_IB = quat_from_euler(0, 0, initial_yaw_rad)

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

        # Compute wind in body frame
        wind_B = C_IB.T @ wind_I

        history.append({
            't': state.t,
            'yaw': np.degrees(yaw),
            'beta': np.degrees(beta),
            'wind_y_B': wind_B[1],
            'yaw_rate': np.degrees(state.w_B[2])
        })

    return history


def main():
    print("=" * 70)
    print("WEATHERCOCK EFFECT TEST (v2)")
    print("=" * 70)
    print("\nNew parameter: c_n_weath models yaw moment from crosswind")
    print("relative to HEADING (not velocity/sideslip)")
    print("\nSetup: Wind 3 m/s from south (headwind)")
    print("       Parafoil starts heading north with 5° yaw offset\n")

    wind_speed = 3.0
    initial_yaw = 5.0  # Small perturbation

    test_cases = [
        (0.0, "No weathercock (c_n_weath=0)"),
        (0.02, "Mild weathercock (c_n_weath=+0.02)"),
        (0.05, "Moderate weathercock (c_n_weath=+0.05)"),
        (0.10, "Strong weathercock (c_n_weath=+0.10)"),
    ]

    for c_n_weath, description in test_cases:
        print(f"\n{'='*60}")
        print(f"Test: {description}")
        print(f"{'='*60}")

        history = simulate_weathercock(initial_yaw, wind_speed, c_n_weath, duration=60.0)

        print(f"\n{'Time':>6} {'Yaw':>10} {'Beta':>10} {'Wind_Y_B':>12} {'Yaw Rate':>12}")
        print("-" * 52)

        for i, h in enumerate(history):
            if i % 300 == 0 or i == len(history) - 1:
                print(f"{h['t']:>6.1f} {h['yaw']:>9.1f}° {h['beta']:>9.2f}° {h['wind_y_B']:>10.2f} m/s {h['yaw_rate']:>10.2f}°/s")

        initial_yaw_val = history[0]['yaw']
        final_yaw_val = history[-1]['yaw']
        yaw_change = final_yaw_val - initial_yaw_val

        # Handle wrap
        while yaw_change > 180:
            yaw_change -= 360
        while yaw_change < -180:
            yaw_change += 360

        print(f"\nTotal yaw change: {yaw_change:.1f}°")

        # Determine if this is stable or unstable equilibrium
        if abs(final_yaw_val) < abs(initial_yaw_val) * 0.5:
            print("→ STABLE: Returns toward headwind (no weathercock)")
        elif abs(yaw_change) > 90:
            print("→ STRONG WEATHERCOCK: Turned significantly toward downwind")
        elif abs(yaw_change) > 30:
            print("→ MODERATE WEATHERCOCK: Turning away from headwind")
        elif abs(yaw_change) > 10:
            print("→ WEAK WEATHERCOCK: Slight instability")
        else:
            print("→ NEAR NEUTRAL: Minimal yaw change")

    print("\n" + "=" * 70)
    print("PHYSICAL INTERPRETATION")
    print("=" * 70)
    print("""
The weathercock effect causes parafoils to turn from headwind to tailwind:

1. Flying into headwind: wind_y_B ≈ 0 when perfectly aligned
2. Small yaw perturbation: creates crosswind component (wind_y_B ≠ 0)
3. Negative c_n_weath: crosswind creates yaw moment AWAY from wind
4. This amplifies the perturbation → unstable equilibrium
5. Eventually parafoil turns to face downwind (stable equilibrium)

With c_n_weath > 0:
- Wind from right (wind_y_B > 0) → positive yaw moment → turn right
- This turns the parafoil AWAY from the wind source (weathercock)
- Equilibrium at yaw=180° (tailwind) is STABLE
- Equilibrium at yaw=0° (headwind) is UNSTABLE
""")


if __name__ == '__main__':
    main()
