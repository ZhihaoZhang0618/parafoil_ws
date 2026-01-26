#!/usr/bin/env python3
"""
Test weathercock effect with initial perturbation.

Key insight: weathercock effect is about stability of equilibrium.
- Flying into headwind with perfect alignment: beta = 0, no yaw moment
- With small initial yaw offset: creates sideslip, which either:
  - UNSTABLE: yaw moment amplifies perturbation → turns toward downwind
  - STABLE: yaw moment opposes perturbation → returns to headwind
"""

import numpy as np
import sys
sys.path.insert(0, '/home/aims/parafoil_ws/src/parafoil_dynamics')

from parafoil_dynamics.state import State, ControlCmd
from parafoil_dynamics.params import Params
from parafoil_dynamics.dynamics import dynamics
from parafoil_dynamics.math3d import quat_from_euler, quat_to_euler, quat_to_rotmat, compute_aero_angles
from parafoil_dynamics.integrators import integrate_with_substeps, IntegratorType


def simulate_perturbed(initial_yaw_deg: float, wind_speed: float, c_nb: float, c_Yb: float, duration: float = 30.0):
    """Simulate flight with initial yaw perturbation."""
    params = Params()
    params.c_nb = c_nb
    params.c_Yb = c_Yb

    # Wind blowing FROM south TO north (headwind for northward flight)
    wind_I = np.array([-wind_speed, 0.0, 0.0])

    p_I = np.array([0.0, 0.0, -500.0])
    v_I = np.array([4.0, 0.0, 1.0])  # Flying north

    # Start with small yaw perturbation (not perfectly into wind)
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

        history.append({
            't': state.t,
            'yaw': np.degrees(yaw),
            'beta': np.degrees(beta),
            'yaw_rate': np.degrees(state.w_B[2])
        })

    return history


def main():
    print("=" * 70)
    print("WEATHERCOCK STABILITY TEST")
    print("=" * 70)
    print("\nParafoil starts with 10° yaw offset from headwind direction")
    print("Wind from south (3 m/s headwind)")
    print("\n- If UNSTABLE: yaw will diverge (turn away from wind)")
    print("- If STABLE: yaw will return to 0° (face into wind)\n")

    wind_speed = 3.0
    initial_yaw = 10.0  # 10 degree perturbation

    # Test current parameters
    print("=" * 60)
    print("Test 1: Current params (c_nb=0.15, c_Yb=-6.8)")
    print("=" * 60)

    history = simulate_perturbed(initial_yaw, wind_speed, c_nb=0.15, c_Yb=-6.8, duration=30.0)

    print(f"\n{'Time':>6} {'Yaw':>10} {'Beta':>10} {'Yaw Rate':>12}")
    print("-" * 40)
    for i, h in enumerate(history):
        if i % 150 == 0 or i == len(history) - 1:
            print(f"{h['t']:>6.1f} {h['yaw']:>9.1f}° {h['beta']:>9.2f}° {h['yaw_rate']:>10.2f}°/s")

    initial_yaw = history[0]['yaw']
    final_yaw = history[-1]['yaw']

    if abs(final_yaw) < abs(initial_yaw) * 0.5:
        print("\n→ STABLE: Parafoil returns toward headwind (unrealistic)")
    elif abs(final_yaw) > abs(initial_yaw) * 2:
        print("\n→ UNSTABLE: Parafoil turns away from wind (weathercock effect)")
    else:
        print("\n→ MARGINAL: Small change in yaw")

    # Test with negative c_nb (unstable weathercock)
    print("\n" + "=" * 60)
    print("Test 2: Negative c_nb (c_nb=-0.15, c_Yb=-6.8)")
    print("Sign reversal: sideslip causes DESTABILIZING yaw moment")
    print("=" * 60)

    history = simulate_perturbed(initial_yaw, wind_speed, c_nb=-0.15, c_Yb=-6.8, duration=30.0)

    print(f"\n{'Time':>6} {'Yaw':>10} {'Beta':>10} {'Yaw Rate':>12}")
    print("-" * 40)
    for i, h in enumerate(history):
        if i % 150 == 0 or i == len(history) - 1:
            print(f"{h['t']:>6.1f} {h['yaw']:>9.1f}° {h['beta']:>9.2f}° {h['yaw_rate']:>10.2f}°/s")

    final_yaw = history[-1]['yaw']
    if abs(final_yaw) > abs(initial_yaw) * 2:
        print("\n→ UNSTABLE: Weathercock effect working!")
    else:
        print("\n→ Still too stable")

    # Test with very negative c_nb and reduced c_Yb
    print("\n" + "=" * 60)
    print("Test 3: Stronger destabilizing (c_nb=-0.5, c_Yb=-2.0)")
    print("=" * 60)

    history = simulate_perturbed(initial_yaw, wind_speed, c_nb=-0.5, c_Yb=-2.0, duration=30.0)

    print(f"\n{'Time':>6} {'Yaw':>10} {'Beta':>10} {'Yaw Rate':>12}")
    print("-" * 40)
    for i, h in enumerate(history):
        if i % 150 == 0 or i == len(history) - 1:
            print(f"{h['t']:>6.1f} {h['yaw']:>9.1f}° {h['beta']:>9.2f}° {h['yaw_rate']:>10.2f}°/s")

    final_yaw = history[-1]['yaw']
    if abs(final_yaw) > 90:
        print("\n→ STRONG weathercock: Turned significantly toward downwind")
    elif abs(final_yaw) > 45:
        print("\n→ MODERATE weathercock effect")
    else:
        print(f"\n→ Yaw change: {final_yaw - initial_yaw:.1f}°")

    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    print("""
The weathercock effect requires that sideslip produces a DESTABILIZING
yaw moment, not a stabilizing one.

Current model: C_n = c_nr * r_hat + c_nda * delta_a + c_nb * beta

With positive c_nb: positive beta (wind from right) → positive yaw moment
                    → turns RIGHT into the wind (STABILIZING)

For weathercock: positive beta (wind from right) → NEGATIVE yaw moment
                 → turns LEFT away from wind (DESTABILIZING)

This means c_nb should be NEGATIVE for realistic weathercock behavior.
""")


if __name__ == '__main__':
    main()
