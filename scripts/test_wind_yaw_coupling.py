#!/usr/bin/env python3
"""
Test wind-yaw coupling: verify that crosswind produces yaw moment.
"""

import numpy as np
import sys
sys.path.insert(0, '/home/aims/parafoil_ws/src/parafoil_dynamics')

from parafoil_dynamics.state import State, ControlCmd
from parafoil_dynamics.params import Params
from parafoil_dynamics.dynamics import dynamics
from parafoil_dynamics.math3d import quat_from_euler, quat_to_euler, quat_to_rotmat, compute_aero_angles
from parafoil_dynamics.integrators import integrate_with_substeps, IntegratorType


def simulate_with_wind(wind_direction: str, wind_speed: float, duration: float = 20.0):
    """Simulate flight with crosswind."""
    params = Params()

    # Start heading north
    p_I = np.array([0.0, 0.0, -500.0])
    v_I = np.array([4.0, 0.0, 1.0])  # Flying north
    q_IB = quat_from_euler(0, 0, 0)  # Heading north (yaw=0)

    state = State(
        p_I=p_I, v_I=v_I, q_IB=q_IB,
        w_B=np.zeros(3),
        delta=np.array([0.0, 0.0]),
        t=0.0
    )
    cmd = ControlCmd(delta_cmd=np.array([0.0, 0.0]))  # No brake input

    # Set wind direction
    if wind_direction == "east":
        wind_I = np.array([0.0, wind_speed, 0.0])  # From east (positive E)
    elif wind_direction == "west":
        wind_I = np.array([0.0, -wind_speed, 0.0])  # From west (negative E)
    elif wind_direction == "north":
        wind_I = np.array([wind_speed, 0.0, 0.0])  # From north (headwind)
    elif wind_direction == "south":
        wind_I = np.array([-wind_speed, 0.0, 0.0])  # From south (tailwind)
    else:
        wind_I = np.zeros(3)

    dt = 0.02
    dt_max = 0.005
    n_steps = int(duration / dt)

    yaw_history = []
    beta_history = []

    for step in range(n_steps):
        state = integrate_with_substeps(
            dynamics, state, cmd, params, dt, dt_max,
            method=IntegratorType.RK4,
            wind_fn=lambda t: wind_I
        )

        roll, pitch, yaw = quat_to_euler(state.q_IB)

        # Compute sideslip angle
        C_IB = quat_to_rotmat(state.q_IB)
        v_rel_I = state.v_I - wind_I
        v_rel_B = C_IB.T @ v_rel_I
        V, alpha, beta = compute_aero_angles(v_rel_B)

        yaw_history.append(np.degrees(yaw))
        beta_history.append(np.degrees(beta))

    return {
        'final_yaw': yaw_history[-1],
        'yaw_change': yaw_history[-1] - yaw_history[0],
        'avg_beta': np.mean(beta_history[-100:]),
        'yaw_history': yaw_history
    }


def main():
    print("=" * 60)
    print("WIND-YAW COUPLING TEST")
    print("=" * 60)
    print(f"\nTesting with c_nb = {Params().c_nb}")
    print("Parafoil heading north (yaw=0), no brake input\n")

    # Note: wind vector is velocity, so [0,+E,0] means air moving east = wind FROM west
    wind_cases = [
        ("none", 0.0, "No wind (baseline)"),
        ("east", 3.0, "Wind FROM west (crosswind from LEFT)"),
        ("west", 3.0, "Wind FROM east (crosswind from RIGHT)"),
        ("north", 3.0, "Wind FROM south (tailwind)"),
        ("south", 3.0, "Wind FROM north (headwind)"),
    ]

    print(f"{'Wind Condition':<40} {'Beta':>10} {'Yaw Change':>12} {'Expected':>12}")
    print("-" * 74)

    for direction, speed, description in wind_cases:
        result = simulate_with_wind(direction, speed)

        # Expected physical behavior:
        # - Wind FROM left -> left canopy has more drag -> left slows down -> turn LEFT
        # - Wind FROM right -> right canopy has more drag -> right slows down -> turn RIGHT
        if direction == "east":  # Wind FROM west (left side)
            expected = "Turn left"
        elif direction == "west":  # Wind FROM east (right side)
            expected = "Turn right"
        else:
            expected = "Straight"

        yaw_change = result['yaw_change']
        actual = "Turn left" if yaw_change < -5 else ("Turn right" if yaw_change > 5 else "Straight")
        match = "OK" if actual == expected else "MISMATCH"

        print(f"{description:<40} {result['avg_beta']:>9.1f}° {yaw_change:>10.1f}° {expected:>12}")

    print("\n" + "-" * 60)
    print("Physical interpretation:")
    print("  - Crosswind from RIGHT -> right canopy has more drag -> turn RIGHT")
    print("  - Crosswind from LEFT -> left canopy has more drag -> turn LEFT")
    print("  - c_nb > 0: models asymmetric drag from crosswind on parafoil canopy")


if __name__ == '__main__':
    main()
