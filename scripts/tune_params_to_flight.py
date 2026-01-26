#!/usr/bin/env python3
"""
Parafoil Parameter Tuning Script

This script automatically tunes the aerodynamic parameters of the 6DoF parafoil
simulation to match actual flight data from log 09_01_06.

Target performance (from flight log analysis):
- Airspeed: ~4.3 m/s
- Sink rate: ~0.85 m/s
- Glide ratio: ~5.0
- Trim alpha: ~10-15 deg (estimated)

Usage:
    python scripts/tune_params_to_flight.py
"""

import numpy as np
import sys
from scipy.optimize import minimize, differential_evolution
from dataclasses import dataclass
from typing import Tuple, Dict, List
import json
import os

sys.path.insert(0, '/home/aims/parafoil_ws/src/parafoil_dynamics')

from parafoil_dynamics.state import State, ControlCmd
from parafoil_dynamics.params import Params
from parafoil_dynamics.dynamics import dynamics
from parafoil_dynamics.math3d import quat_from_euler, quat_to_euler, compute_aero_angles, quat_to_rotmat
from parafoil_dynamics.integrators import integrate_with_substeps, IntegratorType


# =============================================================================
# Target values from actual flight data (09_01_06)
# =============================================================================
@dataclass
class FlightTargets:
    """Target performance metrics from actual flight data"""
    # From README_log09_01_06_analysis.md
    # Zero brake (brake_ratio=0.0): sink=0.82, h_speed=4.20, L/D=5.11
    # Mid brake (brake_ratio=0.5): sink=0.90, h_speed=4.46, L/D=4.97
    # Estimated airspeed (after wind correction): ~4.27 m/s

    airspeed_zero_brake: float = 4.3       # m/s (estimated from circle fit)
    sink_rate_zero_brake: float = 0.85     # m/s
    glide_ratio_zero_brake: float = 5.0    # L/D

    airspeed_mid_brake: float = 4.3        # m/s (similar to zero brake)
    sink_rate_mid_brake: float = 0.90      # m/s
    glide_ratio_mid_brake: float = 4.8     # L/D

    airspeed_full_brake: float = 2.8       # m/s (rough estimate from heavy brake data)
    sink_rate_full_brake: float = 1.32     # m/s
    glide_ratio_full_brake: float = 2.1    # L/D


TARGETS = FlightTargets()


# =============================================================================
# Simulation utilities
# =============================================================================
def simulate_trim(params: Params, brake_ratio: float = 0.0,
                  max_time: float = 30.0, dt: float = 0.02) -> Dict:
    """
    Simulate parafoil to find trim condition at given brake ratio.

    Returns dict with:
        - converged: bool
        - airspeed: float (m/s)
        - sink_rate: float (m/s)
        - glide_ratio: float
        - alpha: float (rad)
        - pitch: float (rad)
    """
    # Initial state - start with reasonable initial velocity
    p_I = np.array([0.0, 0.0, -500.0])  # 500m altitude
    v_I = np.array([4.0, 0.0, 1.0])     # Roughly trimmed initial velocity
    q_IB = quat_from_euler(0, 0, 0)

    state = State(
        p_I=p_I,
        v_I=v_I,
        q_IB=q_IB,
        w_B=np.zeros(3),
        delta=np.array([brake_ratio, brake_ratio]),  # Symmetric brake
        t=0.0
    )

    cmd = ControlCmd(delta_cmd=np.array([brake_ratio, brake_ratio]))

    # History for convergence check
    history_len = 50  # 1 second of data
    airspeed_history = []
    sink_history = []

    dt_max = 0.005
    n_steps = int(max_time / dt)

    for step in range(n_steps):
        # Integrate
        state = integrate_with_substeps(
            dynamics, state, cmd, params, dt, dt_max,
            method=IntegratorType.RK4
        )

        # Compute metrics
        C_IB = quat_to_rotmat(state.q_IB)
        v_B = C_IB.T @ state.v_I
        V, alpha, beta = compute_aero_angles(v_B)

        h_speed = np.sqrt(state.v_I[0]**2 + state.v_I[1]**2)
        sink_rate = state.v_I[2]  # Positive = descending (NED)

        airspeed_history.append(V)
        sink_history.append(sink_rate)

        # Keep only recent history
        if len(airspeed_history) > history_len:
            airspeed_history.pop(0)
            sink_history.pop(0)

        # Check convergence (after some settling time)
        if step > 500 and len(airspeed_history) >= history_len:
            airspeed_std = np.std(airspeed_history)
            sink_std = np.std(sink_history)

            # Converged if variations are small
            if airspeed_std < 0.05 and sink_std < 0.05:
                avg_airspeed = np.mean(airspeed_history)
                avg_sink = np.mean(sink_history)

                # Compute final metrics
                roll, pitch, yaw = quat_to_euler(state.q_IB)

                glide_ratio = h_speed / max(avg_sink, 0.01) if avg_sink > 0.01 else 0.0

                return {
                    'converged': True,
                    'airspeed': avg_airspeed,
                    'sink_rate': avg_sink,
                    'glide_ratio': glide_ratio,
                    'alpha': alpha,
                    'pitch': pitch,
                    'h_speed': h_speed,
                    'time': state.t
                }

        # Check for instability
        if np.any(np.isnan(state.v_I)) or np.any(np.abs(state.v_I) > 100):
            return {
                'converged': False,
                'airspeed': float('inf'),
                'sink_rate': float('inf'),
                'glide_ratio': 0.0,
                'alpha': 0.0,
                'pitch': 0.0,
                'h_speed': 0.0,
                'time': state.t
            }

    # Didn't converge but return final values
    avg_airspeed = np.mean(airspeed_history) if airspeed_history else 0.0
    avg_sink = np.mean(sink_history) if sink_history else 0.0
    h_speed = np.sqrt(state.v_I[0]**2 + state.v_I[1]**2)
    roll, pitch, yaw = quat_to_euler(state.q_IB)
    C_IB = quat_to_rotmat(state.q_IB)
    v_B = C_IB.T @ state.v_I
    V, alpha, beta = compute_aero_angles(v_B)

    return {
        'converged': False,
        'airspeed': avg_airspeed,
        'sink_rate': avg_sink,
        'glide_ratio': h_speed / max(avg_sink, 0.01) if avg_sink > 0.01 else 0.0,
        'alpha': alpha,
        'pitch': pitch,
        'h_speed': h_speed,
        'time': state.t
    }


# =============================================================================
# Cost function for optimization
# =============================================================================
def compute_cost(param_vector: np.ndarray, base_params: Params,
                 verbose: bool = False) -> float:
    """
    Compute cost (error) between simulated trim and target flight data.

    param_vector: [c_L0, c_La, c_D0, c_Da2, c_m0, c_ma]
    """
    # Unpack parameters
    c_L0, c_La, c_D0, c_Da2, c_m0, c_ma = param_vector

    # Apply parameter constraints
    if c_L0 < 0 or c_La < 0 or c_D0 < 0 or c_Da2 < 0:
        return 1e6
    if c_L0 > 1.0 or c_La > 5.0 or c_D0 > 0.5 or c_Da2 > 2.0:
        return 1e6

    # Create modified params
    params = Params()
    params.c_L0 = c_L0
    params.c_La = c_La
    params.c_D0 = c_D0
    params.c_Da2 = c_Da2
    params.c_m0 = c_m0
    params.c_ma = c_ma

    total_cost = 0.0

    # Test zero brake
    result_zero = simulate_trim(params, brake_ratio=0.0, max_time=20.0)
    if not result_zero['converged']:
        return 1e5

    # Zero brake cost
    airspeed_err = (result_zero['airspeed'] - TARGETS.airspeed_zero_brake) / TARGETS.airspeed_zero_brake
    sink_err = (result_zero['sink_rate'] - TARGETS.sink_rate_zero_brake) / TARGETS.sink_rate_zero_brake
    glide_err = (result_zero['glide_ratio'] - TARGETS.glide_ratio_zero_brake) / TARGETS.glide_ratio_zero_brake

    # Weight: airspeed and sink rate are most important
    cost_zero = 2.0 * airspeed_err**2 + 2.0 * sink_err**2 + 1.0 * glide_err**2
    total_cost += cost_zero

    # Test mid brake
    result_mid = simulate_trim(params, brake_ratio=0.5, max_time=20.0)
    if not result_mid['converged']:
        return 1e5

    # Mid brake cost
    sink_err_mid = (result_mid['sink_rate'] - TARGETS.sink_rate_mid_brake) / TARGETS.sink_rate_mid_brake
    glide_err_mid = (result_mid['glide_ratio'] - TARGETS.glide_ratio_mid_brake) / TARGETS.glide_ratio_mid_brake

    cost_mid = 1.5 * sink_err_mid**2 + 1.0 * glide_err_mid**2
    total_cost += cost_mid

    if verbose:
        print(f"\n  Zero brake: V={result_zero['airspeed']:.2f} m/s, "
              f"sink={result_zero['sink_rate']:.2f} m/s, L/D={result_zero['glide_ratio']:.2f}")
        print(f"  Mid brake:  V={result_mid['airspeed']:.2f} m/s, "
              f"sink={result_mid['sink_rate']:.2f} m/s, L/D={result_mid['glide_ratio']:.2f}")
        print(f"  Cost: {total_cost:.4f}")

    return total_cost


def optimize_callback(xk, convergence=None):
    """Callback for differential evolution to show progress"""
    cost = compute_cost(xk, Params(), verbose=False)
    print(f"  Current best: c_L0={xk[0]:.3f}, c_La={xk[1]:.3f}, "
          f"c_D0={xk[2]:.3f}, c_Da2={xk[3]:.3f}, c_m0={xk[4]:.3f}, c_ma={xk[5]:.3f} | cost={cost:.4f}")


# =============================================================================
# Main optimization
# =============================================================================
def main():
    print("=" * 80)
    print("Parafoil Parameter Tuning")
    print("=" * 80)
    print("\nTarget performance (from flight log 09_01_06):")
    print(f"  Zero brake: V={TARGETS.airspeed_zero_brake:.1f} m/s, "
          f"sink={TARGETS.sink_rate_zero_brake:.2f} m/s, L/D={TARGETS.glide_ratio_zero_brake:.1f}")
    print(f"  Mid brake:  V={TARGETS.airspeed_mid_brake:.1f} m/s, "
          f"sink={TARGETS.sink_rate_mid_brake:.2f} m/s, L/D={TARGETS.glide_ratio_mid_brake:.1f}")

    # Current parameters
    current_params = Params()
    print("\n--- Current parameters ---")
    print(f"  c_L0={current_params.c_L0:.3f}, c_La={current_params.c_La:.3f}")
    print(f"  c_D0={current_params.c_D0:.3f}, c_Da2={current_params.c_Da2:.3f}")
    print(f"  c_m0={current_params.c_m0:.3f}, c_ma={current_params.c_ma:.3f}")

    # Evaluate current parameters
    print("\n--- Evaluating current parameters ---")
    current_vector = np.array([
        current_params.c_L0, current_params.c_La,
        current_params.c_D0, current_params.c_Da2,
        current_params.c_m0, current_params.c_ma
    ])
    current_cost = compute_cost(current_vector, current_params, verbose=True)
    print(f"\nCurrent total cost: {current_cost:.4f}")

    # Parameter bounds for optimization
    # [c_L0, c_La, c_D0, c_Da2, c_m0, c_ma]
    bounds = [
        (0.1, 0.8),    # c_L0: zero-alpha lift
        (1.5, 4.0),    # c_La: lift curve slope
        (0.05, 0.25),  # c_D0: parasite drag
        (0.1, 1.0),    # c_Da2: induced drag
        (-0.2, 0.3),   # c_m0: zero-alpha pitch moment
        (-1.5, -0.3),  # c_ma: pitch moment slope (must be negative for stability)
    ]

    print("\n--- Starting optimization (Differential Evolution) ---")
    print("This may take a few minutes...")

    result = differential_evolution(
        lambda x: compute_cost(x, current_params, verbose=False),
        bounds,
        seed=42,
        maxiter=100,
        tol=0.001,
        disp=True,
        callback=optimize_callback,
        workers=1,  # Single thread for stability
        polish=True
    )

    print("\n" + "=" * 80)
    print("OPTIMIZATION RESULTS")
    print("=" * 80)

    optimized = result.x
    print(f"\nOptimized parameters:")
    print(f"  c_L0 = {optimized[0]:.4f}  (was {current_params.c_L0:.4f})")
    print(f"  c_La = {optimized[1]:.4f}  (was {current_params.c_La:.4f})")
    print(f"  c_D0 = {optimized[2]:.4f}  (was {current_params.c_D0:.4f})")
    print(f"  c_Da2 = {optimized[3]:.4f}  (was {current_params.c_Da2:.4f})")
    print(f"  c_m0 = {optimized[4]:.4f}  (was {current_params.c_m0:.4f})")
    print(f"  c_ma = {optimized[5]:.4f}  (was {current_params.c_ma:.4f})")

    print("\n--- Verifying optimized parameters ---")
    final_cost = compute_cost(optimized, current_params, verbose=True)
    print(f"\nFinal cost: {final_cost:.4f} (was {current_cost:.4f})")
    print(f"Improvement: {(current_cost - final_cost) / current_cost * 100:.1f}%")

    # Save results
    results_dict = {
        'optimized_params': {
            'c_L0': float(optimized[0]),
            'c_La': float(optimized[1]),
            'c_D0': float(optimized[2]),
            'c_Da2': float(optimized[3]),
            'c_m0': float(optimized[4]),
            'c_ma': float(optimized[5])
        },
        'original_params': {
            'c_L0': current_params.c_L0,
            'c_La': current_params.c_La,
            'c_D0': current_params.c_D0,
            'c_Da2': current_params.c_Da2,
            'c_m0': current_params.c_m0,
            'c_ma': current_params.c_ma
        },
        'targets': {
            'airspeed_zero_brake': TARGETS.airspeed_zero_brake,
            'sink_rate_zero_brake': TARGETS.sink_rate_zero_brake,
            'glide_ratio_zero_brake': TARGETS.glide_ratio_zero_brake
        },
        'final_cost': float(final_cost),
        'original_cost': float(current_cost)
    }

    output_path = '/home/aims/parafoil_ws/scripts/tuned_params.json'
    with open(output_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Print code snippet to update params.py
    print("\n" + "=" * 80)
    print("CODE SNIPPET - Add to params.py:")
    print("=" * 80)
    print(f"""
    # Tuned aerodynamic coefficients (matched to flight log 09_01_06)
    c_L0: float = {optimized[0]:.4f}    # Zero-alpha lift coefficient
    c_La: float = {optimized[1]:.4f}    # Lift curve slope [1/rad]
    c_D0: float = {optimized[2]:.4f}    # Parasite drag
    c_Da2: float = {optimized[3]:.4f}   # Induced drag factor
    c_m0: float = {optimized[4]:.4f}    # Zero-alpha pitching moment
    c_ma: float = {optimized[5]:.4f}    # Pitching moment slope [1/rad]
""")


if __name__ == '__main__':
    main()
