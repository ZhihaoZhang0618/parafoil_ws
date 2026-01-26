#!/usr/bin/env python3
"""
Quick parameter analysis based on flight physics.

From flight log 09_01_06:
- Airspeed V ≈ 4.3 m/s
- Sink rate ≈ 0.85 m/s
- L/D ≈ 5.0
- Mass m = 2.2 kg
- Wing area S = 1.5 m²

Physics:
- In steady glide: L = W·cos(γ), D = W·sin(γ)
- γ = atan(sink/V_horiz) = atan(0.85/4.22) ≈ 11.4°
- C_L = 2·L / (ρ·V²·S)
- C_D = 2·D / (ρ·V²·S)
"""

import numpy as np
import sys
sys.path.insert(0, '/home/aims/parafoil_ws/src/parafoil_dynamics')

from parafoil_dynamics.state import State, ControlCmd
from parafoil_dynamics.params import Params
from parafoil_dynamics.dynamics import dynamics
from parafoil_dynamics.math3d import quat_from_euler, quat_to_euler, compute_aero_angles, quat_to_rotmat
from parafoil_dynamics.integrators import integrate_with_substeps, IntegratorType


def compute_required_coefficients():
    """Compute required aerodynamic coefficients from flight data."""
    print("=" * 70)
    print("PARAMETER ANALYSIS FROM FLIGHT DATA")
    print("=" * 70)

    # Flight data
    V = 4.3          # m/s airspeed
    sink = 0.85      # m/s sink rate
    m = 2.2          # kg
    S = 1.5          # m² wing area
    rho = 1.29       # kg/m³
    g = 9.81         # m/s²

    # Compute glide angle
    V_horiz = np.sqrt(V**2 - sink**2)
    gamma = np.arctan2(sink, V_horiz)
    print(f"\nFlight conditions:")
    print(f"  Airspeed V = {V:.2f} m/s")
    print(f"  Sink rate = {sink:.2f} m/s")
    print(f"  Horizontal speed = {V_horiz:.2f} m/s")
    print(f"  Glide angle γ = {np.degrees(gamma):.1f}°")
    print(f"  Glide ratio L/D = {V_horiz/sink:.2f}")

    # Forces in steady glide
    W = m * g
    L_required = W * np.cos(gamma)
    D_required = W * np.sin(gamma)
    print(f"\nRequired forces:")
    print(f"  Weight W = {W:.2f} N")
    print(f"  Lift L = {L_required:.2f} N")
    print(f"  Drag D = {D_required:.2f} N")

    # Dynamic pressure
    q_bar = 0.5 * rho * V**2
    print(f"\nDynamic pressure q = {q_bar:.2f} Pa")

    # Required coefficients
    C_L_required = L_required / (q_bar * S)
    C_D_required = D_required / (q_bar * S)
    print(f"\nRequired coefficients (total):")
    print(f"  C_L = {C_L_required:.3f}")
    print(f"  C_D = {C_D_required:.3f}")
    print(f"  C_L/C_D = {C_L_required/C_D_required:.2f}")

    # Estimate trim angle of attack
    # For a parafoil, typical alpha_trim ≈ 8-15°
    # C_L = c_L0 + c_La * alpha
    # Assuming typical c_La ≈ 2-3, and we need C_L ≈ 0.95
    print(f"\nEstimated trim conditions:")

    # Try different combinations
    print("\nTrying different c_L0/c_La combinations:")
    for c_L0 in [0.3, 0.4, 0.5, 0.6]:
        for c_La in [2.0, 2.5, 3.0]:
            alpha_trim = (C_L_required - c_L0) / c_La
            if 0 < alpha_trim < 0.35:  # 0-20 degrees reasonable
                print(f"  c_L0={c_L0:.2f}, c_La={c_La:.2f} -> alpha_trim={np.degrees(alpha_trim):.1f}°")

    # For drag: C_D = c_D0 + c_Da2 * alpha²
    # Assuming alpha ≈ 10-12°, alpha² ≈ 0.03-0.04
    print(f"\nDrag coefficient breakdown:")
    alpha_est = 0.18  # ~10 degrees in radians
    print(f"  Assuming alpha_trim ≈ {np.degrees(alpha_est):.0f}°")
    print(f"  If c_D0 = 0.08, c_Da2 = {(C_D_required - 0.08)/(alpha_est**2):.2f}")
    print(f"  If c_D0 = 0.10, c_Da2 = {(C_D_required - 0.10)/(alpha_est**2):.2f}")

    return {
        'C_L_required': C_L_required,
        'C_D_required': C_D_required,
        'gamma': gamma
    }


def test_params(c_L0, c_La, c_D0, c_Da2, c_m0, c_ma):
    """Test a parameter set by simulating 20 seconds."""
    params = Params()
    params.c_L0 = c_L0
    params.c_La = c_La
    params.c_D0 = c_D0
    params.c_Da2 = c_Da2
    params.c_m0 = c_m0
    params.c_ma = c_ma

    # Initial state
    p_I = np.array([0.0, 0.0, -500.0])
    v_I = np.array([4.0, 0.0, 1.0])
    q_IB = quat_from_euler(0, 0, 0)

    state = State(
        p_I=p_I, v_I=v_I, q_IB=q_IB,
        w_B=np.zeros(3),
        delta=np.zeros(2),
        t=0.0
    )
    cmd = ControlCmd(delta_cmd=np.zeros(2))

    dt = 0.02
    dt_max = 0.005

    # Simulate 20 seconds
    for _ in range(1000):
        state = integrate_with_substeps(
            dynamics, state, cmd, params, dt, dt_max,
            method=IntegratorType.RK4
        )

    # Compute final metrics
    C_IB = quat_to_rotmat(state.q_IB)
    v_B = C_IB.T @ state.v_I
    V, alpha, beta = compute_aero_angles(v_B)
    h_speed = np.sqrt(state.v_I[0]**2 + state.v_I[1]**2)
    sink = state.v_I[2]
    roll, pitch, yaw = quat_to_euler(state.q_IB)

    return {
        'V': V,
        'sink': sink,
        'h_speed': h_speed,
        'glide_ratio': h_speed / sink if sink > 0.01 else 0,
        'alpha_deg': np.degrees(alpha),
        'pitch_deg': np.degrees(pitch)
    }


def grid_search():
    """Grid search for best parameters."""
    print("\n" + "=" * 70)
    print("GRID SEARCH FOR PARAMETERS")
    print("=" * 70)
    print("\nTarget: V≈4.3 m/s, sink≈0.85 m/s, L/D≈5.0")
    print("-" * 70)

    # Target values
    V_target = 4.3
    sink_target = 0.85
    LD_target = 5.0

    best_cost = float('inf')
    best_params = None
    best_result = None

    # Grid search - focus on parameters that most affect steady state
    c_L0_range = [0.35, 0.40, 0.45, 0.50]
    c_La_range = [2.5, 3.0, 3.5]
    c_D0_range = [0.06, 0.08, 0.10]
    c_Da2_range = [0.15, 0.25, 0.35]

    # Fixed pitch moment coefficients (for stability)
    c_m0 = 0.1
    c_ma = -0.72

    print(f"\nSearching {len(c_L0_range)*len(c_La_range)*len(c_D0_range)*len(c_Da2_range)} combinations...")

    for c_L0 in c_L0_range:
        for c_La in c_La_range:
            for c_D0 in c_D0_range:
                for c_Da2 in c_Da2_range:
                    try:
                        result = test_params(c_L0, c_La, c_D0, c_Da2, c_m0, c_ma)

                        # Compute cost
                        V_err = (result['V'] - V_target) / V_target
                        sink_err = (result['sink'] - sink_target) / sink_target
                        LD_err = (result['glide_ratio'] - LD_target) / LD_target

                        cost = V_err**2 + sink_err**2 + 0.5 * LD_err**2

                        if cost < best_cost:
                            best_cost = cost
                            best_params = (c_L0, c_La, c_D0, c_Da2, c_m0, c_ma)
                            best_result = result

                    except Exception as e:
                        continue

    print(f"\nBest parameters found:")
    print(f"  c_L0 = {best_params[0]:.3f}")
    print(f"  c_La = {best_params[1]:.3f}")
    print(f"  c_D0 = {best_params[2]:.3f}")
    print(f"  c_Da2 = {best_params[3]:.3f}")
    print(f"  c_m0 = {best_params[4]:.3f}")
    print(f"  c_ma = {best_params[5]:.3f}")

    print(f"\nSimulated performance:")
    print(f"  Airspeed V = {best_result['V']:.2f} m/s (target: {V_target})")
    print(f"  Sink rate = {best_result['sink']:.2f} m/s (target: {sink_target})")
    print(f"  Glide ratio = {best_result['glide_ratio']:.2f} (target: {LD_target})")
    print(f"  Trim alpha = {best_result['alpha_deg']:.1f}°")
    print(f"  Pitch = {best_result['pitch_deg']:.1f}°")

    return best_params


def compare_with_current():
    """Compare current vs optimized parameters."""
    print("\n" + "=" * 70)
    print("COMPARISON: CURRENT vs OPTIMIZED")
    print("=" * 70)

    # Current parameters
    print("\n--- Current parameters ---")
    current = test_params(0.24, 2.14, 0.12, 0.33, 0.1, -0.72)
    print(f"  V = {current['V']:.2f} m/s, sink = {current['sink']:.2f} m/s, "
          f"L/D = {current['glide_ratio']:.2f}, alpha = {current['alpha_deg']:.1f}°")

    # Optimized (will be filled by grid search)
    best = grid_search()

    print("\n--- Final recommendation ---")
    print(f"Update params.py with:")
    print(f"    c_L0 = {best[0]:.4f}")
    print(f"    c_La = {best[1]:.4f}")
    print(f"    c_D0 = {best[2]:.4f}")
    print(f"    c_Da2 = {best[3]:.4f}")

    return best


if __name__ == '__main__':
    compute_required_coefficients()
    best_params = compare_with_current()
