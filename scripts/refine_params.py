#!/usr/bin/env python3
"""
Refine parameter search to match flight data more closely.
"""

import numpy as np
import sys
sys.path.insert(0, '/home/aims/parafoil_ws/src/parafoil_dynamics')

from parafoil_dynamics.state import State, ControlCmd
from parafoil_dynamics.params import Params
from parafoil_dynamics.dynamics import dynamics
from parafoil_dynamics.math3d import quat_from_euler, quat_to_euler, compute_aero_angles, quat_to_rotmat
from parafoil_dynamics.integrators import integrate_with_substeps, IntegratorType
import json


def test_params(c_L0, c_La, c_D0, c_Da2, c_m0, c_ma, c_Lds=0.0, c_Dds=0.0, brake=0.0):
    """Test a parameter set by simulating 30 seconds."""
    params = Params()
    params.c_L0 = c_L0
    params.c_La = c_La
    params.c_D0 = c_D0
    params.c_Da2 = c_Da2
    params.c_m0 = c_m0
    params.c_ma = c_ma
    params.c_Lds = c_Lds
    params.c_Dds = c_Dds

    # Initial state
    p_I = np.array([0.0, 0.0, -500.0])
    v_I = np.array([4.0, 0.0, 1.0])
    q_IB = quat_from_euler(0, 0, 0)

    state = State(
        p_I=p_I, v_I=v_I, q_IB=q_IB,
        w_B=np.zeros(3),
        delta=np.array([brake, brake]),
        t=0.0
    )
    cmd = ControlCmd(delta_cmd=np.array([brake, brake]))

    dt = 0.02
    dt_max = 0.005

    # Simulate 30 seconds for better convergence
    for _ in range(1500):
        state = integrate_with_substeps(
            dynamics, state, cmd, params, dt, dt_max,
            method=IntegratorType.RK4
        )
        # Check for NaN
        if np.any(np.isnan(state.v_I)):
            return None

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


def fine_grid_search():
    """Fine grid search around the best parameters found."""
    print("=" * 70)
    print("FINE GRID SEARCH")
    print("=" * 70)
    print("\nTargets:")
    print("  Zero brake: V≈4.3 m/s, sink≈0.85 m/s, L/D≈5.0")
    print("  Mid brake:  V≈4.3 m/s, sink≈0.90 m/s, L/D≈4.8")

    # Targets
    V_target = 4.3
    sink_target_zero = 0.85
    sink_target_mid = 0.90
    LD_target = 5.0

    best_cost = float('inf')
    best_params = None
    best_results = None

    # Fine search around previous best
    # c_L0_range = [0.45, 0.50, 0.55, 0.60, 0.65]
    # c_La_range = [2.8, 3.0, 3.2, 3.5]
    # c_D0_range = [0.08, 0.10, 0.12, 0.14]
    # c_Da2_range = [0.30, 0.35, 0.40, 0.50]

    # Even finer - focus on getting V closer
    # To reduce V, we need more drag or less lift
    c_L0_range = [0.55, 0.60, 0.65, 0.70]
    c_La_range = [3.0, 3.2, 3.5, 3.8]
    c_D0_range = [0.12, 0.14, 0.16, 0.18]
    c_Da2_range = [0.35, 0.40, 0.50, 0.60]
    c_Lds_range = [0.1, 0.15, 0.2]
    c_Dds_range = [0.3, 0.4, 0.5]

    # Fixed pitch stability
    c_m0 = 0.1
    c_ma = -0.72

    total = len(c_L0_range) * len(c_La_range) * len(c_D0_range) * len(c_Da2_range)
    print(f"\nSearching {total} base combinations...")

    count = 0
    for c_L0 in c_L0_range:
        for c_La in c_La_range:
            for c_D0 in c_D0_range:
                for c_Da2 in c_Da2_range:
                    count += 1
                    if count % 50 == 0:
                        print(f"  Progress: {count}/{total}")

                    try:
                        # Test zero brake
                        result_zero = test_params(c_L0, c_La, c_D0, c_Da2, c_m0, c_ma, brake=0.0)
                        if result_zero is None:
                            continue

                        # Compute cost (prioritize matching V and sink)
                        V_err = (result_zero['V'] - V_target) / V_target
                        sink_err = (result_zero['sink'] - sink_target_zero) / sink_target_zero
                        LD_err = (result_zero['glide_ratio'] - LD_target) / LD_target

                        # Weight V and sink equally, glide ratio less
                        cost = 2.0 * V_err**2 + 2.0 * sink_err**2 + 0.5 * LD_err**2

                        if cost < best_cost:
                            best_cost = cost
                            best_params = {
                                'c_L0': c_L0,
                                'c_La': c_La,
                                'c_D0': c_D0,
                                'c_Da2': c_Da2,
                                'c_m0': c_m0,
                                'c_ma': c_ma
                            }
                            best_results = {'zero_brake': result_zero}

                    except Exception as e:
                        continue

    print(f"\n" + "=" * 70)
    print("BEST PARAMETERS FOUND")
    print("=" * 70)
    for key, val in best_params.items():
        print(f"  {key} = {val:.4f}")

    print(f"\nZero brake performance:")
    r = best_results['zero_brake']
    print(f"  Airspeed V = {r['V']:.2f} m/s (target: {V_target})")
    print(f"  Sink rate = {r['sink']:.2f} m/s (target: {sink_target_zero})")
    print(f"  Glide ratio = {r['glide_ratio']:.2f} (target: {LD_target})")
    print(f"  Trim alpha = {r['alpha_deg']:.1f}°")
    print(f"  Pitch = {r['pitch_deg']:.1f}°")

    # Now search for brake coefficients
    print(f"\n--- Finding brake effect coefficients ---")
    best_brake_cost = float('inf')
    best_brake_params = None

    for c_Lds in c_Lds_range:
        for c_Dds in c_Dds_range:
            try:
                result_mid = test_params(
                    best_params['c_L0'], best_params['c_La'],
                    best_params['c_D0'], best_params['c_Da2'],
                    best_params['c_m0'], best_params['c_ma'],
                    c_Lds=c_Lds, c_Dds=c_Dds, brake=0.5
                )
                if result_mid is None:
                    continue

                sink_err = (result_mid['sink'] - sink_target_mid) / sink_target_mid
                cost = sink_err**2

                if cost < best_brake_cost:
                    best_brake_cost = cost
                    best_brake_params = {'c_Lds': c_Lds, 'c_Dds': c_Dds}
                    best_mid_result = result_mid

            except:
                continue

    if best_brake_params:
        print(f"\nBrake effect coefficients:")
        print(f"  c_Lds = {best_brake_params['c_Lds']:.3f}")
        print(f"  c_Dds = {best_brake_params['c_Dds']:.3f}")

        print(f"\nMid brake (0.5) performance:")
        print(f"  Airspeed V = {best_mid_result['V']:.2f} m/s")
        print(f"  Sink rate = {best_mid_result['sink']:.2f} m/s (target: {sink_target_mid})")
        print(f"  Glide ratio = {best_mid_result['glide_ratio']:.2f}")

        best_params.update(best_brake_params)

    # Save results
    output = {
        'tuned_params': best_params,
        'performance': {
            'zero_brake': best_results['zero_brake'],
            'mid_brake': best_mid_result if best_brake_params else None
        },
        'targets': {
            'V': V_target,
            'sink_zero': sink_target_zero,
            'sink_mid': sink_target_mid,
            'LD': LD_target
        }
    }

    with open('/home/aims/parafoil_ws/scripts/tuned_params.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n" + "=" * 70)
    print("COPY THESE TO params.py:")
    print("=" * 70)
    print(f"""
    # Tuned aerodynamic coefficients (matched to flight log 09_01_06)
    c_L0: float = {best_params['c_L0']:.4f}
    c_La: float = {best_params['c_La']:.4f}
    c_D0: float = {best_params['c_D0']:.4f}
    c_Da2: float = {best_params['c_Da2']:.4f}
    c_Lds: float = {best_params.get('c_Lds', 0.15):.4f}
    c_Dds: float = {best_params.get('c_Dds', 0.40):.4f}
    c_m0: float = {best_params['c_m0']:.4f}
    c_ma: float = {best_params['c_ma']:.4f}
""")

    return best_params


if __name__ == '__main__':
    fine_grid_search()
