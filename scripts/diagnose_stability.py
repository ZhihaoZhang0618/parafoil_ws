#!/usr/bin/env python3
"""
Diagnose stability issues in parafoil dynamics.
Run this script standalone (no ROS) to isolate dynamics issues.
"""

import sys
import numpy as np
sys.path.insert(0, '/home/aims/parafoil_ws/src/parafoil_dynamics')

from parafoil_dynamics.state import State, ControlCmd
from parafoil_dynamics.params import Params
from parafoil_dynamics.dynamics import dynamics
from parafoil_dynamics.integrators import integrate_with_substeps, IntegratorType


def run_test(test_name, delta_l, delta_r, duration=30.0, ctl_dt=0.02, dt_max=0.005):
    """Run a single test scenario and check for stability."""
    print(f"\n{'='*60}")
    print(f"测试: {test_name}")
    print(f"刹车: L={delta_l:.2f}, R={delta_r:.2f}, 持续={duration}s")
    print('='*60)
    
    params = Params()
    
    # Initial state: 500m altitude, ~6 m/s forward velocity (close to trim)
    state = State(
        p_I=np.array([0.0, 0.0, -500.0]),  # NED: -500m = 500m altitude
        v_I=np.array([5.0, 0.0, 1.0]),      # Close to expected trim speed
        q_IB=np.array([1.0, 0.0, 0.0, 0.0]),
        w_B=np.array([0.0, 0.0, 0.0]),
        delta=np.array([0.0, 0.0]),
        t=0.0
    )
    
    cmd = ControlCmd(delta_cmd=np.array([delta_l, delta_r]))
    
    t = 0.0
    last_print = 0.0
    max_speed = 0.0
    max_omega = 0.0
    
    try:
        while t < duration:
            # Check for instability
            speed = np.linalg.norm(state.v_I)
            omega = np.linalg.norm(state.w_B)
            max_speed = max(max_speed, speed)
            max_omega = max(max_omega, omega)
            
            if not np.all(np.isfinite(state.p_I)) or not np.all(np.isfinite(state.v_I)):
                print(f"  ❌ 发散! t={t:.2f}s: 位置或速度无穷大")
                return False
            
            if not np.all(np.isfinite(state.q_IB)) or not np.all(np.isfinite(state.w_B)):
                print(f"  ❌ 发散! t={t:.2f}s: 姿态或角速度无穷大")
                return False
            
            if speed > 100:
                print(f"  ❌ 发散! t={t:.2f}s: 速度过大 {speed:.1f} m/s")
                return False
            
            if omega > 10:  # rad/s
                print(f"  ❌ 发散! t={t:.2f}s: 角速度过大 {np.degrees(omega):.1f} deg/s")
                return False
            
            # Print status every 2 seconds
            if t - last_print >= 2.0:
                alt = -state.p_I[2]
                v = np.linalg.norm(state.v_I)
                # Compute Euler angles for debugging
                q = state.q_IB
                # Roll, Pitch, Yaw from quaternion
                roll = np.arctan2(2*(q[0]*q[1] + q[2]*q[3]), 1 - 2*(q[1]**2 + q[2]**2))
                pitch = np.arcsin(np.clip(2*(q[0]*q[2] - q[3]*q[1]), -1, 1))
                yaw = np.arctan2(2*(q[0]*q[3] + q[1]*q[2]), 1 - 2*(q[2]**2 + q[3]**2))
                
                print(f"  t={t:5.1f}s | alt={alt:6.1f}m | v={v:5.1f}m/s | "
                      f"roll={np.degrees(roll):6.1f}° pitch={np.degrees(pitch):6.1f}° yaw={np.degrees(yaw):6.1f}°")
                last_print = t
            
            # Integration step using the proper API
            state = integrate_with_substeps(
                dynamics_fn=dynamics,
                state=state,
                cmd=cmd,
                params=params,
                ctl_dt=ctl_dt,
                dt_max=dt_max,
                method=IntegratorType.RK4
            )
            t += ctl_dt
            
    except Exception as e:
        print(f"  ❌ 异常! t={t:.2f}s: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"  ✅ 完成! 最大速度={max_speed:.1f}m/s, 最大角速度={np.degrees(max_omega):.1f}°/s")
    return True


def main():
    print("="*60)
    print("Parafoil 动力学稳定性诊断")
    print("="*60)
    
    tests = [
        ("自由滑翔", 0.0, 0.0, 30),
        ("左刹车 25%", 0.25, 0.0, 20),
        ("右刹车 25%", 0.0, 0.25, 20),
        ("对称刹车 25%", 0.25, 0.25, 20),
        ("左刹车 50%", 0.5, 0.0, 20),
        ("右刹车 50%", 0.0, 0.5, 20),
        ("对称刹车 50%", 0.5, 0.5, 20),
        ("左刹车 100%", 1.0, 0.0, 20),
        ("右刹车 100%", 0.0, 1.0, 20),
        ("对称刹车 100%", 1.0, 1.0, 20),
    ]
    
    results = []
    for name, delta_l, delta_r, duration in tests:
        success = run_test(name, delta_l, delta_r, duration)
        results.append((name, success))
    
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    for name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"  {status}: {name}")
    
    passed = sum(1 for _, s in results if s)
    total = len(results)
    print(f"\n总计: {passed}/{total} 测试通过")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
