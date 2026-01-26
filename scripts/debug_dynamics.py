#!/usr/bin/env python3
"""
调试翼伞动力学：逐步积分并打印关键变量
"""

import numpy as np
import sys
sys.path.insert(0, '/home/aims/parafoil_ws/src/parafoil_dynamics')

from parafoil_dynamics.state import State, StateDot, ControlCmd
from parafoil_dynamics.params import Params
from parafoil_dynamics.dynamics import dynamics
from parafoil_dynamics.math3d import (
    quat_from_euler, quat_to_euler, quat_to_rotmat, 
    compute_aero_angles, normalize_quaternion
)
from parafoil_dynamics.integrators import integrate_with_substeps, IntegratorType


def debug_dynamics():
    """逐步调试动力学"""
    params = Params()
    
    # 初始状态 (NED)
    p_I = np.array([0.0, 0.0, -500.0])  # 500m 高度 (D = -500)
    v_I = np.array([10.0, 0.0, 2.0])    # 北向10m/s，下沉2m/s
    q_IB = quat_from_euler(0, 0, 0)      # 水平姿态，朝北
    
    state = State(
        p_I=p_I,
        v_I=v_I,
        q_IB=q_IB,
        w_B=np.zeros(3),
        delta=np.zeros(2),
        t=0.0
    )
    
    cmd = ControlCmd(delta_cmd=np.zeros(2))
    
    dt = 0.02  # 20ms
    dt_max = 0.005  # 5ms substeps
    
    print("="*80)
    print("翼伞动力学调试")
    print("="*80)
    print(f"初始条件 (NED):")
    print(f"  位置: N={p_I[0]:.1f}, E={p_I[1]:.1f}, D={p_I[2]:.1f}")
    print(f"  速度: Vn={v_I[0]:.1f}, Ve={v_I[1]:.1f}, Vd={v_I[2]:.1f} m/s")
    print(f"  姿态: roll=0°, pitch=0°, yaw=0°")
    print()
    
    # 模拟 5 秒
    for step in range(250):  # 5s at 50Hz
        t = step * dt
        
        # 每 0.5 秒打印一次
        if step % 25 == 0:
            # 当前状态分析
            C_IB = quat_to_rotmat(state.q_IB)
            C_BI = C_IB.T
            v_B = C_BI @ state.v_I  # 体轴系速度
            
            V, alpha, beta = compute_aero_angles(v_B)
            roll, pitch, yaw = quat_to_euler(state.q_IB)
            
            # 计算气动系数
            q_bar = 0.5 * params.rho * V * V
            C_L = params.c_L0 + params.c_La * alpha
            C_D = params.c_D0 + params.c_Da2 * alpha**2
            
            # 升力和阻力
            L = q_bar * params.S * C_L
            D = q_bar * params.S * C_D
            
            # 俯仰力矩系数
            q_hat = state.w_B[1] * params.c / (2.0 * max(V, params.V_min))
            C_m = params.c_m0 + params.c_ma * alpha + params.c_mq * q_hat
            M_pitch = q_bar * params.S * params.c * C_m
            
            # 重力和气动平衡
            W = params.m * params.g  # 重量
            
            print(f"t = {t:.1f}s:")
            print(f"  位置: N={state.p_I[0]:.1f}, E={state.p_I[1]:.1f}, D={state.p_I[2]:.1f}")
            print(f"  速度(I): Vn={state.v_I[0]:.2f}, Ve={state.v_I[1]:.2f}, Vd={state.v_I[2]:.2f} m/s")
            print(f"  速度(B): u={v_B[0]:.2f}, v={v_B[1]:.2f}, w={v_B[2]:.2f} m/s")
            print(f"  气速: V={V:.2f} m/s, alpha={np.degrees(alpha):.2f}°, beta={np.degrees(beta):.2f}°")
            print(f"  姿态: roll={np.degrees(roll):.1f}°, pitch={np.degrees(pitch):.1f}°, yaw={np.degrees(yaw):.1f}°")
            print(f"  角速度: p={np.degrees(state.w_B[0]):.2f}, q={np.degrees(state.w_B[1]):.2f}, r={np.degrees(state.w_B[2]):.2f} °/s")
            print(f"  气动系数: C_L={C_L:.3f}, C_D={C_D:.3f}, L/D={C_L/max(C_D,1e-6):.2f}")
            print(f"  力: L={L:.1f}N, D={D:.1f}N, W={W:.1f}N")
            print(f"  俯仰力矩: C_m={C_m:.4f}, M_pitch={M_pitch:.1f}Nm")
            print()
        
        # 积分一步
        state = integrate_with_substeps(
            dynamics, state, cmd, params, dt, dt_max,
            method=IntegratorType.RK4
        )
    
    print("="*80)
    print("最终状态:")
    print(f"  高度下降: {-state.p_I[2] - 500:.1f}m")
    print(f"  水平距离: {np.sqrt(state.p_I[0]**2 + state.p_I[1]**2):.1f}m")
    roll, pitch, yaw = quat_to_euler(state.q_IB)
    print(f"  最终姿态: roll={np.degrees(roll):.1f}°, pitch={np.degrees(pitch):.1f}°, yaw={np.degrees(yaw):.1f}°")


if __name__ == '__main__':
    debug_dynamics()
