"""
Core parafoil 6DoF dynamics.

This module implements the explicit differential equation:
    x_dot = f(x, u, params, wind)

Physics includes:
- Quasi-static aerodynamics (lift, drag, side force)
- Aerodynamic moments (roll, pitch, yaw with damping)
- Payload drag with moment contribution
- Actuator first-order lag
- 6DoF rigid body dynamics

Coordinate system: NED (North-East-Down)
"""

import numpy as np
from typing import Optional, Callable

from .state import State, StateDot, ControlCmd
from .params import Params
from .math3d import (
    quat_to_rotmat, quat_derivative, normalize_quaternion,
    cross, compute_aero_angles
)


def dynamics(
    state: State,
    cmd: ControlCmd,
    params: Params,
    wind_I: Optional[np.ndarray] = None
) -> StateDot:
    """
    Compute the time derivative of the parafoil state.
    
    This is the core dynamics function: x_dot = f(x, u)
    
    Args:
        state: Current state
        cmd: Control command
        params: Simulation parameters
        wind_I: Wind velocity in inertial frame [m/s], default zeros
        
    Returns:
        StateDot: Time derivative of state
    """
    if wind_I is None:
        wind_I = np.zeros(3)
    
    # Extract state
    v_I = state.v_I
    q_IB = normalize_quaternion(state.q_IB)
    w_B = state.w_B
    delta = np.clip(state.delta, 0.0, 1.0)  # Clamp actuator state
    
    # Rotation matrix: B -> I
    C_IB = quat_to_rotmat(q_IB)
    C_BI = C_IB.T  # I -> B
    
    # Compute relative velocity (subtract wind)
    v_rel_I = v_I - wind_I
    v_rel_B = C_BI @ v_rel_I
    
    # Compute aerodynamic angles with protection
    V, alpha, beta = compute_aero_angles(v_rel_B, eps=params.eps)
    V_safe = max(V, params.V_min)
    
    # Control decomposition
    delta_s = (delta[0] + delta[1]) / 2.0  # Symmetric brake
    # Asymmetric brake: delta_a = delta_L - delta_R
    # Convention: positive delta_a (more left brake) produces negative yaw (turn left)
    # This matches real parafoil behavior: pulling left brake increases left side drag
    delta_a = delta[0] - delta[1]
    
    # -----------------------------------------------------------------
    # Aerodynamic forces (with stall model)
    # -----------------------------------------------------------------
    q_bar = 0.5 * params.rho * V_safe * V_safe  # Dynamic pressure

    # Linear lift coefficient: C_L = c_L0 + c_La * alpha + c_Lds * delta_s
    C_L_linear = params.c_L0 + params.c_La * alpha + params.c_Lds * delta_s

    # Stall model: lift drops and drag increases beyond stall angle
    # Stall angle decreases with brake (heavy brake = earlier stall)
    alpha_stall = params.alpha_stall - params.alpha_stall_brake * delta_s

    if alpha > alpha_stall:
        # Post-stall: lift drops smoothly, drag increases sharply
        stall_ratio = (alpha - alpha_stall) / params.alpha_stall_width
        stall_factor = np.exp(-stall_ratio * stall_ratio)  # Gaussian drop

        # Lift drops to ~30% of max in deep stall
        C_L = C_L_linear * (0.3 + 0.7 * stall_factor)

        # Drag increases significantly in stall
        C_D_stall_add = params.c_D_stall * (1.0 - stall_factor)
        C_D = params.c_D0 + params.c_Da2 * alpha * alpha + params.c_Dds * delta_s + C_D_stall_add
    else:
        C_L = C_L_linear
        # Drag coefficient: C_D = c_D0 + c_Da2 * alpha^2 + c_Dds * delta_s
        C_D = params.c_D0 + params.c_Da2 * alpha * alpha + params.c_Dds * delta_s
    
    # Side force coefficient: C_Y = c_Yb * beta
    C_Y = params.c_Yb * beta
    
    # Forces in stability/wind frame
    L = q_bar * params.S * C_L  # Lift (perpendicular to v_rel)
    D = q_bar * params.S * C_D  # Drag (opposite to v_rel)
    Y = q_bar * params.S * C_Y  # Side force
    
    # Convert to body frame forces
    # In body frame: x=forward, y=right, z=down
    # Lift acts perpendicular to velocity (in x-z plane), opposite to z component
    # Drag acts opposite to velocity direction
    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)
    
    # Aerodynamic force in body frame
    # F_aero_B = [-D*cos(alpha) + L*sin(alpha), Y, -D*sin(alpha) - L*cos(alpha)]
    F_aero_B = np.array([
        -D * cos_alpha + L * sin_alpha,
        Y,
        -D * sin_alpha - L * cos_alpha
    ])
    
    # -----------------------------------------------------------------
    # Aerodynamic moments
    # -----------------------------------------------------------------
    # Non-dimensional angular rates
    p, q, r = w_B[0], w_B[1], w_B[2]
    
    # Protected non-dimensional rates
    p_hat = p * params.b / (2.0 * V_safe)
    q_hat = q * params.c / (2.0 * V_safe)
    r_hat = r * params.b / (2.0 * V_safe)
    
    # Roll moment coefficient
    # C_l = c_lp * p_hat + c_lda * delta_a + c_lb * beta
    # c_lb provides dihedral effect (roll due to sideslip)
    C_l = params.c_lp * p_hat + params.c_lda * delta_a + params.c_lb * beta
    
    # Pitch moment coefficient
    # C_m = c_m0 + c_ma * alpha + c_mq * q_hat
    C_m = params.c_m0 + params.c_ma * alpha + params.c_mq * q_hat
    
    # Yaw moment coefficient
    # C_n = c_nr * r_hat + c_nda * delta_a + c_nb * beta + c_n_weath * wind_y
    # c_nb models wind-yaw coupling: sideslip (from crosswind) produces yaw moment
    # c_n_weath models weathercock: direct crosswind creates destabilizing yaw moment
    C_n = params.c_nr * r_hat + params.c_nda * delta_a + params.c_nb * beta

    # Weathercock effect: yaw moment from crosswind relative to heading (not velocity)
    # This creates an unstable equilibrium when flying into headwind
    # wind_B[1] is the wind component in body Y direction (crosswind)
    wind_B = C_BI @ wind_I
    wind_y_normalized = wind_B[1] / V_safe  # Normalize by airspeed
    C_n += params.c_n_weath * wind_y_normalized
    
    # Moments in body frame
    M_aero_B = q_bar * params.S * np.array([
        C_l * params.b,
        C_m * params.c,
        C_n * params.b
    ])
    
    # -----------------------------------------------------------------
    # Payload drag
    # -----------------------------------------------------------------
    v_rel_norm = np.linalg.norm(v_rel_B)
    if v_rel_norm > params.eps:
        # Payload drag force (opposite to relative velocity)
        F_pd_B = -0.5 * params.rho * params.c_D_pd * params.S_pd * v_rel_norm * v_rel_B
        
        # TEMPORARILY DISABLED: Payload drag moment can cause instability
        # when the payload offset is large relative to the moment of inertia
        # M_pd_B = cross(params.r_pd_B, F_pd_B)
        M_pd_B = np.zeros(3)
    else:
        F_pd_B = np.zeros(3)
        M_pd_B = np.zeros(3)
    
    # -----------------------------------------------------------------
    # Gravity
    # -----------------------------------------------------------------
    # In NED: gravity is [0, 0, +g]
    g_I = np.array([0.0, 0.0, params.g])
    
    # -----------------------------------------------------------------
    # Pendulum restoring moment (payload gravity effect)
    # -----------------------------------------------------------------
    # When the canopy tilts, the payload weight creates a restoring moment
    # The payload is suspended L meters below the canopy
    # M_pendulum = r_payload × F_gravity_payload (in body frame)
    # For small angles: M_pendulum ≈ -m_payload * g * L * [sin(roll), sin(pitch), 0]
    
    # Get roll and pitch from quaternion
    q = q_IB
    roll_approx = 2.0 * (q[0]*q[1] + q[2]*q[3])  # sin(roll) for small angles
    pitch_approx = 2.0 * (q[0]*q[2] - q[1]*q[3])  # sin(pitch) for small angles
    
    # Pendulum restoring moment (provides roll/pitch stability)
    L = params.line_length
    k_pendulum = params.m_payload * params.g * L
    M_pendulum_B = np.array([
        -k_pendulum * roll_approx,   # Roll restoring
        -k_pendulum * pitch_approx,  # Pitch restoring
        0.0                          # No yaw restoring from pendulum
    ])
    
    # -----------------------------------------------------------------
    # Total forces and moments
    # -----------------------------------------------------------------
    # Total force in body frame
    F_total_B = F_aero_B + F_pd_B
    
    # Transform to inertial frame and add gravity
    F_total_I = C_IB @ F_total_B + params.m * g_I
    
    # Total moment in body frame
    M_total_B = M_aero_B + M_pd_B + M_pendulum_B
    
    # -----------------------------------------------------------------
    # State derivatives - Single rigid body model
    # -----------------------------------------------------------------
    # Use full system inertia for all rotations to ensure coordinated turning.
    # Yaw rate must track flight direction change.
    
    # Position derivative: p_dot = v
    p_I_dot = v_I
    
    # Velocity derivative: v_dot = F/m
    v_I_dot = F_total_I / params.m
    
    # Quaternion derivative: q_dot = 0.5 * q ⊗ [0, w_B]
    q_IB_dot = quat_derivative(q_IB, w_B)
    
    # Angular velocity derivative - single rigid body with system inertia
    # w_dot = I_inv @ (M_total - w × Iw)
    Iw = params.I_B @ w_B
    w_B_dot = params.I_B_inv @ (M_total_B - cross(w_B, Iw))
    
    # Actuator dynamics: delta_dot = (delta_cmd - delta) / tau_act
    delta_cmd_clamped = np.clip(cmd.delta_cmd, 0.0, 1.0)
    delta_dot = (delta_cmd_clamped - delta) / params.tau_act
    
    return StateDot(
        p_I_dot=p_I_dot,
        v_I_dot=v_I_dot,
        q_IB_dot=q_IB_dot,
        w_B_dot=w_B_dot,
        delta_dot=delta_dot
    )


def dynamics_array(
    x: np.ndarray,
    cmd: ControlCmd,
    params: Params,
    wind_I: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Array-based dynamics function for use with integrators.
    
    Args:
        x: Flattened state array
        cmd: Control command
        params: Simulation parameters
        wind_I: Wind velocity in inertial frame
        
    Returns:
        Flattened state derivative array
    """
    state = State.from_array(x)
    state_dot = dynamics(state, cmd, params, wind_I)
    return state_dot.to_array()


def get_body_acceleration(
    state: State,
    cmd: ControlCmd,
    params: Params,
    wind_I: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute acceleration in body frame (for sensor simulation).
    
    This returns the specific force (acceleration minus gravity) 
    as would be measured by an accelerometer.
    
    Args:
        state: Current state
        cmd: Control command
        params: Simulation parameters
        wind_I: Wind velocity
        
    Returns:
        Body-frame acceleration [m/s^2]
    """
    state_dot = dynamics(state, cmd, params, wind_I)
    
    # Get rotation matrix
    C_IB = quat_to_rotmat(state.q_IB)
    C_BI = C_IB.T
    
    # Gravity in inertial frame (NED)
    g_I = np.array([0.0, 0.0, params.g])
    
    # Specific force = total acceleration - gravity
    a_I = state_dot.v_I_dot
    specific_force_I = a_I - g_I
    
    # Transform to body frame
    specific_force_B = C_BI @ specific_force_I
    
    return specific_force_B


def compute_glide_ratio(state: State, params: Params) -> float:
    """
    Compute current glide ratio (horizontal distance / vertical descent).
    
    Args:
        state: Current state
        params: Simulation parameters
        
    Returns:
        Glide ratio (positive when descending)
    """
    v_horiz = np.sqrt(state.v_I[0]**2 + state.v_I[1]**2)
    v_vert = state.v_I[2]  # Positive when going down (NED)
    
    if v_vert > params.eps:
        return v_horiz / v_vert
    else:
        return float('inf')


def compute_turn_rate(state: State) -> float:
    """
    Compute current turn rate (yaw rate) in rad/s.
    
    Args:
        state: Current state
        
    Returns:
        Turn rate [rad/s]
    """
    # Get yaw component of body angular velocity
    C_IB = quat_to_rotmat(state.q_IB)
    w_I = C_IB @ state.w_B
    return w_I[2]  # Yaw rate in inertial frame
