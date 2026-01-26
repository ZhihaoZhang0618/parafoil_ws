"""
Numerical integrators for parafoil dynamics.

Supports:
- Euler integration
- Semi-implicit Euler integration
- 4th-order Runge-Kutta (RK4)
- Sub-stepping with dt_max (Flightmare style)

All integrators are decoupled from the dynamics function,
taking a generic dynamics callable.
"""

import numpy as np
from typing import Callable, Optional, Literal
from enum import Enum

from .state import State, StateDot, ControlCmd
from .params import Params
from .math3d import normalize_quaternion


class IntegratorType(Enum):
    """Available integrator types."""
    EULER = "euler"
    SEMI_IMPLICIT = "semi_implicit"
    RK4 = "rk4"


def euler_step(
    dynamics_fn: Callable[[State, ControlCmd, Params, Optional[np.ndarray]], StateDot],
    state: State,
    cmd: ControlCmd,
    params: Params,
    dt: float,
    wind_I: Optional[np.ndarray] = None
) -> State:
    """
    Explicit Euler integration step.
    
    x_{n+1} = x_n + dt * f(x_n, u)
    
    Args:
        dynamics_fn: Dynamics function that computes state derivative
        state: Current state
        cmd: Control command
        params: Simulation parameters
        dt: Time step [s]
        wind_I: Wind velocity in inertial frame
        
    Returns:
        New state after integration step
    """
    # Compute derivative
    state_dot = dynamics_fn(state, cmd, params, wind_I)
    
    # Euler update
    new_state = State(
        p_I=state.p_I + dt * state_dot.p_I_dot,
        v_I=state.v_I + dt * state_dot.v_I_dot,
        q_IB=state.q_IB + dt * state_dot.q_IB_dot,
        w_B=state.w_B + dt * state_dot.w_B_dot,
        delta=state.delta + dt * state_dot.delta_dot,
        t=state.t + dt
    )
    
    # Normalize quaternion
    new_state.q_IB = normalize_quaternion(new_state.q_IB)
    
    # Clamp actuator state
    new_state.delta = np.clip(new_state.delta, 0.0, 1.0)
    
    # Check for finite values
    if not new_state.is_finite():
        raise RuntimeError(f"Non-finite state detected after Euler step at t={state.t}")
    
    return new_state


def semi_implicit_step(
    dynamics_fn: Callable[[State, ControlCmd, Params, Optional[np.ndarray]], StateDot],
    state: State,
    cmd: ControlCmd,
    params: Params,
    dt: float,
    wind_I: Optional[np.ndarray] = None
) -> State:
    """
    Semi-implicit Euler integration step.
    
    Update order (symplectic-style):
    1. Compute derivatives at current state
    2. Update velocities first: v_{n+1} = v_n + dt * v_dot
    3. Update positions using new velocities: p_{n+1} = p_n + dt * v_{n+1}
    
    This provides better energy conservation for oscillatory systems.
    
    Args:
        dynamics_fn: Dynamics function
        state: Current state
        cmd: Control command
        params: Simulation parameters
        dt: Time step [s]
        wind_I: Wind velocity
        
    Returns:
        New state after integration step
    """
    # Compute derivative at current state
    state_dot = dynamics_fn(state, cmd, params, wind_I)
    
    # Update velocities first (including angular velocity)
    new_v_I = state.v_I + dt * state_dot.v_I_dot
    new_w_B = state.w_B + dt * state_dot.w_B_dot
    new_delta = state.delta + dt * state_dot.delta_dot
    
    # Update quaternion (using new angular velocity for better stability)
    # q_dot = 0.5 * q ⊗ [0, w_B]
    from .math3d import quat_derivative
    q_dot_new = quat_derivative(state.q_IB, new_w_B)
    new_q_IB = state.q_IB + dt * q_dot_new
    new_q_IB = normalize_quaternion(new_q_IB)
    
    # Update position using new velocity
    new_p_I = state.p_I + dt * new_v_I
    
    # Create new state
    new_state = State(
        p_I=new_p_I,
        v_I=new_v_I,
        q_IB=new_q_IB,
        w_B=new_w_B,
        delta=np.clip(new_delta, 0.0, 1.0),
        t=state.t + dt
    )
    
    # Check for finite values
    if not new_state.is_finite():
        raise RuntimeError(f"Non-finite state detected after semi-implicit step at t={state.t}")
    
    return new_state


def rk4_step(
    dynamics_fn: Callable[[State, ControlCmd, Params, Optional[np.ndarray]], StateDot],
    state: State,
    cmd: ControlCmd,
    params: Params,
    dt: float,
    wind_I: Optional[np.ndarray] = None
) -> State:
    """
    4th-order Runge-Kutta integration step.
    
    Classic RK4 with 4 function evaluations per step.
    Provides O(dt^5) local error.
    
    Args:
        dynamics_fn: Dynamics function
        state: Current state
        cmd: Control command
        params: Simulation parameters
        dt: Time step [s]
        wind_I: Wind velocity
        
    Returns:
        New state after integration step
    """
    # k1 = f(x_n)
    k1 = dynamics_fn(state, cmd, params, wind_I)
    
    # k2 = f(x_n + 0.5*dt*k1)
    state_k2 = _apply_derivative(state, k1, 0.5 * dt)
    k2 = dynamics_fn(state_k2, cmd, params, wind_I)
    
    # k3 = f(x_n + 0.5*dt*k2)
    state_k3 = _apply_derivative(state, k2, 0.5 * dt)
    k3 = dynamics_fn(state_k3, cmd, params, wind_I)
    
    # k4 = f(x_n + dt*k3)
    state_k4 = _apply_derivative(state, k3, dt)
    k4 = dynamics_fn(state_k4, cmd, params, wind_I)
    
    # Weighted average: x_{n+1} = x_n + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    new_state = State(
        p_I=state.p_I + (dt / 6.0) * (
            k1.p_I_dot + 2*k2.p_I_dot + 2*k3.p_I_dot + k4.p_I_dot
        ),
        v_I=state.v_I + (dt / 6.0) * (
            k1.v_I_dot + 2*k2.v_I_dot + 2*k3.v_I_dot + k4.v_I_dot
        ),
        q_IB=state.q_IB + (dt / 6.0) * (
            k1.q_IB_dot + 2*k2.q_IB_dot + 2*k3.q_IB_dot + k4.q_IB_dot
        ),
        w_B=state.w_B + (dt / 6.0) * (
            k1.w_B_dot + 2*k2.w_B_dot + 2*k3.w_B_dot + k4.w_B_dot
        ),
        delta=state.delta + (dt / 6.0) * (
            k1.delta_dot + 2*k2.delta_dot + 2*k3.delta_dot + k4.delta_dot
        ),
        t=state.t + dt
    )
    
    # Normalize quaternion
    new_state.q_IB = normalize_quaternion(new_state.q_IB)
    
    # Clamp actuator state
    new_state.delta = np.clip(new_state.delta, 0.0, 1.0)
    
    # Check for finite values
    if not new_state.is_finite():
        raise RuntimeError(f"Non-finite state detected after RK4 step at t={state.t}")
    
    return new_state


def _apply_derivative(state: State, state_dot: StateDot, dt: float) -> State:
    """
    Apply a state derivative to create an intermediate state.
    Used internally by RK4.
    """
    new_q = state.q_IB + dt * state_dot.q_IB_dot
    new_q = normalize_quaternion(new_q)
    
    return State(
        p_I=state.p_I + dt * state_dot.p_I_dot,
        v_I=state.v_I + dt * state_dot.v_I_dot,
        q_IB=new_q,
        w_B=state.w_B + dt * state_dot.w_B_dot,
        delta=np.clip(state.delta + dt * state_dot.delta_dot, 0.0, 1.0),
        t=state.t + dt
    )


def integrate_with_substeps(
    dynamics_fn: Callable[[State, ControlCmd, Params, Optional[np.ndarray]], StateDot],
    state: State,
    cmd: ControlCmd,
    params: Params,
    ctl_dt: float,
    dt_max: float,
    method: IntegratorType = IntegratorType.RK4,
    wind_fn: Optional[Callable[[float], np.ndarray]] = None
) -> State:
    """
    Integrate over a control period with sub-stepping for stability.
    
    This is the Flightmare-style integration approach:
    - Control command is held constant over ctl_dt
    - Internal integration uses smaller steps of at most dt_max
    - Ensures numerical stability even with stiff dynamics
    
    Args:
        dynamics_fn: Dynamics function
        state: Current state
        cmd: Control command (held constant over ctl_dt)
        params: Simulation parameters
        ctl_dt: Control period [s]
        dt_max: Maximum sub-step size [s]
        method: Integration method to use
        wind_fn: Optional function that returns wind given time t
        
    Returns:
        New state after integration over ctl_dt
    """
    # Select integrator
    if method == IntegratorType.EULER:
        step_fn = euler_step
    elif method == IntegratorType.SEMI_IMPLICIT:
        step_fn = semi_implicit_step
    elif method == IntegratorType.RK4:
        step_fn = rk4_step
    else:
        raise ValueError(f"Unknown integrator type: {method}")
    
    # Compute number of sub-steps
    n_steps = max(1, int(np.ceil(ctl_dt / dt_max)))
    dt_sub = ctl_dt / n_steps
    
    # Integrate
    current_state = state
    for _ in range(n_steps):
        # Get wind at current time
        if wind_fn is not None:
            wind_I = wind_fn(current_state.t)
        else:
            wind_I = None
        
        # Take integration step
        current_state = step_fn(
            dynamics_fn, current_state, cmd, params, dt_sub, wind_I
        )
    
    return current_state


def get_integrator_step(
    method: IntegratorType
) -> Callable[[Callable, State, ControlCmd, Params, float, Optional[np.ndarray]], State]:
    """
    Get the step function for a given integrator type.
    
    Args:
        method: Integration method
        
    Returns:
        Step function
    """
    if method == IntegratorType.EULER:
        return euler_step
    elif method == IntegratorType.SEMI_IMPLICIT:
        return semi_implicit_step
    elif method == IntegratorType.RK4:
        return rk4_step
    else:
        raise ValueError(f"Unknown integrator type: {method}")


def parse_integrator_type(name: str) -> IntegratorType:
    """
    Parse integrator type from string.
    
    Args:
        name: Integrator name string
        
    Returns:
        IntegratorType enum value
    """
    name_lower = name.lower().strip()
    
    if name_lower in ["euler", "explicit_euler"]:
        return IntegratorType.EULER
    elif name_lower in ["semi_implicit", "semi-implicit", "symplectic"]:
        return IntegratorType.SEMI_IMPLICIT
    elif name_lower in ["rk4", "runge_kutta", "runge-kutta"]:
        return IntegratorType.RK4
    else:
        raise ValueError(
            f"Unknown integrator type: {name}. "
            f"Valid options: euler, semi_implicit, rk4"
        )
