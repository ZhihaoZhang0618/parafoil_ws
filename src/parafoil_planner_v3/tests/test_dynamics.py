import numpy as np
from numpy.testing import assert_allclose

from parafoil_dynamics.dynamics import dynamics as sim_dynamics
from parafoil_dynamics.integrators import IntegratorType, integrate_with_substeps
from parafoil_dynamics.params import Params
from parafoil_dynamics.state import ControlCmd, State as SimState
from parafoil_planner_v3.dynamics.parafoil_6dof import SixDOFDynamics
from parafoil_planner_v3.types import Control, State, Wind
from parafoil_planner_v3.utils.quaternion_utils import quat_to_rpy


def test_6dof_f_vector_shape_and_finite():
    dyn = SixDOFDynamics()
    x0 = State(
        p_I=np.array([0.0, 0.0, -100.0]),
        v_I=np.array([4.5, 0.0, 0.9]),
        q_IB=np.array([1.0, 0.0, 0.0, 0.0]),
        w_B=np.zeros(3),
        t=0.0,
    ).to_vector()
    u = np.array([0.2, 0.2])
    xdot = dyn.f_vector(x0, u, 0.0, wind_I=np.zeros(3))
    assert xdot.shape == (13,)
    assert np.all(np.isfinite(xdot))


def test_6dof_step_keeps_quaternion_normalized():
    dyn = SixDOFDynamics()
    state = State(
        p_I=np.array([0.0, 0.0, -80.0]),
        v_I=np.array([4.0, 0.0, 1.0]),
        q_IB=np.array([1.0, 0.0, 0.0, 0.0]),
        w_B=np.array([0.01, 0.02, 0.03]),
        t=0.0,
    )
    nxt = dyn.step(state, Control(0.2, 0.2), Wind(np.zeros(3)), dt=0.02)
    assert np.all(np.isfinite(nxt.to_vector()))
    assert_allclose(np.linalg.norm(nxt.q_IB), 1.0, atol=1e-6)


def test_turn_response_has_expected_yaw_sign():
    dyn = SixDOFDynamics()
    wind = Wind(np.zeros(3))

    def yaw_after(control: Control) -> float:
        state = State(
            p_I=np.array([0.0, 0.0, -80.0]),
            v_I=np.array([4.5, 0.0, 0.9]),
            q_IB=np.array([1.0, 0.0, 0.0, 0.0]),
            w_B=np.zeros(3),
            t=0.0,
        )
        dt = 0.02
        for _ in range(int(1.0 / dt)):
            state = dyn.step(state, control, wind, dt=dt)
        _, _, yaw = quat_to_rpy(state.q_IB)
        return float(yaw)

    yaw_left = yaw_after(Control(0.5, 0.0))  # delta_a>0 => yaw_rate<0
    yaw_right = yaw_after(Control(0.0, 0.5))  # delta_a<0 => yaw_rate>0

    assert yaw_left < -0.02
    assert yaw_right > 0.02


def test_discretization_accuracy_compares_to_rk4_reference():
    params = Params()
    cmd = ControlCmd(delta_cmd=np.array([0.3, 0.4]))
    wind = np.array([1.0, -0.5, 0.0], dtype=float)
    state0 = SimState(
        p_I=np.array([0.0, 0.0, -100.0]),
        v_I=np.array([6.0, 0.5, 1.0]),
        q_IB=np.array([1.0, 0.0, 0.0, 0.0]),
        w_B=np.array([0.02, -0.01, 0.03]),
        delta=np.array([0.2, 0.2]),
        t=0.0,
    )

    def propagate(method: IntegratorType, dt_max: float, total_s: float = 2.0, ctl_dt: float = 0.1) -> SimState:
        state = state0.copy()
        t = 0.0
        while t < total_s - 1e-9:
            state = integrate_with_substeps(
                sim_dynamics,
                state,
                cmd,
                params,
                ctl_dt,
                dt_max,
                method,
                wind_fn=lambda _t: wind,
            )
            t += ctl_dt
        return state

    ref = propagate(IntegratorType.RK4, dt_max=0.002)
    euler = propagate(IntegratorType.EULER, dt_max=0.02)
    semi = propagate(IntegratorType.SEMI_IMPLICIT, dt_max=0.02)
    rk4 = propagate(IntegratorType.RK4, dt_max=0.02)

    err_euler = float(np.linalg.norm(euler.p_I - ref.p_I))
    err_semi = float(np.linalg.norm(semi.p_I - ref.p_I))
    err_rk4 = float(np.linalg.norm(rk4.p_I - ref.p_I))

    assert err_rk4 < err_euler
    assert err_rk4 < err_semi
    assert err_rk4 < 1.0e-3
