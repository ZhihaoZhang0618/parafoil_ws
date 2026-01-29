from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from parafoil_dynamics.dynamics import dynamics as sim_dynamics
from parafoil_dynamics.integrators import IntegratorType, integrate_with_substeps
from parafoil_dynamics.params import Params
from parafoil_dynamics.state import ControlCmd, State as SimState

from parafoil_planner_v3.types import Control, State, Wind


@dataclass(frozen=True)
class IntegratorConfig:
    method: IntegratorType = IntegratorType.RK4
    dt_max: float = 0.01


class SixDOFDynamics:
    """
    6-DOF parafoil dynamics wrapper.

    Internally reuses `parafoil_dynamics` (same model as the simulator) and exposes:
      - f_vector(x,u,t): 13D state derivative (actuator assumed instantaneous)
      - step(state, control, wind, dt): integrate forward
    """

    def __init__(self, params: Optional[Params] = None, integrator: Optional[IntegratorConfig] = None) -> None:
        self.params = params or Params()
        self.integrator = integrator or IntegratorConfig()

    @staticmethod
    def _to_sim_state(state: State, control: Control) -> SimState:
        # Actuator assumed instantaneous for planning: delta == commanded.
        delta = control.clipped().as_array
        return SimState(
            p_I=state.p_I,
            v_I=state.v_I,
            q_IB=state.q_IB,
            w_B=state.w_B,
            delta=delta,
            t=float(state.t),
        )

    @staticmethod
    def _from_sim_state(sim_state: SimState) -> State:
        return State(
            p_I=sim_state.p_I,
            v_I=sim_state.v_I,
            q_IB=sim_state.q_IB,
            w_B=sim_state.w_B,
            t=float(sim_state.t),
        )

    def f_vector(self, x: np.ndarray, u: np.ndarray, t: float, wind_I: Optional[np.ndarray] = None) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(13)
        u = np.asarray(u, dtype=float).reshape(2)
        state = State.from_vector(x, t=float(t))
        control = Control(float(u[0]), float(u[1])).clipped()
        sim_state = self._to_sim_state(state, control)
        cmd = ControlCmd(delta_cmd=control.as_array)
        state_dot = sim_dynamics(sim_state, cmd, self.params, wind_I=wind_I)
        # Return 13D derivative (ignore actuator delta_dot).
        x_dot = np.concatenate(
            [
                state_dot.p_I_dot,
                state_dot.v_I_dot,
                state_dot.q_IB_dot,
                state_dot.w_B_dot,
            ],
            axis=0,
        )
        return np.asarray(x_dot, dtype=float).reshape(13)

    def step(self, state: State, control: Control, wind: Wind, dt: float) -> State:
        sim_state = self._to_sim_state(state, control)
        cmd = ControlCmd(delta_cmd=control.clipped().as_array)

        wind_I = wind.v_I
        wind_fn = (lambda _t: wind_I) if wind is not None else None
        sim_next = integrate_with_substeps(
            sim_dynamics,
            sim_state,
            cmd,
            self.params,
            ctl_dt=float(dt),
            dt_max=float(self.integrator.dt_max),
            method=self.integrator.method,
            wind_fn=wind_fn,
        )
        return self._from_sim_state(sim_next)

