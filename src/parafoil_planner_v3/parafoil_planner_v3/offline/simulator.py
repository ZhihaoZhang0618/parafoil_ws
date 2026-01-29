from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from parafoil_planner_v3.types import Control, State, Wind


class DynamicsModel(Protocol):
    def step(self, state: State, control: Control, wind: Wind, dt: float) -> State:  # noqa: D102
        ...


@dataclass
class OfflineSimulator:
    dynamics: DynamicsModel
    state: State
    wind: Wind

    def reset(self, state: State) -> None:
        self.state = state.copy()

    def step(self, control: Control, dt: float) -> State:
        self.state = self.dynamics.step(self.state, control, self.wind, dt=float(dt))
        return self.state

    def get_state(self) -> State:
        return self.state
