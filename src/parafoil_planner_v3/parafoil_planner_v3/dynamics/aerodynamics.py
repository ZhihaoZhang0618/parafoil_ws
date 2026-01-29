from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

from parafoil_planner_v3.utils.interpolation import PiecewiseLinear


@dataclass(frozen=True)
class PolarTable:
    """
    Simple polar table for the current simulator parameter set.

    Data points are taken from the workspace root README (steady-state sims).
    """

    brake: np.ndarray = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=float)
    airspeed: np.ndarray = np.array([4.44, 4.19, 3.97, 3.78, 3.61, 3.47, 3.33, 3.22, 3.11, 3.01, 2.92], dtype=float)
    sink: np.ndarray = np.array([0.90, 1.03, 1.13, 1.20, 1.26, 1.30, 1.33, 1.36, 1.39, 1.40, 1.42], dtype=float)
    _airspeed_interp: PiecewiseLinear = field(init=False, repr=False)
    _sink_interp: PiecewiseLinear = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "brake", np.asarray(self.brake, dtype=float).reshape(-1))
        object.__setattr__(self, "airspeed", np.asarray(self.airspeed, dtype=float).reshape(-1))
        object.__setattr__(self, "sink", np.asarray(self.sink, dtype=float).reshape(-1))
        object.__setattr__(self, "_airspeed_interp", PiecewiseLinear.from_points(self.brake, self.airspeed))
        object.__setattr__(self, "_sink_interp", PiecewiseLinear.from_points(self.brake, self.sink))

    def interpolate(self, brake: float) -> Tuple[float, float]:
        b = float(np.clip(brake, float(self.brake[0]), float(self.brake[-1])))
        V = self._airspeed_interp.eval(b)
        w = self._sink_interp.eval(b)
        return float(V), float(w)

    def slope(self, brake: float) -> float:
        """Vertical/horizontal slope (sink / airspeed) in still air."""
        V, w = self.interpolate(brake)
        return float(w / max(V, 1e-6))

    def select_brake_for_required_slope(self, k_req: float) -> float:
        """
        Select symmetric brake that best matches required descent slope k_req = H/D.

        This uses still-air slope (sink/airspeed) as a conservative proxy.
        """
        k_req = float(max(k_req, 0.0))
        ks = np.array([self.slope(b) for b in self.brake], dtype=float)
        idx = int(np.argmin(np.abs(ks - k_req)))
        return float(self.brake[idx])
