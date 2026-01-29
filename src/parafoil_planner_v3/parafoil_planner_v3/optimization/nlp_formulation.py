from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np

from .gpm_collocation import GPMCollocation


@dataclass(frozen=True)
class GPMNLP:
    """
    Lightweight NLP formulation helper for GPM collocation.

    This class is intentionally minimal; `solver_interface.GPMSolver` is the default
    entry point for solving.
    """

    gpm: GPMCollocation
    f: Callable[[np.ndarray, np.ndarray, float], np.ndarray]

    def dynamics_defects(self, X: np.ndarray, U: np.ndarray, tf: float) -> np.ndarray:
        return self.gpm.discretize_dynamics(self.f, X, U, 0.0, float(tf))

    def time_grid(self, tf: float) -> np.ndarray:
        return np.array([self.gpm.tau_to_time(float(tau), 0.0, float(tf)) for tau in self.gpm.tau], dtype=float)

