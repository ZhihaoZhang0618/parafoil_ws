from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def lerp(a: float, b: float, t: float) -> float:
    return float((1.0 - t) * a + t * b)


def lerp_vec(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return (1.0 - t) * a + t * b


@dataclass(frozen=True)
class PiecewiseLinear:
    xs: np.ndarray
    ys: np.ndarray

    @staticmethod
    def from_points(xs: Sequence[float], ys: Sequence[float]) -> "PiecewiseLinear":
        xs_arr = np.asarray(xs, dtype=float).reshape(-1)
        ys_arr = np.asarray(ys, dtype=float).reshape(-1)
        if xs_arr.shape != ys_arr.shape:
            raise ValueError("xs and ys must have same length")
        if xs_arr.size < 2:
            raise ValueError("need at least 2 points")
        if np.any(np.diff(xs_arr) <= 0):
            raise ValueError("xs must be strictly increasing")
        return PiecewiseLinear(xs=xs_arr, ys=ys_arr)

    def eval(self, x: float) -> float:
        x = float(x)
        if x <= float(self.xs[0]):
            return float(self.ys[0])
        if x >= float(self.xs[-1]):
            return float(self.ys[-1])
        i = int(np.searchsorted(self.xs, x, side="right") - 1)
        x0 = float(self.xs[i])
        x1 = float(self.xs[i + 1])
        y0 = float(self.ys[i])
        y1 = float(self.ys[i + 1])
        t = 0.0 if abs(x1 - x0) < 1e-12 else (x - x0) / (x1 - x0)
        return lerp(y0, y1, t)

