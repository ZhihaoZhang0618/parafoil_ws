from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WindFilterConfig:
    enable: bool = True
    tau_s: float = 2.0
    max_delta_mps: float = 2.0
    gust_threshold_mps: float = 3.0
    gust_hold_s: float = 1.0


class WindFilter:
    def __init__(self, config: WindFilterConfig | None = None) -> None:
        self.config = config or WindFilterConfig()
        self._filtered: np.ndarray | None = None
        self._gust_time: float | None = None

    def reset(self) -> None:
        self._filtered = None
        self._gust_time = None

    def update(self, wind_vec: np.ndarray, dt: float) -> np.ndarray:
        wind_vec = np.asarray(wind_vec, dtype=float).reshape(3)
        if not bool(self.config.enable):
            self._filtered = wind_vec.copy()
            self._gust_time = None
            return wind_vec.copy()

        if self._filtered is None:
            self._filtered = wind_vec.copy()
            self._gust_time = None
            return wind_vec.copy()

        dt = float(max(dt, 0.0))
        delta = wind_vec - self._filtered
        delta_norm = float(np.linalg.norm(delta))
        if delta_norm > float(self.config.gust_threshold_mps):
            self._gust_time = 0.0

        if self._gust_time is not None and self._gust_time < float(self.config.gust_hold_s):
            self._gust_time += dt
            wind_used = self._filtered.copy()
        else:
            self._gust_time = None
            wind_used = wind_vec

        tau = float(max(self.config.tau_s, 1e-3))
        alpha = dt / (tau + dt) if dt > 0.0 else 1.0
        filtered = self._filtered + alpha * (wind_used - self._filtered)

        max_delta = float(self.config.max_delta_mps)
        if max_delta > 0.0 and dt > 0.0:
            max_step = max_delta * dt
            diff = filtered - self._filtered
            diff_norm = float(np.linalg.norm(diff))
            if diff_norm > max_step and diff_norm > 1e-9:
                filtered = self._filtered + diff / diff_norm * max_step

        self._filtered = np.asarray(filtered, dtype=float)
        return self._filtered.copy()
