from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from parafoil_planner_v3.types import Scenario, Trajectory, TrajectoryType, Wind
from parafoil_planner_v3.utils.quaternion_utils import wrap_pi


@dataclass(frozen=True)
class LibraryTrajectory:
    scenario: Scenario
    trajectory_type: TrajectoryType
    trajectory: Trajectory
    cost: float = 0.0
    metadata: Optional[Dict] = None


class TrajectoryLibrary:
    def __init__(self, trajectories: Optional[Iterable[LibraryTrajectory]] = None) -> None:
        self._trajectories: List[LibraryTrajectory] = list(trajectories or [])
        self._kdtree: Optional[cKDTree] = None
        self._features: Optional[np.ndarray] = None
        self._feature_weights: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self._trajectories)

    def __getitem__(self, idx: int) -> LibraryTrajectory:
        return self._trajectories[idx]

    @staticmethod
    def _scenario_to_features(s: Scenario) -> np.ndarray:
        bearing = float(np.deg2rad(s.target_bearing_deg))
        wind_dir = float(np.deg2rad(s.wind_direction_deg))
        rel_wind = wrap_pi(wind_dir - bearing)
        return np.array(
            [s.initial_altitude_m, s.target_distance_m, bearing, s.wind_speed, rel_wind],
            dtype=float,
        )

    def build_index(self, feature_weights: Optional[Dict[str, float]] = None) -> None:
        if not self._trajectories:
            self._kdtree = None
            self._features = None
            return

        weights = feature_weights or {
            "altitude": 0.2,
            "distance": 0.25,
            "bearing": 0.15,
            "wind_speed": 0.2,
            "wind_angle": 0.2,
        }
        w = np.array(
            [
                float(weights["altitude"]),
                float(weights["distance"]),
                float(weights["bearing"]),
                float(weights["wind_speed"]),
                float(weights["wind_angle"]),
            ],
            dtype=float,
        )
        self._feature_weights = w

        feats = np.stack([self._scenario_to_features(t.scenario) for t in self._trajectories], axis=0)
        self._features = feats
        scaled = feats * w
        self._kdtree = cKDTree(scaled)

    def query_knn(self, features: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        if self._kdtree is None or self._feature_weights is None:
            raise RuntimeError("TrajectoryLibrary index not built")
        f = np.asarray(features, dtype=float).reshape(5)
        dist, idx = self._kdtree.query(f * self._feature_weights, k=min(int(k), len(self._trajectories)))
        return np.asarray(dist), np.asarray(idx, dtype=int)

    def save(self, path: str) -> None:
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(path_obj, "wb") as f:
            pickle.dump(self._trajectories, f)

    @staticmethod
    def load(path: str) -> "TrajectoryLibrary":
        with open(path, "rb") as f:
            trajectories = pickle.load(f)
        lib = TrajectoryLibrary(trajectories)
        lib.build_index()
        return lib

