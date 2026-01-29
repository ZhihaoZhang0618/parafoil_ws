from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


class GuidancePhase(Enum):
    CRUISE = "cruise"
    APPROACH = "approach"
    FLARE = "flare"
    LANDED = "landed"
    ABORT = "abort"


class TrajectoryType(Enum):
    DIRECT = "DIRECT"
    S_TURN = "S_TURN"
    RACETRACK = "RACETRACK"
    SPIRAL = "SPIRAL"


@dataclass(frozen=True)
class Control:
    """Brake control (left/right) in [0, 1]."""

    delta_L: float
    delta_R: float

    def clipped(self, lo: float = 0.0, hi: float = 1.0) -> "Control":
        return Control(
            delta_L=float(np.clip(self.delta_L, lo, hi)),
            delta_R=float(np.clip(self.delta_R, lo, hi)),
        )

    @property
    def as_array(self) -> np.ndarray:
        return np.array([self.delta_L, self.delta_R], dtype=float)


@dataclass
class State:
    """
    6-DOF planning state (13D):
      [p_N, p_E, p_D, v_N, v_E, v_D, q_w, q_x, q_y, q_z, w_x, w_y, w_z]
    Frame: NED.
    Quaternion convention: [w, x, y, z].
    """

    p_I: np.ndarray  # (3,)
    v_I: np.ndarray  # (3,)
    q_IB: np.ndarray  # (4,)
    w_B: np.ndarray  # (3,)
    t: float = 0.0

    def __post_init__(self) -> None:
        self.p_I = np.asarray(self.p_I, dtype=float).reshape(3)
        self.v_I = np.asarray(self.v_I, dtype=float).reshape(3)
        self.q_IB = np.asarray(self.q_IB, dtype=float).reshape(4)
        self.w_B = np.asarray(self.w_B, dtype=float).reshape(3)

    def copy(self) -> "State":
        return State(
            p_I=self.p_I.copy(),
            v_I=self.v_I.copy(),
            q_IB=self.q_IB.copy(),
            w_B=self.w_B.copy(),
            t=float(self.t),
        )

    @property
    def position_xy(self) -> np.ndarray:
        return self.p_I[:2]

    @property
    def altitude(self) -> float:
        # NED: altitude is -Down.
        return float(-self.p_I[2])

    @property
    def speed_horizontal(self) -> float:
        return float(np.linalg.norm(self.v_I[:2]))

    def to_vector(self) -> np.ndarray:
        return np.concatenate([self.p_I, self.v_I, self.q_IB, self.w_B]).astype(float)

    @staticmethod
    def from_vector(x: Sequence[float], t: float = 0.0) -> "State":
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.shape[0] != 13:
            raise ValueError(f"Expected 13D state vector, got {x.shape[0]}")
        return State(
            p_I=x[0:3],
            v_I=x[3:6],
            q_IB=x[6:10],
            w_B=x[10:13],
            t=float(t),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "position": self.p_I.tolist(),
            "velocity": self.v_I.tolist(),
            "quaternion": self.q_IB.tolist(),
            "angular_rate": self.w_B.tolist(),
            "t": float(self.t),
        }


@dataclass(frozen=True)
class Target:
    p_I: np.ndarray  # (3,) NED

    def __post_init__(self) -> None:
        object.__setattr__(self, "p_I", np.asarray(self.p_I, dtype=float).reshape(3))

    @property
    def position_xy(self) -> np.ndarray:
        return self.p_I[:2]

    @property
    def altitude(self) -> float:
        return float(-self.p_I[2])


@dataclass(frozen=True)
class Wind:
    v_I: np.ndarray  # (3,) NED

    def __post_init__(self) -> None:
        object.__setattr__(self, "v_I", np.asarray(self.v_I, dtype=float).reshape(3))

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.v_I[:2]))

    @property
    def direction_hat_xy(self) -> np.ndarray:
        v = self.v_I[:2]
        n = float(np.linalg.norm(v))
        if n < 1e-9:
            return np.array([1.0, 0.0], dtype=float)
        return v / n


@dataclass(frozen=True)
class Waypoint:
    t: float
    state: State

    def to_dict(self) -> Dict[str, Any]:
        return {"t": float(self.t), "state": self.state.to_dict()}


@dataclass
class Trajectory:
    waypoints: List[Waypoint]
    controls: List[Control]
    trajectory_type: TrajectoryType = TrajectoryType.DIRECT
    metadata: Optional[Dict[str, Any]] = None

    @property
    def duration(self) -> float:
        if not self.waypoints:
            return 0.0
        return float(self.waypoints[-1].t - self.waypoints[0].t)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trajectory_type": self.trajectory_type.value,
            "waypoints": [w.to_dict() for w in self.waypoints],
            "controls": [{"delta_L": c.delta_L, "delta_R": c.delta_R} for c in self.controls],
            "metadata": self.metadata or {},
        }


@dataclass(frozen=True)
class PhaseTransition:
    from_phase: GuidancePhase
    to_phase: GuidancePhase
    triggered: bool
    reason: str = ""


@dataclass(frozen=True)
class Scenario:
    wind_speed: float
    wind_direction_deg: float
    initial_altitude_m: float
    target_distance_m: float
    target_bearing_deg: float

    def to_feature_vector(self) -> np.ndarray:
        return np.array(
            [
                self.initial_altitude_m,
                self.target_distance_m,
                np.deg2rad(self.target_bearing_deg),
                self.wind_speed,
                np.deg2rad(self.wind_direction_deg),
            ],
            dtype=float,
        )

