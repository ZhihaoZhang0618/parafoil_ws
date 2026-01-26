"""
State and Control Command definitions for parafoil simulation.

Coordinate system: NED (North-East-Down)
- x = North, y = East, z = Down
- Gravity is +9.81 along z-positive
- Altitude h = -z
- Ground contact when z > 0
"""

from dataclasses import dataclass, field
import numpy as np
from typing import Optional


@dataclass
class State:
    """
    Parafoil 6DoF state in NED coordinate system.
    
    Attributes:
        p_I: Position in inertial (NED) frame [m], shape (3,)
        v_I: Velocity in inertial (NED) frame [m/s], shape (3,)
        q_IB: Quaternion representing B->I rotation, format [w, x, y, z], shape (4,)
        w_B: Angular velocity in body frame [rad/s], shape (3,)
        delta: Actuator internal state [delta_l, delta_r], range [0,1], shape (2,)
        t: Current simulation time [s]
    """
    p_I: np.ndarray = field(default_factory=lambda: np.zeros(3))
    v_I: np.ndarray = field(default_factory=lambda: np.array([10.0, 0.0, 2.0]))
    q_IB: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    w_B: np.ndarray = field(default_factory=lambda: np.zeros(3))
    delta: np.ndarray = field(default_factory=lambda: np.zeros(2))
    t: float = 0.0
    
    def __post_init__(self):
        """Ensure all arrays are numpy arrays with correct dtype."""
        self.p_I = np.asarray(self.p_I, dtype=np.float64)
        self.v_I = np.asarray(self.v_I, dtype=np.float64)
        self.q_IB = np.asarray(self.q_IB, dtype=np.float64)
        self.w_B = np.asarray(self.w_B, dtype=np.float64)
        self.delta = np.asarray(self.delta, dtype=np.float64)
    
    def copy(self) -> 'State':
        """Create a deep copy of the state."""
        return State(
            p_I=self.p_I.copy(),
            v_I=self.v_I.copy(),
            q_IB=self.q_IB.copy(),
            w_B=self.w_B.copy(),
            delta=self.delta.copy(),
            t=self.t
        )
    
    def to_array(self) -> np.ndarray:
        """Flatten state to a 1D array for integration."""
        return np.concatenate([
            self.p_I,       # 0:3
            self.v_I,       # 3:6
            self.q_IB,      # 6:10
            self.w_B,       # 10:13
            self.delta,     # 13:15
            [self.t]        # 15
        ])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'State':
        """Construct state from a 1D array."""
        return cls(
            p_I=arr[0:3].copy(),
            v_I=arr[3:6].copy(),
            q_IB=arr[6:10].copy(),
            w_B=arr[10:13].copy(),
            delta=arr[13:15].copy(),
            t=float(arr[15])
        )
    
    @property
    def altitude(self) -> float:
        """Get altitude (h = -z in NED)."""
        return -self.p_I[2]
    
    @property
    def is_on_ground(self) -> bool:
        """Check if parafoil has landed (z > 0 means below ground)."""
        return self.p_I[2] > 0.0
    
    def is_finite(self) -> bool:
        """Check if all state values are finite (no NaN or Inf)."""
        return (
            np.all(np.isfinite(self.p_I)) and
            np.all(np.isfinite(self.v_I)) and
            np.all(np.isfinite(self.q_IB)) and
            np.all(np.isfinite(self.w_B)) and
            np.all(np.isfinite(self.delta)) and
            np.isfinite(self.t)
        )


@dataclass
class StateDot:
    """
    Time derivative of the parafoil state.
    
    Attributes:
        p_I_dot: Velocity in inertial frame [m/s]
        v_I_dot: Acceleration in inertial frame [m/s^2]
        q_IB_dot: Quaternion derivative
        w_B_dot: Angular acceleration in body frame [rad/s^2]
        delta_dot: Actuator rate [1/s]
    """
    p_I_dot: np.ndarray = field(default_factory=lambda: np.zeros(3))
    v_I_dot: np.ndarray = field(default_factory=lambda: np.zeros(3))
    q_IB_dot: np.ndarray = field(default_factory=lambda: np.zeros(4))
    w_B_dot: np.ndarray = field(default_factory=lambda: np.zeros(3))
    delta_dot: np.ndarray = field(default_factory=lambda: np.zeros(2))
    
    def to_array(self) -> np.ndarray:
        """Flatten state derivative to a 1D array."""
        return np.concatenate([
            self.p_I_dot,       # 0:3
            self.v_I_dot,       # 3:6
            self.q_IB_dot,      # 6:10
            self.w_B_dot,       # 10:13
            self.delta_dot,     # 13:15
            [1.0]               # 15: dt/dt = 1
        ])


@dataclass
class ControlCmd:
    """
    Control command for parafoil.
    
    Attributes:
        delta_cmd: Commanded left/right brake [delta_l_cmd, delta_r_cmd], range [0,1]
    """
    delta_cmd: np.ndarray = field(default_factory=lambda: np.zeros(2))
    
    def __post_init__(self):
        """Ensure array is numpy array and clamp to valid range."""
        self.delta_cmd = np.asarray(self.delta_cmd, dtype=np.float64)
        self.delta_cmd = np.clip(self.delta_cmd, 0.0, 1.0)
    
    @property
    def delta_s(self) -> float:
        """Symmetric brake: affects lift/drag -> glide ratio, sink rate."""
        return (self.delta_cmd[0] + self.delta_cmd[1]) / 2.0
    
    @property
    def delta_a(self) -> float:
        """Asymmetric brake: affects roll/yaw -> heading, turn radius."""
        return self.delta_cmd[1] - self.delta_cmd[0]
    
    @classmethod
    def from_left_right(cls, delta_l: float, delta_r: float) -> 'ControlCmd':
        """Create control command from left and right brake values."""
        return cls(delta_cmd=np.array([delta_l, delta_r]))
