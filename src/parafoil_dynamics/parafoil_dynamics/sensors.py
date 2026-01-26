"""
Sensor simulation for parafoil.

Generates sensor measurements from true state with configurable noise.
Supports:
- Position (GPS-like)
- Accelerometer (body-frame specific force)
- Gyroscope (body-frame angular velocity)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple

from .state import State, ControlCmd
from .params import Params
from .math3d import quat_to_rotmat


@dataclass
class SensorConfig:
    """
    Configuration for sensor noise models.
    
    All noise values are standard deviations.
    """
    # Position sensor (GPS-like)
    position_noise_std: np.ndarray = field(
        default_factory=lambda: np.array([0.5, 0.5, 1.0])  # [m]
    )
    position_enabled: bool = True
    
    # Accelerometer
    accel_noise_std: np.ndarray = field(
        default_factory=lambda: np.array([0.02, 0.02, 0.02])  # [m/s^2]
    )
    accel_bias: np.ndarray = field(
        default_factory=lambda: np.zeros(3)  # [m/s^2]
    )
    accel_enabled: bool = True
    
    # Gyroscope
    gyro_noise_std: np.ndarray = field(
        default_factory=lambda: np.array([0.001, 0.001, 0.001])  # [rad/s]
    )
    gyro_bias: np.ndarray = field(
        default_factory=lambda: np.zeros(3)  # [rad/s]
    )
    gyro_enabled: bool = True
    
    # Random seed
    seed: Optional[int] = None
    
    def __post_init__(self):
        self.position_noise_std = np.asarray(self.position_noise_std, dtype=np.float64)
        self.accel_noise_std = np.asarray(self.accel_noise_std, dtype=np.float64)
        self.accel_bias = np.asarray(self.accel_bias, dtype=np.float64)
        self.gyro_noise_std = np.asarray(self.gyro_noise_std, dtype=np.float64)
        self.gyro_bias = np.asarray(self.gyro_bias, dtype=np.float64)


@dataclass
class SensorMeasurement:
    """
    Container for sensor measurements.
    
    Attributes:
        position: Position measurement in NED [m]
        body_acc: Body-frame acceleration (specific force) [m/s^2]
        body_ang_vel: Body-frame angular velocity [rad/s]
        timestamp: Measurement timestamp [s]
    """
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    body_acc: np.ndarray = field(default_factory=lambda: np.zeros(3))
    body_ang_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    timestamp: float = 0.0
    
    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=np.float64)
        self.body_acc = np.asarray(self.body_acc, dtype=np.float64)
        self.body_ang_vel = np.asarray(self.body_ang_vel, dtype=np.float64)


class SensorModel:
    """
    Sensor simulation model.
    
    Generates noisy measurements from true state.
    
    Usage:
        config = SensorConfig(accel_noise_std=[0.1, 0.1, 0.1])
        sensor = SensorModel(config)
        
        # In simulation loop:
        measurement = sensor.get_measurement(state, params, a_B)
    """
    
    def __init__(self, config: Optional[SensorConfig] = None):
        """
        Initialize sensor model.
        
        Args:
            config: Sensor configuration (uses defaults if None)
        """
        self.config = config if config is not None else SensorConfig()
        self.rng = np.random.default_rng(self.config.seed)
    
    def reset(self, seed: Optional[int] = None) -> None:
        """
        Reset sensor model (for new episode).
        
        Args:
            seed: New random seed
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        elif self.config.seed is not None:
            self.rng = np.random.default_rng(self.config.seed)
        else:
            self.rng = np.random.default_rng()
    
    def get_measurement(
        self,
        state: State,
        params: Params,
        body_acc: Optional[np.ndarray] = None
    ) -> SensorMeasurement:
        """
        Generate sensor measurements from true state.
        
        Args:
            state: True parafoil state
            params: Simulation parameters
            body_acc: Body-frame acceleration (if None, computed from gravity only)
            
        Returns:
            SensorMeasurement with noisy measurements
        """
        # Position measurement
        if self.config.position_enabled:
            position = self._measure_position(state)
        else:
            position = state.p_I.copy()
        
        # Accelerometer measurement
        if self.config.accel_enabled:
            if body_acc is None:
                # Compute specific force from gravity
                C_IB = quat_to_rotmat(state.q_IB)
                C_BI = C_IB.T
                g_I = np.array([0.0, 0.0, params.g])
                g_B = C_BI @ g_I
                body_acc = -g_B  # Specific force is opposite to gravity
            
            measured_acc = self._measure_accel(body_acc)
        else:
            measured_acc = body_acc if body_acc is not None else np.zeros(3)
        
        # Gyroscope measurement
        if self.config.gyro_enabled:
            measured_gyro = self._measure_gyro(state.w_B)
        else:
            measured_gyro = state.w_B.copy()
        
        return SensorMeasurement(
            position=position,
            body_acc=measured_acc,
            body_ang_vel=measured_gyro,
            timestamp=state.t
        )
    
    def _measure_position(self, state: State) -> np.ndarray:
        """Generate noisy position measurement."""
        noise = self.config.position_noise_std * self.rng.standard_normal(3)
        return state.p_I + noise
    
    def _measure_accel(self, true_acc: np.ndarray) -> np.ndarray:
        """Generate noisy accelerometer measurement."""
        noise = self.config.accel_noise_std * self.rng.standard_normal(3)
        return true_acc + self.config.accel_bias + noise
    
    def _measure_gyro(self, true_gyro: np.ndarray) -> np.ndarray:
        """Generate noisy gyroscope measurement."""
        noise = self.config.gyro_noise_std * self.rng.standard_normal(3)
        return true_gyro + self.config.gyro_bias + noise
    
    def get_true_body_acceleration(
        self,
        state: State,
        state_dot_v_I: np.ndarray,
        params: Params
    ) -> np.ndarray:
        """
        Compute true body-frame specific force (what accelerometer measures).
        
        Specific force = total acceleration - gravity
        
        Args:
            state: Current state
            state_dot_v_I: Inertial frame acceleration (v_I_dot)
            params: Simulation parameters
            
        Returns:
            Body-frame specific force [m/s^2]
        """
        # Gravity in inertial frame
        g_I = np.array([0.0, 0.0, params.g])
        
        # Specific force in inertial frame
        specific_force_I = state_dot_v_I - g_I
        
        # Transform to body frame
        C_IB = quat_to_rotmat(state.q_IB)
        C_BI = C_IB.T
        specific_force_B = C_BI @ specific_force_I
        
        return specific_force_B


def create_sensor_model_from_dict(config_dict: dict) -> SensorModel:
    """
    Create sensor model from configuration dictionary.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        SensorModel instance
    """
    config = SensorConfig(
        position_noise_std=np.array(
            config_dict.get('position_noise_std', [0.5, 0.5, 1.0])
        ),
        position_enabled=config_dict.get('position_enabled', True),
        accel_noise_std=np.array(
            config_dict.get('accel_noise_std', [0.02, 0.02, 0.02])
        ),
        accel_bias=np.array(
            config_dict.get('accel_bias', [0.0, 0.0, 0.0])
        ),
        accel_enabled=config_dict.get('accel_enabled', True),
        gyro_noise_std=np.array(
            config_dict.get('gyro_noise_std', [0.001, 0.001, 0.001])
        ),
        gyro_bias=np.array(
            config_dict.get('gyro_bias', [0.0, 0.0, 0.0])
        ),
        gyro_enabled=config_dict.get('gyro_enabled', True),
        seed=config_dict.get('seed', None)
    )
    
    return SensorModel(config)
