"""
Wind models for parafoil simulation.

Supports three layers of wind disturbance:
1. Steady wind: Constant wind vector
2. Gust: Discrete step gusts at random intervals
3. Colored wind: First-order filtered white noise (continuous spectrum)

All models support seeding for reproducibility and can be reset
for new episodes (domain randomization).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class WindConfig:
    """
    Configuration for wind model.
    
    Attributes:
        enable_steady: Enable steady wind component
        enable_gust: Enable discrete gust component
        enable_colored: Enable colored noise component
        
        steady_wind: Steady wind vector in NED [m/s]
        
        gust_interval: Mean interval between gusts [s]
        gust_duration: Gust duration [s]
        gust_magnitude: Maximum gust magnitude [m/s]
        
        colored_tau: Time constant for colored noise filter [s]
        colored_sigma: Standard deviation of colored noise [m/s]
        
        seed: Random seed for reproducibility (None for random)
    """
    # Enable flags (default: all off for clean simulation)
    enable_steady: bool = False
    enable_gust: bool = False
    enable_colored: bool = False
    
    # Steady wind (constant)
    steady_wind: np.ndarray = field(default_factory=lambda: np.array([2.0, 0.0, 0.0]))
    
    # Gust parameters
    gust_interval: float = 10.0       # Mean time between gusts [s]
    gust_duration: float = 2.0        # Gust duration [s]
    gust_magnitude: float = 3.0       # Max gust speed [m/s]
    
    # Colored noise parameters
    colored_tau: float = 2.0          # Filter time constant [s]
    colored_sigma: float = 1.0        # Noise standard deviation [m/s]
    
    # Random seed
    seed: Optional[int] = None
    
    def __post_init__(self):
        self.steady_wind = np.asarray(self.steady_wind, dtype=np.float64)


class WindModel:
    """
    Wind model combining steady, gust, and colored noise components.
    
    Usage:
        config = WindConfig(enable_steady=True, steady_wind=[2, 0, 0])
        wind = WindModel(config)
        wind.reset()  # Call at episode start
        
        # In simulation loop:
        wind_vec = wind.get_wind(t, dt)
    """
    
    def __init__(self, config: Optional[WindConfig] = None):
        """
        Initialize wind model.
        
        Args:
            config: Wind configuration (uses defaults if None)
        """
        self.config = config if config is not None else WindConfig()
        
        # Initialize random generator
        self.rng = np.random.default_rng(self.config.seed)
        
        # Gust state
        self._gust_active = False
        self._gust_start_time = 0.0
        self._gust_end_time = 0.0
        self._next_gust_time = 0.0
        self._current_gust = np.zeros(3)
        
        # Colored noise state
        self._colored_state = np.zeros(3)
        
        # Last update time
        self._last_t = 0.0
    
    def reset(self, seed: Optional[int] = None) -> None:
        """
        Reset wind model for a new episode.
        
        Args:
            seed: New random seed (uses config seed if None)
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        elif self.config.seed is not None:
            self.rng = np.random.default_rng(self.config.seed)
        else:
            self.rng = np.random.default_rng()
        
        # Reset gust state
        self._gust_active = False
        self._gust_start_time = 0.0
        self._gust_end_time = 0.0
        self._next_gust_time = self._sample_next_gust_time(0.0)
        self._current_gust = np.zeros(3)
        
        # Reset colored noise state
        self._colored_state = np.zeros(3)
        
        self._last_t = 0.0
    
    def get_wind(self, t: float, dt: float = 0.01) -> np.ndarray:
        """
        Get total wind vector at time t.
        
        Args:
            t: Current simulation time [s]
            dt: Time step for colored noise update [s]
            
        Returns:
            Wind velocity in NED frame [m/s]
        """
        wind = np.zeros(3)
        
        # Steady wind
        if self.config.enable_steady:
            wind += self.config.steady_wind
        
        # Gust wind
        if self.config.enable_gust:
            wind += self._get_gust(t)
        
        # Colored noise wind
        if self.config.enable_colored:
            wind += self._get_colored(t, dt)
        
        self._last_t = t
        return wind
    
    def _get_gust(self, t: float) -> np.ndarray:
        """Get gust component at time t."""
        # Check if current gust has ended
        if self._gust_active and t > self._gust_end_time:
            self._gust_active = False
            self._next_gust_time = self._sample_next_gust_time(t)
        
        # Check if new gust should start
        if not self._gust_active and t >= self._next_gust_time:
            self._gust_active = True
            self._gust_start_time = t
            self._gust_end_time = t + self.config.gust_duration
            self._current_gust = self._sample_gust_vector()
        
        # Return gust if active
        if self._gust_active:
            # Smooth ramp-up and ramp-down
            progress = (t - self._gust_start_time) / self.config.gust_duration
            
            # Use smooth window function
            window = self._smooth_window(progress)
            return window * self._current_gust
        
        return np.zeros(3)
    
    def _get_colored(self, t: float, dt: float) -> np.ndarray:
        """Get colored noise component at time t."""
        # First-order filter: dx/dt = -x/tau + sigma*sqrt(2/tau)*w
        # where w is white noise
        
        dt_actual = t - self._last_t
        if dt_actual <= 0:
            dt_actual = dt
        
        tau = self.config.colored_tau
        sigma = self.config.colored_sigma
        
        # Discrete update for first-order filter
        alpha = np.exp(-dt_actual / tau)
        noise_std = sigma * np.sqrt(1 - alpha**2)
        
        self._colored_state = (
            alpha * self._colored_state + 
            noise_std * self.rng.standard_normal(3)
        )
        
        return self._colored_state.copy()
    
    def _sample_next_gust_time(self, current_t: float) -> float:
        """Sample time for next gust (exponential distribution)."""
        interval = self.rng.exponential(self.config.gust_interval)
        return current_t + interval
    
    def _sample_gust_vector(self) -> np.ndarray:
        """Sample a random gust vector."""
        # Random direction on sphere
        direction = self.rng.standard_normal(3)
        direction = direction / (np.linalg.norm(direction) + 1e-10)
        
        # Random magnitude
        magnitude = self.rng.uniform(0, self.config.gust_magnitude)
        
        return magnitude * direction
    
    def _smooth_window(self, progress: float) -> float:
        """
        Smooth window function for gust ramp.
        
        Args:
            progress: 0 to 1 progress through gust
            
        Returns:
            Window value 0 to 1
        """
        # Hann window for smooth ramp
        return 0.5 * (1 - np.cos(2 * np.pi * progress))
    
    def set_steady_wind(self, wind: np.ndarray) -> None:
        """Set steady wind vector."""
        self.config.steady_wind = np.asarray(wind, dtype=np.float64)
    
    def randomize_steady_wind(
        self,
        mean: np.ndarray,
        std: float
    ) -> np.ndarray:
        """
        Randomize steady wind around a mean (for domain randomization).
        
        Args:
            mean: Mean wind vector
            std: Standard deviation of perturbation
            
        Returns:
            New steady wind vector
        """
        perturbation = std * self.rng.standard_normal(3)
        self.config.steady_wind = np.asarray(mean, dtype=np.float64) + perturbation
        return self.config.steady_wind.copy()


def create_wind_model_from_dict(config_dict: dict) -> WindModel:
    """
    Create wind model from configuration dictionary.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        WindModel instance
    """
    config = WindConfig(
        enable_steady=config_dict.get('enable_steady', False),
        enable_gust=config_dict.get('enable_gust', False),
        enable_colored=config_dict.get('enable_colored', False),
        steady_wind=np.array(config_dict.get('steady_wind', [0.0, 0.0, 0.0])),
        gust_interval=config_dict.get('gust_interval', 10.0),
        gust_duration=config_dict.get('gust_duration', 2.0),
        gust_magnitude=config_dict.get('gust_magnitude', 3.0),
        colored_tau=config_dict.get('colored_tau', 2.0),
        colored_sigma=config_dict.get('colored_sigma', 1.0),
        seed=config_dict.get('seed', None)
    )
    
    model = WindModel(config)
    model.reset()
    return model
